"""
MDLM SFT — instruction fine-tune of the masked-diffusion LM (DiffuSeq-style).

Loads pretrained MDLM weights (66.8M, vocab_full=16001) and fine-tunes on
(prompt, response) pairs. Unlike the continuous-diffusion SFT, there are NO new
parameters — the architecture is identical to pretrain, so we load strict=True.

Conditioning (DiffuSeq for discrete diffusion):
  - Sequence packed as: [BOS, prompt..., SEP, response..., EOS, PAD...] length 512
  - mask.bin marks response positions (1 = response, incl. EOS)
  - Each training step: sample t ~ U(T_MIN, T_MAX); mask ONLY response positions
    with per-token prob t. Prompt + PAD stay clean (the conditioning signal).
  - Loss: CE on masked response positions only.

At sampling time (sample_mdlm.py with --prompt) the prompt header is clamped
clean and only response positions are iteratively unmasked — the exact mirror
of this training objective.

Outputs:
  - checkpoints/mdlm_sft/step_NNNNN/
  - logs/mdlm_sft.log

Run (after mdlm_v1 pretrain finishes):
  .venv/Scripts/python.exe raw_data/004_train/scripts/train_mdlm_sft.py
"""
import builtins
import io
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

ROOT = Path('D:/001_PROJECTS/psyche_diffuse_60m')
TRAIN_ROOT = ROOT / 'raw_data/004_train'
TOK_DIR = TRAIN_ROOT / 'tokenizer'
SFT_DATA_DIR = TRAIN_ROOT / 'datasets' / 'sft_en'

# RUN_NAME isolates this SFT run; PRETRAIN_RUN selects which pretrain to fine-tune.
# Defaults: SFT the 66.8M mdlm_v1. For the 120M pipeline set MDLM_SFT_RUN=mdlm_sft_120m
# MDLM_PRETRAIN=mdlm_120m and the matching arch env vars (see arch section).
RUN_NAME = os.environ.get('MDLM_SFT_RUN', 'mdlm_sft')
PRETRAIN_RUN = os.environ.get('MDLM_PRETRAIN', 'mdlm_v1')
PRETRAIN_CKPT = TRAIN_ROOT / 'checkpoints' / PRETRAIN_RUN / 'final'
CKPT_DIR = TRAIN_ROOT / 'checkpoints' / RUN_NAME
LOGS_DIR = TRAIN_ROOT / 'logs'
for d in (CKPT_DIR, LOGS_DIR): d.mkdir(parents=True, exist_ok=True)

# ---- vocabulary (must match train_mdlm.py) ----
VOCAB_CLEAN = 16_000
MASK_ID = VOCAB_CLEAN
VOCAB_FULL = VOCAB_CLEAN + 1

# ---- arch (MUST match the pretrain run being fine-tuned; env-overridable) ----
# 66.8M default:  HIDDEN=512 N_LAYERS=12 N_HEADS=8  MLP_INTER=1408
# 120M:           HIDDEN=640 N_LAYERS=15 N_HEADS=10 MLP_INTER=1664
SEQ_LEN = 512
HIDDEN = int(os.environ.get('MDLM_HIDDEN', 512))
N_LAYERS = int(os.environ.get('MDLM_LAYERS', 12))
N_HEADS = int(os.environ.get('MDLM_HEADS', 8))
MLP_INTER = int(os.environ.get('MDLM_MLP_INTER', 1408))
TIME_EMB_DIM = 256
MAX_POS = SEQ_LEN
assert HIDDEN % N_HEADS == 0, f'HIDDEN {HIDDEN} not divisible by N_HEADS {N_HEADS}'

# ---- SFT training ----
PER_DEVICE_BS = 32
ACCUM_STEPS = 8
LR = 1e-4              # lower than pretrain (4e-4) — fine-tuning
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.1
ADAM_B1, ADAM_B2 = 0.9, 0.99
MAX_GRAD_NORM = 1.0
MAX_STEPS = 8000
SAVE_STEPS = 500
LOG_STEPS = 20
SAVE_KEEP = 3
AUTO_RESUME = True
T_MIN = 0.01
T_MAX = 0.99

# ---- logger tee ----
_orig_print = builtins.print
_LOG_FH = open(LOGS_DIR / f'{RUN_NAME}.log', 'a', encoding='utf-8', buffering=1)
def _tee(*a, **kw):
    kw.setdefault('flush', True)
    _orig_print(*a, **kw)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        _LOG_FH.write(f'[{ts}] ' + ' '.join(str(x) for x in a) + '\n')
        _LOG_FH.flush(); os.fsync(_LOG_FH.fileno())
    except Exception:
        pass
builtins.print = _tee

print(f'=== MDLM SFT [{RUN_NAME}] start {datetime.now().isoformat(timespec="seconds")} ===')
print(f'pretrain ckpt: {PRETRAIN_CKPT}')
print(f'arch: HIDDEN={HIDDEN} N_LAYERS={N_LAYERS} N_HEADS={N_HEADS} MLP_INTER={MLP_INTER}')
print(f'data: {SFT_DATA_DIR}')

# ---- cuda ----
assert torch.cuda.is_available()
device = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print('cuda:', torch.cuda.get_device_name(0))


# ---- model (identical to train_mdlm.py) ----
class TimestepEmbedding(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = (t.float() * 1000.0)[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm.to(x.dtype)) * self.weight

class DiTAttention(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)
    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.out(out.transpose(1, 2).contiguous().view(B, N, D))

class SwiGLU(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.gate = nn.Linear(hidden, inter, bias=False)
        self.up = nn.Linear(hidden, inter, bias=False)
        self.down = nn.Linear(inter, hidden, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class DiTBlock(nn.Module):
    def __init__(self, hidden, n_heads, inter):
        super().__init__()
        self.norm1 = RMSNorm(hidden)
        self.attn = DiTAttention(hidden, n_heads)
        self.norm2 = RMSNorm(hidden)
        self.mlp = SwiGLU(hidden, inter)
        self.adaLN = nn.Linear(hidden, 6 * hidden, bias=True)
        nn.init.zeros_(self.adaLN.weight); nn.init.zeros_(self.adaLN.bias)
    def forward(self, x, c):
        s_a, sc_a, g_a, s_m, sc_m, g_m = self.adaLN(F.silu(c)).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + sc_a.unsqueeze(1)) + s_a.unsqueeze(1)
        x = x + g_a.unsqueeze(1) * self.attn(h)
        h = self.norm2(x) * (1 + sc_m.unsqueeze(1)) + s_m.unsqueeze(1)
        x = x + g_m.unsqueeze(1) * self.mlp(h)
        return x

class MDLMDenoiser(nn.Module):
    def __init__(self, vocab_full, hidden, n_layers, n_heads, inter, max_pos, time_dim):
        super().__init__()
        self.vocab_full = vocab_full
        self.tok_embed = nn.Embedding(vocab_full, hidden)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_pos, hidden))
        self.time_mlp = TimestepEmbedding(time_dim, hidden)
        self.blocks = nn.ModuleList([DiTBlock(hidden, n_heads, inter) for _ in range(n_layers)])
        self.final_norm = RMSNorm(hidden)
        self.final_adaLN = nn.Linear(hidden, 2 * hidden, bias=True)
        nn.init.zeros_(self.final_adaLN.weight); nn.init.zeros_(self.final_adaLN.bias)
        self.out_bias = nn.Parameter(torch.zeros(vocab_full))
        nn.init.normal_(self.tok_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.use_grad_ckpt = True

    def forward(self, ids, t):
        c = self.time_mlp(t)
        h = self.tok_embed(ids) + self.pos_embed[:, :ids.size(1), :]
        for blk in self.blocks:
            if self.training and self.use_grad_ckpt:
                h = torch.utils.checkpoint.checkpoint(blk, h, c, use_reentrant=False)
            else:
                h = blk(h, c)
        s, sc = self.final_adaLN(F.silu(c)).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + sc.unsqueeze(1)) + s.unsqueeze(1)
        logits = h @ self.tok_embed.weight.t() + self.out_bias
        return logits


# ---- data ----
meta = json.loads((SFT_DATA_DIR / 'meta.json').read_text())
print(f'SFT data: {meta["n_rows"]:,} rows, seq_len={meta["seq_len"]}')
assert meta['seq_len'] == SEQ_LEN
N_ROWS = meta['n_rows']
tok_arr = np.memmap(SFT_DATA_DIR / 'train.bin', dtype=np.uint16, mode='r').reshape(N_ROWS, SEQ_LEN)
mask_arr = np.memmap(SFT_DATA_DIR / 'mask.bin', dtype=np.uint8, mode='r').reshape(N_ROWS, SEQ_LEN)
print(f'mmap: tok shape={tok_arr.shape}, resp_mask shape={mask_arr.shape}')


class SFTDataset(IterableDataset):
    def __init__(self, tok_arr, mask_arr):
        self.tok = tok_arr
        self.mask = mask_arr
        self.n = len(tok_arr)
    def __iter__(self):
        rng = np.random.default_rng()
        while True:
            i = int(rng.integers(0, self.n))
            ids = self.tok[i].astype(np.int64)
            m = self.mask[i].astype(np.int64)
            yield torch.from_numpy(ids), torch.from_numpy(m)

loader = DataLoader(SFTDataset(tok_arr, mask_arr), batch_size=PER_DEVICE_BS, num_workers=0)

# ---- model build ----
model = MDLMDenoiser(VOCAB_FULL, HIDDEN, N_LAYERS, N_HEADS, MLP_INTER, MAX_POS, TIME_EMB_DIM)
model = model.to(torch.bfloat16).to(device)

# Load pretrain MDLM weights — identical arch, strict=True
if PRETRAIN_CKPT.exists():
    w = torch.load(PRETRAIN_CKPT / 'model.pt', map_location='cpu', weights_only=True)
    missing, unexpected = model.load_state_dict(w, strict=True)
    print(f'loaded pretrain MDLM strict=True (missing={missing}, unexpected={unexpected})')
else:
    print(f'WARNING: pretrain ckpt {PRETRAIN_CKPT} missing — training from scratch!')

# Auto-resume from SFT checkpoint
resume_from = None
resumed_step = 0
if AUTO_RESUME:
    sd = sorted((d for d in CKPT_DIR.glob('step_*') if d.is_dir()),
                key=lambda d: int(d.name.split('_')[-1]))
    if sd:
        resume_from = str(sd[-1])
        resumed_step = int(sd[-1].name.split('_')[-1])
        w = torch.load(Path(resume_from) / 'model.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(w, strict=True)
        print(f'[auto-resume] {resume_from} step={resumed_step}')

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'params: {n_params/1e6:.1f}M')

# ---- optimizer ----
try:
    from bitsandbytes.optim import AdamW8bit
    optim = AdamW8bit(model.parameters(), lr=LR, betas=(ADAM_B1, ADAM_B2), weight_decay=WEIGHT_DECAY)
    print('opt: AdamW8bit')
except Exception as e:
    print(f'AdamW8bit unavailable ({e}); using torch AdamW')
    optim = torch.optim.AdamW(model.parameters(), lr=LR, betas=(ADAM_B1, ADAM_B2), weight_decay=WEIGHT_DECAY)

from transformers import get_cosine_schedule_with_warmup
warmup_steps = int(WARMUP_RATIO * MAX_STEPS)
sched = get_cosine_schedule_with_warmup(optim, warmup_steps, MAX_STEPS)

if resume_from:
    state_pt = Path(resume_from) / 'training_state.pt'
    if state_pt.exists():
        state = torch.load(state_pt, map_location='cpu', weights_only=False)
        optim.load_state_dict(state['optimizer'])
        sched.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng_state'])
        if state.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng_state'])
        if state.get('numpy_rng_state') is not None:
            np.random.set_state(state['numpy_rng_state'])
        resumed_step = state['step']
        print(f'resumed: step={resumed_step}, lr={sched.get_last_lr()[0]:.2e}')


def save_checkpoint(ckpt_dir, step):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / 'model.pt')
    torch.save({
        'step': step,
        'optimizer': optim.state_dict(),
        'scheduler': sched.state_dict(),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'numpy_rng_state': np.random.get_state(),
    }, ckpt_dir / 'training_state.pt')

def rotate_checkpoints(keep=3):
    dirs = sorted((d for d in CKPT_DIR.glob('step_*') if d.is_dir()),
                  key=lambda d: int(d.name.split('_')[-1]))
    for d in dirs[:-keep]:
        shutil.rmtree(d, ignore_errors=True)


# ---- train ----
step = resumed_step
micro = 0
running = {'loss': 0.0, 'resp_mask_frac': 0.0, 'tok_acc': 0.0}
t0 = time.time()
model.train()
print(f'training: step={step} -> {MAX_STEPS}, save every {SAVE_STEPS}, accum={ACCUM_STEPS}')

try:
    for ids_clean, resp_mask in loader:
        ids_clean = ids_clean.to(device, non_blocking=True)   # (B, N) int64
        resp_mask = resp_mask.to(device, non_blocking=True)    # (B, N) int64, 1=response
        B, N = ids_clean.shape

        # Sample t per sequence; mask ONLY response positions with per-token prob t
        t_per_seq = torch.rand(B, device=device) * (T_MAX - T_MIN) + T_MIN  # (B,)
        mask_prob = t_per_seq.view(B, 1) * resp_mask.float()  # 0 outside response
        token_mask = torch.rand(B, N, device=device) < mask_prob  # (B, N) bool

        ids_masked = torch.where(token_mask, torch.tensor(MASK_ID, device=device), ids_clean)

        logits = model(ids_masked, t_per_seq)  # (B, N, V_full)
        flat_logits = logits.reshape(-1, VOCAB_FULL)
        flat_targets = ids_clean.reshape(-1)
        flat_mask = token_mask.reshape(-1).float()
        n_masked = flat_mask.sum().clamp_min(1.0)
        ce_per = F.cross_entropy(flat_logits, flat_targets, reduction='none')
        loss = (ce_per * flat_mask).sum() / n_masked

        (loss / ACCUM_STEPS).backward()

        running['loss'] += loss.item()
        running['resp_mask_frac'] += flat_mask.mean().item()
        with torch.no_grad():
            preds = flat_logits.argmax(dim=-1)
            tok_acc = ((preds == flat_targets).float() * flat_mask).sum() / n_masked
            running['tok_acc'] += tok_acc.item()
        micro += 1

        if micro % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            step += 1

            if step % LOG_STEPS == 0:
                elapsed = max(time.time() - t0, 1)
                done = step - resumed_step
                tok_per_sec = (done * ACCUM_STEPS * PER_DEVICE_BS * SEQ_LEN) / elapsed
                lr = sched.get_last_lr()[0]
                denom = LOG_STEPS * ACCUM_STEPS
                print(f'step {step}/{MAX_STEPS}  loss={running["loss"]/denom:.4f}  '
                      f'tok_acc={running["tok_acc"]/denom:.3f}  '
                      f'resp_mask={running["resp_mask_frac"]/denom:.3f}  '
                      f'lr={lr:.2e}  tok/s={tok_per_sec:.0f}')
                running = {k: 0.0 for k in running}

            if step % SAVE_STEPS == 0:
                ckpt = CKPT_DIR / f'step_{step:07d}'
                save_checkpoint(ckpt, step)
                rotate_checkpoints(keep=SAVE_KEEP)
                print(f'  saved: {ckpt}')

            if step >= MAX_STEPS:
                break
except KeyboardInterrupt:
    print(f'[interrupted at step {step}]')
finally:
    if step >= MAX_STEPS:
        final = CKPT_DIR / 'final'
        save_checkpoint(final, step)
        print(f'Done. Final: {final}')
    print(f'[end] step={step}, this session={step - resumed_step}')
