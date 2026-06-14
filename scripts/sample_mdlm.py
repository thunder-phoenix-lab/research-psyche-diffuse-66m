"""
MDLM sampler — iterative unmasking.

Algorithm (Sahoo et al. 2024):
  Start: x = all [MASK].
  For k = K, K-1, ..., 1:
    t_k = k/K, t_prev = (k-1)/K
    logits = model(x, t_k)
    sample x0_hat per position (top-p + temp)
    unmask_prob = (t_k - t_prev) / t_k
    for each masked position, with prob unmask_prob: replace with x0_hat[pos]
  Final t=0: fill any remaining masks.

Supports:
  - Unconditional generation (no prompt, all positions masked)
  - Prefix conditioning: prompt tokens at start clamped to clean ids,
    remaining response positions masked.

Usage:
  .venv/Scripts/python.exe raw_data/004_train/scripts/sample_mdlm.py \
      --prompt "user: I feel anxious about work. assistant:" \
      --resp_len 128 --steps 128 --temperature 0.8 --top_p 0.92

  # Unconditional:
  .venv/Scripts/python.exe raw_data/004_train/scripts/sample_mdlm.py \
      --unconditional --resp_len 256 --steps 256
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import io
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path('D:/001_PROJECTS/psyche_diffuse_60m')
TRAIN_ROOT = ROOT / 'raw_data/004_train'
TOK_MODEL = TRAIN_ROOT / 'tokenizer' / 'spm_en.model'
DEFAULT_CKPT = TRAIN_ROOT / 'checkpoints' / 'mdlm_v1' / 'final'

VOCAB_CLEAN = 16_000
MASK_ID = VOCAB_CLEAN
VOCAB_FULL = VOCAB_CLEAN + 1
SEQ_LEN = 512
# Arch env-overridable — MUST match the checkpoint being sampled.
# 66.8M default: HIDDEN=512 N_LAYERS=12 N_HEADS=8 MLP_INTER=1408
# 120M:          HIDDEN=640 N_LAYERS=15 N_HEADS=10 MLP_INTER=1664
HIDDEN = int(os.environ.get('MDLM_HIDDEN', 512))
N_LAYERS = int(os.environ.get('MDLM_LAYERS', 12))
N_HEADS = int(os.environ.get('MDLM_HEADS', 8))
MLP_INTER = int(os.environ.get('MDLM_MLP_INTER', 1408))
TIME_EMB_DIM = 256
MAX_POS = SEQ_LEN

BOS_ID = 2
SEP_ID = 1
EOS_ID = 3
PAD_ID = 0


# ============================================================
# Model (must match train_mdlm.py exactly)
# ============================================================
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
        self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm.to(x.dtype)) * self.weight

class DiTAttention(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.n_heads = n_heads; self.head_dim = hidden // n_heads
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
        self.norm1 = RMSNorm(hidden); self.attn = DiTAttention(hidden, n_heads)
        self.norm2 = RMSNorm(hidden); self.mlp = SwiGLU(hidden, inter)
        self.adaLN = nn.Linear(hidden, 6 * hidden, bias=True)
    def forward(self, x, c):
        s_a, sc_a, g_a, s_m, sc_m, g_m = self.adaLN(F.silu(c)).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + sc_a.unsqueeze(1)) + s_a.unsqueeze(1)
        x = x + g_a.unsqueeze(1) * self.attn(h)
        h = self.norm2(x) * (1 + sc_m.unsqueeze(1)) + s_m.unsqueeze(1)
        x = x + g_m.unsqueeze(1) * self.mlp(h)
        return x

class MDLMDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embed = nn.Embedding(VOCAB_FULL, HIDDEN)
        self.pos_embed = nn.Parameter(torch.zeros(1, MAX_POS, HIDDEN))
        self.time_mlp = TimestepEmbedding(TIME_EMB_DIM, HIDDEN)
        self.blocks = nn.ModuleList([DiTBlock(HIDDEN, N_HEADS, MLP_INTER) for _ in range(N_LAYERS)])
        self.final_norm = RMSNorm(HIDDEN)
        self.final_adaLN = nn.Linear(HIDDEN, 2 * HIDDEN, bias=True)
        self.out_bias = nn.Parameter(torch.zeros(VOCAB_FULL))

    def forward(self, ids, t):
        c = self.time_mlp(t)
        h = self.tok_embed(ids) + self.pos_embed[:, :ids.size(1), :]
        for blk in self.blocks:
            h = blk(h, c)
        s, sc = self.final_adaLN(F.silu(c)).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + sc.unsqueeze(1)) + s.unsqueeze(1)
        logits = h @ self.tok_embed.weight.t() + self.out_bias
        return logits


# ============================================================
# Sampling utilities
# ============================================================
def sample_from_logits(logits_2d, temperature, top_p):
    """logits_2d: (N, V). Returns (N,) sampled ids."""
    if temperature <= 0:
        return logits_2d.argmax(dim=-1)
    logits = logits_2d.float() / max(temperature, 1e-4)
    # ban MASK_ID from being predicted
    logits[..., MASK_ID] = float('-inf')
    if 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumprobs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        mask = cumprobs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = torch.where(mask, torch.full_like(sorted_logits, float('-inf')), sorted_logits)
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = probs.sum(dim=-1, keepdim=True)
    bad = (row_sums.squeeze(-1) <= 0)
    if bad.any():
        # fallback to argmax for any row that became all-inf
        argmax_ids = logits_2d.float().argmax(dim=-1)
        ids = torch.multinomial(probs.clamp_min(1e-12), 1).squeeze(-1)
        ids = torch.where(bad, argmax_ids, ids)
    else:
        ids = torch.multinomial(probs, 1).squeeze(-1)
    return ids


@torch.no_grad()
def generate(model, sp, prompt: str | None, resp_len: int, n_steps: int,
             temperature: float, top_p: float, device, seed: int = 0,
             unconditional: bool = False):
    torch.manual_seed(seed)

    # Build initial token sequence and mask layout
    if unconditional or not prompt:
        # All masked
        total_len = min(resp_len, SEQ_LEN)
        ids = torch.full((1, total_len), MASK_ID, dtype=torch.long, device=device)
        resp_start = 0
        resp_end = total_len
        prompt_ids = []
        header = []
    else:
        prompt_ids = sp.encode(prompt, out_type=int)
        header = [BOS_ID] + prompt_ids + [SEP_ID]
        if len(header) >= SEQ_LEN - 16:
            keep = SEQ_LEN - 16 - 2
            prompt_ids = prompt_ids[-keep:]
            header = [BOS_ID] + prompt_ids + [SEP_ID]
        resp_start = len(header)
        resp_end = min(resp_start + resp_len, SEQ_LEN)
        total_len = resp_end
        ids = torch.full((1, total_len), MASK_ID, dtype=torch.long, device=device)
        ids[0, :len(header)] = torch.tensor(header, device=device)

    # Initial mask = positions currently equal to MASK_ID
    masked = (ids == MASK_ID)

    # Time grid: t goes 1 -> 0 over n_steps. Linear schedule.
    for k in range(n_steps, 0, -1):
        t_k = k / n_steps
        t_prev = (k - 1) / n_steps
        t_b = torch.tensor([t_k], device=device, dtype=torch.float32)

        logits = model(ids, t_b)  # (1, N, V)
        # Sample x0_hat for all positions
        flat_logits = logits[0]  # (N, V)
        x0_hat = sample_from_logits(flat_logits, temperature, top_p)  # (N,)
        x0_hat = x0_hat.unsqueeze(0)  # (1, N)

        # Decide which masked positions to unmask
        # MDLM: with prob (t_k - t_prev) / t_k, unmask
        if t_k > 0:
            unmask_prob = (t_k - t_prev) / t_k
        else:
            unmask_prob = 1.0
        rand = torch.rand_like(ids, dtype=torch.float32)
        do_unmask = masked & (rand < unmask_prob)
        ids = torch.where(do_unmask, x0_hat, ids)
        masked = (ids == MASK_ID)

        if not masked.any():
            break

    # Final pass: any remaining masks → argmax sample at t=0
    if masked.any():
        t_b = torch.tensor([0.001], device=device, dtype=torch.float32)
        logits = model(ids, t_b)
        x0_hat = sample_from_logits(logits[0], temperature, top_p).unsqueeze(0)
        ids = torch.where(masked, x0_hat, ids)

    # Decode response portion
    resp_ids = ids[0, resp_start:resp_end].tolist()
    # Clean special tokens
    clean = [t for t in resp_ids if t < VOCAB_CLEAN and t not in (PAD_ID, BOS_ID, SEP_ID)]
    # Stop at first EOS
    if EOS_ID in clean:
        clean = clean[:clean.index(EOS_ID)]
    text = sp.decode(clean)
    return {
        'text': text,
        'ids': resp_ids,
        'full_ids': ids[0].tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', type=str, default=None)
    ap.add_argument('--unconditional', action='store_true')
    ap.add_argument('--resp_len', type=int, default=128)
    ap.add_argument('--steps', type=int, default=128)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--top_p', type=float, default=0.92)
    ap.add_argument('--checkpoint', type=str, default=str(DEFAULT_CKPT))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_samples', type=int, default=1)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sp = spm.SentencePieceProcessor(); sp.load(str(TOK_MODEL))

    model = MDLMDenoiser().to(torch.bfloat16).to(device).eval()
    ckpt_path = Path(args.checkpoint) / 'model.pt'
    w = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(w, strict=True)
    print(f'loaded {ckpt_path}')

    for s in range(args.n_samples):
        out = generate(model, sp, args.prompt, args.resp_len, args.steps,
                       args.temperature, args.top_p, device, seed=args.seed + s,
                       unconditional=args.unconditional)
        print('=' * 70)
        print(f'[seed={args.seed + s}] {out["text"]}')


if __name__ == '__main__':
    main()
