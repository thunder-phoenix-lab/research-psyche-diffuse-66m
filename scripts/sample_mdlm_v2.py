"""
MDLM sampler v2 — confidence-based unmasking + repetition control.

Three additions over sample_mdlm.py (no weight changes — same checkpoint):

  1. CONFIDENCE-BASED UNMASKING (MaskGIT / Chang et al. 2022, standard for MDLM):
     Instead of randomly unmasking positions each step, compute the model's
     confidence (max prob) at every masked position and unmask the MOST
     confident ones first. Number unmasked per step follows a cosine schedule.
     This is the single biggest decode-quality lever for masked diffusion.

  2. REPETITION PENALTY (global, over already-committed tokens) + NEIGHBOR BAN:
     When scoring a masked position, penalize tokens already committed in the
     response, and hard-ban a token equal to an immediate neighbour (±1, ±2).
     Directly kills the "people people people" / "ACT-CT-CT" collapse.

  3. NO-REPEAT-NGRAM (size N): ban a candidate token if committing it would
     complete a contiguous N-gram that already exists in the committed response.

Time conditioning: t = (#masked response positions) / (response length), which
matches the training objective (t ≈ fraction of response masked).

Usage:
  .venv/Scripts/python.exe raw_data/004_train/scripts/sample_mdlm_v2.py \
      --prompt "user: I feel lonely even around people. assistant:" \
      --resp_len 110 --steps 110 --temperature 0.9 --top_p 0.9 \
      --rep_penalty 1.3 --no_repeat_ngram 3 --neighbor_ban 2

Env arch vars (match checkpoint): MDLM_HIDDEN / MDLM_LAYERS / MDLM_HEADS / MDLM_MLP_INTER.
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
DEFAULT_CKPT = TRAIN_ROOT / 'checkpoints' / 'mdlm_sft' / 'final'

VOCAB_CLEAN = 16_000
MASK_ID = VOCAB_CLEAN
VOCAB_FULL = VOCAB_CLEAN + 1
SEQ_LEN = 512
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
# Model (identical to train_mdlm.py)
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
# v2 decoding
# ============================================================
def apply_repetition_controls(logits_row, committed_tokens, pos, ids_list,
                              rep_penalty, neighbor_ban, no_repeat_ngram):
    """
    logits_row: (V,) float logits for one masked position `pos`.
    committed_tokens: set of token ids already committed in the response.
    ids_list: full current id list (python), for neighbor/ngram checks.
    Returns modified logits_row.
    """
    logits = logits_row.clone()
    logits[MASK_ID] = float('-inf')  # never emit MASK

    # 1) global repetition penalty over committed tokens
    if rep_penalty != 1.0 and committed_tokens:
        idx = torch.tensor(sorted(committed_tokens), device=logits.device, dtype=torch.long)
        vals = logits[idx]
        vals = torch.where(vals > 0, vals / rep_penalty, vals * rep_penalty)
        logits[idx] = vals

    # 2) neighbor ban — hard-ban tokens equal to committed neighbours within ±neighbor_ban
    if neighbor_ban > 0:
        for d in range(1, neighbor_ban + 1):
            for nb in (pos - d, pos + d):
                if 0 <= nb < len(ids_list):
                    tok = ids_list[nb]
                    if tok != MASK_ID and tok < VOCAB_CLEAN:
                        logits[tok] = float('-inf')

    # 3) no-repeat-ngram — ban candidate that completes a repeated contiguous N-gram
    #    using the committed left context [pos-(N-1) .. pos-1]
    if no_repeat_ngram and no_repeat_ngram > 1:
        n = no_repeat_ngram
        left = ids_list[max(0, pos - (n - 1)):pos]
        if len(left) == n - 1 and all(t != MASK_ID for t in left):
            # find all existing n-grams in the committed sequence starting with `left`
            seq = ids_list
            banned = set()
            for j in range(len(seq) - n + 1):
                if seq[j:j + n - 1] == left and seq[j + n - 1] != MASK_ID:
                    banned.add(seq[j + n - 1])
            for b in banned:
                if b < VOCAB_FULL:
                    logits[b] = float('-inf')
    return logits


def filter_top_p(logits, top_p):
    if not (0 < top_p < 1.0):
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumprobs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    mask = cumprobs > top_p
    mask[1:] = mask[:-1].clone()
    mask[0] = False
    sorted_logits[mask] = float('-inf')
    out = torch.full_like(logits, float('-inf'))
    out[sorted_idx] = sorted_logits
    return out


@torch.no_grad()
def generate(model, sp, prompt: str | None, resp_len: int, n_steps: int,
             temperature: float, top_p: float, rep_penalty: float,
             neighbor_ban: int, no_repeat_ngram: int, device, seed: int = 0,
             unconditional: bool = False):
    torch.manual_seed(seed)

    if unconditional or not prompt:
        total_len = min(resp_len, SEQ_LEN)
        ids = torch.full((1, total_len), MASK_ID, dtype=torch.long, device=device)
        resp_start, resp_end = 0, total_len
        header = []
    else:
        prompt_ids = sp.encode(prompt, out_type=int)
        header = [BOS_ID] + prompt_ids + [SEP_ID]
        if len(header) >= SEQ_LEN - 16:
            keep = SEQ_LEN - 16 - 2
            header = [BOS_ID] + prompt_ids[-keep:] + [SEP_ID]
        resp_start = len(header)
        resp_end = min(resp_start + resp_len, SEQ_LEN)
        total_len = resp_end
        ids = torch.full((1, total_len), MASK_ID, dtype=torch.long, device=device)
        ids[0, :len(header)] = torch.tensor(header, device=device)

    resp_positions = list(range(resp_start, resp_end))
    n_resp = len(resp_positions)

    def n_masked_now():
        return int((ids[0, resp_start:resp_end] == MASK_ID).sum().item())

    for step in range(n_steps):
        cur_masked = n_masked_now()
        if cur_masked == 0:
            break
        # time = fraction of response still masked
        t_val = cur_masked / n_resp
        t_b = torch.tensor([t_val], device=device, dtype=torch.float32)
        logits = model(ids, t_b)[0].float()  # (N, V)

        # cosine schedule: how many masked positions should REMAIN after this step
        frac_after = math.cos(math.pi / 2 * (step + 1) / n_steps)
        target_remaining = int(math.floor(n_resp * frac_after))
        n_to_unmask = max(1, cur_masked - target_remaining)
        n_to_unmask = min(n_to_unmask, cur_masked)

        ids_list = ids[0].tolist()
        committed = set(t for t in ids_list[resp_start:resp_end] if t != MASK_ID and t < VOCAB_CLEAN)

        # score every masked position: pick a token + confidence
        cand_positions = [p for p in resp_positions if ids_list[p] == MASK_ID]
        chosen_tok = {}
        chosen_conf = {}
        for p in cand_positions:
            lg = apply_repetition_controls(logits[p], committed, p, ids_list,
                                           rep_penalty, neighbor_ban, no_repeat_ngram)
            lg_t = lg / max(temperature, 1e-4)
            lg_t = filter_top_p(lg_t, top_p)
            probs = torch.softmax(lg_t, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            if probs.sum() <= 0:
                tok = int(lg.argmax().item()); conf = 0.0
            else:
                tok = int(torch.multinomial(probs, 1).item())
                conf = float(probs[tok].item())
            chosen_tok[p] = tok
            chosen_conf[p] = conf

        # unmask the top-confidence positions
        ranked = sorted(cand_positions, key=lambda p: chosen_conf[p], reverse=True)
        for p in ranked[:n_to_unmask]:
            ids[0, p] = chosen_tok[p]

    # fill any leftover masks greedily
    if n_masked_now() > 0:
        t_b = torch.tensor([0.001], device=device, dtype=torch.float32)
        logits = model(ids, t_b)[0].float()
        ids_list = ids[0].tolist()
        for p in resp_positions:
            if ids_list[p] == MASK_ID:
                lg = logits[p].clone(); lg[MASK_ID] = float('-inf')
                ids[0, p] = int(lg.argmax().item())

    resp_ids = ids[0, resp_start:resp_end].tolist()
    clean = [t for t in resp_ids if t < VOCAB_CLEAN and t not in (PAD_ID, BOS_ID, SEP_ID)]
    if EOS_ID in clean:
        clean = clean[:clean.index(EOS_ID)]
    return sp.decode(clean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', type=str, default=None)
    ap.add_argument('--unconditional', action='store_true')
    ap.add_argument('--resp_len', type=int, default=110)
    ap.add_argument('--steps', type=int, default=110)
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--rep_penalty', type=float, default=1.3)
    ap.add_argument('--neighbor_ban', type=int, default=2)
    ap.add_argument('--no_repeat_ngram', type=int, default=3)
    ap.add_argument('--checkpoint', type=str, default=str(DEFAULT_CKPT))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_samples', type=int, default=1)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sp = spm.SentencePieceProcessor(); sp.load(str(TOK_MODEL))

    model = MDLMDenoiser().to(torch.bfloat16).to(device).eval()
    w = torch.load(Path(args.checkpoint) / 'model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(w, strict=True)
    print(f'loaded {args.checkpoint}  (HIDDEN={HIDDEN} L={N_LAYERS})')

    for s in range(args.n_samples):
        txt = generate(model, sp, args.prompt, args.resp_len, args.steps,
                       args.temperature, args.top_p, args.rep_penalty,
                       args.neighbor_ban, args.no_repeat_ngram, device,
                       seed=args.seed + s, unconditional=args.unconditional)
        print('=' * 70)
        print(f'[seed={args.seed + s}] {txt}')


if __name__ == '__main__':
    main()
