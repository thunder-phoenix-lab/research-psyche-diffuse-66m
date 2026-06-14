"""
Export MDLM-SFT (66.8M, clean) to a mobile TorchScript .ptl for on-device inference.

- Loads mdlm_sft_clean/final (best 66.8M chat model).
- fp32 cast (mobile interpreter has no bf16), optional int8 dynamic-quant on Linear.
- TorchScript trace of forward(ids, t) -> logits.
- optimize_for_mobile + _save_for_lite_interpreter -> app assets.
- Copies spm_en.model into assets.

The iterative confidence-unmasking sampler runs in Kotlin (calls this forward ~N times).
"""
from __future__ import annotations
import math, shutil, sys, io
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path('D:/001_PROJECTS/psyche_diffuse_60m')
TRAIN = ROOT / 'raw_data/004_train'
CKPT = TRAIN / 'checkpoints' / 'mdlm_sft_clean' / 'final' / 'model.pt'
ASSETS = ROOT / 'experimental' / 'app' / 'src' / 'main' / 'assets'
ASSETS.mkdir(parents=True, exist_ok=True)

VOCAB_CLEAN = 16_000; MASK_ID = VOCAB_CLEAN; VOCAB_FULL = VOCAB_CLEAN + 1
SEQ_LEN = 512; HIDDEN = 512; N_LAYERS = 12; N_HEADS = 8; MLP_INTER = 1408; TIME_DIM = 256


class TimestepEmbedding(nn.Module):
    def __init__(s, dim, hidden):
        super().__init__(); s.dim = dim
        s.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
    def forward(s, t):
        half = s.dim // 2
        fr = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        a = (t.float() * 1000.0)[:, None] * fr[None, :]
        return s.mlp(torch.cat([torch.sin(a), torch.cos(a)], -1))

class RMSNorm(nn.Module):
    def __init__(s, d, eps=1e-5):
        super().__init__(); s.weight = nn.Parameter(torch.ones(d)); s.eps = eps
    def forward(s, x):
        n = x.float().pow(2).mean(-1, keepdim=True).add(s.eps).rsqrt()
        return (x * n) * s.weight

class Attn(nn.Module):
    def __init__(s, h, nh):
        super().__init__(); s.nh = nh; s.hd = h // nh
        s.qkv = nn.Linear(h, 3*h, bias=False); s.out = nn.Linear(h, h, bias=False)
    def forward(s, x):
        B, N, D = x.shape
        q, k, v = s.qkv(x).view(B, N, 3, s.nh, s.hd).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v)
        return s.out(o.transpose(1, 2).reshape(B, N, D))

class SwiGLU(nn.Module):
    def __init__(s, h, i):
        super().__init__(); s.gate = nn.Linear(h, i, bias=False); s.up = nn.Linear(h, i, bias=False); s.down = nn.Linear(i, h, bias=False)
    def forward(s, x): return s.down(F.silu(s.gate(x)) * s.up(x))

class Block(nn.Module):
    def __init__(s, h, nh, i):
        super().__init__(); s.norm1 = RMSNorm(h); s.attn = Attn(h, nh); s.norm2 = RMSNorm(h); s.mlp = SwiGLU(h, i)
        s.adaLN = nn.Linear(h, 6*h, bias=True)
    def forward(s, x, c):
        sa, sca, ga, sm, scm, gm = s.adaLN(F.silu(c)).chunk(6, -1)
        x = x + ga.unsqueeze(1) * s.attn(s.norm1(x) * (1+sca.unsqueeze(1)) + sa.unsqueeze(1))
        x = x + gm.unsqueeze(1) * s.mlp(s.norm2(x) * (1+scm.unsqueeze(1)) + sm.unsqueeze(1))
        return x

class MDLM(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok_embed = nn.Embedding(VOCAB_FULL, HIDDEN)
        s.pos_embed = nn.Parameter(torch.zeros(1, SEQ_LEN, HIDDEN))
        s.time_mlp = TimestepEmbedding(TIME_DIM, HIDDEN)
        s.blocks = nn.ModuleList([Block(HIDDEN, N_HEADS, MLP_INTER) for _ in range(N_LAYERS)])
        s.final_norm = RMSNorm(HIDDEN); s.final_adaLN = nn.Linear(HIDDEN, 2*HIDDEN, bias=True)
        s.out_bias = nn.Parameter(torch.zeros(VOCAB_FULL))
    def forward(s, ids, t):
        c = s.time_mlp(t)
        h = s.tok_embed(ids) + s.pos_embed[:, :ids.size(1), :]
        for b in s.blocks: h = b(h, c)
        sh, sc = s.final_adaLN(F.silu(c)).chunk(2, -1)
        h = s.final_norm(h) * (1+sc.unsqueeze(1)) + sh.unsqueeze(1)
        return h @ s.tok_embed.weight.t() + s.out_bias


def main():
    print(f'loading {CKPT}')
    m = MDLM().eval()
    w = torch.load(CKPT, map_location='cpu', weights_only=True)
    m.load_state_dict({k: v.float() for k, v in w.items()}, strict=True)
    m = m.float()
    print('params:', sum(p.numel() for p in m.parameters())/1e6, 'M')

    # int8 dynamic quant on Linear (shrinks ~4x, faster CPU)
    try:
        mq = torch.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)
        print('dynamic int8 quant: OK')
    except Exception as e:
        print('quant failed, fp32 fallback:', e); mq = m

    ex_ids = torch.randint(0, VOCAB_FULL, (1, SEQ_LEN), dtype=torch.long)
    ex_t = torch.tensor([0.5], dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(mq, (ex_ids, ex_t), check_trace=False)
    from torch.utils.mobile_optimizer import optimize_for_mobile
    opt = optimize_for_mobile(traced)
    out = ASSETS / 'mdlm_chat.ptl'
    opt._save_for_lite_interpreter(str(out))
    print(f'saved {out}  ({out.stat().st_size/1e6:.1f} MB)')

    spm = TRAIN / 'tokenizer' / 'spm_en.model'
    shutil.copy(spm, ASSETS / 'spm_en.model')
    print(f'copied tokenizer -> {ASSETS/"spm_en.model"}')

    # sanity: run traced forward once
    with torch.no_grad():
        o = opt(ex_ids, ex_t)
    print('forward OK, logits shape:', tuple(o.shape))


if __name__ == '__main__':
    main()
