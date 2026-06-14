"""
Paired significance: 120M-SFT vs 66.8M-SFT on the SAME 725 (prompt, gold) pairs.
Recompute per-pair MiniLM cosine-to-gold for both models, take the paired difference
d = cos120 - cos66, and report mean, 95% bootstrap CI, paired-t, and win-rate.
Paired (same prompts) -> far more powerful than comparing the two aggregate means.
"""
import json, re, os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

EVALDIR = Path('D:/001_PROJECTS/psyche_diffuse_60m/raw_data/004_train/eval')
EMB = 'sentence-transformers/all-MiniLM-L6-v2'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
A_RUN = os.environ.get('PAIR_A', 'sft120_clean')   # reports d = cos(A) - cos(B)
B_RUN = os.environ.get('PAIR_B', 'sft66_clean')

def norm(q):
    return re.sub(r'[^a-z0-9 ]', '', q.lower()).strip()[:100]

def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]

def main():
    pairs = {norm(r['q']): r['gold'] for r in load_jsonl(EVALDIR / 'eval_pairs.jsonl')}
    rA = {norm(r['q']): r['resp'] for r in load_jsonl(EVALDIR / f'eval1000_{A_RUN}.jsonl')}
    rB = {norm(r['q']): r['resp'] for r in load_jsonl(EVALDIR / f'eval1000_{B_RUN}.jsonl')}
    keys = [k for k in pairs if k in rA and k in rB]
    golds = [pairs[k] for k in keys]
    a120 = [rA[k] for k in keys]
    a66 = [rB[k] for k in keys]
    print(f'paired n = {len(keys)}')

    tok = AutoTokenizer.from_pretrained(EMB, local_files_only=True)
    emb = AutoModel.from_pretrained(EMB, local_files_only=True).to(DEV).eval()

    @torch.no_grad()
    def E(texts):
        out = []
        for i in range(0, len(texts), 64):
            b = [t if t.strip() else '.' for t in texts[i:i+64]]
            enc = tok(b, padding=True, truncation=True, max_length=256, return_tensors='pt').to(DEV)
            h = emb(**enc).last_hidden_state
            m = enc['attention_mask'].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu())
        return torch.cat(out)

    G, A, B = E(golds), E(a120), E(a66)
    c120 = (A * G).sum(-1).numpy()
    c66 = (B * G).sum(-1).numpy()
    d = c120 - c66
    n = len(d)

    mean_d = float(d.mean())
    sd = float(d.std(ddof=1))
    t = mean_d / (sd / np.sqrt(n))
    rng = np.random.default_rng(0)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    win = float((d > 0).mean())

    print(f'cos[{A_RUN}] mean = {c120.mean():.4f}')
    print(f'cos[{B_RUN}] mean = {c66.mean():.4f}')
    print(f'paired d ({A_RUN} - {B_RUN}): mean={mean_d:+.4f}  sd={sd:.3f}')
    print(f'95% bootstrap CI : [{lo:+.4f}, {hi:+.4f}]')
    print(f'paired t({n-1})   : {t:+.2f}')
    print(f'win-rate A>B     : {win:.3f}')
    verdict = 'TIE (CI spans 0)' if lo < 0 < hi else (f'{A_RUN} better' if lo > 0 else f'{B_RUN} better')
    print(f'VERDICT          : {verdict}')

if __name__ == '__main__':
    main()
