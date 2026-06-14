"""
1000-question automated eval for an MDLM-SFT checkpoint.

Loads a model ONCE, generates a response to each of the 1000 in-distribution psych
prompts (eval_1000_questions.txt) with the v2 sampler (confidence-unmask + rep-controls),
then computes diversity/collapse + degeneration metrics. Run it twice with matched
settings (66.8M-clean and 120M-clean) to read off the scale effect.

Env:
  EVAL_RUN     label for output files (e.g. sft66_clean / sft120_clean)
  EVAL_CKPT    checkpoint dir containing model.pt
  EVAL_DEVICE  cuda|cpu  (default cuda if available)
  EVAL_LIMIT   cap #prompts (0 = all; use small for smoke tests)
  MDLM_HIDDEN/MDLM_LAYERS/MDLM_HEADS/MDLM_MLP_INTER  arch (must match ckpt)

Sampler settings are HARD-CODED here so the two model runs are exactly matched.
"""
from __future__ import annotations
import os, sys, re, json, time
from pathlib import Path

ROOT = Path('D:/001_PROJECTS/psyche_diffuse_60m')
SCRIPTS = ROOT / 'raw_data/004_train/scripts'
EVALDIR = ROOT / 'raw_data/004_train/eval'
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(EVALDIR))

import torch
import sentencepiece as spm
# Both modules below reassign sys.stdout to a fresh TextIOWrapper at import. Hold a
# reference to the first one across the second import, else it is GC'd and closes the
# shared stdout buffer ("I/O operation on closed file").
import sample_mdlm_v2 as S
_KEEP_STDOUT = sys.stdout
from compute_collapse_metrics import sample_metrics

# ---- matched sampler settings (identical for every model we compare) ----
RESP_LEN, STEPS = 110, 28
TEMP, TOP_P, REP, NB, NRG = 0.9, 0.9, 1.3, 2, 3

# ---- reference-based (neural cosine to gold answer) ----
EMBED_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
PAIRS_FILE = EVALDIR / 'eval_pairs.jsonl'

def _norm(q):
    return re.sub(r'[^a-z0-9 ]', '', q.lower()).strip()[:100]

def embed_cosines(responses, golds, device):
    """Mean-pooled MiniLM cosine between each response and its gold. Cached model, no net."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(EMBED_NAME, local_files_only=True)
    emb = AutoModel.from_pretrained(EMBED_NAME, local_files_only=True).to(device).eval()

    @torch.no_grad()
    def embed(texts):
        vecs = []
        for i in range(0, len(texts), 64):
            batch = [t if t.strip() else '.' for t in texts[i:i+64]]
            enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            out = emb(**enc).last_hidden_state
            mask = enc['attention_mask'].unsqueeze(-1).float()
            v = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vecs.append(torch.nn.functional.normalize(v, dim=-1).cpu())
        return torch.cat(vecs)

    R, G = embed(responses), embed(golds)
    return (R * G).sum(-1).tolist()   # cosine per pair (unit vectors)

RUN = os.environ.get('EVAL_RUN', 'run')
CKPT = Path(os.environ.get('EVAL_CKPT', str(S.DEFAULT_CKPT)))
DEVICE = os.environ.get('EVAL_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
LIMIT = int(os.environ.get('EVAL_LIMIT', '0'))

def nonascii_frac(t):
    return sum(1 for c in t if ord(c) > 127) / max(1, len(t))

def main():
    qs = [l.strip() for l in (EVALDIR / 'eval_1000_questions.txt').read_text(encoding='utf-8').splitlines() if l.strip()]
    if LIMIT:
        qs = qs[:LIMIT]
    print(f'[{RUN}] device={DEVICE} ckpt={CKPT} n={len(qs)} arch H={S.HIDDEN} L={S.N_LAYERS}', flush=True)

    sp = spm.SentencePieceProcessor(); sp.load(str(S.TOK_MODEL))
    model = S.MDLMDenoiser().to(torch.bfloat16).to(DEVICE).eval()
    w = torch.load(CKPT / 'model.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(w, strict=True)

    out_jsonl = EVALDIR / f'eval1000_{RUN}.jsonl'
    results, t0 = [], time.time()
    with out_jsonl.open('w', encoding='utf-8') as f:
        for i, q in enumerate(qs):
            resp = S.generate(model, sp, f'user: {q} assistant:', RESP_LEN, STEPS,
                              TEMP, TOP_P, REP, NB, NRG, DEVICE, seed=0, unconditional=False)
            results.append(resp)
            f.write(json.dumps({'q': q, 'resp': resp}, ensure_ascii=False) + '\n')
            if (i + 1) % 50 == 0:
                dt = time.time() - t0
                print(f'  {i+1}/{len(qs)}  {dt/(i+1):.2f}s/q  eta {dt/(i+1)*(len(qs)-i-1)/60:.1f}min', flush=True)

    ms = [sample_metrics(r) for r in results]
    n = len(ms)
    def avg(k): return sum(m[k] for m in ms) / n
    empty = sum(1 for r in results if len(r.strip()) < 3) / n
    langmix = sum(1 for r in results if nonascii_frac(r) > 0.05) / n
    confab_ct = sum(m['confab'] for m in ms)

    lines = [
        f'### 1000-Q EVAL — {RUN}  (n={n}, ckpt={CKPT.parent.name}/{CKPT.name})',
        f'sampler: resp_len={RESP_LEN} steps={STEPS} temp={TEMP} top_p={TOP_P} rep={REP} nb={NB} nrg={NRG}',
        f'avg_len(words) : {avg("n"):.1f}',
        f'distinct-1     : {avg("d1"):.3f}',
        f'distinct-2     : {avg("d2"):.3f}',
        f'distinct-3     : {avg("d3"):.3f}',
        f'rep-1          : {avg("rep1"):.3f}',
        f'seq-rep-4      : {avg("seqrep4"):.3f}   <- collapse signal',
        f'max-tok-freq   : {avg("mtf"):.3f}',
        f'confab-rate    : {avg("confab"):.3f}   ({confab_ct}/{n} prescreen)',
        f'empty-rate     : {empty:.3f}',
        f'lang-mix-rate  : {langmix:.3f}   (>5% non-ascii)',
    ]

    # reference-based: neural cosine vs gold (only prompts that have a real gold answer)
    if PAIRS_FILE.exists():
        golds = {}
        for ln in PAIRS_FILE.read_text(encoding='utf-8').splitlines():
            if ln.strip():
                p = json.loads(ln); golds[_norm(p['q'])] = p['gold']
        pr, pg = [], []
        for q, r in zip(qs, results):
            g = golds.get(_norm(q))
            if g:
                pr.append(r); pg.append(g)
        if pr:
            print(f'  embedding {len(pr)} paired (response, gold) for cosine...', flush=True)
            cos = sorted(embed_cosines(pr, pg, DEVICE))
            k = len(cos)
            lines += [
                f'--- reference-based (MiniLM cosine vs gold, n_paired={k}) ---',
                f'cos_mean       : {sum(cos)/k:.3f}',
                f'cos_median     : {cos[k//2]:.3f}',
                f'cos_p10 / p90  : {cos[k//10]:.3f} / {cos[k*9//10]:.3f}',
            ]

    summary = '\n'.join(lines)
    (EVALDIR / f'eval1000_{RUN}_metrics.txt').write_text(summary + '\n', encoding='utf-8')
    print('\n' + summary, flush=True)
    print(f'\n-> {out_jsonl.name} + eval1000_{RUN}_metrics.txt', flush=True)

if __name__ == '__main__':
    main()
