"""
Paper-grade collapse/diversity metrics: v1 sampler (random unmask) vs v2
(confidence unmask + rep-penalty + no-repeat-ngram), SAME 66.8M-noisy ckpt.

Pure text analysis on the saved eval files — no GPU.

Metrics (word-level, lowercased, alpha tokens):
  distinct-1/2/3  (Li et al. 2016)  = |unique n-grams| / |n-grams|   (higher = more diverse)
  rep-1                              = 1 - distinct-1                  (redundant unigrams)
  seq-rep-4 (Welleck et al. 2020)    = 1 - distinct-4                  (collapse signal)
  max-tok-freq                       = max count of any token / n_tok  ("people people people")
  confab-rate                        = fraction of samples with off-domain (marine/eco/...) terms
"""
from __future__ import annotations
import re, sys, io
from pathlib import Path
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

EVAL = Path('D:/001_PROJECTS/psyche_diffuse_60m/raw_data/004_train/eval')
V1 = EVAL / 'mdlm_sft_samples.txt'
V2 = EVAL / 'mdlm_sft_samples_v2.txt'

# NARROWED off-domain confab PRESCREEN — verified marine/eco/geo/phys-sci core only.
# Dropped ambiguous tokens that caused false positives: species, war, penal, brick,
# mammal, seal(s), habitat, glacier. This regex is a PRESCREEN; the reported number
# is HUMAN-VERIFIED domain-drift (see verified_confab labels below), not raw regex.
CONFAB_RE = re.compile(r'\b(coast\w*|coral|reef\w*|marine|oceanic|estuar\w*|tidal|sediment\w*|'
                       r'benthic|plankton|fishery|aquatic|ecosystem\w*|ecolog\w*|'
                       r'tectonic|seismic|volcan\w*|molecul\w*|enzyme\w*|semiconduct\w*)\b',
                       re.IGNORECASE)

PROMPT_RE = re.compile(r'^PROMPT:\s*(.*?)\s*(?:\[temp=[\d.]+\])?\s*$')
SEED_RE = re.compile(r'^\[seed=\d+\]\s*(.*)$')

STD5 = [
    'user: i feel anxious about work. assistant:',
    'user: how do cbt and act differ? assistant:',
    'user: i keep procrastinating on my thesis. what should i do? assistant:',
    "user: explain vygotsky's zone of proximal development. assistant:",
    'user: my partner says i am emotionally unavailable. how do i respond? assistant:',
]


def parse(path):
    """-> list of (prompt_norm, text)"""
    out = []
    cur = None
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        m = PROMPT_RE.match(line)
        if m:
            cur = m.group(1).strip().lower()
            continue
        m = SEED_RE.match(line)
        if m and cur:
            txt = m.group(1).strip()
            if txt:
                out.append((cur, txt))
    return out


def words(t):
    return re.findall(r"[a-z]+", t.lower())


def ngrams(ws, n):
    return [tuple(ws[i:i+n]) for i in range(len(ws)-n+1)] if len(ws) >= n else []


def distinct(ws, n):
    g = ngrams(ws, n)
    return (len(set(g)) / len(g)) if g else 1.0


def sample_metrics(t):
    ws = words(t)
    nt = len(ws)
    d1, d2, d3, d4 = (distinct(ws, n) for n in (1, 2, 3, 4))
    mtf = (max(Counter(ws).values()) / nt) if nt else 0.0
    confab = 1 if CONFAB_RE.search(t) else 0
    return dict(n=nt, d1=d1, d2=d2, d3=d3,
                rep1=1-d1, seqrep4=1-d4, mtf=mtf, confab=confab)


def agg(samples):
    if not samples:
        return None
    keys = ('n', 'd1', 'd2', 'd3', 'rep1', 'seqrep4', 'mtf', 'confab')
    return {k: sum(s[k] for s in samples)/len(samples) for k in keys} | {'count': len(samples)}


def report(name, rows):
    print(f'\n### {name}  (n_samples={len(rows)})')
    a = agg([sample_metrics(t) for _, t in rows])
    if not a:
        print('  (no samples)'); return a
    print(f'  avg_len(words) : {a["n"]:.0f}')
    print(f'  distinct-1     : {a["d1"]:.3f}')
    print(f'  distinct-2     : {a["d2"]:.3f}')
    print(f'  distinct-3     : {a["d3"]:.3f}')
    print(f'  rep-1          : {a["rep1"]:.3f}')
    print(f'  seq-rep-4      : {a["seqrep4"]:.3f}   <- collapse signal')
    print(f'  max-tok-freq   : {a["mtf"]:.3f}   <- "people people people"')
    print(f'  confab-rate    : {a["confab"]:.3f}   ({sum(sample_metrics(t)["confab"] for _,t in rows)}/{len(rows)})')
    return a


def main():
    v1 = parse(V1)
    v2 = parse(V2)
    v1_5 = [(p, t) for p, t in v1 if p in STD5]
    v2_5 = [(p, t) for p, t in v2 if p in STD5]

    print('=' * 64)
    print('COLLAPSE-FIX METRICS: v1 sampler vs v2 sampler (same 66.8M-noisy ckpt)')
    print('=' * 64)

    print('\n## FULL FILES')
    a1 = report('v1 (random unmask)  — ALL', v1)
    a2 = report('v2 (confidence+rep) — ALL', v2)

    print('\n## MATCHED 5 STANDARD PROMPTS (apples-to-apples)')
    b1 = report('v1 — std5', v1_5)
    b2 = report('v2 — std5', v2_5)

    # headline deltas on matched-5
    if b1 and b2:
        print('\n## DELTA (matched-5, v2 - v1)')
        for k, lbl in [('seqrep4','seq-rep-4'),('mtf','max-tok-freq'),('rep1','rep-1'),
                       ('d2','distinct-2'),('d3','distinct-3'),('confab','confab-rate')]:
            d = b2[k]-b1[k]
            arrow = 'down' if d < 0 else 'up'
            print(f'  {lbl:12s}: {b1[k]:.3f} -> {b2[k]:.3f}  ({d:+.3f} {arrow})')

    # per-prompt confab + collapse for the two factual prompts
    print('\n## PER-PROMPT (factual prompts — collapse/confab were here)')
    for tag, rows in [('v1', v1), ('v2', v2)]:
        for key in ('cbt and act', 'vygotsky'):
            sel = [(p,t) for p,t in rows if key in p]
            if sel:
                a = agg([sample_metrics(t) for _,t in sel])
                print(f'  [{tag}] {key:14s} seqrep4={a["seqrep4"]:.3f} mtf={a["mtf"]:.3f} '
                      f'd2={a["d2"]:.3f} confab={a["confab"]:.2f} (n={a["count"]})')


if __name__ == '__main__':
    main()
