"""
Score the blind scale-traps annotation: decode S-ids -> source (120m / 66m), compute
per-source DRIFT rate per annotator + consensus (>=2 of 3), Wilson 95% CIs, Newcombe diff
CI, Fisher exact two-sided, and pairwise Cohen kappa (inter-annotator agreement).
Numpy/stdlib only.
"""
import json, math
from pathlib import Path

EVAL = Path('D:/001_PROJECTS/psyche_diffuse_60m/raw_data/004_train/eval')
KEY = json.loads((EVAL / 'blind_traps_scale_key.json').read_text(encoding='utf-8'))
RATERS = ['A', 'B', 'C']

def load_annot(r):
    d = {}
    for line in (EVAL / f'annot_{r}.txt').read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if ':' in line and line.upper().startswith('S'):
            sid, v = line.split(':', 1)
            d[sid.strip()] = 1 if 'yes' in v.strip().lower() else 0
    return d

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return p, max(0, c-h), min(1, c+h)

def newcombe(k1, n1, k2, n2, z=1.96):
    p1, l1, u1 = wilson(k1, n1, z)
    p2, l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1-l1)**2 + (u2-p2)**2)
    hi = d + math.sqrt((u1-p1)**2 + (p2-l2)**2)
    return d, lo, hi

def fisher_two_sided(a, b, c, d):
    n = a+b+c+d
    r1, r2, c1 = a+b, c+d, a+c
    def hyp(x):
        return (math.comb(r1, x) * math.comb(r2, c1-x)) / math.comb(n, c1)
    lo, hi = max(0, c1-r2), min(r1, c1)
    p_obs = hyp(a)
    return sum(hyp(x) for x in range(lo, hi+1) if hyp(x) <= p_obs + 1e-12)

def cohen_kappa(d1, d2):
    ids = [s for s in d1 if s in d2]
    n = len(ids)
    both1 = sum(1 for s in ids if d1[s]==1 and d2[s]==1)
    both0 = sum(1 for s in ids if d1[s]==0 and d2[s]==0)
    po = (both1+both0)/n
    p1a = sum(d1[s] for s in ids)/n; p1b = sum(d2[s] for s in ids)/n
    pe = p1a*p1b + (1-p1a)*(1-p1b)
    return (po-pe)/(1-pe) if pe < 1 else 1.0

def rate_by_source(annot):
    out = {'120m': [0,0], '66m': [0,0]}   # [drift, total]
    for sid, src in ((s, KEY[s]['corpus']) for s in KEY):
        if sid in annot:
            out[src][1] += 1
            out[src][0] += annot[sid]
    return out

def main():
    annots = {r: load_annot(r) for r in RATERS}
    print('=== per-annotator DRIFT rate by source (n=28 each) ===')
    for r in RATERS:
        rs = rate_by_source(annots[r])
        a = rs['120m']; b = rs['66m']
        print(f'  {r}: 120m {a[0]}/{a[1]}={a[0]/a[1]:.2f}   66m {b[0]}/{b[1]}={b[0]/b[1]:.2f}')

    print('\n=== inter-annotator agreement (Cohen kappa) ===')
    for i in range(len(RATERS)):
        for j in range(i+1, len(RATERS)):
            k = cohen_kappa(annots[RATERS[i]], annots[RATERS[j]])
            print(f'  kappa({RATERS[i]},{RATERS[j]}) = {k:.3f}')

    # consensus = majority (>=2 of 3)
    cons = {}
    for sid in KEY:
        votes = [annots[r].get(sid) for r in RATERS if sid in annots[r]]
        if votes:
            cons[sid] = 1 if sum(votes) >= 2 else 0
    rs = rate_by_source(cons)
    a, b = rs['120m'], rs['66m']
    p1, l1, u1 = wilson(a[0], a[1]); p2, l2, u2 = wilson(b[0], b[1])
    d, dlo, dhi = newcombe(a[0], a[1], b[0], b[1])
    fp = fisher_two_sided(a[0], a[1]-a[0], b[0], b[1]-b[0])

    print('\n=== CONSENSUS (majority of 3) — SCALE comparison ===')
    print(f'  120M-clean drift : {a[0]}/{a[1]} = {p1:.3f}  Wilson95 [{l1:.3f}, {u1:.3f}]')
    print(f'  66.8M-clean drift: {b[0]}/{b[1]} = {p2:.3f}  Wilson95 [{l2:.3f}, {u2:.3f}]')
    print(f'  diff (120-66)    : {d:+.3f}  Newcombe95 [{dlo:+.3f}, {dhi:+.3f}]')
    print(f'  Fisher exact 2-sided p = {fp:.3f}')
    verdict = 'NO scale effect (CI spans 0, p>0.05)' if (dlo < 0 < dhi or fp > 0.05) else 'scale effect present'
    print(f'  VERDICT: {verdict}')

if __name__ == '__main__':
    main()
