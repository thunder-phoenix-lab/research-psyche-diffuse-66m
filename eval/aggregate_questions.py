"""Aggregate curated_*.txt shards -> deduped, shuffled, capped-to-1000 eval set."""
import re, random
from pathlib import Path

OUTDIR = Path('D:/001_PROJECTS/psyche_diffuse_60m/raw_data/004_train/eval/q_shards')
ALL = Path('D:/001_PROJECTS/psyche_diffuse_60m/raw_data/004_train/eval/eval_questions_all.txt')
FINAL = Path('D:/001_PROJECTS/psyche_diffuse_60m/raw_data/004_train/eval/eval_1000_questions.txt')
CAP = 1000

NUMPFX = re.compile(r'^\s*\d+[.)]\s*')
WS = re.compile(r'\s+')

def norm(q):
    return re.sub(r'[^a-z0-9 ]', '', q.lower()).strip()[:100]

def main():
    seen, out = set(), []
    files = sorted(OUTDIR.glob('curated_*.txt'))
    for f in files:
        for line in f.read_text(encoding='utf-8', errors='ignore').splitlines():
            q = WS.sub(' ', NUMPFX.sub('', line)).strip().strip('"').strip()
            if len(q) < 20:
                continue
            k = norm(q)
            if k in seen:
                continue
            seen.add(k)
            out.append(q)
    ALL.write_text('\n'.join(out) + '\n', encoding='utf-8')
    random.seed(7)
    random.shuffle(out)
    final = out[:CAP]
    FINAL.write_text('\n'.join(final) + '\n', encoding='utf-8')
    print(f'curated files: {len(files)}')
    print(f'unique total: {len(out)}  (-> {ALL.name})')
    print(f'final capped:  {len(final)}  (-> {FINAL.name})')

if __name__ == '__main__':
    main()
