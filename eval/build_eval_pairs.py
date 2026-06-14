"""
Attach gold answers to the eval questions -> eval_pairs.jsonl  {q, gold, gold_words}.
Keeps only questions whose source row has a REAL assistant answer (>=25 words, not an
imhi classification label). These pairs enable reference-based scoring (model response
vs a real answer). Golds are training answers -> similarity reflects target-recall, not
unbiased quality; only the 120M-vs-66.8M delta is interpretable.
"""
import json, re
from pathlib import Path

ROOT = Path('D:/001_PROJECTS/psyche_diffuse_60m/raw_data')
SRC = ROOT / 'extracted' / '03_sft_dialogue'
FILES = [SRC / f for f in ('en_sft_combined.jsonl', 'mh_sft_combined.jsonl',
                           'zh_psy_sft_translated_en_clean.jsonl')]
EVAL = ROOT / '004_train' / 'eval' / 'eval_1000_questions.txt'
OUT = ROOT / '004_train' / 'eval' / 'eval_pairs.jsonl'

ASSIST = re.compile(r'\n?\s*assistant\s*:', re.I)
USERPFX = re.compile(r'^\s*user\s*:\s*', re.I)
POST = re.compile(r'post\s*:\s*(.*?)\s*question\s*:', re.I | re.S)
WS = re.compile(r'\s+')
CLS = re.compile(r'(this post shows|the poster|mental disorder|no mental|yes[.,]|^(depression|anxiety|suicid))', re.I)

def norm(q):
    return re.sub(r'[^a-z0-9 ]', '', q.lower()).strip()[:100]

def answer_of(text):
    m = ASSIST.search(text)
    return WS.sub(' ', text[m.end():]).strip() if m else ''

def main():
    m = {}
    for path in FILES:
        if not path.exists():
            continue
        with path.open(encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                text = rec.get('text', '') or ''
                ans = answer_of(text)
                if not ans:
                    continue
                am = ASSIST.search(text)
                uq = WS.sub(' ', USERPFX.sub('', text[:am.start()] if am else text)).strip()
                if uq:
                    m.setdefault(norm(uq), ans)
                pm = POST.search(text)
                if pm:
                    m.setdefault(norm(WS.sub(' ', pm.group(1)).strip()), ans)

    qs = [l.strip() for l in EVAL.read_text(encoding='utf-8').splitlines() if l.strip()]
    pairs = []
    for q in qs:
        a = m.get(norm(q))
        if a and len(a.split()) >= 25 and not CLS.match(a):
            pairs.append({'q': q, 'gold': a, 'gold_words': len(a.split())})
    with OUT.open('w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f'eval questions : {len(qs)}')
    print(f'Q/A pairs kept : {len(pairs)}  -> {OUT.name}')

if __name__ == '__main__':
    main()
