"""
Feasibility: can we attach a GOLD answer to each of the 1000 eval questions?

Build norm(question) -> assistant answer from the 3 sft_en source files (matching BOTH
the conversational extraction AND the mh 'Post:' extraction we used), then look up every
line of eval_1000_questions.txt. Report match rate + answer-length distribution so we can
tell real answers apart from imhi classification labels (which are short / templated).
"""
import json, re
from pathlib import Path

ROOT = Path('D:/001_PROJECTS/psyche_diffuse_60m/raw_data')
SRC = ROOT / 'extracted' / '03_sft_dialogue'
FILES = [SRC / f for f in ('en_sft_combined.jsonl', 'mh_sft_combined.jsonl',
                           'zh_psy_sft_translated_en_clean.jsonl')]
EVAL = ROOT / '004_train' / 'eval' / 'eval_1000_questions.txt'

ASSIST = re.compile(r'\n?\s*assistant\s*:', re.I)
USERPFX = re.compile(r'^\s*user\s*:\s*', re.I)
POST = re.compile(r'post\s*:\s*(.*?)\s*question\s*:', re.I | re.S)
WS = re.compile(r'\s+')
CLS = re.compile(r'(this post shows|the poster|mental disorder|no mental|yes[.,]|depression|anxiety|suicid)', re.I)

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
                # conversational question (user turn up to assistant)
                am = ASSIST.search(text)
                uq = WS.sub(' ', USERPFX.sub('', text[:am.start()] if am else text)).strip()
                if uq and ans:
                    m.setdefault(norm(uq), ans)
                # mh 'Post:' disclosure variant
                pm = POST.search(text)
                if pm and ans:
                    m.setdefault(norm(WS.sub(' ', pm.group(1)).strip()), ans)

    qs = [l.strip() for l in EVAL.read_text(encoding='utf-8').splitlines() if l.strip()]
    matched = real = cls = 0
    lens = []
    for q in qs:
        a = m.get(norm(q))
        if a is None:
            continue
        matched += 1
        lens.append(len(a.split()))
        if len(a.split()) >= 25 and not CLS.match(a):
            real += 1
        else:
            cls += 1
    lens.sort()
    print(f'eval questions      : {len(qs)}')
    print(f'matched to a source : {matched}  ({matched/len(qs)*100:.0f}%)')
    print(f'  real answer (>=25w): {real}')
    print(f'  short/classific.   : {cls}')
    if lens:
        print(f'answer len(words) median={lens[len(lens)//2]} p10={lens[len(lens)//10]} p90={lens[len(lens)*9//10]}')

if __name__ == '__main__':
    main()
