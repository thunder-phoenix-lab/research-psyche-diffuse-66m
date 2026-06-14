"""
Build candidate question shards for the 1000-question in-distribution eval.

Source = the EXACT 3 files that built sft_en (so questions are in-distribution):
  extracted/03_sft_dialogue/{en_sft_combined, mh_sft_combined, zh_psy_sft_translated_en_clean}.jsonl

We extract the USER prompt (everything up to the first assistant turn), clean it,
psych-filter the general `en` set, balance across sources, dedup, shuffle (seed=42),
and write 20 shards x 150 candidates -> raw_data/004_train/eval/q_shards/shard_NN.txt.
Each shard also keeps source tags in q_shards/shard_NN.jsonl for traceability.
20 sub-agents then curate the best 50 distinct, well-formed psychology questions per shard
-> 1000 total.
"""
import json, re, random, sys
from pathlib import Path

ROOT = Path('D:/001_PROJECTS/psyche_diffuse_60m/raw_data')
SRC = ROOT / 'extracted' / '03_sft_dialogue'
OUT = ROOT / '004_train' / 'eval' / 'q_shards'
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    'en':  SRC / 'en_sft_combined.jsonl',
    'mh':  SRC / 'mh_sft_combined.jsonl',
    'zh':  SRC / 'zh_psy_sft_translated_en_clean.jsonl',
}

N_SHARDS = 20
TARGET = 3000                   # candidate pool target (sub-agent picks best 50/shard)
MIN_LEN, MAX_LEN = 25, 850      # chars of the cleaned question (psyqa disclosures run long)

# psych/MH relevance gate (only applied to the general `en` set)
PSY = re.compile(r'\b(anxi|depress|therap|psycholog|emotion|stress|trauma|grief|'
                 r'cope|coping|mental health|self[- ]?esteem|panic|lonely|loneli|'
                 r'anger|burnout|insomnia|addict|relationship|breakup|self[- ]?harm|'
                 r'suicid|ptsd|ocd|adhd|bipolar|schizo|counsel|mindful|cbt|'
                 r'feel(ing)? (sad|empty|anxious|worthless|numb|guilty)|'
                 r'overwhelm|motivat|procrastinat|confidence|boundaries)\b', re.I)

# drop dataset classification-task prompts (imhi) and other non-chat templates
JUNK = re.compile(r'(the answer to the question|does the poster|this post shows|'
                  r'mental disorder of|consider this post|\bclassify\b|\blabel\b|'
                  r'select the (correct|right)|multiple[- ]?choice|true or false|'
                  r'the poster (suffers|is|feels|seems)|reasoning:|\bpost:\s)', re.I)

ASSISTANT_RE = re.compile(r'\n?\s*assistant\s*:', re.I)
USERPFX_RE = re.compile(r'^\s*user\s*:\s*', re.I)
WS_RE = re.compile(r'\s+')
NONLATIN = re.compile(r'[^\x00-\x7f]')

def extract_prompt(rec):
    # Prefer a structured `messages` list if present; else parse `text`.
    msgs = rec.get('messages')
    if isinstance(msgs, list) and msgs:
        for m in msgs:
            if str(m.get('role', '')).lower() in ('user', 'human'):
                return str(m.get('content', ''))
        return ''
    text = rec.get('text', '') or ''
    m = ASSISTANT_RE.search(text)
    user_part = text[:m.start()] if m else text
    user_part = USERPFX_RE.sub('', user_part)
    return user_part

def clean(q):
    q = WS_RE.sub(' ', q).strip()
    return q

def norm_key(q):
    return re.sub(r'[^a-z0-9 ]', '', q.lower())[:120]

def looks_like_question(q):
    if '?' in q:
        return True
    # help-seeking / first-person statements common in psych prompts
    return bool(re.match(r'(i\b|i\'m|i am|i\'ve|my |how |what |why |should |can |is it|does |when )', q, re.I))

def load(tag, path, limit=None, psy_gate=False):
    out = []
    if not path.exists():
        print(f'  WARN missing {path}', file=sys.stderr)
        return out
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            q = clean(extract_prompt(rec))
            if not (MIN_LEN <= len(q) <= MAX_LEN):
                continue
            # drop residual non-Latin heavy lines (e.g. untranslated CJK)
            if len(NONLATIN.findall(q)) > 4:
                continue
            if not looks_like_question(q):
                continue
            if JUNK.search(q):     # classification-task / template prompts
                continue
            if psy_gate and not PSY.search(q):
                continue
            out.append({'q': q, 'src': tag, 'doc_id': rec.get('doc_id', '')})
            if limit and len(out) >= limit:
                break
    return out

def main():
    random.seed(42)
    # balanced pull: zh natural psych (take all), en psych-gated, mh capped (its
    # imhi classification prompts are weaker than zh/en natural questions).
    zh = load('zh', FILES['zh'], limit=None, psy_gate=False)
    mh = load('mh', FILES['mh'], limit=1400, psy_gate=False)   # natural mh only (JUNK drops imhi templates)
    en = load('en', FILES['en'], limit=1400, psy_gate=True)
    print(f'extracted: zh={len(zh)} mh={len(mh)} en={len(en)}')

    # dedup across all (normalized)
    seen, pool = set(), []
    for grp in (zh, mh, en):
        random.shuffle(grp)
        for r in grp:
            k = norm_key(r['q'])
            if k in seen:
                continue
            seen.add(k)
            pool.append(r)
    print(f'after dedup: {len(pool)}')

    # balanced interleave so each shard sees a zh/mh/en mix, then trim to TARGET
    by = {'zh': [], 'mh': [], 'en': []}
    for r in pool:
        by[r['src']].append(r)
    inter = []
    while len(inter) < TARGET and any(by.values()):
        for t in ('zh', 'mh', 'en'):
            if by[t]:
                inter.append(by[t].pop())
            if len(inter) >= TARGET:
                break
    print(f'interleaved pool: {len(inter)} (zh-left={len(by["zh"])} mh-left={len(by["mh"])} en-left={len(by["en"])})')

    if len(inter) < TARGET:
        print(f'note: {len(inter)} candidates (< {TARGET} target) — sharding evenly', file=sys.stderr)

    # write 20 shards, distributing candidates EVENLY (no empty tail shards)
    per_shard = -(-len(inter) // N_SHARDS)   # ceil
    for s in range(N_SHARDS):
        chunk = inter[s*per_shard:(s+1)*per_shard]
        txt = OUT / f'shard_{s:02d}.txt'
        jsl = OUT / f'shard_{s:02d}.jsonl'
        with txt.open('w', encoding='utf-8') as f:
            for i, r in enumerate(chunk, 1):
                f.write(f'{i}. {r["q"]}\n')
        with jsl.open('w', encoding='utf-8') as f:
            for r in chunk:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    counts = {t: sum(1 for r in inter if r['src'] == t) for t in ('zh', 'mh', 'en')}
    print(f'wrote {N_SHARDS} shards x ~{per_shard} -> {OUT}')
    print(f'source mix in candidates: {counts}')

if __name__ == '__main__':
    main()
