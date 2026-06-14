"""
Build a clean EN pretrain corpus:
  - Read raw_data/corpus/en/shard_*.jsonl.zst
  - Filter: only allow-listed sources (no SFT/dialogue, no chat-style)
  - Filter: drop docs where Cyrillic ratio > 5% or any CJK characters
  - Drop boilerplate prefix from known SFT-like rows (defensive)
  - Concat with the cleaned translated dataset
  - Write to raw_data/corpus/en_clean/shard_*.jsonl.zst (target ~50MB/shard)
"""
import json
import os
import re
import sys
import io
from pathlib import Path

import zstandard as zstd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / 'corpus' / 'en'
TRANSLATED = ROOT / 'extracted' / '01_pretrain_en' / 'dissercat_psyjournals_translated_en_clean.jsonl'

# STRICT_PILE=1 → use psych-dominant re-filtered pile/pes2o and write to en_clean_v2,
# leaving the original en_clean (66.8M baseline corpus) untouched.
STRICT_PILE = os.environ.get('STRICT_PILE', '') == '1'
OUT_DIR = ROOT / 'corpus' / ('en_clean_v2' if STRICT_PILE else 'en_clean')
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_SOURCES = {
    'openalex_psychology',
    'pmc_oa_psychology',
    'wikipedia_en_psychology',
    'project_gutenberg',
    'psyarxiv_osf',
    'arxiv_query',
    'big_five_essays',
    # New (added when reaching for 2B+ tokens — psy-filtered subsets of large general datasets):
    'pile_sample',
    'pes2o_sample',
    'tiny_textbooks',
    's2orc_psychology',
}
# All translated content is allowed via the translated jsonl below

CYR_RE = re.compile(r'[Ѐ-ӿ]')
HAN_RE = re.compile(r'[一-鿿]')
LAT_RE = re.compile(r'[A-Za-z]')

# Boilerplate prefixes occasionally present in SFT-style rows (defensive)
BOILERPLATE_PATTERNS = [
    re.compile(r'^You are a helpful mental health[^\n]*\n+', re.IGNORECASE),
    re.compile(r'^If you are a licensed psychologist[^\n]*\n+', re.IGNORECASE),
    re.compile(r'^(user|assistant|User|Assistant):\s*', re.MULTILINE),
]


def is_clean_lang(text: str) -> bool:
    if not text:
        return False
    cyr = len(CYR_RE.findall(text))
    han = len(HAN_RE.findall(text))
    lat = len(LAT_RE.findall(text))
    if han > 0:
        return False
    if lat == 0:
        return False
    if cyr / max(lat + cyr, 1) > 0.05:
        return False
    return True


def strip_boilerplate(text: str) -> str:
    for pat in BOILERPLATE_PATTERNS:
        text = pat.sub('', text)
    return text


def iter_corpus_en():
    dctx = zstd.ZstdDecompressor()
    for shard in sorted(SRC_DIR.glob('shard_*.jsonl.zst')):
        with shard.open('rb') as f:
            with dctx.stream_reader(f) as reader:
                buf = b''
                while True:
                    chunk = reader.read(1 << 20)
                    if not chunk:
                        break
                    buf += chunk
                    while b'\n' in buf:
                        line, _, buf = buf.partition(b'\n')
                        if not line.strip():
                            continue
                        try:
                            d = json.loads(line)
                        except Exception:
                            continue
                        yield d


def iter_translated():
    with TRANSLATED.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def iter_extracted_jsonl(path):
    """Stream a flat .jsonl (used for pile_sample.jsonl, pes2o_sample.jsonl,
    wikipedia_ru_translated_en.jsonl)."""
    if not path.exists():
        return
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main():
    src_stats = {}
    kept_docs = 0
    kept_chars = 0
    dropped_source = 0
    dropped_lang = 0
    dropped_short = 0

    # We'll stream-write shards
    SHARD_SIZE = 50 * 1024 * 1024  # 50 MB compressed-ish
    shard_idx = 0
    cur_buf = []
    cur_bytes = 0

    cctx = zstd.ZstdCompressor(level=10)

    def flush(force=False):
        nonlocal shard_idx, cur_buf, cur_bytes
        if not cur_buf:
            return
        if not force and cur_bytes < SHARD_SIZE:
            return
        out_path = OUT_DIR / f'shard_{shard_idx:05d}.jsonl.zst'
        raw = ('\n'.join(cur_buf) + '\n').encode('utf-8')
        compressed = cctx.compress(raw)
        out_path.write_bytes(compressed)
        print(f'  wrote {out_path.name}: {len(cur_buf):,} docs, {len(raw)/1e6:.1f} MB raw, {len(compressed)/1e6:.1f} MB zst')
        shard_idx += 1
        cur_buf = []
        cur_bytes = 0

    def emit(doc):
        nonlocal kept_docs, kept_chars, cur_bytes
        line = json.dumps(doc, ensure_ascii=False)
        cur_buf.append(line)
        cur_bytes += len(line) + 1
        kept_docs += 1
        kept_chars += len(doc.get('text', ''))
        if cur_bytes >= SHARD_SIZE:
            flush()

    print('Streaming corpus/en/ with filters...')
    for d in iter_corpus_en():
        src = d.get('source', '?')
        if src not in ALLOWED_SOURCES:
            dropped_source += 1
            continue
        text = d.get('text', '')
        if len(text) < 200:
            dropped_short += 1
            continue
        if not is_clean_lang(text):
            dropped_lang += 1
            continue
        text = strip_boilerplate(text).strip()
        if len(text) < 200:
            dropped_short += 1
            continue
        d['text'] = text
        emit(d)
        src_stats.setdefault(src, 0)
        src_stats[src] += 1
        if kept_docs % 50000 == 0:
            print(f'  ... kept={kept_docs:,}  dropped_src={dropped_source:,}  dropped_lang={dropped_lang:,}  dropped_short={dropped_short:,}')

    print('\nAppending translated content (dissercat + psyjournals -> EN)...')
    trans_added = 0
    for d in iter_translated():
        text = d.get('text', '')
        if len(text) < 200:
            dropped_short += 1
            continue
        if not is_clean_lang(text):
            dropped_lang += 1
            continue
        emit(d)
        trans_added += 1
        src_stats.setdefault(d.get('source', 'translated'), 0)
        src_stats[d.get('source', 'translated')] += 1
    print(f'  added {trans_added:,} translated docs')

    # Stream new extracted sources (Pile psy-filter, PES2O psy-filter, RU wiki translated)
    pile_name = 'pile_sample_strict.jsonl' if STRICT_PILE else 'pile_sample.jsonl'
    pes2o_name = 'pes2o_sample_strict.jsonl' if STRICT_PILE else 'pes2o_sample.jsonl'
    print(f'  [extras] pile={pile_name}  pes2o={pes2o_name}  (STRICT_PILE={STRICT_PILE})')
    extras = [
        ROOT / 'extracted' / '01_pretrain_en' / pile_name,
        ROOT / 'extracted' / '01_pretrain_en' / pes2o_name,
        ROOT / 'extracted' / '01_pretrain_en' / 'wikipedia_ru_translated_en.jsonl',
    ]
    for p in extras:
        if not p.exists():
            print(f'  [skip] {p.name} missing')
            continue
        print(f'Appending {p.name}...')
        n_added = 0
        for d in iter_extracted_jsonl(p):
            text = d.get('text', '')
            if len(text) < 200:
                dropped_short += 1
                continue
            if not is_clean_lang(text):
                dropped_lang += 1
                continue
            emit(d)
            n_added += 1
            src_stats.setdefault(d.get('source', '?'), 0)
            src_stats[d.get('source', '?')] += 1
            if n_added % 50_000 == 0:
                print(f'  ... {p.name}: +{n_added:,} (total kept {kept_docs:,})')
        print(f'  added {n_added:,} from {p.name}')

    flush(force=True)

    print('\n=== SUMMARY ===')
    print(f'Total kept docs: {kept_docs:,}')
    print(f'Total kept chars: {kept_chars/1e9:.2f} GB (~{kept_chars/4/1e6:.0f}M tokens est.)')
    print(f'Dropped (source not in allowlist): {dropped_source:,}')
    print(f'Dropped (language filter): {dropped_lang:,}')
    print(f'Dropped (too short): {dropped_short:,}')
    print(f'Shards written: {shard_idx}')
    print('\nKept by source:')
    for s, n in sorted(src_stats.items(), key=lambda x: -x[1]):
        print(f'  {s:35s} {n:>8,}')


if __name__ == '__main__':
    main()
