"""
Build a fresh .bin from corpus/en_clean/ using the EXISTING SPM tokenizer.
Output: raw_data/004_train/datasets/pretrain_en_diff_clean/train.bin
"""
import json
import os
import sys
import io
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import zstandard as zstd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
# STRICT_PILE=1 → build from the cleaned en_clean_v2 corpus into a separate
# dataset dir, leaving the original 66.8M baseline dataset intact.
STRICT_PILE = os.environ.get('STRICT_PILE', '') == '1'
DATA_SRC = ROOT / 'corpus' / ('en_clean_v2' if STRICT_PILE else 'en_clean')
TOK_MODEL = ROOT / '004_train' / 'tokenizer' / 'spm_en.model'
OUT_DIR = ROOT / '004_train' / 'datasets' / ('pretrain_en_clean_v2' if STRICT_PILE else 'pretrain_en_diff_clean')
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_BIN = OUT_DIR / 'train.bin'
META_JSON = OUT_DIR / 'meta.json'

VOCAB_SIZE = 16000

sp = spm.SentencePieceProcessor()
sp.load(str(TOK_MODEL))
assert sp.vocab_size() == VOCAB_SIZE, f'tok vocab mismatch: {sp.vocab_size()} vs {VOCAB_SIZE}'
EOS_ID = sp.eos_id()
print(f'tokenizer: {TOK_MODEL.name}  vocab={sp.vocab_size()}  eos={EOS_ID}')


def iter_clean_docs():
    dctx = zstd.ZstdDecompressor()
    for shard in sorted(DATA_SRC.glob('shard_*.jsonl.zst')):
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
                        t = (d.get('text') or '').strip()
                        if len(t) >= 50:
                            yield t


t0 = time.time()
tokens_written = 0
docs = 0
buf = []
BUF_FLUSH = 4 * 1024 * 1024
print(f'writing to {TRAIN_BIN}')
with TRAIN_BIN.open('wb') as out:
    for t in iter_clean_docs():
        ids = sp.encode(t, out_type=int)
        ids.append(EOS_ID)
        buf.extend(ids)
        docs += 1
        if len(buf) >= BUF_FLUSH:
            arr = np.asarray(buf, dtype=np.uint16)
            arr.tofile(out)
            tokens_written += len(arr)
            buf.clear()
            if docs % 50_000 == 0:
                print(f'  ... docs={docs:,}  tokens={tokens_written/1e6:.1f}M  elapsed={time.time()-t0:.0f}s')
    if buf:
        arr = np.asarray(buf, dtype=np.uint16)
        arr.tofile(out)
        tokens_written += len(arr)

META_JSON.write_text(json.dumps({
    'lang': 'en_clean',
    'docs': docs,
    'train_tokens': tokens_written,
    'dtype': 'uint16',
    'eos_id': EOS_ID,
    'vocab_size': VOCAB_SIZE,
}, indent=2))
print(f'done: {docs:,} docs, {tokens_written/1e6:.1f}M tokens in {time.time()-t0:.0f}s')
print(f'output: {TRAIN_BIN}  ({TRAIN_BIN.stat().st_size/1e6:.0f} MB)')
