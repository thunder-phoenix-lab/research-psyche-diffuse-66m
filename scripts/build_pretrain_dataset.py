"""
Build a streaming pretrain dataset from corpus/<lang>/ zstd shards.
Writes tokenized .bin files for fast mmapped training.

Output: 004_train/datasets/pretrain_<lang>/{train.bin, val.bin, meta.json}
        with uint32 (or uint16 if vocab<65K) token ids.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
import numpy as np
import sentencepiece as spm
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).parent))
from _train_common import CORPUS, TOKENIZER_DIR, TRAIN_ROOT


def shard_iter(shard_path: Path):
    with shard_path.open("rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            buf = b""
            while True:
                chunk = reader.read(1 << 20)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, _, buf = buf.partition(b"\n")
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    yield d


def tokenize_corpus(lang: str, tok: spm.SentencePieceProcessor,
                    val_ratio: float, max_docs: int | None) -> dict:
    lang_dir = CORPUS / lang
    if not lang_dir.exists():
        print(f"  skip: {lang_dir} missing")
        return {"docs": 0, "tokens": 0}
    out_dir = TRAIN_ROOT / "datasets" / f"pretrain_{lang}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    eos_id = tok.eos_id()
    # use uint32 to allow vocab up to ~4B
    f_train = train_path.open("wb")
    f_val = val_path.open("wb")
    n_train_tokens = 0
    n_val_tokens = 0
    n_docs = 0
    n_train_docs = 0
    n_val_docs = 0
    # deterministic split: every 1/val_ratio-th doc → val
    val_every = max(int(1 / val_ratio), 2) if val_ratio > 0 else 9999

    for shard in sorted(lang_dir.glob("shard_*.jsonl.zst")):
        for d in shard_iter(shard):
            text = d.get("text") or ""
            if len(text) < 50:
                continue
            ids = tok.encode(text)
            if not ids:
                continue
            ids.append(eos_id)
            arr = np.array(ids, dtype=np.uint32).tobytes()
            n_docs += 1
            if n_docs % val_every == 0:
                f_val.write(arr)
                n_val_tokens += len(ids)
                n_val_docs += 1
            else:
                f_train.write(arr)
                n_train_tokens += len(ids)
                n_train_docs += 1
            if max_docs and n_docs >= max_docs:
                break
        if max_docs and n_docs >= max_docs:
            break

    f_train.close(); f_val.close()
    info = {
        "lang": lang,
        "docs": n_docs,
        "train_docs": n_train_docs, "val_docs": n_val_docs,
        "train_tokens": n_train_tokens, "val_tokens": n_val_tokens,
        "dtype": "uint32",
        "eos_id": eos_id,
        "vocab_size": tok.vocab_size(),
    }
    (out_dir / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"  {lang}: {n_docs} docs, train={n_train_tokens:,} val={n_val_tokens:,} tokens")
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["ru", "en", "zh"])
    ap.add_argument("--val-ratio", type=float, default=0.005)
    ap.add_argument("--max-docs", type=int, default=None,
                    help="Cap docs per lang (for quick tests)")
    args = ap.parse_args()
    spm_path = TOKENIZER_DIR / "spm.model"
    if not spm_path.exists():
        print(f"Tokenizer not found at {spm_path}; run train_tokenizer.py first")
        return 1
    tok = spm.SentencePieceProcessor()
    tok.load(str(spm_path))
    print(f"Loaded tokenizer (vocab={tok.vocab_size()})")
    total = {"docs": 0, "train_tokens": 0, "val_tokens": 0}
    for lang in args.langs:
        info = tokenize_corpus(lang, tok, args.val_ratio, args.max_docs)
        for k in ("docs", "train_tokens", "val_tokens"):
            total[k] = total.get(k, 0) + info.get(k, 0)
    print(f"\n== total docs={total['docs']}, "
          f"train={total['train_tokens']:,}, val={total['val_tokens']:,} ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
