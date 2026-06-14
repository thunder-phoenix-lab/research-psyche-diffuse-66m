"""
Train SentencePiece BPE 32K tokenizer on the multilingual corpus.

Reads .jsonl.zst shards from corpus/<lang>/ for ru + en + zh (primary),
plus a small slice of "other" languages for coverage.

Output: 004_train/tokenizer/spm.model + spm.vocab
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import zstandard as zstd

sys.path.insert(0, str(Path(__file__).parent))
from _train_common import CORPUS, TOKENIZER_DIR, LOGS

LANGS_PRIMARY = ["ru", "en", "zh"]
LANGS_AUX = ["en_lowconf", "ru_lowconf", "af", "ar", "de", "es", "fr", "nl"]


def iter_texts_from_shard(shard_path: Path, max_docs: int) -> list[str]:
    out: list[str] = []
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
                    t = d.get("text") or ""
                    if t and len(t) > 50:
                        out.append(t)
                        if len(out) >= max_docs:
                            return out
    return out


def write_training_text(target_file: Path, max_per_lang: int) -> dict:
    target_file.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}
    bytes_written = 0
    with target_file.open("w", encoding="utf-8") as out:
        for lang in LANGS_PRIMARY + LANGS_AUX:
            lang_dir = CORPUS / lang
            if not lang_dir.exists():
                continue
            shards = sorted(lang_dir.glob("shard_*.jsonl.zst"))
            if not shards:
                continue
            collected = 0
            target = max_per_lang if lang in LANGS_PRIMARY else max(max_per_lang // 10, 500)
            for sh in shards:
                if collected >= target:
                    break
                texts = iter_texts_from_shard(sh, target - collected)
                for t in texts:
                    # SentencePiece input: 1 doc per line; strip newlines
                    line = t.replace("\n", " ").replace("\r", " ").strip()
                    if len(line) < 50:
                        continue
                    out.write(line + "\n")
                    bytes_written += len(line) + 1
                    collected += 1
            stats[lang] = collected
            print(f"  {lang}: {collected} docs from {len(shards)} shards")
    print(f"Wrote {target_file} ({bytes_written/1e6:.1f} MB)")
    return stats


def train_spm(input_file: Path, output_prefix: Path, vocab_size: int = 32000) -> None:
    import sentencepiece as spm
    args = dict(
        input=str(input_file),
        model_prefix=str(output_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9999,    # high to cover RU/CJK/etc.
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        num_threads=8,
        train_extremely_large_corpus=True,
        byte_fallback=True,           # any unseen bytes -> bytefallback (no OOV)
        normalization_rule_name="nmt_nfkc",
        remove_extra_whitespaces=False,
        max_sentence_length=16384,
        # special tokens
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
        user_defined_symbols=[
            "<|user|>", "<|assistant|>", "<|system|>",
            "<|im_start|>", "<|im_end|>",
            "<think>", "</think>",
            "[USER]", "[EMAIL]", "[PHONE]", "[REDACTED]",
        ],
    )
    print(f"Training SentencePiece: vocab={vocab_size} on {input_file}")
    spm.SentencePieceTrainer.train(**args)
    print(f"Wrote {output_prefix}.model and {output_prefix}.vocab")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--max-per-lang", type=int, default=200_000,
                    help="Max documents to sample per primary language")
    args = ap.parse_args()
    t0 = time.time()
    train_corpus = TOKENIZER_DIR / "_train_text.txt"
    print(f"== gathering training text from {CORPUS} ==")
    stats = write_training_text(train_corpus, args.max_per_lang)
    print(f"== training SentencePiece ({args.vocab_size}) ==")
    train_spm(train_corpus, TOKENIZER_DIR / "spm", args.vocab_size)
    # write stats sidecar
    info = {
        "vocab_size": args.vocab_size,
        "max_per_lang": args.max_per_lang,
        "docs_per_lang": stats,
        "training_text_path": str(train_corpus),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (TOKENIZER_DIR / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    print(f"\n== done in {info['elapsed_sec']}s ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
