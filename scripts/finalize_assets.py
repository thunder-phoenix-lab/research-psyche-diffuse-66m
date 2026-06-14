"""
Finalize Android assets:
  1. int8 dynamic-quantize mdlm_chat.onnx -> mdlm_chat_int8.onnx (smaller, faster CPU)
  2. dump SentencePiece vocab -> vocab.json  {pieces:[...], unk,bos,eos,pad ids, mask id}
     (Kotlin tokenizer: greedy longest-match encode + piece-join decode)
"""
from __future__ import annotations
import json, sys, io
from pathlib import Path
import sentencepiece as spm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = Path('D:/001_PROJECTS/psyche_diffuse_60m')
ASSETS = ROOT / 'experimental' / 'app' / 'src' / 'main' / 'assets'
SPM = ROOT / 'raw_data/004_train/tokenizer/spm_en.model'
VOCAB_CLEAN = 16_000

# 1) int8 quant
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    src = ASSETS / 'mdlm_chat.onnx'
    dst = ASSETS / 'mdlm_chat_int8.onnx'
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    print(f'int8: {dst.name} {dst.stat().st_size/1e6:.1f} MB (from {src.stat().st_size/1e6:.0f} MB)')
except Exception as e:
    print('int8 quant failed (keep fp32):', e)

# 2) vocab dump
sp = spm.SentencePieceProcessor(); sp.load(str(SPM))
pieces = [sp.id_to_piece(i) for i in range(sp.vocab_size())]
meta = {
    'pieces': pieces,
    'vocab_size': sp.vocab_size(),
    'unk_id': sp.unk_id(), 'bos_id': sp.bos_id(), 'eos_id': sp.eos_id(), 'pad_id': sp.pad_id(),
    'sep_id': 1,            # unk reused as turn boundary (matches build_sft_dataset.py)
    'mask_id': VOCAB_CLEAN, # 16000 special [MASK] added for MDLM
}
(ASSETS / 'vocab.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
print(f'vocab.json: {len(pieces)} pieces, mask_id={meta["mask_id"]}, bos={meta["bos_id"]} sep={meta["sep_id"]} eos={meta["eos_id"]}')
print('sample pieces[:8]:', pieces[:8])
print('assets now:', sorted(p.name for p in ASSETS.iterdir()))
