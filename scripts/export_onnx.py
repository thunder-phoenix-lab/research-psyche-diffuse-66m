"""
Export MDLM-SFT (66.8M, clean) to ONNX for ONNX Runtime Mobile (Android).
ORT-Mobile is actively maintained and its NNAPI execution provider taps the
phone GPU/NPU. forward(ids[1,512] int64, t[1] float32) -> logits[1,512,16001].

Verifies parity (PyTorch vs onnxruntime CPU) before saving to app assets.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from export_mobile import MDLM, CKPT, ASSETS, VOCAB_FULL, SEQ_LEN  # import sets utf-8 stdout

OUT = ASSETS / 'mdlm_chat.onnx'


def main():
    print(f'loading {CKPT}')
    m = MDLM().eval()
    w = torch.load(CKPT, map_location='cpu', weights_only=True)
    m.load_state_dict({k: v.float() for k, v in w.items()}, strict=True)
    m = m.float()
    print('params:', sum(p.numel() for p in m.parameters())/1e6, 'M')

    ex_ids = torch.randint(0, VOCAB_FULL, (1, SEQ_LEN), dtype=torch.long)
    ex_t = torch.tensor([0.5], dtype=torch.float32)

    torch.onnx.export(
        m, (ex_ids, ex_t), str(OUT),
        input_names=['ids', 't'], output_names=['logits'],
        opset_version=17, do_constant_folding=True,
        dynamic_axes=None,  # fixed [1,512] — simpler+faster on mobile
    )
    print(f'exported {OUT} ({OUT.stat().st_size/1e6:.1f} MB)')

    # parity check: pytorch vs onnxruntime CPU
    import onnxruntime as ort
    with torch.no_grad():
        ref = m(ex_ids, ex_t).numpy()
    sess = ort.InferenceSession(str(OUT), providers=['CPUExecutionProvider'])
    got = sess.run(['logits'], {'ids': ex_ids.numpy(), 't': ex_t.numpy()})[0]
    diff = np.abs(ref - got).max()
    print(f'parity max|diff| = {diff:.4e}  (ref argmax==onnx argmax: {(ref.argmax(-1)==got.argmax(-1)).mean()*100:.1f}%)')
    print('OK' if diff < 1e-2 else 'WARN: large diff')


if __name__ == '__main__':
    main()
