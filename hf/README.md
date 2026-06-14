---
language: en
license: apache-2.0
library_name: pytorch
pipeline_tag: text-generation
tags:
  - diffusion
  - masked-diffusion
  - mdlm
  - psychology
  - mental-health
  - small-model
  - on-device
---

# psyche-diffuse-66m (production) & psyche-diffuse-120m

66.8M-parameter **masked diffusion language model** (MDLM, DiT backbone) for English psychology / mental-health dialogue, trained on one RTX 3090. The 120M sibling is published to make the paper's scaling null independently checkable. Paper: *Small-Scale Text Diffusion for a Psychology Assistant* (Denis Kim, 2026), PDF in the GitHub repo.

## ⚠️ Intended use & safety

Research artifact for studying small-scale text diffusion. **Not a therapist, not a medical device, not for crisis use.** Outputs are frequently vague, repetitive, and can be factually wrong (paper §4–8: blind homonym-drift floor 25–32%). For crisis support call 988 (US) or local equivalents. No real conversations or PII in training data.

## Files

| file | what |
|---|---|
| `model_66m_clean_sft.pt` | production checkpoint (DiT 512×12×8, MLP 1408; bf16) |
| `model_120m_clean_sft.pt` | scale checkpoint (640×15×10, 1664; bf16) |
| `spm_en.model` | 16k SentencePiece BPE (+`[MASK]`=16000) |
| `mdlm_chat.onnx` | fp32 mobile export of 66m |

## Use

```bash
git clone <github-repo>
python scripts/sample_mdlm_v2.py --checkpoint model_66m_clean_sft.pt \
  --prompt "user: I feel anxious about work. assistant:" --resp_len 110 --steps 28 \
  --temperature 0.9 --top_p 0.9   # + rep_penalty 1.3, neighbor-ban 2, no-repeat-3gram (defaults)
# 120m: add MDLM_HIDDEN=640 MDLM_LAYERS=15 MDLM_HEADS=10 MDLM_MLP_INTER=1664
```

Use confidence-ordered unmasking (v2 sampler). Random-order unmasking re-introduces repetition collapse (paper §4).

## Training

MDLM objective (mask prob t∼U(0.01,0.99), CE on masked); DiffuSeq-style SFT (response-only noising). Pretrain 20k steps / SFT 8k, batch 256×512, LR 4e-4 cosine, bf16 + AdamW8bit. Data: strict-filtered psychology corpus 0.774B tokens; SFT 261k pairs (assistant + mental-health + translated psychology dialogue). Corpora not redistributed; build scripts in repo.

## Eval (paper, audited)

725 paired prompts, MiniLM cosine-to-gold: 66m-clean 0.500; 120m 0.490 (d=−0.0094, win 0.479 → null at fixed data). Blind traps (3 annotators, κ .77–.87): 66m 9/28 vs 120m 7/28, p=0.77. distinct-2 0.993; rep-1 0.137; collapse 0.

## Citation

```bibtex
@misc{kim2026psyche, title={Small-Scale Text Diffusion for a Psychology Assistant}, author={Kim, Denis}, year={2026}, note={arXiv preprint}}
```
