# Iteration 1 — Diffusion v3 Pretrain + SFT scenario C

## TL;DR
Pipeline работает end-to-end, но 60M continuous diffusion @ 2.5B токенов даёт **topic-coherent word salad**. Scenario C трюки (DiffuSeq + self-cond + CFG) измеримо помогают на тематике, но грамматики нет.

## Training
| stage | start ce | end ce | steps | params | tokens | duration |
|-------|----------|--------|-------|--------|--------|----------|
| Pretrain v3 | scratch | 1.54 | 20 000 | 60.8M | 2.56B | ~7.5 ч |
| SFT scenario C | 1.66 | 1.52 | 8 000 | 60.9M | 261 k pairs | ~3.1 ч |

Pretrain ckpt: `raw_data/004_train/checkpoints/diffusion_v3/final/`
SFT ckpt: `raw_data/004_train/checkpoints/diffusion_sft/final/`

## Architecture
- DiT, 12 layers × hidden=512 × heads=8 × MLP=1408
- d_emb=128, seq_len=512, vocab=16k (SPM BPE)
- AdaLN-zero timestep conditioning
- Cosine noise schedule, T=1000
- SFT adds `in_proj_sc` (zero-init) for self-conditioning input

## Sampler (v1)
DDIM x0-parameterization, 50 steps, parallel cond/uncond CFG streams with prompt anchor re-paste every step. Self-cond carries `x0_hat` from previous step.

## Eval results (5 prompts × 3 cfg, seed=0)
Файл: `sft_samples_v1.txt`

| prompt | cfg=1.0 | cfg=2.0 | cfg=3.5 |
|---|---|---|---|
| anxious about work | food drift | темат-сдвиг (stressful, hesitate) | punctuation soup |
| CBT vs ACT | food drift | technical noise | punctuation soup |
| **procrastinating thesis** | food drift | **best**: "boundaries jobs", "Reasoning heal expectations" | **best**: chain advice pronouns |
| Vygotsky ZPD | food drift | broken | punctuation soup |
| emotionally unavailable | food drift | broken | punctuation soup |

## Verified working
- ✓ pretrain → SFT load: `missing=['in_proj_sc.weight']`, embeddings preserved (`emb_norm=11.293`)
- ✓ CFG gives directional control (cfg=1.0 falls into Pile cooking drift, cfg=2-3.5 pushes toward advice vocab)
- ✓ self-conditioning fires correctly (sc=0.5 throughout training)
- ✓ DDIM+anchor sampling completes without NaN/inf

## Known limitations
- **Word salad syntax**: tokens sampled near-independently at each position via argmax
- **No long-range structure**: continuous diffusion at 60M / 2.5B tokens is undertrained per Chinchilla (optimal ≈125M for this token budget)
- **Pile drift at cfg=1.0**: pile_sample contributed ~1.7B tokens of "wellness/lifestyle" that poison unconditional generation
- **Hard-knowledge prompts fail**: model lacks capacity to retain facts (CBT/Vygotsky)

## Path forward
**Free wins (sampler-only, no retrain)**: clamp-trick (Li et al.), MBR sampling, more DDIM steps, CFG sweep, constrained argmax → being applied as v2.

**Medium**: switch to SEDD/MDLM discrete diffusion (architectural — best ROI for this scale).

**Big**: scale to 125M params + 80k pretrain steps + drop pile_sample.

---
Date: 2026-05-29
