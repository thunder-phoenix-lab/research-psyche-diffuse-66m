# Iteration 3 — Extended sampler experiments (no retrain) — NEGATIVE RESULT

## Summary
Added 5 sampling techniques on top of v2: final top-p+temp+rep_penalty, CFG cosine ramp, stochastic DDIM (eta>0), iterative refinement, short response_max. **All 5 together made outputs strictly worse** than v2. Sampling-only ceiling = v2.

## What got worse
| metric | v2 | v3 |
|---|---|---|
| therapy register lock-in | 3/5 | 0/5 |
| on-topic word density | ~35% | ~15% |
| Unicode/emoji garbage | none | rampant |
| readability | broken sentences | random tokens |

## Per-technique post-mortem
1. **CFG cosine ramp (1.0 → 3.0)** — Hypothesis was that early low-CFG explores, late high-CFG sharpens. Reality: early low-CFG falls into the same Pile food-drift mode v1 had. By the time CFG ramps up, x_t has drifted off the advice manifold. **Constant cfg=2.5 throughout (v2) was correct.**

2. **Stochastic DDIM (eta=0.2-0.4)** — Did give MBR diversity but each individual trajectory was noisier. MBR scores still collapsed (~-0.027). Net effect: lower quality without scoring signal to pick winners.

3. **Final top-p+temp sampling** — The killer. SPM 16k vocab includes rare Unicode tokens (CJK, emoji, accented chars) the model assigns very small but non-zero prob. top-p=0.92 was wide enough to occasionally sample them. argmax in v2 never touched these. **rep_penalty couldn't undo this** — each rare token was a one-shot pick.

4. **Iterative refinement (2-3 passes)** — Re-noising to t=300 then denoising erased v2's structure. Useful for image diffusion where pixel structure regenerates from coarse-to-fine; useless here because text is brittle to noise.

5. **Short response_max=128 vs 384** — Neutral. Length wasn't the bottleneck.

## What this rules out
- Inference-time exploration ≠ better text for this model class at this scale.
- The model's logits are NOT well-calibrated: rare tokens get non-trivial probability mass that diversifies output but breaks coherence.
- MBR self-scoring won't work — all paths give similar mean log-prob because constraint + clamp force similar entropy.

## What v3 confirmed (good knowledge)
- The numerical-stability bug at eta>0 (NaN from `(1-ab_p)/(1-ab_t)` near small-t) is real and we fixed it (`nan_to_num` + clamp).
- Full v3 pipeline runs end-to-end (~1.3 min/prompt at refine=2, n_mbr=8, steps=200).

## Sampling-only ceiling = v2
Best regime remains: **constant cfg=2.5, clamp+constrain, argmax decode, no refinement, MBR-8**.

Further quality requires real retrain:
- Option A: continue pretrain → 80k–150k steps total (×4-8 GPU time)
- Option B: discrete diffusion (SEDD / MDLM) from scratch — fundamentally better architecture at small scale
- Option C: scale to 125M params (Chinchilla optimal) + retrain
- Option D: distill from existing English LM (e.g. Pythia-160M)

## Artifacts
- `sample_sft_v3.py` — has all features, kept for reference
- `sft_samples_v3.txt` — 5 prompts × 3 configs, all worse than v2
- This file

---
Date: 2026-05-29
Status: NEGATIVE result. v2 remains current best inference setup.
