# Iteration 2 — Sampler improvements (no retrain)

## What changed
Pure inference-time fixes on the **same SFT checkpoint** as v1:

| change | effect |
|---|---|
| **Clamp-trick** (Li et al.) with top-k=32 sampling | Project x0_hat → embed(sample(logits)) on response positions every step after 70% denoising |
| **Temperature on clamp** (T=0.7) | Avoid stopword collapse (`your/you/to/the`) at high CFG |
| **DDIM steps 50 → 200** | Give clamping more iterations to converge |
| **MBR n=8** | Generate 8 parallel samples, pick highest mean log-prob |
| **Constrained decode** | Forbid PAD/BOS/SEP in response; EOS only after 16 tokens; anti-3-gram-repeat |
| **CFG sweep** | Test 1.2 / 1.7 / 2.5 (was 1.0 / 2.0 / 3.5) |

## Best regime
**CFG=2.5  clamp_temp=0.7  clamp_after=0.7** on advice/coping prompts.

## v1 → v2 comparison
| prompt | v1 (cfg=2.0) | v2 (cfg=2.5 best) |
|---|---|---|
| anxious about work | `pepper, baking, delicious` food drift | `feel support breathing talking joy happy schedule work navigate options give` |
| procrastinating thesis | "boundaries jobs Reasoning heal" fragments | `support kind seek wish experiences talking helpful steps listening therapist learn approach` |
| **emotionally unavailable** | broken | **best v2 result** — therapy register: `Therapy counseling partner understanding communication care friends sessions self meditation coping family relationship` |
| CBT vs ACT | technical noise | ❌ still punctuation soup |
| Vygotsky ZPD | broken | ❌ still punctuation soup |

## Quantitative wins
- **Vocabulary density** (on-topic content words / total tokens): v1 ~10%, v2 ~35% on advice prompts
- **Food/cooking drift**: eliminated at cfg≥1.7
- **Therapy register lock-in**: visible at cfg=2.5 on 3/5 prompts

## Remaining limitations
- **Grammar still broken** — fundamental at 60M params / 2.5B tokens for continuous diffusion
- **Knowledge prompts fail completely** (CBT, Vygotsky) — model has no factual capacity at this scale
- **MBR scores too uniform** (~-0.023 across all 8 samples) — constrain pass + clamp force all paths to similar log-prob; need diversity-aware scoring or external scorer
- **CFG=1.7 mid-range still erratic** — works at 2.5 (strong push to advice manifold) or fails

## Sample (best of v2): "emotionally unavailable" cfg=2.5
> "own's feel time me you without feel what I library It to to? can one' **Therapy**. I lead might to every you ways me ownll bring **husband**udg Is willingness on **feelings** this any wrong requires **help** fordown. you share' if need possible If your You you just do yourself you provide goalful I yourself. You overall through here your to else you see **communication understand care friends** you employees dailybeing to stop buying else. way Create an. I to no You **counseling** Ill your You see what answers, your or It **therapy**. **partner** understanding way own someone is have remind have happy. know rather see self own work make have take you seeking collected, matters overall for bring we a didn **therapy** I meaningful our situation if I I, practical un your canounts always I with explore issues I. effort and and your Additionally you moment our together protect' important **support conversation** your know to previous you to, and whether and **partner** you time What get **meditation** self however try you to this to activities might you going work this Remember care how because you..."

## What's NOT achievable from sampler alone
1. Real grammar → needs more pretrain (80–150k steps) or discrete diffusion (SEDD/MDLM)
2. Factual knowledge (CBT/Vygotsky) → needs ×10 token budget or distill from existing LM
3. Coherent multi-sentence structure → architectural limit of continuous diffusion at this scale

---
Date: 2026-05-29
Files: `sft_samples_v2.txt`, `sample_sft_v2.py`
