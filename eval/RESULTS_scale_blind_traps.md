# Scale result — blind homonym-trap drift (120M-clean vs 66.8M-clean, SFT)

Date: 2026-05-31. Complement to `RESULTS_scale_1000q.md`. Tests the SPECIFIC failure the
aggregate 1000-Q metric could miss: on rare technical psych terms (homonyms), does the model
drift into a non-psychology domain — and does 2× scale move that floor?

## Method (locked blind protocol — no self-labeling)
- 14 homonym-bait prompts (`confab_trap_prompts.txt`), n=2 seeds each → 28 samples/model.
- Pool 28×120M + 28×66.8M, shuffled, source-stripped → `blind_traps_scale.txt` (S000–S055).
- **3 fresh independent sub-agents** blind-labeled DRIFT=yes/no (psychology vs non-psych domain;
  repetition/incoherence alone ≠ drift). They did not know source, hypothesis, or direction.
- Decoded via `blind_traps_scale_key.json`; `score_blind_traps.py` computes the stats.

## Inter-annotator agreement (reliability)
Cohen κ: (A,B)=0.767, (A,C)=0.867, (B,C)=0.825 — substantial→almost-perfect. The drift
judgment is reproducible (unlike the earlier non-blind self-labeling that inflated a corpus effect).

## Per-annotator drift rate (n=28/source)
| rater | 120M | 66.8M |
|---|---|---|
| A | 6/28 = 0.21 | 8/28 = 0.29 |
| B | 6/28 = 0.21 | 9/28 = 0.32 |
| C | 8/28 = 0.29 | 9/28 = 0.32 |

All three: 120M ≤ 66.8M (120M drifts slightly *less*, not more).

## Consensus (majority of 3) — scale comparison
- 120M-clean drift : **7/28 = 0.250**, Wilson95 [0.127, 0.434]
- 66.8M-clean drift: **9/28 = 0.321**, Wilson95 [0.179, 0.507]
- diff (120−66) = **−0.071**, Newcombe95 **[−0.294, +0.161]**, Fisher exact p = **0.768**

## Verdict
**No significant scale effect on the homonym-drift floor.** The floor sits at **~25–32% for
both 66.8M and 120M**; the 120M is numerically a touch lower but well within noise (CI spans 0,
p=0.77). Doubling parameters did not move it.

## Convergence with the 1000-Q result
- 1000-Q aggregate quality: 120M ≈ 66.8M (tie, 120M a hair worse).
- Blind homonym traps: 120M ≈ 66.8M (no sig. difference, 120M a hair better).
Both evals agree: **66.8M → 120M is a null at this data budget.**

## Caveat (same as 1000-Q)
Both models are data-bound: 0.774B tokens → 66.8M=11.6, 120M=6.4 tok/param (Chinchilla ≈20).
This shows 2× params at FIXED 0.77B tokens doesn't move the floor — NOT that scale can never
move it. A Chinchilla-fed 120M (~2.4B tokens) is the untested next lever.

Artifacts: `blind_traps_scale.txt`, `blind_traps_scale_key.json`, `annot_{A,B,C}.txt`, `score_blind_traps.py`.
