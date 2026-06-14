# Matched comparison — corpus (noisy vs clean) x decode (v1 vs v2 sampler)

Core paper result. Factorial design isolating the two axes we identified:
- **Decode axis** (sampler v1 random-unmask vs v2 confidence+rep) at FIXED temp=0.9
  -> tests Finding 1 (collapse = decode). Should reproduce on either corpus.
- **Corpus axis** (66.8M-noisy vs 66.8M-clean, SAME arch, SAME SFT) -> tests Finding 2
  (confab = data). Should DROP confab noisy->clean.

All 66.8M, HIDDEN=512/L12/H8/MLP1408. Sampler settings fixed: temp=0.9 top_p=0.9
(v2 adds rep_penalty=1.3 neighbor_ban=2 no_repeat_ngram=3). Confab = HUMAN-labeled
domain drift on the 14 trap prompts (`confab_trap_prompts.txt`), n=28/checkpoint.

---

## A. Fail-fast pretrain-level probe (DONE 00:48) — strong early signal
Vygotsky completion on PRETRAIN checkpoints (before SFT), same v2 sampler:
- **noisy-pretrain:** drifts off-domain -> "reactionary islands / tropical space / skin
  region / anterior" (geography/anatomy)
- **clean-pretrain:** stays in domain -> "anterior proximal growth / distal development /
  developmental delay / brain maturation / gestational age / Infancy Growth" — NO marine/island
=> corpus cleaning removed the proximal->off-domain association IN THE BASE MODEL.
This is matched pretrain-level evidence for Finding 2, independent of the SFT n=2 samples.
- Note: clean-pretrain UNCONDITIONAL degenerates ("the of the and", "effects effects") —
  small/homogeneous corpus (openalex abstracts) + all-masked init. SFT product is always
  conditioned, where clean is coherent — flagged, not blocking.

---

## B. Finding 1 — collapse = DECODE (temp-controlled), on noisy-SFT  [DONE]
v1 sampler @ temp0.9 vs v2 sampler @ temp0.9 (same temp -> isolates unmask-order+rep-control).
std-5, n=10 each.

| metric | v1@0.9 (random) | v2@0.9 (conf+rep) | delta |
|---|---|---|---|
| distinct-1 | 0.647 | 0.835 | +29% |
| distinct-2 | 0.929 | 0.971 | +0.042 |
| rep-1 | 0.353 | 0.165 | -53% |
| max-tok-freq | 0.090 | 0.064 | -29% |

**Honest attenuation vs the temp-confounded draft:** the earlier comparison (v1 @ {0.7,1.0})
reported rep-1 -62% / distinct-1 +46% / mtf -42%. At FIXED temp=0.9 the effect is SMALLER
(rep-1 -53% / distinct-1 +29% / mtf -29%) -> part of the original delta WAS temperature
(v1@0.7 is more repetitive than v1@0.9). Confound confirmed and removed. The decode effect
nonetheless remains large and clearly significant. Finding 1 holds as temperature-controlled.
Files: `mdlm_sft_samples_v1_temp09.txt` (v1@0.9), `mdlm_sft_samples_v2.txt` (v2@0.9).

---

## C. Finding 2 — confab = DATA (corpus axis), human-labeled traps
v2 sampler on the 14 trap prompts (homonym-bait: proximal/distal, attachment/bonding,
cognitive load, control group, reinforcement, conditioning, transference, projection,
regression, fixation, drive, defense, arousal, reflection), 2 seeds each = 28 samples.

| corpus | human-verified confab / 28 | notes |
|---|---|---|
| noisy-66.8M | **16/28 (57%)** [DONE] | confab_traps_noisy66.txt |
| clean-66.8M | **8/28 (29%)** [DONE] | confab_traps_clean66.txt — HALVED vs noisy |

### RETRACTED: the non-blind "57% -> 29%" reduction was ANNOTATOR BIAS
My first labeling was non-blind (I knew which file was clean and expected a drop). Three
checks were run before publishing:

**Check 1 — CIs.** noisy 16/28 Wilson [0.39,0.74]; clean 8/28 [0.15,0.47]; diff Newcombe
[0.03,0.50]; Fisher p=0.058 (NOT sig at 0.05). "Halved/-49%" was over-precise already.

**Check 2 — BLIND re-annotation (decisive).** All 56 samples pooled, shuffled, corpus-stripped,
labeled by TWO independent sub-agents that knew neither corpus nor hypothesis:
| annotator | noisy | clean | diff | Fisher p |
|---|---|---|---|---|
| me (non-blind) | 16/28 (57%) | 8/28 (29%) | +0.28 | 0.058 |
| A (blind) | 11/28 (39%) | 10/28 (36%) | +0.04 | 1.000 |
| B (blind) | 7/28 (25%) | 7/28 (25%) | 0.00 | 1.000 |
Inter-annotator agreement A vs B: Cohen kappa 0.71 (the criterion is reliable; the two blind
labelers simply find NO corpus effect). => The corpus difference I "found" was confirmation
bias: liberal "drift" on noisy, lenient on clean. **Corpus cleaning produced NO measurable
reduction in trap-confab under blind annotation.** This is consistent with the traps being
mostly homonym-bait (a corpus-invariant capacity floor), per section C's own caveat.

**Check 3 — specificity (refutes a different confound).** Coping answers, noisy vs clean SFT:
TTR 0.899 vs 0.900, content-word% 0.556 vs 0.562, hapax 0.910 vs 0.907, avg-word-len 4.69 vs
4.99. Clean coping is NOT blander — so wherever effects exist they are not generic specificity loss.

### Finding 2 status: NOT ESTABLISHED by the trap metric
- Trap-confab shows no corpus effect blind. The headline reduction is retracted.
- Probe A (pretrain Vygotsky: clean stays developmental, noisy -> islands/tropical) is suggestive
  but ALSO non-blind and n=3; it cannot carry Finding 2 alone.
- Corpus composition (69% pile) remains a real but circumstantial prior, not a measured effect.
=> Honest claim: at 66.8M, corpus cleaning did NOT measurably reduce confabulation. Confab here
is dominated by a homonym/capacity floor that data cleaning at this scale does not move.

### CRITICAL nuance found during labeling: trap set mixes TWO drift types
The 14 trap prompts turned out to be mostly HOMONYM-bait, and noisy drift splits as:
- **Contamination-type** (corpus pulls off-domain text): Vygotsky -> "coastal/coral/shoreal"
  (pile marine ecology). ~2/16. THIS is what corpus cleaning targets.
- **Homonym/capacity-type** (the word has a strong non-psych sense the 60M model defaults to):
  drive->driving/vehicles, defense->military/court, transference->circumference,
  projection->injection, fixation->problem-solving/math, control->programming/units,
  reinforcement->body-morphology. ~14/16. This is a LEXICAL/CAPACITY floor — present
  regardless of corpus, NOT expected to be fixed by cleaning (the homonym exists either way).

### Consequence for reading the clean result
Corpus cleaning should drop CONTAMINATION-type drift, not the homonym floor. So:
- If clean trap-confab stays ~high -> likely the homonym floor; do NOT misread as "cleaning failed".
- The clean test of the CORPUS effect is specifically: (a) does Vygotsky->coastal disappear
  (already YES in the pretrain probe, section A); (b) does any off-domain-TEXT pull drop.
- Homonym confab is a 60M capacity issue; may partially improve at 120M scale (better
  disambiguation), NOT from 66.8M corpus cleaning. Report the two subsets separately.

Per-prompt noisy drift (YES=drift): Vygotsky 2/2, control 2/2, transference 2/2, projection 2/2,
fixation 2/2, drive 2/2, defense 2/2, reinforcement 1/2, classical 1/2; clean(no drift):
attachment 0/2, cognitive-load 0/2, regression 0/2, arousal 0/2, reflection 0/2.

---

## D. 120M-clean — LAUNCHED 2026-05-30 04:56 (as a planned scale experiment)
Reframed after the blind check demolished the confab-reduction justification.
- Original gate point 1 (confab down) — RETRACTED (blind annotation: no corpus effect).
- Point 2 (coping preserved, 14/14, specificity identical, CBT no longer collapses) — HOLDS.
- Point 3 (conditioned coherent) — HOLDS.
=> Justification for 120M-clean is NOT "cleaning fixes confab". It is: (a) the clean corpus is
demonstrably NOT WORSE (coping fluency + specificity preserved, collapse-fix reproduces), and
(b) 120M is the SCALE point of the L-design, run as a planned experiment (user-authorized),
to test whether scale moves the homonym/capacity floor (Vygotsky/fixation/classical) that
66.8M corpus cleaning did not. NOT run as a hope for a quality jump — connectivity/facts are
expected to remain beyond 120M too.
Config: MDLM_RUN=mdlm_120m MDLM_DATA=pretrain_en_clean_v2 arch 640/15/10/1664. PID running.

## Coverage note (for the paper)
This is an L-design (3 cells): noisy-66 / clean-66 / [pending] clean-120.
- noisy-66 vs clean-66 = DATA effect at fixed size: collapse-fix reproduces (decode), confab
  halves (data). Clean factorial on the two axes.
- clean-66 vs clean-120 = SCALE effect at fixed clean data (pending launch).
- Not run: noisy-120 (would complete the 2x2 but is expensive and off the critical question).

---
Date: 2026-05-30
Status: A done (clean wins at pretrain level). B/C pending GPU batches (running, sharing card
with clean-SFT — no throughput hit). D auto-evaluated when C completes.
