# Collapse-fix in numbers — v1 vs v2 sampler (same 66.8M-noisy ckpt)

Word-level metrics on saved eval files (`mdlm_sft_samples.txt` = v1 random-unmask;
`mdlm_sft_samples_v2.txt` = v2 confidence-unmask + rep-penalty + no-repeat-ngram).
Script: `compute_collapse_metrics.py`. No weight change — pure decode comparison.

## FINDING 1 — Repetition collapse = DECODE problem -> v2 FIXES it (SOLID)
Matched-5 prompts:
| metric | v1 (random) | v2 (conf+rep) | delta |
|---|---|---|---|
| distinct-1 | 0.571 | 0.835 | +46% |
| distinct-2 | 0.879 | 0.971 | +0.092 |
| distinct-3 | 0.957 | 0.993 | +0.036 |
| rep-1 | 0.429 | 0.165 | -62% |
| max-tok-freq | 0.111 | 0.064 | -42% |

- Large effects, correct metrics for this failure mode. Holds.
- Per-prompt where v1 collapsed hardest: CBT max-tok-freq 0.234->0.135 (distinct-2 0.740->0.957);
  Vygotsky max-tok-freq 0.103->0.070.
- seq-rep-4 is a WEAK discriminator here (0.011->0.001): collapse is token-level/short-cycle,
  not clean 4-gram loops -> max-tok-freq + rep-1 are the right metrics. (Kept in text on purpose.)

### Caveat that downgrades Finding 1 from "matched" to "directional"
NOT temperature-controlled: v1 used temp{0.7,1.0}, v2 used temp 0.9. Matched-5 controls the
prompt set, NOT temperature -> part of the diversity gain COULD be temperature, not the
confidence-unmask. Effect sizes are large (rep-1 -62%) so the DIRECTION is robust, but this is
"directionally robust, not temperature-controlled." A clean ablation needs v1 @ temp=0.9 (one
variable: unmask-order + rep-control). -> queued GPU TODO. Do not call this "matched" until done.

## FINDING 2 — Confabulation = DATA problem, NOT decode (DIRECTIONAL, weaker — handle honestly)

### The auto-regex is unreliable — demonstrated, not assumed
Narrowed marine/eco regex prescreen: v1 = 0/20, v2 = 2/18. This is MISLEADING: it would imply v2
confabulates more. Reality (human-read all 38 samples): drift domain VARIES, so a marine-only
regex cannot track it. Auto-confab metric on homonym prompts is principled noise -> human labels
mandatory. (This failure is itself evidence for the point.)

### Human-verified domain-drift (every sample labeled by reading)
| split | drift / total | where |
|---|---|---|
| v1 full | 3/20 | all Vygotsky: "seal...wilderland" (wildlife), "war/seism/brick" (geopolitics), "software/AI/operational" (tech) |
| v2 full | 2/18 | both Vygotsky: "coral/coastal/shoreal" (marine) |
| Vygotsky only | v1 3/4, v2 2/2 | both samplers drift hard on this homonym prompt |

### Honest claim
- Confab PERSISTS under both samplers and is NOT reduced by the decode fix — consistent with the
  data hypothesis (decode changed everything EXCEPT this).
- BUT n is tiny (Vygotsky n=4 / n=2). The earlier draft's "0.50->1.00 increase" was noise and is
  RETRACTED. No rate comparison is warranted at this n.
- Robust evidence that confab=data, independent of these samples: (a) corpus composition — pile =
  69% of tokens, loosely psych-filtered; (b) confab appears SPECIFICALLY on the factual/homonym
  prompt (Vygotsky: proximal/distal -> anatomy/marine) and NOT on coping prompts, under BOTH
  samplers. The samples are illustrative; the corpus stat is the load-bearing argument.

## Status of the two findings
- #1 (collapse=decode): solid effect sizes + right metrics; downgrade to "directionally robust,
  not temperature-controlled" until v1@temp0.9 ablation.
- #2 (confab=data): idea correct and supported by corpus composition, but the SAMPLE-level
  quantification is anecdotal (n<=4). Needs n>=20 trap-prompt set + human verification to stand alone.

## Queued (GPU — run when control pretrain frees the card)
1. v1 @ temp=0.9 re-gen on std-5 -> makes Finding 1 a true single-variable ablation.
2. Confab trap-set (confab_trap_prompts.txt, ~12 psych/homonym prompts x 2 seeds, both samplers)
   -> n>=20 human-labeled confab subset. Run on noisy AND clean ckpts.
3. Re-run these metrics on clean-66.8M -> 2-D table (decode x corpus): expect collapse-fix to
   reproduce on clean, and human-verified confab to DROP from corpus cleaning (the real test of #2).

---
Date: 2026-05-29
Takeaway: #1 (collapse=decode) strong, near-final pending temp ablation. #2 (confab=data)
directionally right via corpus composition but sample-quantification is anecdotal — strengthen
with n>=20 human-labeled traps on the clean checkpoint. Auto-regex confab retired for human labels.
