# Data-axis result — 1000-Q sensitive eval (clean-0.77B vs noisy-2.56B, 66.8M SFT)

Date: 2026-05-31. Eval-only run (no training) added after review: replaces §7's ceiling
metrics (14/14 binary, TTR 0.899/0.900) with the sensitive 1000-Q embedding-cosine already
used on the scale axis (§8.1 / `RESULTS_scale_1000q.md`). Same harness, same sampler, same
725 (prompt, gold) pairs. **Still confounded** (clean and noisy differ in BOTH volume and
purity) — not a token-controlled claim; it only upgrades the metric.

## Numbers

| metric | clean-0.77B SFT | noisy-2.56B SFT |
|---|---|---|
| cos-to-gold mean | **0.500** | **0.490** |
| distinct-2 | 0.993 | 0.991 |
| rep-1 | 0.137 | 0.141 |
| max-tok-freq | 0.038 | 0.042 |
| seq-rep-4 / empty / lang-mix | 0 / .001 / .002 | .001 / 0 / 0 |

### Paired significance (cosine, same 725 prompts; `paired_sig.py PAIR_A=sft66_clean PAIR_B=sft66_noisy`)
- paired d (clean − noisy) = **+0.0094**, sd 0.126
- 95% bootstrap CI = **[+0.0003, +0.0186]**, paired t(724) = **+2.01**
- win-rate clean>noisy = **0.527**

## Verdict
The clean corpus (3× fewer tokens) is **no worse — marginally better**, *just* clearing
significance (CI lower bound +0.0003) but **negligible in size** (win-rate 0.527, a coin-flip).
So across 0.77–2.56B the metrics do not meaningfully separate the corpora; **cleaning's benefit
is cost, with at most a slim general-fluency edge** — not a meaningful quality gain.

## Symmetry with the scale axis (a note on the metric, not the effects)
Mirror image of `RESULTS_scale_1000q.md`: data d=+0.0094 [+0.0003,+0.0186] (clean>noisy) vs
scale d=−0.0094 [−0.0188,−0.0001] (66.8M>120M) — **same magnitude, same boundary, opposite
sign.** Whatever else this shows, it rules out floor-effect blindness: the cosine *does* resolve
~0.01-scale differences; it simply finds both contrasts at the edge of detectability. We do **not**
claim the signs are stable (a different seed could flip either, §8.3) — only that both effects,
*if real*, are this small.

## Consistency with §6
Any edge here is in **general fluency**; it does not contradict the blind trap result (§6 /
`RESULTS_matched_clean_vs_noisy.md`), where cleaning produced no reduction in homonym-confab —
different failure surfaces.

Artifacts: `eval1000_sft66_noisy.jsonl`, `eval1000_sft66_noisy_metrics.txt`, `paired_sig.py`.
