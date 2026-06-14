# Scale result — 1000-Q matched eval (120M-clean vs 66.8M-clean, SFT)

Date: 2026-05-31. Matched 1000 in-distribution psych prompts (`eval_1000_questions.txt`),
identical v2 sampler (resp_len=110, steps=28, temp=0.9, top_p=0.9, rep=1.3, nb=2, nrg=3).
Reference-free metrics on all 1000; reference-based MiniLM cosine-to-gold on 725 (q,gold) pairs.

## Numbers

| metric | 120M-SFT | 66.8M-SFT | Δ (120−66) |
|---|---|---|---|
| avg_len (words) | 92.9 | 94.4 | −1.5 |
| distinct-1 | 0.853 | 0.863 | −0.010 |
| distinct-2 | 0.989 | 0.993 | −0.004 |
| distinct-3 | 0.998 | 0.999 | −0.001 |
| rep-1 | 0.147 | 0.137 | +0.010 |
| seq-rep-4 | 0.000 | 0.000 | 0 |
| max-tok-freq | 0.042 | 0.038 | +0.004 |
| confab-rate (prescreen) | 0.002 (2/1000) | 0.000 (0/1000) | +0.002 |
| empty-rate | 0.000 | 0.001 | ~0 |
| lang-mix-rate | 0.002 | 0.002 | 0 |
| **cos-to-gold mean** | **0.490** | **0.500** | **−0.0094** |

### Paired significance (cosine, same 725 prompts on both models)
- paired d (120−66) = **−0.0094**, sd 0.129
- 95% bootstrap CI = **[−0.0188, −0.0001]**, paired t(724) = **−1.96**
- win-rate 120>66 = **0.479** (near coin-flip)

## Verdict
**Scaling 66.8M → 120M produced NO practical quality gain on this eval — marginally
negative, right at the p≈0.05 boundary.** The cosine CI barely excludes 0 (upper edge
−0.0001) and the win-rate is 47.9%, so the two models are effectively interchangeable here;
"66.8M better" would overstate a 0.0094 effect. Every reference-free metric agrees (120M a
hair worse). Both are healthy: zero seq-rep-4 collapse, ~0 lang-mix, ~0 empty.

## Most likely mechanism — data-bound (Chinchilla), not a capacity ceiling
Both models pretrained on the SAME 0.7739B-token clean corpus:
- 66.8M → **11.6 tok/param**
- 121.4M → **6.4 tok/param**  (Chinchilla-optimal ≈ 20)

Both are under-trained; the 120M is **~1.8× more data-starved**, so its extra capacity
cannot be realized at fixed data — the expected data-bound regime where bigger ≠ better.
This is NOT evidence that scale is useless; it is evidence that 120M needs more tokens
(~2.4B for Chinchilla-optimal, ≈3× current) to pay off.

## Caveats
- **In-distribution** prompts (from the SFT sources), not held-out → measures target-recall, not generalization.
- **Gold = training answers** → cosine reflects recall of the target distribution, not unbiased quality.
- **Cosine ceiling is low (~0.49)** because outputs are ~90 words vs ~389-word golds and are rough; the metric may under-resolve quality differences. Win-rate (≈0.5) is the cleanest summary: the models are interchangeable on this eval.
- Complementary test pending: **blind homonym/capacity traps** (n≥20, fresh sub-agents) — does scale move the *specific* failure floor, which an aggregate metric may miss.

Artifacts: `eval1000_sft120_clean.jsonl`, `eval1000_sft66_clean.jsonl`,
`eval1000_sft{120,66}_clean_metrics.txt`, `paired_sig.py`.
