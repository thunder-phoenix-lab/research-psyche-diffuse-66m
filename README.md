# Psyche Diffuse — small-scale text diffusion for a psychology assistant

Code, evaluation artifacts, and paper for the preprint:

> **Small-Scale Text Diffusion for a Psychology Assistant: A Discrete-Diffusion Pipeline, a Blind-Annotation Retraction, and an Honest Data-Bound Scaling Null** — Denis Kim, 2026. [`paper/preprint.pdf`](paper/preprint.pdf)

A 66.8M-parameter masked-diffusion LM (MDLM) for English psychology / mental-health dialogue, trained end-to-end on one RTX 3090 and deployed on-device (Android, ONNX). The paper reports, with matched controls and paired statistics: an architecture result (discrete ≫ continuous at this scale), a decoding result (repetition collapse is a decode artifact), a data result (strict cleaning costs nothing), a scale null (66.8M→120M at fixed data), and a self-caught annotation bias (the headline corpus effect was retracted after blind re-annotation, Fisher p=1.0).

**Weights:** 66.8M-clean (production) and 120M checkpoints — on Hugging Face (see `hf/README.md` model card).

## Headline numbers

| axis | result |
|---|---|
| decode (v1→v2 sampler, same weights, temp 0.9) | rep-1 0.353→0.165 (−53%), distinct-1 +29% |
| data (clean 0.77B vs noisy 2.56B, 725 paired) | cos d=+0.0094 [+0.0003,+0.0186], win 0.527 — cleaning is 3.3× cheaper, no quality loss |
| scale (120M vs 66.8M at fixed data, 725 paired) | cos d=−0.0094 [−0.0188,−0.0001], win 0.479; blind traps 7/28 vs 9/28, p=0.768 — null |
| retraction (non-blind→blind) | 57%→29% became 39%/36% & 25%/25%, p=1.0; κ=0.71 |

## Repo map

```
scripts/   train_mdlm.py, train_mdlm_sft.py     pretrain / SFT (env-overridable arch)
           sample_mdlm.py, sample_mdlm_v2.py    v1 / v2 samplers (v2 = production)
           build_clean_pretrain.py, build_*     corpus build recipe (tokenizer ships frozen: spm_en.model on HF)
           export_onnx.py, export_mobile.py     on-device export
eval/      run_1000_eval.py, paired_sig.py      1000-prompt harness + paired significance
           score_blind_traps.py                 blind-trap stats (Wilson/Newcombe/Fisher/kappa)
           blind_traps*.txt, *_key.json         pooled blind sets + decode keys
           annotations/annot_{A,B,C}.txt        raw blind labels (scale round)
           RESULTS_*.md                         primary result records
paper/     preprint.tex, preprint.pdf           the paper
app/       Android app sources (no model asset)
hf/        Hugging Face model card
```

## Reproduce

```bash
# pretrain 66.8M (defaults) / 120M (env)
python scripts/train_mdlm.py
MDLM_RUN=mdlm_120m MDLM_HIDDEN=640 MDLM_LAYERS=15 MDLM_HEADS=10 MDLM_MLP_INTER=1664 python scripts/train_mdlm.py
python scripts/train_mdlm_sft.py
python scripts/sample_mdlm_v2.py --prompt "user: I feel anxious about work. assistant:" --checkpoint <ckpt>
# stats reproduce bit-exact from shipped artifacts
python eval/score_blind_traps.py
```

## Data note

Corpora are not redistributed: the noisy mix is ~69% Pile-derived and the SFT mix has heterogeneous licenses. We ship the exact filter/build scripts and counts (noisy 2,559,832,146 tokens / clean 773,925,936 / SFT 261,443 pairs). The 1000-question eval set ships as prompts; gold answers can be rebuilt with `eval/build_eval_pairs.py` from source datasets.

## License & contact

Code and eval artifacts: Apache-2.0. Paper text and figures (`paper/`): CC-BY-4.0. Not a medical device; outputs may be wrong; not for crisis use. Crisis support: 988 (US), 112/local equivalents. Contact: kdleonidovich04@gmail.com.
