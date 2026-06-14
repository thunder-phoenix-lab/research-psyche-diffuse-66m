#!/usr/bin/env bash
# Supervisor for the 120M-clean MDLM SFT (DiffuSeq response-masking).
# train_mdlm_sft.py loads pretrain mdlm_120m/final strict=True and has its own
# auto-resume from checkpoints/mdlm_sft_120m/step_*. Same rationale as the pretrain
# supervisor: a transient CUDA fault shouldn't idle the GPU on an unattended ~6h run.
# Identical script/config/state on resume -> training dynamics unchanged.
set -u
cd /d/001_PROJECTS/psyche_diffuse_60m

export MDLM_SFT_RUN=mdlm_sft_120m
export MDLM_PRETRAIN=mdlm_120m
export MDLM_HIDDEN=640
export MDLM_LAYERS=15
export MDLM_HEADS=10
export MDLM_MLP_INTER=1664

SUP=raw_data/004_train/logs/mdlm_sft_120m_supervisor.log
OUT=raw_data/004_train/logs/mdlm_sft_120m_stdout.log
MAX_ATTEMPTS=12

echo "[supervisor] === start $(date) === cap=$MAX_ATTEMPTS" >> "$SUP"
for i in $(seq 1 $MAX_ATTEMPTS); do
  echo "[supervisor] launch attempt $i $(date)" >> "$SUP"
  .venv/Scripts/python.exe raw_data/004_train/scripts/train_mdlm_sft.py >> "$OUT" 2>&1
  rc=$?
  echo "[supervisor] attempt $i exited rc=$rc $(date)" >> "$SUP"
  if [ $rc -eq 0 ]; then
    echo "[supervisor] clean finish (rc=0) — stopping" >> "$SUP"
    break
  fi
  echo "[supervisor] non-zero exit — auto-resume in 15s" >> "$SUP"
  sleep 15
done
echo "[supervisor] === end $(date) ===" >> "$SUP"
