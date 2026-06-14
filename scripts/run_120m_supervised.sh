#!/usr/bin/env bash
# Supervisor for the 120M-clean MDLM pretrain.
# train_mdlm.py has auto-resume (loads latest checkpoints/mdlm_120m/step_* with
# optimizer+scheduler+step). So if a transient CUDA fault kills the process, we just
# relaunch and it continues from the last 500-step checkpoint. This does NOT change
# training dynamics (same script/config/state) — it only removes wall-clock gaps from
# transient GPU faults on an unattended overnight run.
#
# Cap restarts so a *deterministic* failure can't spin forever (it would crash fast
# each attempt -> ~min of thrash, visible in the supervisor log, then stop for a human).
set -u
cd /d/001_PROJECTS/psyche_diffuse_60m

export MDLM_RUN=mdlm_120m
export MDLM_DATA=pretrain_en_clean_v2
export MDLM_HIDDEN=640
export MDLM_LAYERS=15
export MDLM_HEADS=10
export MDLM_MLP_INTER=1664
export MDLM_MAX_STEPS=20000

SUP=raw_data/004_train/logs/mdlm_120m_supervisor.log
OUT=raw_data/004_train/logs/mdlm_120m_stdout.log
MAX_ATTEMPTS=12

echo "[supervisor] === start $(date) === cap=$MAX_ATTEMPTS" >> "$SUP"
for i in $(seq 1 $MAX_ATTEMPTS); do
  echo "[supervisor] launch attempt $i $(date)" >> "$SUP"
  .venv/Scripts/python.exe raw_data/004_train/scripts/train_mdlm.py >> "$OUT" 2>&1
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
