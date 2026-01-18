#!/bin/bash

# Phase 1 Long Horizon Pipeline Evaluation - Wrist Camera Only
# Evaluates the long horizon manipulation pipeline using wrist camera observations

CPUS_PER_TASK=8
GPUS=1
JOB_NAME="eval_long_horizon_pipeline_wrist"
LOG_DIR="$ENDPOINT/pkgs_baselines/openvla-oft/logs"
LOG_FILE="eval_long_horizon_pipeline_wrist_%j.out"

# VLA checkpoint path (from atomic skills training)
VLA_CHECKPOINT="/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/atomic_skills/1.0.1/openvla-7b+libero_atomic_skills+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--200000_chkpt"

# Task options: "Cooking Preparation Setup", "Complete Kitchen Organization", "Switch Table Objects"
TASK_NAME="Cooking Preparation Setup"

sbatch \
  --nodelist=arcee.ib \
  --cpus-per-task=$CPUS_PER_TASK \
  --gpus=$GPUS \
  -o "$LOG_DIR/$LOG_FILE" \
  -J $JOB_NAME \
  --wrap="python scripts/phase1/execute_long_horizon_pipeline.py \
      --task_name \"$TASK_NAME\" \
      --vla_checkpoint \"$VLA_CHECKPOINT\" \
      --wrist_only"
