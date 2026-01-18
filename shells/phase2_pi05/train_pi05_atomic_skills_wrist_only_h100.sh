#!/bin/bash

# Pi0.5 Atomic Skills Wrist-Only LoRA Training (H100 Version)
# Fine-tuning Pi0.5 on Atomic Skills Augmented Closer Dataset with LoRA adapters
# Hardware: 4x H100 (80GB each) with FSDP
# Memory: ~22.5GB per GPU (LoRA) vs ~70GB (full fine-tuning)

# ============================================================================
# Configuration Parameters
# ============================================================================
BATCH_SIZE=64  # Per-device batch size (default: 64)
                     # Effective batch size = BATCH_SIZE * GPUS
                     # Examples:
                     #   - batch_size=32  -> effective=128
                     #   - batch_size=64  -> effective=256 (default)
                     #   - batch_size=128 -> effective=512
                     #   - batch_size=256 -> effective=1024

CPUS_PER_TASK=32
GPUS=2
JOB_NAME="finetune_pi05_atomic_skills_wrist_lora_h100"
LOG_DIR="$ENDPOINT/pkgs_baselines/openvla-oft/logs/pi05"
LOG_FILE="train_pi05_atomic_skills_wrist_lora_h100_bs${BATCH_SIZE}_%j.out"

# ============================================================================
# Calculated Values
# ============================================================================
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GPUS))

echo "============================================================================"
echo "Pi0.5 LoRA Training Configuration (H100)"
echo "============================================================================"
echo "Config name:            pi05_atomic_skills_wrist_only_closer_lora"
echo "Per-device batch size:  $BATCH_SIZE"
echo "Number of GPUs:         $GPUS (H100 80GB)"
echo "Effective batch size:   $EFFECTIVE_BATCH_SIZE"
echo "Log file:               $LOG_FILE"
echo "============================================================================"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Training Pi0.5 with LoRA on 4 H100 GPUs
# - Uses FSDP (fsdp_devices=4) for distributed training across 4 GPUs
# - LoRA adapters for memory-efficient fine-tuning
# - Wrist camera only (base camera masked)
# - H100 has 80GB VRAM, can use larger batch sizes than A6000
sbatch \
  --partition=h100 \
  --cpus-per-task=$CPUS_PER_TASK \
  --gpus=$GPUS \
  -o "$LOG_DIR/$LOG_FILE" \
  -J $JOB_NAME \
  --wrap="cd externals/openpi && \
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 && \
    export OPENPI_DATA_HOME=/mnt/arc/yygx/.cache/openpi && \
    export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 && \
    export JAX_ENABLE_COMPILATION_CACHE=0 && \
    bash apply_lerobot_fix.sh && \
    uv run scripts/train.py pi05_atomic_skills_wrist_only_closer_lora \
      --batch-size=$BATCH_SIZE \
      --fsdp-devices=$GPUS \
      --exp-name=atomic_skills_wrist_closer_lora_h100_bs${BATCH_SIZE}_${GPUS}gpu \
      --overwrite"
