#!/bin/bash

# Atomic Skills Above Long ID 9 Training - OpenVLA-OFT LoRA Finetuning on Above Atomic Long ID 9 Dataset
# Direct execution (no SLURM) - for shared GPU nodes
#
# Note: This dataset contains selected augmented demos starting from "above region" poses.
# Task 9: [Task description to be updated]
# All demos are augmented (some shifted, some non-shifted).
# Balanced sampling can be used to balance shifted vs non-shifted demos if needed.

GPUS=8
LOG_DIR="logs"
LOG_FILE="train_above_atomic_id_9_direct_$(date +%Y%m%d_%H%M%S).out"
mkdir -p "$LOG_DIR"

# NCCL settings for direct GPU access (avoiding SLURM conflicts)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_TIMEOUT=3600
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# WandB settings - use offline mode to avoid communication errors
export WANDB_MODE=offline

# Training on the above atomic long ID 9 dataset
torchrun --standalone --nnodes 1 --nproc-per-node $GPUS vla-scripts/finetune.py \
  --is_local_policy True \
  --num_images_in_input 1 \
  --vla_path openvla/openvla-7b \
  --data_root_dir datasets/rlds_datasets \
  --dataset_name libero_above_atomic_long_id9 \
  --run_root_dir runs/libero_above_atomic_long_id9/1.0.0 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --use_proprio True \
  --grad_accumulation_steps 2 \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 100005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity 'yygx' \
  --wandb_project 'openvla-oft-atomic-skills-above-id-9' \
  --run_id_note 'atomic_skills_above_id_9--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state' \
  2>&1 | tee "$LOG_DIR/$LOG_FILE"
