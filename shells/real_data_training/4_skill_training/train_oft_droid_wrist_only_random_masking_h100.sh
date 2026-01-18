#!/bin/bash

# Real Robot DROID Training - OpenVLA-OFT LoRA Finetuning on DROID Dataset (Wrist Camera Only)
#
# This script trains on real robot data collected in DROID format.
# Uses only wrist camera view (single image input).
# Dataset location: datasets/rlds_datasets/real_datasets/droid

CPUS_PER_TASK=64
GPUS=4
JOB_NAME="droid_wrist_only"
LOG_DIR="$ENDPOINT/pkgs_baselines/openvla-oft/logs"
LOG_FILE="train_droid_wrist_only_%j.out"

# Training on DROID dataset with wrist camera only
sbatch \
  --partition=h100 \
  --cpus-per-task=$CPUS_PER_TASK \
  --gpus=$GPUS \
  -o "$LOG_DIR/$LOG_FILE" \
  -J $JOB_NAME \
  --wrap="torchrun --standalone --nnodes 1 --nproc-per-node $GPUS vla-scripts/finetune.py \
    --is_local_policy True \
    --num_images_in_input 1 \
    --vla_path openvla/openvla-7b \
    --data_root_dir datasets/rlds_datasets/real_datasets/2026-01-07_4_skills \
    --dataset_name droid \
    --run_root_dir runs/droid_wrist_only_4_skills_masking/1.0.2 \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --use_proprio True \
    --grad_accumulation_steps 1 \
    --batch_size 12 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 20005 \
    --save_freq 2000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --wandb_entity 'yygx' \
    --wandb_project 'openvla-oft-real-robot-droid' \
    --run_id_note 'droid_real_robot--8_acts_chunk--continuous_acts--L1_regression--wrist_only--proprio_state'"

