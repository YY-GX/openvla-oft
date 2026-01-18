#!/bin/bash

# Real Robot DROID Training - OpenVLA-OFT LoRA Finetuning on DROID Dataset (Both Views)
#
# This script trains on real robot data collected in DROID format.
# Uses both exterior camera (primary) and wrist camera views.
# Dataset location: datasets/rlds_datasets/real_datasets/droid

CPUS_PER_TASK=32
GPUS=2
JOB_NAME="droid_both_views"
LOG_DIR="$ENDPOINT/pkgs_baselines/openvla-oft/logs"
LOG_FILE="train_droid_both_views_%j.out"

# Increase NCCL timeout to prevent collective operation timeouts (30 minutes)
export NCCL_TIMEOUT=1800

sbatch \
  --partition=h100 \
  --cpus-per-task=$CPUS_PER_TASK \
  --gpus=$GPUS \
  -o "$LOG_DIR/$LOG_FILE" \
  -J $JOB_NAME \
  --wrap="torchrun --standalone --nnodes 1 --nproc-per-node $GPUS vla-scripts/finetune.py \
    --num_images_in_input 2 \
    --vla_path openvla/openvla-7b \
    --data_root_dir datasets/rlds_datasets/real_datasets/2026-01-03 \
    --dataset_name droid \
    --run_root_dir runs/droid_both_views_pick_red_cup_pick_yellow_mustard/1.0.0 \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --use_proprio True \
    --grad_accumulation_steps 1 \
    --batch_size 12 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 10000 \
    --max_steps 10005 \
    --save_freq 1000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --wandb_entity 'yygx' \
    --wandb_project 'openvla-oft-real-robot-droid' \
    --run_id_note 'droid_real_robot--8_acts_chunk--continuous_acts--L1_regression--both_views--proprio_state'"

