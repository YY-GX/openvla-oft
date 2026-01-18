#!/bin/bash

# Atomic Skills Above Long ID 10 Wrist Mask Training - OpenVLA-OFT LoRA Finetuning on Above Atomic Long ID 10 Wrist Mask Dataset (Both Views)
#
# Note: This dataset contains selected augmented demos starting from "above region" poses.
# Task 10: [Task description to be updated]
# All demos are augmented (some shifted, some non-shifted).
# Balanced sampling can be used to balance shifted vs non-shifted demos if needed.
# Uses both agentview and wrist camera views.

CPUS_PER_TASK=64
GPUS=4
JOB_NAME="wrist_mask"
LOG_DIR="$ENDPOINT/pkgs_baselines/openvla-oft/logs"
LOG_FILE="train_above_atomic_id_10_wrist_mask_%j.out"

# Training on the above atomic long ID 10 wrist mask dataset with both views
sbatch \
  --partition=h100 \
  --cpus-per-task=$CPUS_PER_TASK \
  --gpus=$GPUS \
  -o "$LOG_DIR/$LOG_FILE" \
  -J $JOB_NAME \
  --wrap="export WANDB_DISABLE_SERVICE=true && \
export TMPDIR=/mnt/arc/yygx/tmp && \
mkdir -p \$TMPDIR && \
torchrun --standalone --nnodes 1 --nproc-per-node $GPUS vla-scripts/finetune.py \
    --is_local_policy False \
    --num_images_in_input 2 \
    --vla_path openvla/openvla-7b \
    --data_root_dir datasets/rlds_datasets \
    --dataset_name libero_above_atomic_long_id10_wrist_mask \
    --run_root_dir runs/libero_above_atomic_long_id10_wrist_mask/1.0.0 \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --use_proprio True \
    --grad_accumulation_steps 1 \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 100005 \
    --save_freq 10000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --wandb_entity 'yygx' \
    --wandb_project 'openvla-oft-atomic-skills-above-id-10-wrist-mask-both-views' \
    --run_id_note 'atomic_skills_above_id_10_wrist_mask--8_acts_chunk--continuous_acts--L1_regression--both_views--proprio_state'"

