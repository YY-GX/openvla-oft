#!/bin/bash

# Atomic Skills Augmented Closer Training - OpenVLA-OFT LoRA Finetuning on Augmented Closer Dataset
#
# Note: This dataset has a severe imbalance (90% augmented, 10% original demos).
# Balanced sampling is now ENABLED by default (50/50 original vs augmented).
# To disable balanced sampling, remove:
#   --use_balanced_sampling True \
#   --original_demo_ratio 0.5 \

CPUS_PER_TASK=32
GPUS=4
JOB_NAME="finetune_openvla_atomic_skills_augmented_closer"
LOG_DIR="$ENDPOINT/pkgs_baselines/openvla-oft/logs"
LOG_FILE="train_atomic_skills_augmented_closer_%j.out"

# Training on the new augmented closer dataset
sbatch \
  --partition=h100 \
  --cpus-per-task=$CPUS_PER_TASK \
  --gpus=$GPUS \
  -o "$LOG_DIR/$LOG_FILE" \
  -J $JOB_NAME \
  --wrap="torchrun --standalone --nnodes 1 --nproc-per-node $GPUS vla-scripts/finetune.py \
    --is_local_policy True \
    --grad_accumulation_steps 1 \
    --num_images_in_input 1 \
    --vla_path openvla/openvla-7b \
    --data_root_dir datasets/rlds_datasets \
    --dataset_name libero_atomic_skills_augmented_closer \
    --run_root_dir runs/libero_atomic_skills_augmented_closer/1.0.0 \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --use_proprio True \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 100005 \
    --save_freq 10000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --wandb_entity 'yygx' \
    --wandb_project 'openvla-oft-atomic-skills-augmented-closer' \
    --run_id_note 'atomic_skills_augmented_closer--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state' \\
    --use_balanced_sampling True \\
    --original_demo_ratio 0.5"
