#!/bin/bash

# Atomic Skills Augmented Debug Training V1 - OpenVLA-OFT LoRA Finetuning on Augmented Debug Dataset
# Resumes from 200k checkpoint of atomic skills training

CPUS_PER_TASK=32
GPUS=8
JOB_NAME="finetune_openvla_atomic_skills_augmented_debug_v1"
LOG_DIR="$ENDPOINT/pkgs_baselines/openvla-oft/logs"
LOG_FILE="train_atomic_skills_augmented_debug_v1_%j.out"

# Training on the new augmented debug dataset
sbatch \
  --nodelist=mirage.ib \
  --cpus-per-task=$CPUS_PER_TASK \
  --gpus=$GPUS \
  -o "$LOG_DIR/$LOG_FILE" \
  -J $JOB_NAME \
  --wrap="torchrun --standalone --nnodes 1 --nproc-per-node $GPUS vla-scripts/finetune.py \
    --is_local_policy True \
    --grad_accumulation_steps 2 \
    --num_images_in_input 1 \
    --vla_path /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/atomic_skills/1.0.1/openvla-7b+libero_atomic_skills+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--200000_chkpt \
    --resume True \
    --resume_step 200000 \
    --data_root_dir datasets/rlds_datasets \
    --dataset_name libero_atomic_skills_augmented_debug \
    --run_root_dir runs/atomic_skills_augmented_debug/1.0.1 \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --use_proprio True \
    --batch_size 4 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 100005 \
    --save_freq 10000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --wandb_entity 'yygx' \
    --wandb_project 'openvla-oft-atomic-skills-augmented-debug' \
    --run_id_note 'atomic_skills_augmented_debug--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state'"
