#!/bin/bash

# Atomic Skills Above Long ID 2 Training - OpenVLA-OFT LoRA Finetuning on Above Atomic Long ID 2 Dataset
#
# Note: This dataset contains selected augmented demos starting from "above region" poses.
# Task 2: Complete Kitchen Organization (6 skills: pick black bowl, place black bowl on plate,
#         open top drawer, pick ketchup, place ketchup in drawer, close drawer).
# All demos are augmented (some shifted, some non-shifted).
# Balanced sampling can be used to balance shifted vs non-shifted demos if needed.

CPUS_PER_TASK=64
GPUS=8
JOB_NAME="above_atomic_id_2"
LOG_DIR="$ENDPOINT/pkgs_baselines/openvla-oft/logs"
LOG_FILE="train_above_atomic_id_2_%j.out"

# Training on the above atomic long ID 2 dataset (Task 2: Complete Kitchen Organization)
sbatch \
  --nodelist=mirage.ib \
  --cpus-per-task=$CPUS_PER_TASK \
  --gpus=$GPUS \
  -o "$LOG_DIR/$LOG_FILE" \
  -J $JOB_NAME \
  --wrap="torchrun --standalone --nnodes 1 --nproc-per-node $GPUS vla-scripts/finetune.py \
    --is_local_policy True \
    --num_images_in_input 1 \
    --vla_path openvla/openvla-7b \
    --data_root_dir datasets/rlds_datasets \
    --dataset_name libero_above_atomic_long_id2 \
    --run_root_dir runs/libero_above_atomic_long_id2/1.0.0 \
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
    --wandb_project 'openvla-oft-atomic-skills-above-id-2' \
    --run_id_note 'atomic_skills_above_id_2--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state'"
