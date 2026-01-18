#!/bin/bash

# Baseline OFT-OBJ Long ID 10 Training - OpenVLA-OFT LoRA Finetuning on Baseline OFT-OBJ Dataset (H100)
#
# Note: This is a baseline dataset for comparison with the augmentation approach.
# Unlike the "above_atomic" datasets that use augmented demos starting from "above region" poses,
# this dataset provides target object pose directly in proprioception (14D state).
# State: EE pose (6D) + gripper (2D) + target object pose (6D) = 14D
# Task 10: Cooking Preparation Setup (6 skills: pick moka pot, place on stove, turn on stove,
#          pick frying pan, place on stove, open microwave).
# All demos are augmented (some shifted, some non-shifted).

CPUS_PER_TASK=64
GPUS=4
JOB_NAME="baseline_obj_id10_h100"
LOG_DIR="$ENDPOINT/pkgs_baselines/openvla-oft/logs"
LOG_FILE="train_baseline_obj_id10_h100_%j.out"

# Training on the baseline OFT-OBJ long ID 10 dataset (Task 10: Cooking Preparation Setup)
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
    --is_local_policy True \
    --num_images_in_input 1 \
    --vla_path openvla/openvla-7b \
    --data_root_dir datasets/rlds_datasets \
    --dataset_name libero_oft_obj_long_id10 \
    --run_root_dir runs/libero_oft_obj_long_id10/1.0.0 \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --use_proprio True \
    --grad_accumulation_steps 1 \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 100005 \
    --save_freq 10000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --wandb_entity 'yygx' \
    --wandb_project 'openvla-oft-baseline-obj-id-10' \
    --run_id_note 'baseline_obj_id_10--8_acts_chunk--continuous_acts--L1_regression--wrist_img--14d_state_with_target_obj_pose'"
