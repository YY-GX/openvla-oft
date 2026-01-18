#!/bin/bash

# Atomic Skills Above Training - Resume from 40000 checkpoint
#
# Note: This resumes training from the 40000 checkpoint.

CPUS_PER_TASK=64
GPUS=8
JOB_NAME="above_atomic_resume"
LOG_DIR="$ENDPOINT/pkgs_baselines/openvla-oft/logs"
LOG_FILE="train_above_atomic_resume_%j.out"

# Resume training from 40000 checkpoint
sbatch \
  --nodelist=mirage.ib \
  --cpus-per-task=$CPUS_PER_TASK \
  --gpus=$GPUS \
  -o "$LOG_DIR/$LOG_FILE" \
  -J $JOB_NAME \
  --wrap="torchrun --standalone --nnodes 1 --nproc-per-node $GPUS vla-scripts/finetune.py \
    --resume True \
    --resume_step 40000 \
    --vla_path 'runs/libero_above_atomic/1.0.0/openvla-7b+libero_above_atomic+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--40000_chkpt' \
    --grad_accumulation_steps 2 \
    --data_root_dir datasets/rlds_datasets \
    --dataset_name libero_above_atomic \
    --run_root_dir runs/libero_above_atomic/1.0.0 \
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
    --wandb_project 'openvla-oft-atomic-skills-above' \
    --run_id_note 'atomic_skills_above--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state' \
    --num_images_in_input 1 \
    --is_local_policy True"
