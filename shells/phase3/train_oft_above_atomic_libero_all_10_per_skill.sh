#!/bin/bash

# Atomic Skills Above Libero All (10 Per Skill) Training - OpenVLA-OFT LoRA Finetuning
#
# Note: This dataset contains 10 demos per skill (downsampled from full dataset).
# All 27 unique atomic skills from long-horizon tasks (IDs 8, 9, 10) + LIBERO composite tasks
# Downsampling: 4 non-shifted + 3 standard_shift + 3 z_only_shift = 10 demos per skill

CPUS_PER_TASK=64
GPUS=8
JOB_NAME="above_atomic_libero_all_10_per_skill"
LOG_DIR="$ENDPOINT/pkgs_baselines/openvla-oft/logs"
LOG_FILE="train_above_atomic_libero_all_10_per_skill_%j.out"

# Training on the above atomic libero all dataset (10 demos per skill)
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
    --dataset_name libero_above_atomic_libero_all_10_per_skill \
    --run_root_dir runs/libero_above_atomic_libero_all_10_per_skill/1.0.0 \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --use_proprio True \
    --grad_accumulation_steps 2 \
    --batch_size 4 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 50005 \
    --save_freq 10000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --wandb_entity 'yygx' \
    --wandb_project 'openvla-oft-atomic-skills-above-libero-all-10-per-skill' \
    --run_id_note 'atomic_skills_above_libero_all_10_per_skill--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state'"
