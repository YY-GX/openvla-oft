#!/bin/bash

# Atomic Skills Above Training - OpenVLA-OFT LoRA Finetuning on Above Atomic Dataset (Both Views)
#
# Note: This dataset contains augmented demos starting from "above region" poses.
# All demos are augmented (some shifted, some non-shifted).
# Balanced sampling can be used to balance shifted vs non-shifted demos if needed.
# Uses both agentview and wrist camera views.

CPUS_PER_TASK=64
GPUS=8
JOB_NAME="above_atomic_both_views"
LOG_DIR="$ENDPOINT/pkgs_baselines/openvla-oft/logs"
LOG_FILE="train_above_atomic_both_views_%j.out"

# Training on the new above atomic dataset with both views
sbatch \
  --nodelist=mirage.ib \
  --cpus-per-task=$CPUS_PER_TASK \
  --gpus=$GPUS \
  -o "$LOG_DIR/$LOG_FILE" \
  -J $JOB_NAME \
  --wrap="torchrun --standalone --nnodes 1 --nproc-per-node $GPUS vla-scripts/finetune.py \
    --num_images_in_input 2 \
    --vla_path openvla/openvla-7b \
    --data_root_dir datasets/rlds_datasets \
    --dataset_name libero_above_atomic_both_view \
    --run_root_dir runs/libero_above_atomic_both_view/1.0.0 \
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
    --wandb_project 'openvla-oft-atomic-skills-above-both-views' \
    --run_id_note 'atomic_skills_above--8_acts_chunk--continuous_acts--L1_regression--both_views--proprio_state'"

