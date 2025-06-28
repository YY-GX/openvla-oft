#!/bin/bash

CPUS_PER_TASK=4
GPUS=1
JOB_NAME="finetune_pose_vlm_single"
LOG_DIR="$ENDPOINT/pkgs_baselines/openvla-oft/logs"
LOG_FILE="train_pose_vlm_single_%j.out"

# Enable PyTorch elastic error tracebacks
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1

sbatch \
  --nodelist=megatron.ib \
  --cpus-per-task=$CPUS_PER_TASK \
  --gpus=$GPUS \
  -o "$LOG_DIR/$LOG_FILE" \
  -J $JOB_NAME \
  --wrap="python vla-scripts/finetune_pose.py \
    --num_images_in_input 1 \
    --resume False \
    --vla_path 'openvla/openvla-7b' \
    --data_root_dir datasets/local_pairs_datasets/ \
    --dataset_name pose_dataset \
    --run_root_dir runs/pose_vlm/1.0.0 \
    --pose_head_type gmm \
    --gmm_num_components 3 \
    --pose_dim 6 \
    --num_pose_tokens 6 \
    --use_film False \
    --use_proprio False \
    --batch_size 2 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 1000 \
    --save_freq 500 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --pose_aug True \
    --pose_aug_position_std 0.02 \
    --pose_aug_orientation_std 0.1 \
    --lora_rank 32 \
    --wandb_entity 'your-wandb-entity' \
    --wandb_project 'openvla-oft-pose' \
    --run_id_note 'single_gpu_test'" 