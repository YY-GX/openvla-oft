#!/bin/bash

CPUS_PER_TASK=32
GPUS=8
JOB_NAME="finetune_pose_vlm_hungarian"
LOG_DIR="$ENDPOINT/pkgs_baselines/openvla-oft/logs"
LOG_FILE="train_pose_vlm_hungarian_%j.out"

# Enable PyTorch elastic error tracebacks
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1

sbatch \
  --nodelist=megatron.ib \
  --cpus-per-task=$CPUS_PER_TASK \
  --gpus=$GPUS \
  -o "$LOG_DIR/$LOG_FILE" \
  -J $JOB_NAME \
  --wrap="torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune_pose.py \
    --num_images_in_input 1 \
    --resume False \
    --vla_path 'openvla/openvla-7b' \
    --data_root_dir datasets/local_pairs_datasets/ \
    --splits_folder splits \
    --dataset_name pose_dataset \
    --run_root_dir runs/pose_vlm/1.0.0 \
    --pose_head_type hungarian \
    --hungarian_weight 0.1 \
    --pose_dim 6 \
    --num_pose_tokens 6 \
    --use_film False \
    --use_proprio False \
    --batch_size 4 \
    --grad_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --num_steps_before_decay 100000 \
    --max_steps 20000 \
    --save_freq 2500 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --pose_aug True \
    --pose_aug_position_std 0.02 \
    --pose_aug_orientation_std 0.1 \
    --lora_rank 32 \
    --wandb_entity 'yygx' \
    --wandb_project 'openvla-oft-pose' \
    --run_id_note 'hungarian_pose_head--direct_prediction--6_gt_poses' \
    --use_val_set True \
    --val_freq 5000 \
    --val_time_limit 180" 