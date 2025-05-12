#!/bin/bash

CPUS_PER_TASK=32
GPUS=8
JOB_NAME="finetune_openvla"
LOG_DIR="$ENDPOINT/pkgs_baselines/openvla-oft/logs"
LOG_FILE="train_libero_local1_local_policy_only_wrist_%j.out"

sbatch \
  --cpus-per-task=$CPUS_PER_TASK \
  --gpus=$GPUS \
  -o "$LOG_DIR/$LOG_FILE" \
  -J $JOB_NAME \
  --wrap="torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
    --resume True \
    --resume_step 20000 \
    --vla_path '/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/view_wrist/1.0.0/openvla-7b+libero_local1+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--20000_chkpt' \
    --is_local_policy True \
    --num_images_in_input 1 \
    --data_root_dir datasets/rlds_datasets/ \
    --dataset_name libero_local1 \
    --run_root_dir runs/view_wrist/1.0.0 \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --use_proprio True \
    --batch_size 4 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 150005 \
    --save_freq 10000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 32 \
    --wandb_entity 'yygx' \
    --wandb_project 'openvla-oft-libero-wrist' \
    --run_id_note 'parallel_dec--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state'"
