#!/bin/zsh

echo "Submitting job..."

# Define variables
CPUS_PER_TASK=32
GPUS=8
LOG_DIR="$ENDPOINT/pkgs_baselines/openvla/logs"
LOG_FILE="train_libero_local1_local_policy_only_wrist_%j.out"
JOB_NAME="openvla_libero_local1_local_policy_only_wrist"
SCRIPT_PATH="/mnt/arc/yygx/pkgs_baselines/openvla/vla-scripts/finetune.py"
VLA_PATH="openvla/openvla-7b"
DATA_ROOT_DIR="/mnt/arc/yygx/pkgs_baselines/openvla/datasets"
DATASET_NAME="libero_local1"
RUN_ROOT_DIR="/mnt/arc/yygx/pkgs_baselines/openvla/runs/view_wrist/1.0.0"
LORA_RANK=32
BATCH_SIZE=4
LEARNING_RATE=5e-4
IMAGE_AUG=True
WANDB_PROJECT="openvla-oft-libero-only-wrist"
WANDB_ENTITY="yygx"
USE_L1_REGRESSION=True
USE_DIFFUSION=False
USE_FILM=False
NUM_IMAGES_IN_INPUT=1
USE_PROPRIO=True
USE_LOCAL_POLICY=True
NUM_STEPS_BEFORE_DECAY=100000
MAX_STEPS=150005
SAVE_FREQ=10000
SAVE_LATEST_CHECKPOINT_ONLY=False
RUN_ID_NOTE="parallel_dec--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state"

echo "Parameterize finish"

# Submit the job
sbatch \
  --cpus-per-task=$CPUS_PER_TASK \
  --gpus=$GPUS \
#  -o "$LOG_DIR/$LOG_FILE" \
  -o "tmp.out" \
  -J $JOB_NAME \
  --wrap="torchrun --standalone --nnodes 1 --nproc-per-node $GPUS $SCRIPT_PATH \
    --is_local_policy $USE_LOCAL_POLICY \
    --num_images_in_input $NUM_IMAGES_IN_INPUT \
    --vla_path $VLA_PATH \
    --data_root_dir $DATA_ROOT_DIR \
    --dataset_name $DATASET_NAME \
    --run_root_dir $RUN_ROOT_DIR \
    --use_l1_regression $USE_L1_REGRESSION \
    --use_diffusion $USE_DIFFUSION \
    --use_film $USE_FILM \
    --use_proprio $USE_PROPRIO \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_steps_before_decay $NUM_STEPS_BEFORE_DECAY \
    --max_steps $MAX_STEPS \
    --save_freq $SAVE_FREQ \
    --save_latest_checkpoint_only $SAVE_LATEST_CHECKPOINT_ONLY \
    --image_aug $IMAGE_AUG \
    --lora_rank $LORA_RANK \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --run_id_note \"$RUN_ID_NOTE\""
