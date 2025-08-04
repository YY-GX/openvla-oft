#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate openvla-oft

# Launch 8 evaluation jobs for different checkpoints and settings
# Each job runs on a different GPU and logs output to a unique file in the checkpoint folder

# Helper: get log file name from ckpt path and eval type
get_logfile() {
  ckpt_path="$1"
  eval_type="$2"
  # Remove trailing .pt and replace / with _
  base_name=$(basename "$ckpt_path" .pt)
  ckpt_dir=$(dirname "$ckpt_path")
  echo "$ckpt_dir/eval_${eval_type}_${base_name}.log"
}

# Arguments common to all runs
DATA_ROOT="datasets/local_pairs_datasets"
BATCH_SIZE=32
NUM_WORKERS=32
SAMPLE_SIZE=1000
NUM_RUNS=1
SEED=42

# Specify the total number of samples for each split
TOTAL_TRAIN=1000   # <-- FILL IN with actual number of train samples
TOTAL_VAL=1000     # <-- FILL IN with actual number of val samples
TOTAL_TEST=1000    # <-- FILL IN with actual number of test samples

# List of jobs: (ckpt_path|eval_type|gpu_id|split)
jobs=(
  # simple head, smaller_splits
  "runs/pose_vlm/1.0.0/openvla-7b+pose_dataset+b8+lr-1e-05+pose_simple+lora-r32+dropout-0.0--image_aug--pose_aug--simple_pose_head_pose_aug--smaller_splits/vla--15000_checkpoint.pt|skill_wise|0|train"
  "runs/pose_vlm/1.0.0/openvla-7b+pose_dataset+b8+lr-1e-05+pose_simple+lora-r32+dropout-0.0--image_aug--pose_aug--simple_pose_head_pose_aug--smaller_splits/vla--15000_checkpoint.pt|demo_wise|1|train"
  "runs/pose_vlm/1.0.0/openvla-7b+pose_dataset+b8+lr-1e-05+pose_simple+lora-r32+dropout-0.0--image_aug--pose_aug--simple_pose_head_pose_aug--smaller_splits/vla--20000_checkpoint.pt|skill_wise|2|train"
  "runs/pose_vlm/1.0.0/openvla-7b+pose_dataset+b8+lr-1e-05+pose_simple+lora-r32+dropout-0.0--image_aug--pose_aug--simple_pose_head_pose_aug--smaller_splits/vla--20000_checkpoint.pt|demo_wise|3|train"
  # gmm head, smaller_splits
  "runs/pose_vlm/1.0.0/openvla-7b+pose_dataset+b8+lr-1e-05+pose_gmm_c3+lora-r32+dropout-0.0--image_aug--pose_aug--gmm_pose_head--3_components--pose_aug--smaller_splits/vla--15000_checkpoint.pt|skill_wise|4|train"
  # gmm head, splits
  "runs/pose_vlm/1.0.0/openvla-7b+pose_dataset+b8+lr-1e-05+pose_gmm_c3+lora-r32+dropout-0.0--image_aug--pose_aug--gmm_pose_head--3_components--pose_aug--splits/vla--30000_checkpoint.pt|skill_wise|5|val"
  "runs/pose_vlm/1.0.0/openvla-7b+pose_dataset+b8+lr-1e-05+pose_gmm_c3+lora-r32+dropout-0.0--image_aug--pose_aug--gmm_pose_head--3_components--pose_aug--splits/vla--30000_checkpoint.pt|demo_wise|6|val"
  # gmm head, splits, different lr
  "runs/pose_vlm/1.0.0/openvla-7b+pose_dataset+b8+lr-0.0005+pose_gmm_c3+lora-r32+dropout-0.0--image_aug--pose_aug--gmm_pose_head--3_components--pose_aug--splits/vla--10000_checkpoint.pt|skill_wise|7|test"
)

# Generate sample indices for all splits in one file
python utils/generate_sample_indices.py $SAMPLE_SIZE $SEED $TOTAL_TRAIN $TOTAL_VAL $TOTAL_TEST
SAMPLE_INDICES_FILE="runs/eval_results/vlm_pose_generator/sample_indices.json"

# Launch jobs
for job in "${jobs[@]}"; do
  IFS='|' read -r CKPT_PATH EVAL_TYPE GPU_ID SPLIT <<< "$job"

  # Determine splits_folder from checkpoint path
  if [[ "$CKPT_PATH" == *"--smaller_splits"* ]]; then
    SPLITS_FOLDER="smaller_splits"
  else
    SPLITS_FOLDER="splits"
  fi

  # Select script
  if [[ "$EVAL_TYPE" == "skill_wise" ]]; then
    SCRIPT="vla-scripts/evaluate_pose_vlm_skill_aware.py"
  else
    SCRIPT="vla-scripts/evaluate_pose_vlm_demo_aware.py"
  fi

  LOGFILE=$(get_logfile "$CKPT_PATH" "$EVAL_TYPE")

  echo "Launching $SCRIPT on GPU $GPU_ID for $CKPT_PATH ($EVAL_TYPE), log: $LOGFILE"
  CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT \
    --checkpoint_path "$CKPT_PATH" \
    --data_root "$DATA_ROOT" \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --sample_size $SAMPLE_SIZE \
    --splits_folder $SPLITS_FOLDER \
    --num_runs $NUM_RUNS \
    --sample_indices_file $SAMPLE_INDICES_FILE \
    --split $SPLIT \
    > "$LOGFILE" 2>&1 &
done

# Print tail commands for real-time log monitoring
echo -e "\nTo monitor logs in real time, use:"
for job in "${jobs[@]}"; do
  IFS='|' read -r CKPT_PATH EVAL_TYPE GPU_ID SPLIT <<< "$job"
  LOGFILE=$(get_logfile "$CKPT_PATH" "$EVAL_TYPE")
  echo "tail -f $LOGFILE"
done

wait 