#!/bin/bash

# Demo-aware evaluation script for PoseVLM
# This script evaluates the model using the correct demo-aware metrics

CHECKPOINT_PATH="runs/pose_vlm/1.0.0/openvla-7b+pose_dataset+b8+lr-0.0005+pose_gmm_c3+lora-r32+dropout-0.0--image_aug--pose_aug--gmm_pose_head--3_components--pose_aug/vla--90000_checkpoint.pt"
DATA_ROOT="datasets/local_pairs_datasets"
OUTPUT_FILE="runs/pose_vlm/1.0.0/openvla-7b+pose_dataset+b8+lr-0.0005+pose_gmm_c3+lora-r32+dropout-0.0--image_aug--pose_aug--gmm_pose_head--3_components--pose_aug/evaluation_results_demo_aware.json"

echo "Starting demo-aware evaluation of PoseVLM..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Data root: $DATA_ROOT"
echo "Output file: $OUTPUT_FILE"
echo ""

python vla-scripts/evaluate_pose_vlm_demo_aware.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --data_root "$DATA_ROOT" \
    --batch_size 32 \
    --num_workers 4 \
    --device cuda \
    --output_file "$OUTPUT_FILE"

echo ""
echo "Evaluation completed! Results saved to $OUTPUT_FILE" 