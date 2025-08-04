#!/bin/bash
# Object-aware evaluation for two checkpoints on test split, each on a different GPU on the same node

# Set 1: Simple pose head checkpoint (GPU 0)
CKPT1="runs/pose_vlm/1.0.0/openvla-7b+pose_dataset+b8+lr-1e-05+pose_simple+lora-r32+dropout-0.0--image_aug--pose_aug--simple_pose_head_pose_aug--smaller_splits/vla--20000_checkpoint.pt"
# Set 2: GMM pose head checkpoint (GPU 1)
CKPT2="runs/pose_vlm/1.0.0/openvla-7b+pose_dataset+b8+lr-1e-05+pose_gmm_c3+lora-r32+dropout-0.0--image_aug--pose_aug--gmm_pose_head--3_components--pose_aug--smaller_splits/vla--20000_checkpoint.pt"

DATA_ROOT="datasets/local_pairs_datasets"
SAMPLED_POSES_DIR="runs/eval_results/vlm_pose_generator/"
SAMPLE_INDICES_FILE="runs/eval_results/vlm_pose_generator/sample_indices.json"
BATCH_SIZE=32
SAMPLE_SIZE=1000
NUM_RUNS=1
NODELIST="megatron.ib"  # Change this to your desired node if needed

# Evaluate on test for CKPT1 (GPU 0)
sbatch --gres=gpu:1 --cpus-per-task 8 --mem 32G --nodelist=$NODELIST \
  -o outs/out_object_aware_simple_test_%j.out \
  -J eval_obj_aware_simple_test \
  --wrap="export CUDA_VISIBLE_DEVICES=0; python utils/evaluate_pose_vlm_object_aware.py \
    --checkpoint_path $CKPT1 \
    --data_root $DATA_ROOT \
    --split test \
    --batch_size $BATCH_SIZE \
    --sample_size $SAMPLE_SIZE \
    --num_runs $NUM_RUNS \
    --sample_indices_file $SAMPLE_INDICES_FILE \
    --sampled_target_object_poses_dir $SAMPLED_POSES_DIR"

echo "Submitted job for simple pose head checkpoint (GPU 0, test split)."

# Evaluate on test for CKPT2 (GPU 1)
sbatch --gres=gpu:1 --cpus-per-task 8 --mem 32G --nodelist=$NODELIST \
  -o outs/out_object_aware_gmm_test_%j.out \
  -J eval_obj_aware_gmm_test \
  --wrap="export CUDA_VISIBLE_DEVICES=1; python utils/evaluate_pose_vlm_object_aware.py \
    --checkpoint_path $CKPT2 \
    --data_root $DATA_ROOT \
    --split test \
    --batch_size $BATCH_SIZE \
    --sample_size $SAMPLE_SIZE \
    --num_runs $NUM_RUNS \
    --sample_indices_file $SAMPLE_INDICES_FILE \
    --sampled_target_object_poses_dir $SAMPLED_POSES_DIR"

echo "Submitted job for GMM pose head checkpoint (GPU 1, test split)." 