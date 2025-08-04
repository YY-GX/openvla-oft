#!/bin/bash

# Combined PoseVLM + OpenVLA-OFT Evaluation Script
# This script runs the combined evaluation pipeline

# Default parameters
POSE_VLM_CHECKPOINT=""
OPENVLA_CHECKPOINT=""
TASK_SUITE_NAME="boss_44"
USE_POSE_VLM_SMALLER_SPLIT_TASKS_ONLY=false
NUM_EPISODES_PER_TASK=20
MAX_IK_ATTEMPTS=5
SAVE_DEBUG_IMAGES=true
DEBUG_IMAGE_DIR="runs/eval_results/combined_eval/debug_images"
DEVICE="cuda"
SEED=42

# Function to print usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Required options:"
    echo "  --pose_vlm_checkpoint PATH    Path to PoseVLM checkpoint"
    echo "  --openvla_checkpoint PATH     Path to OpenVLA-OFT checkpoint"
    echo ""
    echo "Optional options:"
    echo "  --task_suite_name NAME        LIBERO task suite name (default: boss_44)"
    echo "  --use_pose_vlm_smaller_split_tasks_only  Only evaluate on PoseVLM smaller split training tasks (default: false)"
    echo "  --wrist_only                   Use only wrist camera view for OpenVLA-OFT"
    echo "  --third_person_view_only       Use only third-person camera view for OpenVLA-OFT"
    echo "  --both_views                   Use both wrist and third-person camera views for OpenVLA-OFT"
    echo "  --num_episodes_per_task N     Number of episodes per task (default: 20)"
    echo "  --max_ik_attempts N           Maximum IK attempts per pose prediction (default: 5)"
    echo "  --save_debug_images BOOL      Whether to save debug images (default: true)"
    echo "  --debug_image_dir PATH        Directory to save debug images"
    echo "  --device DEVICE               Device to run evaluation on (default: cuda)"
    echo "  --seed SEED                   Random seed for reproducibility (default: 42)"
    echo "  --help                        Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --pose_vlm_checkpoint runs/pose_vlm/1.0.0/.../vla--90000_checkpoint.pt \\"
    echo "      --openvla_checkpoint runs/view_wrist/1.0.3/.../50000_chkpt \\"
    echo "      --num_episodes_per_task 10 --max_ik_attempts 3"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pose_vlm_checkpoint)
            POSE_VLM_CHECKPOINT="$2"
            shift 2
            ;;
        --openvla_checkpoint)
            OPENVLA_CHECKPOINT="$2"
            shift 2
            ;;
        --task_suite_name)
            TASK_SUITE_NAME="$2"
            shift 2
            ;;
        --use_pose_vlm_smaller_split_tasks_only)
            USE_POSE_VLM_SMALLER_SPLIT_TASKS_ONLY="$2"
            shift 2
            ;;
        --wrist_only)
            WRIST_ONLY=true
            shift
            ;;
        --third_person_view_only)
            THIRD_PERSON_VIEW_ONLY=true
            shift
            ;;
        --both_views)
            BOTH_VIEWS=true
            shift
            ;;
        --num_episodes_per_task)
            NUM_EPISODES_PER_TASK="$2"
            shift 2
            ;;
        --max_ik_attempts)
            MAX_IK_ATTEMPTS="$2"
            shift 2
            ;;
        --save_debug_images)
            SAVE_DEBUG_IMAGES="$2"
            shift 2
            ;;
        --debug_image_dir)
            DEBUG_IMAGE_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check required parameters
if [[ -z "$POSE_VLM_CHECKPOINT" ]]; then
    echo "Error: --pose_vlm_checkpoint is required"
    print_usage
    exit 1
fi

if [[ -z "$OPENVLA_CHECKPOINT" ]]; then
    echo "Error: --openvla_checkpoint is required"
    print_usage
    exit 1
fi

# Check if checkpoints exist
if [[ ! -f "$POSE_VLM_CHECKPOINT" ]]; then
    echo "Error: PoseVLM checkpoint not found: $POSE_VLM_CHECKPOINT"
    exit 1
fi

if [[ ! -f "$OPENVLA_CHECKPOINT" ]]; then
    echo "Error: OpenVLA checkpoint not found: $OPENVLA_CHECKPOINT"
    exit 1
fi

# Print configuration
echo "Combined PoseVLM + OpenVLA-OFT Evaluation"
echo "=========================================="
echo "PoseVLM checkpoint: $POSE_VLM_CHECKPOINT"
echo "OpenVLA checkpoint: $OPENVLA_CHECKPOINT"
echo "Task suite: $TASK_SUITE_NAME"
echo "Use PoseVLM smaller split tasks only: $USE_POSE_VLM_SMALLER_SPLIT_TASKS_ONLY"
echo "Wrist only: $WRIST_ONLY"
echo "Third person view only: $THIRD_PERSON_VIEW_ONLY"
echo "Both views: $BOTH_VIEWS"
echo "Episodes per task: $NUM_EPISODES_PER_TASK"
echo "Max IK attempts: $MAX_IK_ATTEMPTS"
echo "Save debug images: $SAVE_DEBUG_IMAGES"
echo "Debug image dir: $DEBUG_IMAGE_DIR"
echo "Device: $DEVICE"
echo "Seed: $SEED"
echo ""

# Create output directories
mkdir -p "runs/eval_results/combined_eval"
mkdir -p "$DEBUG_IMAGE_DIR"

# Run the evaluation
echo "Starting combined evaluation..."
python vla-evaluation/combined_eval/run_combined_eval.py \
    --pose_vlm_checkpoint "$POSE_VLM_CHECKPOINT" \
    --openvla_checkpoint "$OPENVLA_CHECKPOINT" \
    --task_suite_name "$TASK_SUITE_NAME" \
    --use_pose_vlm_smaller_split_tasks_only "$USE_POSE_VLM_SMALLER_SPLIT_TASKS_ONLY" \
    --wrist_only "$WRIST_ONLY" \
    --third_person_view_only "$THIRD_PERSON_VIEW_ONLY" \
    --both_views "$BOTH_VIEWS" \
    --num_episodes_per_task "$NUM_EPISODES_PER_TASK" \
    --max_ik_attempts "$MAX_IK_ATTEMPTS" \
    --save_debug_images "$SAVE_DEBUG_IMAGES" \
    --debug_image_dir "$DEBUG_IMAGE_DIR" \
    --device "$DEVICE" \
    --seed "$SEED"

echo ""
echo "Combined evaluation completed!"
echo "Results saved to: runs/eval_results/combined_eval/"
echo "Debug images saved to: $DEBUG_IMAGE_DIR" 