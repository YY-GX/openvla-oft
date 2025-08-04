#!/bin/bash

# Test Combined Pipeline on PoseVLM Smaller Split Tasks
# This script tests the combined PoseVLM + IK + OpenVLA-OFT pipeline
# specifically on the 6 tasks used for PoseVLM smaller split training.

# Default parameters
POSE_VLM_CHECKPOINT="runs/pose_vlm/1.0.0/openvla-7b+pose_dataset+b8+lr-1e-05+pose_simple+lora-r32+dropout-0.0--image_aug--pose_aug--simple_pose_head_pose_aug--smaller_splits/vla--20000_checkpoint.pt"
OPENVLA_CHECKPOINT=""
NUM_EPISODES=5
MAX_IK_ATTEMPTS=3
SAVE_DEBUG_IMAGES=true
DEBUG_IMAGE_DIR="runs/eval_results/combined_eval/pose_vlm_tasks_debug"
DEVICE="cuda"
SEED=42

# Function to print usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Required options:"
    echo "  --openvla_checkpoint PATH     Path to OpenVLA-OFT checkpoint"
    echo ""
    echo "Optional options:"
    echo "  --pose_vlm_checkpoint PATH    Path to PoseVLM checkpoint (default: $POSE_VLM_CHECKPOINT)"
    echo "  --num_episodes N              Number of episodes per task (default: $NUM_EPISODES)"
    echo "  --max_ik_attempts N           Maximum IK attempts (default: $MAX_IK_ATTEMPTS)"
    echo "  --save_debug_images BOOL      Whether to save debug images (default: $SAVE_DEBUG_IMAGES)"
    echo "  --debug_image_dir PATH        Directory to save debug images (default: $DEBUG_IMAGE_DIR)"
    echo "  --device DEVICE               Device to use (default: $DEVICE)"
    echo "  --seed SEED                   Random seed (default: $SEED)"
    echo "  --help                        Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --openvla_checkpoint runs/view_wrist/1.0.3/.../50000_chkpt \\"
    echo "      --num_episodes 10 --max_ik_attempts 5"
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
        --num_episodes)
            NUM_EPISODES="$2"
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
echo "Testing Combined Pipeline on PoseVLM Smaller Split Tasks"
echo "========================================================"
echo "PoseVLM checkpoint: $POSE_VLM_CHECKPOINT"
echo "OpenVLA checkpoint: $OPENVLA_CHECKPOINT"
echo "Episodes per task: $NUM_EPISODES"
echo "Max IK attempts: $MAX_IK_ATTEMPTS"
echo "Save debug images: $SAVE_DEBUG_IMAGES"
echo "Debug image dir: $DEBUG_IMAGE_DIR"
echo "Device: $DEVICE"
echo "Seed: $SEED"
echo ""

# Create output directories
mkdir -p "runs/eval_results/combined_eval"
mkdir -p "$DEBUG_IMAGE_DIR"

# Run the test
echo "Starting PoseVLM tasks test..."
python vla-evaluation/combined_eval/test_pose_vlm_tasks.py \
    --pose_vlm_checkpoint "$POSE_VLM_CHECKPOINT" \
    --openvla_checkpoint "$OPENVLA_CHECKPOINT" \
    --num_episodes "$NUM_EPISODES" \
    --max_ik_attempts "$MAX_IK_ATTEMPTS" \
    --save_debug_images "$SAVE_DEBUG_IMAGES" \
    --debug_image_dir "$DEBUG_IMAGE_DIR" \
    --device "$DEVICE" \
    --seed "$SEED"

echo ""
echo "PoseVLM tasks test completed!"
echo "Results saved to: runs/eval_results/combined_eval/"
echo "Debug images saved to: $DEBUG_IMAGE_DIR" 