#!/bin/bash
# Convert atomic skills HDF5 dataset to LeRobot format
# Saves locally to datasets/lerobot_datasets/

set -e

# Set custom LeRobot home directory
export HF_LEROBOT_HOME="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/lerobot_datasets"

# Create directory if it doesn't exist
mkdir -p "$HF_LEROBOT_HOME"

echo "Converting atomic skills dataset to LeRobot format..."
echo "Input:  datasets/hdf5_datasets/atomic_local_demos_augmented_pick_10_place_15_atomic_15/combined_by_language"
echo "Output: $HF_LEROBOT_HOME/atomic_skills_wrist_only_closer"
echo ""
echo "This will process:"
echo "  - 64 HDF5 files"
echo "  - ~35,495 episodes"
echo "  - Estimated time: ~30-60 minutes"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Run conversion (WITHOUT --push_to_hub, so it only saves locally)
uv run examples/libero/convert_atomic_skills_to_lerobot.py \
    --data_dir ../../datasets/hdf5_datasets/atomic_local_demos_augmented_pick_10_place_15_atomic_15/combined_by_language \
    --output_repo_id atomic_skills_wrist_only_closer

echo ""
echo "✓ Conversion complete!"
echo "Dataset saved to: $HF_LEROBOT_HOME/atomic_skills_wrist_only_closer"
echo ""
echo "To use this dataset in training, make sure to set:"
echo "  export HF_LEROBOT_HOME=$HF_LEROBOT_HOME"
