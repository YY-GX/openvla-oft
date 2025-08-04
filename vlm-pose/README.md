# VLM Pose Prediction System

This directory contains all Vision-Language-Model (VLM) components for pose prediction in robotics applications. The VLM system predicts 6D poses (position + rotation) from vision and language inputs.

## Overview

The VLM pose prediction pipeline consists of four main components:

1. **Training**: Scripts for fine-tuning PoseVLM models
2. **Data Preparation**: Scripts for processing and preparing pose datasets
3. **Evaluation**: Both standalone VLM evaluation and combined VLA+VLM evaluation
4. **Visualization**: Tools for visualizing pose predictions and analysis
5. **Utils**: Utility functions and debugging tools

## Quick Start

### 1. Data Preparation
```zsh
# Extract target object poses from demonstrations
python vlm-pose/data_preparation/extract_target_object_poses.py

# Create Hungarian matching dataset
python vlm-pose/data_preparation/create_hungarian_dataset.py

# Analyze pose statistics
python vlm-pose/data_preparation/analyze_pose_statistics.py
```

### 2. Training
```zsh
# Basic pose VLM training
zsh vlm-pose/training/scripts/train_pose_vlm.sh

# Training with Hungarian loss
zsh vlm-pose/training/scripts/train_pose_vlm_hungarian.sh

# Single GPU training
zsh vlm-pose/training/scripts/train_pose_vlm_single_gpu.sh
```

### 3. Evaluation
```zsh
# Standalone VLM evaluation (demo-aware)
python vlm-pose/evaluation/standalone/evaluate_pose_vlm_demo_aware.py \
    --checkpoint_path runs/pose_vlm/checkpoint.pt \
    --data_root datasets/local_pairs_datasets \
    --batch_size 8

# Combined VLA+VLM evaluation
python vlm-pose/evaluation/combined/run_combined_eval.py \
    --config vlm-pose/evaluation/combined/configs/combined_eval_config.yaml
```

### 4. Visualization
```zsh
# Visualize pose predictions
python vlm-pose/visualization/visualize_vlm_pose_predictions.py \
    --checkpoint_path runs/pose_vlm/checkpoint.pt \
    --output_dir visualization_results

# Visualize skill pose diversity
python vlm-pose/visualization/visualize_skill_pose_diversity.py \
    --data_root datasets/local_pairs_datasets
```

## Directory Structure

```
vlm-pose/
├── README.md                    # This file
├── training/                    # Model training components
│   ├── README.md
│   ├── finetune_pose.py        # Main training script
│   └── scripts/                # Training shell scripts
├── evaluation/                  # Model evaluation components
│   ├── README.md
│   ├── standalone/             # Individual VLM evaluation
│   └── combined/               # VLA+VLM combined evaluation
├── data_preparation/           # Data processing and preparation
│   ├── README.md
│   └── *.py                    # Data preparation scripts
├── visualization/              # Visualization and analysis tools
│   ├── README.md
│   └── *.py                    # Visualization scripts
└── utils/                      # Utility functions and debugging
    ├── README.md
    └── *.py                    # Utility scripts
```

## Key Features

- **Multi-head Support**: GMM, Simple, and Hungarian pose heads
- **Demo-aware Evaluation**: Considers all valid poses in demonstrations
- **Pose Augmentation**: Built-in data augmentation for poses
- **Visualization Tools**: Comprehensive pose analysis and visualization
- **Combined Pipeline**: Integration with VLA for complete robot control
- **Flexible Training**: Support for various training configurations

## Common Workflows

### Training a New Model
1. Prepare your dataset using `data_preparation/` scripts
2. Configure training parameters in the shell scripts
3. Run training with appropriate script from `training/scripts/`
4. Monitor training with wandb logs

### Evaluating a Trained Model
1. Use standalone evaluation for VLM-specific metrics
2. Use combined evaluation for end-to-end robot performance
3. Visualize results with visualization tools

### Data Analysis
1. Use `analyze_pose_statistics.py` to understand your data
2. Use `compute_target_object_diversity.py` for diversity analysis
3. Use visualization tools to inspect pose predictions

## Requirements

- PyTorch with CUDA support
- Transformers library
- OpenVLA dependencies
- LIBERO environment (for evaluation)
- Additional requirements in main project requirements.txt

For detailed usage of each component, see the README files in the respective subdirectories.