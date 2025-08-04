# VLM Training

This directory contains training scripts and configurations for fine-tuning PoseVLM models.

## Overview

The training system supports multiple pose prediction heads and training configurations:
- **GMM Head**: Gaussian Mixture Model for probabilistic pose prediction
- **Simple Head**: Direct pose regression
- **Hungarian Head**: Hungarian algorithm-based matching loss

## Files

- `finetune_pose.py`: Main training script with full configuration options
- `scripts/`: Shell scripts for different training configurations

## Training Scripts

### Basic Training
```bash
# Standard GMM-based training
bash vlm-pose/training/scripts/train_pose_vlm.sh
```

### Hungarian Loss Training
```bash
# Training with Hungarian matching loss
bash vlm-pose/training/scripts/train_pose_vlm_hungarian.sh
```

### Simple Head Training  
```bash
# Training with simple regression head
bash vlm-pose/training/scripts/train_pose_vlm_simple_head.sh
```

### Single GPU Training
```bash
# For smaller experiments or limited resources
bash vlm-pose/training/scripts/train_pose_vlm_single_gpu.sh
```

## Direct Training Command

```bash
python vlm-pose/training/finetune_pose.py \
    --vla_path 'openvla/openvla-7b' \
    --data_root_dir datasets/local_pairs_datasets/ \
    --dataset_name pose_dataset \
    --run_root_dir runs/pose_vlm/1.0.0 \
    --pose_head_type gmm \
    --gmm_num_components 6 \
    --gmm_entropy_weight 0.1 \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --max_steps 90000 \
    --save_interval 10000 \
    --image_aug \
    --pose_aug \
    --lora_rank 32 \
    --lora_dropout 0.0
```

## Key Parameters

### Model Configuration
- `--vla_path`: Base VLA model to fine-tune from
- `--pose_head_type`: Choose from 'gmm', 'simple', 'hungarian'
- `--lora_rank`: LoRA adaptation rank (default: 32)
- `--lora_dropout`: LoRA dropout rate

### GMM Head Parameters
- `--gmm_num_components`: Number of mixture components (default: 6)
- `--gmm_entropy_weight`: Entropy regularization weight
- `--gmm_min_epsilon`: Minimum covariance value

### Training Configuration
- `--batch_size`: Training batch size
- `--learning_rate`: Learning rate for optimization
- `--max_steps`: Maximum training steps
- `--save_interval`: Steps between checkpoint saves
- `--eval_interval`: Steps between evaluations

### Data Augmentation
- `--image_aug`: Enable image augmentation
- `--pose_aug`: Enable pose-specific augmentation
- `--num_images_in_input`: Number of input images (default: 1)

### Dataset Configuration
- `--data_root_dir`: Root directory for pose datasets
- `--dataset_name`: Dataset type (pose_dataset, hungarian_pose_dataset)
- `--splits_folder`: Subfolder containing data splits

## Output Structure

Training outputs are saved to `--run_root_dir` with the following structure:
```
runs/pose_vlm/1.0.0/{experiment_name}/
├── vla--{step}_checkpoint.pt     # Model checkpoints
├── config.yaml                   # Training configuration
├── metrics.json                  # Training metrics
└── logs/                        # Training logs
```

## Monitoring Training

- **Weights & Biases**: Automatic logging of training metrics
- **Console Output**: Real-time training progress
- **Checkpoint Files**: Saved every `--save_interval` steps

## Tips for Training

1. **Start Small**: Use single GPU training first to validate configuration
2. **Monitor Loss**: Watch for overfitting, especially with small datasets
3. **Experiment with Components**: Try different GMM component numbers
4. **Use Augmentation**: Image and pose augmentation often improve generalization
5. **Adjust Learning Rate**: Start with 5e-4, reduce if training is unstable

## Common Issues

- **CUDA Memory**: Reduce batch size if encountering OOM errors
- **Slow Training**: Ensure proper GPU utilization and data loading
- **NaN Loss**: Reduce learning rate or check data preprocessing
- **Poor Convergence**: Try different pose head types or augmentation settings