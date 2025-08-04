# VLM Evaluation

This directory contains evaluation scripts for PoseVLM models, including both standalone VLM evaluation and combined VLA+VLM pipeline evaluation.

## Overview

The evaluation system provides two types of evaluation:

1. **Standalone Evaluation** (`standalone/`): Direct VLM pose prediction evaluation
2. **Combined Evaluation** (`combined/`): End-to-end VLA+VLM robot control evaluation

## Standalone Evaluation

Located in `standalone/`, these scripts evaluate VLM pose prediction accuracy.

### Demo-Aware Evaluation
```bash
python vlm-pose/evaluation/standalone/evaluate_pose_vlm_demo_aware.py \
    --checkpoint_path runs/pose_vlm/checkpoint.pt \
    --data_root datasets/local_pairs_datasets \
    --batch_size 8 \
    --num_workers 4 \
    --sample_size 1000 \
    --num_runs 3
```

### Skill-Aware Evaluation
```bash
python vlm-pose/evaluation/standalone/evaluate_pose_vlm_skill_aware.py \
    --checkpoint_path runs/pose_vlm/checkpoint.pt \
    --data_root datasets/local_pairs_datasets \
    --batch_size 8 \
    --skills_subset "close_drawer,open_drawer" \
    --target_objects_subset "drawer_handle"
```

### Object-Aware Evaluation
```bash
python vlm-pose/evaluation/standalone/evaluate_pose_vlm_object_aware.py \
    --checkpoint_path runs/pose_vlm/checkpoint.pt \
    --data_root datasets/local_pairs_datasets \
    --batch_size 8 \
    --object_types "drawer_handle,cabinet_door_handle"
```

### Regularization/Hungarian Evaluation
```bash
python vlm-pose/evaluation/standalone/evaluate_pose_vlm_skill_aware_regu_or_hung.py \
    --checkpoint_path runs/pose_vlm/checkpoint.pt \
    --data_root datasets/local_pairs_datasets \
    --evaluation_type "hungarian" \
    --batch_size 8
```

## Combined Evaluation

Located in `combined/`, these scripts evaluate the complete VLA+VLM pipeline in robot environments.

### Main Combined Evaluation
```bash
python vlm-pose/evaluation/combined/run_combined_eval.py \
    --config vlm-pose/evaluation/combined/configs/combined_eval_config.yaml \
    --vla_model_path "openvla/openvla-7b" \
    --pose_vlm_checkpoint runs/pose_vlm/checkpoint.pt \
    --num_episodes 50 \
    --max_steps_per_episode 100
```

### Quick Combined Evaluation Scripts
```bash
# Run with H100 GPU configuration
bash vlm-pose/evaluation/combined/scripts/run_combined_eval_h100.sh

# Run simplified evaluation
bash vlm-pose/evaluation/combined/scripts/run_combined_eval_simple.sh

# Debug single skill
bash vlm-pose/evaluation/combined/scripts/test_debug_single_skill.sh
```

### Test Scripts
```bash
# Test pose VLM tasks
python vlm-pose/evaluation/combined/test_pose_vlm_tasks.py

# Debug single skill execution  
python vlm-pose/evaluation/combined/test_debug_single_skill.py

# Test complete pipeline
python vlm-pose/evaluation/combined/test_combined_pipeline.py
```

## Key Evaluation Metrics

### Standalone Metrics
- **Position Error**: L2 distance between predicted and target positions
- **Rotation Error**: Angular difference between predicted and target orientations  
- **Success Rate**: Percentage of predictions within error thresholds
- **Demo-Aware Accuracy**: Success considering all valid poses in demonstration

### Combined Metrics
- **Task Success Rate**: Percentage of successfully completed robot tasks
- **Episode Length**: Average steps to complete tasks
- **Pose Prediction Accuracy**: Accuracy of pose predictions during execution
- **IK Success Rate**: Success rate of inverse kinematics solutions

## Configuration Files

### Combined Evaluation Config (`combined/configs/combined_eval_config.yaml`)
```yaml
# Environment settings
env_name: "KITCHEN_SCENE4"
max_episodes: 50
max_steps_per_episode: 100

# Model settings
vla_model: "openvla/openvla-7b"
pose_vlm_checkpoint: "runs/pose_vlm/checkpoint.pt"

# Evaluation settings
debug_mode: false
save_videos: true
output_dir: "evaluation_results"
```

## Output Structure

Evaluation results are saved with timestamps and detailed metrics:

```
evaluation_results/
├── {timestamp}_evaluation_report.json    # Detailed metrics
├── {timestamp}_summary.txt              # Human-readable summary  
├── debug_images/                        # Debug visualizations
├── videos/                              # Episode recordings (if enabled)
└── logs/                                # Evaluation logs
```

## Advanced Usage

### Custom Skill Evaluation
```bash
python vlm-pose/evaluation/standalone/evaluate_pose_vlm_skill_aware.py \
    --checkpoint_path runs/pose_vlm/checkpoint.pt \
    --skills_subset "custom_skill_1,custom_skill_2" \
    --custom_config custom_eval_config.yaml
```

### Batch Evaluation
```bash
# Evaluate multiple checkpoints
for checkpoint in runs/pose_vlm/*/vla--*_checkpoint.pt; do
    python vlm-pose/evaluation/standalone/evaluate_pose_vlm_demo_aware.py \
        --checkpoint_path "$checkpoint" \
        --output_suffix "$(basename $(dirname $checkpoint))"
done
```

### Ablation Studies
```bash
# Compare different pose head types
python vlm-pose/evaluation/standalone/evaluate_pose_vlm_skill_aware_regu_or_hung.py \
    --checkpoint_path runs/pose_vlm/gmm_checkpoint.pt \
    --evaluation_type "gmm" \
    --output_dir "ablation_gmm"

python vlm-pose/evaluation/standalone/evaluate_pose_vlm_skill_aware_regu_or_hung.py \
    --checkpoint_path runs/pose_vlm/hungarian_checkpoint.pt \
    --evaluation_type "hungarian" \
    --output_dir "ablation_hungarian"
```

## Tips for Evaluation

1. **Use Demo-Aware**: For most accurate VLM evaluation, use demo-aware metrics
2. **Sample for Speed**: Use `--sample_size` for faster evaluation during development
3. **Multiple Runs**: Run evaluation multiple times for statistical significance
4. **Debug Mode**: Enable debug visualizations when investigating failures
5. **Combined Pipeline**: Use combined evaluation for final robot performance assessment

## Troubleshooting

- **CUDA Memory**: Reduce batch size if encountering OOM during evaluation
- **Slow Loading**: Ensure datasets are on fast storage (SSD recommended)
- **Environment Issues**: Check LIBERO environment setup for combined evaluation
- **Checkpoint Loading**: Verify checkpoint paths and model compatibility