# VLM Utils

This directory contains utility functions and debugging tools for the VLM pose prediction system.

## Overview

The utility scripts provide essential support functions for:

1. **Debugging Tools**: Debug IK pipeline, pose evaluation, and system components
2. **Pose Processing**: Pose normalization and validation utilities  
3. **Data Tools**: Sample generation and data manipulation utilities
4. **Kinematics**: TraCIK (Track-Based Inverse Kinematics) tools

## Scripts

### Debug IK Pipeline
```bash
python vlm-pose/utils/debug_ik_pipeline.py \
    --pose_file target_poses.json \
    --robot_config robot_config.yaml \
    --output_dir debug_results \
    --visualize_solutions \
    --save_debug_info
```

### Debug Pose Evaluation
```bash
python vlm-pose/utils/debug_pose_evaluation.py \
    --predictions_file predictions.json \
    --ground_truth_file ground_truth.json \
    --output_dir evaluation_debug \
    --error_threshold 0.05 \
    --generate_plots
```

### Test Pose Normalization
```bash
python vlm-pose/utils/test_pose_normalization.py \
    --data_root datasets/local_pairs_datasets \
    --test_samples 100 \
    --normalization_method "quaternion" \
    --output_report normalization_test.txt
```

### Generate Sample Indices
```bash
python vlm-pose/utils/generate_sample_indices.py \
    --data_root datasets/local_pairs_datasets \
    --output_file sample_indices.json \
    --sampling_strategy "uniform" \
    --num_samples 1000 \
    --stratify_by_skill
```

### TraCIK Tools
```bash
python vlm-pose/utils/tracik_tools.py \
    --robot_urdf robot.urdf \
    --target_poses poses.json \
    --output_file ik_solutions.json \
    --base_link "base_link" \
    --end_effector_link "gripper_link"
```

## Detailed Usage

### Debugging Tools

#### Debug IK Pipeline
Debug inverse kinematics issues and visualize solutions:
```bash
# Basic IK debugging
python vlm-pose/utils/debug_ik_pipeline.py \
    --pose_file target_poses.json \
    --robot_config configs/robot_config.yaml \
    --output_dir ik_debug

# Advanced IK debugging with visualization
python vlm-pose/utils/debug_ik_pipeline.py \
    --pose_file target_poses.json \
    --robot_config configs/robot_config.yaml \
    --output_dir ik_debug_detailed \
    --visualize_solutions \
    --save_joint_trajectories \
    --check_collisions \
    --timeout 5.0 \
    --max_iterations 100
```

#### Debug Pose Evaluation
Analyze pose prediction errors and debugging information:
```bash
# Basic evaluation debugging
python vlm-pose/utils/debug_pose_evaluation.py \
    --predictions_file results/predictions.json \
    --ground_truth_file datasets/ground_truth.json \
    --output_dir eval_debug

# Comprehensive evaluation debugging
python vlm-pose/utils/debug_pose_evaluation.py \
    --predictions_file results/predictions.json \
    --ground_truth_file datasets/ground_truth.json \
    --output_dir eval_debug_detailed \
    --error_threshold 0.02 \
    --generate_plots \
    --save_error_analysis \
    --skill_breakdown \
    --object_breakdown \
    --statistical_tests
```

### Data Processing Utilities

#### Test Pose Normalization
Validate pose normalization procedures:
```bash
# Test quaternion normalization
python vlm-pose/utils/test_pose_normalization.py \
    --data_root datasets/local_pairs_datasets \
    --test_samples 500 \
    --normalization_method "quaternion" \
    --output_report quaternion_test.txt

# Test multiple normalization methods
python vlm-pose/utils/test_pose_normalization.py \
    --data_root datasets/local_pairs_datasets \
    --test_samples 1000 \
    --normalization_methods "quaternion,euler,rotation_matrix" \
    --output_report normalization_comparison.txt \
    --generate_plots \
    --statistical_analysis
```

#### Generate Sample Indices
Create sampling indices for datasets:
```bash
# Uniform sampling
python vlm-pose/utils/generate_sample_indices.py \
    --data_root datasets/local_pairs_datasets \
    --output_file uniform_samples.json \
    --sampling_strategy "uniform" \
    --num_samples 2000

# Stratified sampling by skill
python vlm-pose/utils/generate_sample_indices.py \
    --data_root datasets/local_pairs_datasets \
    --output_file stratified_samples.json \
    --sampling_strategy "stratified" \
    --num_samples 2000 \
    --stratify_by_skill \
    --min_samples_per_skill 100

# Balanced sampling across objects
python vlm-pose/utils/generate_sample_indices.py \
    --data_root datasets/local_pairs_datasets \
    --output_file balanced_samples.json \
    --sampling_strategy "balanced" \
    --num_samples 1500 \
    --balance_by_object \
    --seed 42
```

### Kinematics Tools

#### TraCIK Integration
Use Track-Based Inverse Kinematics for robot control:
```bash
# Basic IK solving
python vlm-pose/utils/tracik_tools.py \
    --robot_urdf configs/robot.urdf \
    --target_poses target_poses.json \
    --output_file ik_solutions.json \
    --base_link "base_link" \
    --end_effector_link "gripper_link"

# Advanced IK with constraints
python vlm-pose/utils/tracik_tools.py \
    --robot_urdf configs/robot.urdf \
    --target_poses target_poses.json \
    --output_file constrained_ik.json \
    --base_link "base_link" \
    --end_effector_link "gripper_link" \
    --position_tolerance 0.001 \
    --orientation_tolerance 0.01 \
    --timeout 2.0 \
    --max_iterations 500 \
    --joint_limits configs/joint_limits.yaml
```

## Common Workflows

### Debugging Failed Poses
```bash
# 1. Debug pose evaluation to identify issues
python vlm-pose/utils/debug_pose_evaluation.py \
    --predictions_file failed_predictions.json \
    --ground_truth_file ground_truth.json \
    --output_dir pose_debug

# 2. Test pose normalization
python vlm-pose/utils/test_pose_normalization.py \
    --data_root datasets/local_pairs_datasets \
    --test_samples 100 \
    --focus_on_failures

# 3. Debug IK pipeline for problematic poses
python vlm-pose/utils/debug_ik_pipeline.py \
    --pose_file problematic_poses.json \
    --robot_config configs/robot_config.yaml \
    --output_dir ik_debug \
    --visualize_solutions
```

### Dataset Sampling Workflow
```bash
# 1. Generate balanced sample indices
python vlm-pose/utils/generate_sample_indices.py \
    --data_root datasets/local_pairs_datasets \
    --output_file training_samples.json \
    --sampling_strategy "stratified" \
    --num_samples 5000 \
    --stratify_by_skill

# 2. Test normalization on samples
python vlm-pose/utils/test_pose_normalization.py \
    --data_root datasets/local_pairs_datasets \
    --sample_file training_samples.json \
    --normalization_method "quaternion" \
    --output_report sample_normalization.txt
```

### IK Pipeline Validation
```bash
# 1. Generate test poses
python vlm-pose/utils/tracik_tools.py \
    --robot_urdf configs/robot.urdf \
    --generate_test_poses \
    --num_poses 100 \
    --workspace_bounds workspace.yaml \
    --output_file test_poses.json

# 2. Debug IK solutions
python vlm-pose/utils/debug_ik_pipeline.py \
    --pose_file test_poses.json \
    --robot_config configs/robot_config.yaml \
    --output_dir ik_validation \
    --check_reachability \
    --validate_solutions
```

## Key Parameters

### Debug IK Pipeline
- `--pose_file`: JSON file containing target poses
- `--robot_config`: Robot configuration YAML file
- `--visualize_solutions`: Generate visualization of IK solutions
- `--check_collisions`: Enable collision checking
- `--timeout`: IK solver timeout in seconds

### Debug Pose Evaluation
- `--predictions_file`: Model predictions JSON file
- `--ground_truth_file`: Ground truth poses JSON file
- `--error_threshold`: Error threshold for success/failure classification
- `--skill_breakdown`: Analyze errors by skill type
- `--statistical_tests`: Run statistical significance tests

### Generate Sample Indices
- `--sampling_strategy`: "uniform", "stratified", or "balanced"
- `--num_samples`: Total number of samples to generate  
- `--stratify_by_skill`: Ensure representation across skills
- `--balance_by_object`: Balance across different objects

### TraCIK Tools
- `--robot_urdf`: Robot URDF file path
- `--base_link`: Base link name for IK chain
- `--end_effector_link`: End effector link name
- `--position_tolerance`: Position tolerance for IK solutions
- `--orientation_tolerance`: Orientation tolerance for IK solutions

## Output Structure

### Debug Output
```
debug_results/
├── debug_summary.json          # Overall debug summary
├── error_analysis/             # Detailed error analysis
│   ├── position_errors.csv     # Position error breakdown
│   ├── orientation_errors.csv  # Orientation error breakdown
│   └── error_plots/            # Error visualization plots
├── ik_solutions/              # IK solution analysis
│   ├── successful_ik.json     # Successful IK solutions
│   ├── failed_ik.json         # Failed IK attempts
│   └── solution_visualizations/ # IK solution plots
└── validation_reports/        # Validation test results
    ├── normalization_test.txt  # Pose normalization test results
    └── sampling_validation.txt # Sampling validation results
```

## Tips for Using Utils

1. **Start with Basic Commands**: Use default parameters first, then customize
2. **Save Debug Info**: Always save debug information for later analysis
3. **Use Visualization**: Enable visualization for better understanding
4. **Validate Data**: Run validation scripts before training
5. **Monitor Performance**: Use timing and profiling tools for optimization

## Troubleshooting

- **IK Solution Failures**: Check robot URDF and joint limits
- **Pose Normalization Issues**: Verify pose format and coordinate systems
- **Memory Usage**: Reduce batch sizes for large datasets
- **Missing Dependencies**: Ensure TraCIK and robotics libraries are installed
- **File Format Errors**: Validate JSON file formats and data structures