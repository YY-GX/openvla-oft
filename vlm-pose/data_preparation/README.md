# VLM Data Preparation

This directory contains scripts for processing and preparing pose datasets for VLM training and evaluation.

## Overview

The data preparation pipeline transforms raw demonstration data into structured datasets suitable for pose prediction training. Key steps include:

1. **Pose Extraction**: Extract target object poses from demonstrations
2. **Dataset Creation**: Create training/validation splits
3. **Data Analysis**: Analyze pose statistics and diversity
4. **Augmentation**: Prepare augmented datasets

## Scripts

### Core Data Processing

#### Extract Target Object Poses
```bash
python vlm-pose/data_preparation/extract_target_object_poses.py \
    --demo_root_dir /path/to/demonstrations \
    --output_dir datasets/extracted_poses \
    --env_name KITCHEN_SCENE4 \
    --target_objects "drawer_handle,cabinet_door_handle"
```

#### Extract Local Pairs
```bash
python vlm-pose/data_preparation/extract_local_pairs.py \
    --input_dir datasets/raw_demos \
    --output_dir datasets/local_pairs_datasets \
    --pair_distance_threshold 0.1 \
    --max_pairs_per_demo 50
```

#### Split Pose Data
```bash
python vlm-pose/data_preparation/split_pose_data.py \
    --data_root datasets/local_pairs_datasets \
    --output_dir datasets/local_pairs_datasets/splits \
    --train_ratio 0.8 \
    --val_ratio 0.15 \
    --test_ratio 0.05 \
    --random_seed 42
```

### Specialized Dataset Creation

#### Create Hungarian Dataset
```bash
python vlm-pose/data_preparation/create_hungarian_dataset.py \
    --input_dir datasets/local_pairs_datasets \
    --output_dir datasets/hungarian_datasets \
    --max_targets_per_sample 5 \
    --distance_threshold 0.05
```

#### Extract Sampled Target Object Poses
```bash
python vlm-pose/data_preparation/extract_sampled_target_object_poses.py \
    --demo_root_dir /path/to/demonstrations \
    --output_dir datasets/sampled_poses \
    --sample_rate 0.1 \
    --min_samples_per_skill 100
```

### Data Analysis

#### Analyze Pose Statistics
```bash
python vlm-pose/data_preparation/analyze_pose_statistics.py \
    --data_root datasets/local_pairs_datasets \
    --output_dir analysis_results \
    --generate_plots \
    --save_statistics
```

#### Analyze Skill Pose Statistics
```bash
python vlm-pose/data_preparation/analyze_skill_pose_stats.py \
    --data_root datasets/local_pairs_datasets \
    --skills_filter "close_drawer,open_drawer" \
    --output_dir skill_analysis
```

#### Compute Target Object Diversity
```bash
python vlm-pose/data_preparation/compute_target_object_diversity.py \
    --data_root datasets/local_pairs_datasets \
    --target_objects "drawer_handle,cabinet_door_handle" \
    --diversity_metrics "position,orientation,combined" \
    --output_file diversity_analysis.json
```

#### Check Target Object Poses
```bash
python vlm-pose/data_preparation/check_target_object_poses.py \
    --data_root datasets/local_pairs_datasets \
    --validate_poses \
    --check_duplicates \
    --output_report pose_validation_report.txt
```

## Data Pipeline Workflow

### Complete Data Preparation Pipeline
```bash
# 1. Extract poses from raw demonstrations
python vlm-pose/data_preparation/extract_target_object_poses.py \
    --demo_root_dir /path/to/raw_demos \
    --output_dir datasets/extracted_poses

# 2. Create local pairs dataset
python vlm-pose/data_preparation/extract_local_pairs.py \
    --input_dir datasets/extracted_poses \
    --output_dir datasets/local_pairs_datasets

# 3. Split data into train/val/test
python vlm-pose/data_preparation/split_pose_data.py \
    --data_root datasets/local_pairs_datasets \
    --output_dir datasets/local_pairs_datasets/splits

# 4. Analyze data quality and statistics
python vlm-pose/data_preparation/analyze_pose_statistics.py \
    --data_root datasets/local_pairs_datasets \
    --output_dir analysis_results

# 5. Validate poses and check for issues
python vlm-pose/data_preparation/check_target_object_poses.py \
    --data_root datasets/local_pairs_datasets \
    --output_report validation_report.txt

# 6. (Optional) Create Hungarian matching dataset
python vlm-pose/data_preparation/create_hungarian_dataset.py \
    --input_dir datasets/local_pairs_datasets \
    --output_dir datasets/hungarian_datasets
```

## Output Data Structure

### Standard Pose Dataset
```
datasets/local_pairs_datasets/
├── metadata.json                    # Dataset metadata
├── splits/                         # Data splits
│   ├── train.json
│   ├── val.json
│   └── test.json
├── images/                         # RGB images
│   ├── {scene}_{episode}_{step}.png
│   └── ...
├── poses/                          # 6D pose annotations
│   ├── {scene}_{episode}_{step}.json
│   └── ...
└── demonstrations/                 # Original demonstration data
    ├── {scene}_{episode}/
    └── ...
```

### Hungarian Dataset
```
datasets/hungarian_datasets/
├── metadata.json                   # Dataset metadata with Hungarian info
├── samples/                        # Hungarian matching samples
│   ├── {sample_id}.json           # Multiple target poses per sample
│   └── ...
├── images/                         # Shared image directory
└── mapping.json                    # Sample to image/pose mappings
```

## Key Parameters

### Pose Extraction
- `--target_objects`: Objects to extract poses for
- `--pose_threshold`: Minimum pose quality threshold
- `--distance_filter`: Filter poses by distance to robot

### Dataset Creation
- `--pair_distance_threshold`: Maximum distance for pose pairs
- `--max_pairs_per_demo`: Limit pairs per demonstration
- `--train_ratio`: Training set proportion

### Analysis
- `--diversity_metrics`: Types of diversity analysis
- `--generate_plots`: Create visualization plots
- `--save_statistics`: Save detailed statistics

## Data Quality Checks

### Automated Validation
```bash
# Run comprehensive data validation
python vlm-pose/data_preparation/check_target_object_poses.py \
    --data_root datasets/local_pairs_datasets \
    --validate_poses \
    --check_duplicates \
    --check_outliers \
    --pose_error_threshold 0.01 \
    --output_report full_validation.txt
```

### Statistical Analysis
```bash
# Generate comprehensive statistics
python vlm-pose/data_preparation/analyze_pose_statistics.py \
    --data_root datasets/local_pairs_datasets \
    --output_dir detailed_analysis \
    --generate_plots \
    --compute_correlations \
    --save_raw_data
```

## Tips for Data Preparation

1. **Start Small**: Process a subset first to validate the pipeline
2. **Check Quality**: Always run validation after data preparation
3. **Monitor Diversity**: Use diversity analysis to ensure varied training data
4. **Balance Skills**: Ensure balanced representation across different skills
5. **Validate Splits**: Check that train/val/test splits are appropriate
6. **Document Changes**: Keep track of preprocessing parameters

## Common Issues

- **Memory Usage**: Large datasets may require processing in batches
- **Disk Space**: Ensure sufficient storage for processed datasets
- **Pose Quality**: Filter out low-quality or corrupted poses
- **Class Imbalance**: Monitor distribution across skills and objects
- **File Organization**: Maintain consistent naming conventions