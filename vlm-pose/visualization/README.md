# VLM Visualization

This directory contains visualization and analysis tools for VLM pose predictions and dataset analysis.

## Overview

The visualization tools help understand model behavior, analyze predictions, and debug issues in the pose prediction pipeline. Key capabilities include:

1. **Pose Prediction Visualization**: Visualize model predictions vs ground truth
2. **Skill Diversity Analysis**: Analyze pose diversity across different skills
3. **Dataset Exploration**: Interactive exploration of pose datasets

## Scripts

### Visualize VLM Pose Predictions
```bash
python vlm-pose/visualization/visualize_vlm_pose_predictions.py \
    --checkpoint_path runs/pose_vlm/checkpoint.pt \
    --data_root datasets/local_pairs_datasets \
    --output_dir visualization_results \
    --num_samples 100 \
    --batch_size 8 \
    --save_predictions \
    --generate_comparison_plots
```

### Visualize Skill Pose Diversity
```bash
python vlm-pose/visualization/visualize_skill_pose_diversity.py \
    --data_root datasets/local_pairs_datasets \
    --skills_subset "close_drawer,open_drawer,pick_up_object" \
    --output_dir skill_diversity_analysis \
    --plot_type "3d_scatter" \
    --color_by_skill \
    --generate_statistics
```

## Detailed Usage

### Pose Prediction Visualization

#### Basic Visualization
```bash
# Visualize predictions for a trained model
python vlm-pose/visualization/visualize_vlm_pose_predictions.py \
    --checkpoint_path runs/pose_vlm/1.0.0/openvla-7b+pose_dataset+b8+lr-0.0005+gmm/vla--90000_checkpoint.pt \
    --data_root datasets/local_pairs_datasets \
    --output_dir results/predictions_vis \
    --num_samples 50
```

#### Advanced Visualization Options
```bash
# Comprehensive prediction analysis
python vlm-pose/visualization/visualize_vlm_pose_predictions.py \
    --checkpoint_path runs/pose_vlm/checkpoint.pt \
    --data_root datasets/local_pairs_datasets \
    --output_dir results/detailed_analysis \
    --num_samples 200 \
    --batch_size 16 \
    --skills_filter "close_drawer,open_drawer" \
    --objects_filter "drawer_handle" \
    --error_threshold 0.05 \
    --save_predictions \
    --save_debug_images \
    --generate_error_plots \
    --plot_format "png,pdf"
```

#### Specific Skill Analysis
```bash
# Focus on specific skills
python vlm-pose/visualization/visualize_vlm_pose_predictions.py \
    --checkpoint_path runs/pose_vlm/checkpoint.pt \
    --data_root datasets/local_pairs_datasets \
    --output_dir results/skill_specific \
    --skills_filter "close_drawer" \
    --num_samples 100 \
    --detailed_analysis \
    --save_failure_cases
```

### Skill Diversity Analysis

#### Basic Diversity Visualization
```bash
# Analyze pose diversity across skills
python vlm-pose/visualization/visualize_skill_pose_diversity.py \
    --data_root datasets/local_pairs_datasets \
    --output_dir results/diversity_analysis \
    --plot_type "3d_scatter"
```

#### Comprehensive Diversity Analysis
```bash
# Detailed diversity analysis with multiple metrics
python vlm-pose/visualization/visualize_skill_pose_diversity.py \
    --data_root datasets/local_pairs_datasets \
    --skills_subset "close_drawer,open_drawer,pick_up_object,put_down_object" \
    --output_dir results/comprehensive_diversity \
    --plot_types "3d_scatter,heatmap,distribution" \
    --diversity_metrics "position,orientation,combined" \
    --color_by_skill \
    --generate_statistics \
    --save_raw_data \
    --interactive_plots
```

#### Scene-Specific Analysis
```bash
# Analyze diversity within specific scenes
python vlm-pose/visualization/visualize_skill_pose_diversity.py \
    --data_root datasets/local_pairs_datasets \
    --scenes_filter "KITCHEN_SCENE4,KITCHEN_SCENE1" \
    --output_dir results/scene_diversity \
    --group_by_scene \
    --generate_comparison_plots
```

## Output Structure

### Prediction Visualization Output
```
visualization_results/
├── prediction_summary.json         # Overall prediction statistics
├── individual_predictions/          # Per-sample visualizations
│   ├── sample_001_prediction.png   # Prediction vs ground truth
│   ├── sample_001_error.png        # Error visualization
│   └── ...
├── aggregate_plots/                 # Summary visualizations
│   ├── error_distribution.png      # Error distribution plots
│   ├── skill_performance.png       # Performance by skill
│   └── confusion_matrix.png        # Prediction confusion matrix
├── debug_images/                   # Debug visualizations
│   ├── attention_maps/             # Attention visualizations
│   ├── feature_maps/               # Feature map visualizations
│   └── failure_cases/              # Failed prediction analysis
└── data/                          # Raw prediction data
    ├── predictions.json            # All predictions in JSON format
    ├── errors.csv                  # Error metrics in CSV
    └── statistics.json             # Summary statistics
```

### Diversity Analysis Output
```
skill_diversity_analysis/
├── diversity_summary.json          # Overall diversity metrics
├── skill_plots/                    # Per-skill visualizations
│   ├── close_drawer_poses.png      # 3D pose scatter plot
│   ├── open_drawer_poses.png       # 3D pose scatter plot
│   └── ...
├── comparison_plots/               # Cross-skill comparisons
│   ├── position_diversity.png      # Position diversity comparison
│   ├── orientation_diversity.png   # Orientation diversity comparison
│   └── combined_diversity.png      # Combined diversity metrics
├── statistics/                     # Detailed statistics
│   ├── diversity_metrics.json      # Quantitative diversity metrics
│   ├── skill_statistics.csv        # Per-skill statistics
│   └── correlation_analysis.json   # Correlation analysis
└── interactive/                    # Interactive visualizations
    ├── pose_explorer.html          # Interactive pose explorer
    └── diversity_dashboard.html     # Interactive diversity dashboard
```

## Visualization Features

### Prediction Visualization Features
- **Side-by-side Comparison**: Ground truth vs predicted poses
- **Error Heatmaps**: Spatial distribution of prediction errors
- **Attention Visualization**: Model attention maps (if available)
- **Failure Case Analysis**: Detailed analysis of prediction failures
- **Statistical Summaries**: Quantitative performance metrics

### Diversity Analysis Features
- **3D Scatter Plots**: 3D visualization of pose distributions
- **Heatmaps**: Density maps of pose concentrations
- **Distribution Plots**: Statistical distribution visualizations
- **Interactive Plots**: Web-based interactive exploration
- **Comparison Charts**: Side-by-side skill comparisons

## Advanced Usage

### Custom Visualization
```bash
# Create custom visualization with specific parameters
python vlm-pose/visualization/visualize_vlm_pose_predictions.py \
    --checkpoint_path runs/pose_vlm/checkpoint.pt \
    --data_root datasets/local_pairs_datasets \
    --output_dir custom_vis \
    --custom_config custom_vis_config.yaml \
    --plot_style "publication" \
    --figure_size "12,8" \
    --dpi 300
```

### Batch Visualization
```bash
# Visualize multiple checkpoints
for checkpoint in runs/pose_vlm/*/vla--*_checkpoint.pt; do
    exp_name=$(basename $(dirname $checkpoint))
    python vlm-pose/visualization/visualize_vlm_pose_predictions.py \
        --checkpoint_path "$checkpoint" \
        --data_root datasets/local_pairs_datasets \
        --output_dir "results/batch_vis/$exp_name" \
        --num_samples 50
done
```

### Comparative Analysis
```bash
# Compare two different models
python vlm-pose/visualization/visualize_vlm_pose_predictions.py \
    --checkpoint_path runs/pose_vlm/model_a/checkpoint.pt \
    --comparison_checkpoint runs/pose_vlm/model_b/checkpoint.pt \
    --data_root datasets/local_pairs_datasets \
    --output_dir results/model_comparison \
    --generate_comparison_plots
```

## Tips for Visualization

1. **Start with Overview**: Use summary plots before diving into details
2. **Focus on Failures**: Analyze failure cases to understand model limitations
3. **Use Interactive Tools**: Interactive plots help explore complex relationships
4. **Save Raw Data**: Keep raw visualization data for further analysis
5. **Multiple Formats**: Generate plots in multiple formats for different uses
6. **Document Findings**: Keep notes on interesting patterns observed

## Troubleshooting

- **Memory Issues**: Reduce batch size or number of samples if running out of memory
- **Slow Rendering**: Use lower DPI or smaller figure sizes for faster rendering
- **Missing Dependencies**: Ensure matplotlib, seaborn, and plotly are installed
- **Large Output**: Use compression or selective saving for large visualization outputs