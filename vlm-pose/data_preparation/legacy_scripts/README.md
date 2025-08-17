# Legacy Data Preparation Scripts

This directory contains the original data preparation scripts that were used before the v2 pipeline.

## Scripts

- `extract_local_pairs.py` - Original pose extraction script
- `analyze_pose_statistics.py` - Pose statistics analysis
- `analyze_skill_pose_stats.py` - Skill-specific pose statistics  
- `check_target_object_poses.py` - Target object pose validation
- `compute_target_object_diversity.py` - Diversity computation
- `create_hungarian_dataset.py` - Hungarian matching dataset creation
- `extract_sampled_target_object_poses.py` - Sampled target pose extraction
- `extract_target_object_poses.py` - Target object pose extraction
- `split_pose_data.py` - Data splitting utilities

## Migration

These scripts have been superseded by the v2 pipeline in `../v2_pipeline/`. 

For new dataset curation work, please use the v2 pipeline which provides:
- Better contact detection
- More precise pose extraction
- Improved consistency validation
- Cleaner architecture

## Compatibility

These scripts are kept for compatibility with existing workflows and analysis pipelines that may depend on them.