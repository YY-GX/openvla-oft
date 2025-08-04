# File Migration Plan for vla-evaluation Structure

## Overview
This document outlines the migration of existing evaluation files to the new `vla-evaluation/` structure. All existing files remain in their current locations - this is purely a copy operation for better organization.

## Migration Details

### 1. PoseVLM Evaluation Files

**Source: `utils/`**
- `evaluate_pose_vlm_object_aware.py` → `vla-evaluation/pose_vlm_eval/`
- `visualize_vlm_pose_predictions.py` → `vla-evaluation/pose_vlm_eval/`
- `visualize_skill_pose_diversity.py` → `vla-evaluation/pose_vlm_eval/`

**Source: `vla-scripts/`**
- `evaluate_pose_vlm_skill_aware.py` → `vla-evaluation/pose_vlm_eval/`
- `evaluate_pose_vlm_demo_aware.py` → `vla-evaluation/pose_vlm_eval/`

### 2. OpenVLA Evaluation Files

**Source: `experiments/robot/libero/`**
- `run_libero_eval_local.py` → `vla-evaluation/openvla_eval/`
- `run_libero_eval.py` → `vla-evaluation/openvla_eval/`

### 3. Shared Utilities

**Source: `utils/`**
- `tracik_tools.py` → `vla-evaluation/shared/`
- `debug_ik_pipeline.py` → `vla-evaluation/shared/` (for reference)

### 4. New Combined Evaluation Files

**New files to be created:**
- `vla-evaluation/combined_eval/run_combined_eval.py`
- `vla-evaluation/combined_eval/test_combined_pipeline.py`
- `vla-evaluation/combined_eval/pose_to_joint_converter.py`
- `vla-evaluation/combined_eval/debug_image_saver.py`
- `vla-evaluation/combined_eval/evaluation_metrics.py`

### 5. Configuration Files

**New files to be created:**
- `vla-evaluation/pose_vlm_eval/configs/pose_vlm_eval_config.yaml`
- `vla-evaluation/openvla_eval/configs/openvla_eval_config.yaml`
- `vla-evaluation/combined_eval/configs/combined_eval_config.yaml`

### 6. Shell Scripts

**New files to be created:**
- `vla-evaluation/scripts/run_pose_vlm_eval.sh`
- `vla-evaluation/scripts/run_openvla_eval.sh`
- `vla-evaluation/scripts/run_combined_eval.sh`

**Note:** These will reference the existing `shells/` folder structure.

## File Structure After Migration

```
vla-evaluation/
├── pose_vlm_eval/
│   ├── evaluate_pose_vlm_skill_aware.py          # Copied from vla-scripts/
│   ├── evaluate_pose_vlm_demo_aware.py           # Copied from vla-scripts/
│   ├── evaluate_pose_vlm_object_aware.py         # Copied from utils/
│   ├── visualize_vlm_pose_predictions.py         # Copied from utils/
│   ├── visualize_skill_pose_diversity.py         # Copied from utils/
│   └── configs/
│       └── pose_vlm_eval_config.yaml             # New
│
├── openvla_eval/
│   ├── run_libero_eval_local.py                  # Copied from experiments/robot/libero/
│   ├── run_libero_eval.py                        # Copied from experiments/robot/libero/
│   └── configs/
│       └── openvla_eval_config.yaml              # New
│
├── combined_eval/
│   ├── run_combined_eval.py                      # New
│   ├── test_combined_pipeline.py                 # New
│   ├── pose_to_joint_converter.py                # New
│   ├── debug_image_saver.py                      # New
│   ├── evaluation_metrics.py                     # New
│   └── configs/
│       └── combined_eval_config.yaml             # New
│
├── shared/
│   ├── tracik_tools.py                          # Copied from utils/
│   └── debug_ik_pipeline.py                     # Copied from utils/
│
└── scripts/
    ├── run_pose_vlm_eval.sh                     # New
    ├── run_openvla_eval.sh                      # New
    └── run_combined_eval.sh                     # New
```

## Implementation Notes

1. **No file deletion**: All original files remain in their current locations
2. **Import path updates**: New files will need updated import paths
3. **Shell script references**: New shell scripts will reference existing `shells/` folder
4. **Configuration standardization**: New config files will standardize evaluation parameters
5. **Shared utilities**: Common functionality extracted to `shared/` folder

## Benefits

1. **Clear organization**: Each evaluation type has its own folder
2. **Easy navigation**: Related files are co-located
3. **Shared code**: Common utilities in `shared/` folder
4. **Scalability**: Easy to add new evaluation types
5. **Testing**: Test scripts co-located with main scripts 