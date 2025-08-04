# VLA Evaluation Framework

This directory contains organized evaluation scripts for different VLA (Vision-Language-Action) models and their combinations.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Evaluation Types](#evaluation-types)
- [Quick Test Commands](#quick-test-commands)
- [Usage Examples](#usage-examples)
- [Metrics](#metrics)
- [Configuration](#configuration)
- [File Migration](#file-migration)
- [Notes](#notes)

## Directory Structure

```
vla-evaluation/
├── pose_vlm_eval/          # PoseVLM standalone evaluation
├── openvla_eval/           # OpenVLA-OFT standalone evaluation  
├── combined_eval/          # Combined PoseVLM + OpenVLA-OFT evaluation
├── shared/                 # Shared utilities
└── scripts/                # Shell scripts for running evaluations
```

## Evaluation Types

### 1. PoseVLM Evaluation (`pose_vlm_eval/`)
- **Purpose**: Evaluate PoseVLM models for local EE pose prediction
- **Scripts**:
  - `evaluate_pose_vlm_skill_aware.py` - Skill-aware evaluation
  - `evaluate_pose_vlm_demo_aware.py` - Demo-aware evaluation
  - `evaluate_pose_vlm_object_aware.py` - Object-aware evaluation
  - `visualize_vlm_pose_predictions.py` - Visualize pose predictions
  - `visualize_skill_pose_diversity.py` - Visualize skill pose diversity

### 2. OpenVLA Evaluation (`openvla_eval/`)
- **Purpose**: Evaluate OpenVLA-OFT models for task completion
- **Scripts**:
  - `run_libero_eval_local.py` - LIBERO evaluation with local initialization
  - `run_libero_eval.py` - Standard LIBERO evaluation

### 3. Combined Evaluation (`combined_eval/`)
- **Purpose**: Evaluate the full pipeline: PoseVLM → IK → OpenVLA-OFT
- **Pipeline**: LIBERO Task → Reset Image + Language → PoseVLM → Local EE Pose → IK → Joint Positions → OpenVLA-OFT → Task Success
- **Scripts**:
  - `run_combined_eval.py` - Main combined evaluation script
  - `test_combined_pipeline.py` - Test script for single task
  - `test_pose_vlm_tasks.py` - Test script for PoseVLM smaller split tasks only
  - `configs/combined_eval_config.yaml` - Configuration file
- **Features**:
  - Task filtering option (`--use_pose_vlm_smaller_split_tasks_only`) for evaluating only the 5 tasks used in PoseVLM smaller split training (intersection with BOSS_44)
  - View selection options (`--wrist_only`, `--third_person_view_only`, `--both_views`) for OpenVLA-OFT
  - IK retry logic with pose re-prediction
  - Debug image saving with proper naming
  - Dual failure metrics (IK failures vs VLA failures)

### 4. Shared Utilities (`shared/`)
- **Purpose**: Common utilities used across evaluations
- **Files**:
  - `tracik_tools.py` - IK solving utilities
  - `debug_ik_pipeline.py` - IK debugging utilities

### 5. Shell Scripts (`scripts/`)
- **Purpose**: Easy-to-use shell scripts for running evaluations
- **Scripts**:
  - `run_combined_eval.sh` - Run combined evaluation
  - `test_pose_vlm_tasks.sh` - Test combined pipeline on PoseVLM training tasks

## Quick Test Commands

### Single Task Test (Fastest)
```bash
# Test on single task with wrist-only view (first intersection task)
python vla-evaluation/combined_eval/test_combined_pipeline.py \
    --pose_vlm_checkpoint runs/pose_vlm/1.0.0/openvla-7b+pose_dataset+b8+lr-1e-05+pose_simple+lora-r32+dropout-0.0--image_aug--pose_aug--simple_pose_head_pose_aug--smaller_splits/vla--20000_checkpoint.pt \
    --openvla_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/view_wrist/1.0.3/openvla-7b+libero44_local+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt \
    --task_id 16 \
    --num_episodes 3 \
    --max_ik_attempts 2
```

### PoseVLM Smaller Split Tasks Test (Medium Speed)
```bash
# Test on 5 PoseVLM smaller split tasks (intersection with BOSS_44)
./vla-evaluation/scripts/test_pose_vlm_tasks.sh \
    --openvla_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/view_wrist/1.0.3/openvla-7b+libero44_local+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt \
    --num_episodes 5 \
    --max_ik_attempts 3
```

### Full Evaluation (Slowest)
```bash
# Evaluate on all BOSS_44 tasks
./vla-evaluation/scripts/run_combined_eval.sh \
    --pose_vlm_checkpoint runs/pose_vlm/1.0.0/openvla-7b+pose_dataset+b8+lr-1e-05+pose_simple+lora-r32+dropout-0.0--image_aug--pose_aug--simple_pose_head_pose_aug--smaller_splits/vla--20000_checkpoint.pt \
    --openvla_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/view_wrist/1.0.3/openvla-7b+libero44_local+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt \
    --wrist_only \
    --num_episodes_per_task 10 \
    --max_ik_attempts 3
```

## Usage Examples

### Combined Evaluation
```bash
# Run combined evaluation
./vla-evaluation/scripts/run_combined_eval.sh \
    --pose_vlm_checkpoint runs/pose_vlm/1.0.0/.../vla--90000_checkpoint.pt \
    --openvla_checkpoint runs/view_wrist/1.0.3/.../50000_chkpt \
    --num_episodes_per_task 10 \
    --max_ik_attempts 3

# Test single task
python vla-evaluation/combined_eval/test_combined_pipeline.py \
    --pose_vlm_checkpoint runs/pose_vlm/1.0.0/.../vla--90000_checkpoint.pt \
    --openvla_checkpoint runs/view_wrist/1.0.3/.../50000_chkpt \
    --task_id 0 \
    --num_episodes 5

# Test PoseVLM smaller split tasks only
./vla-evaluation/scripts/test_pose_vlm_tasks.sh \
    --openvla_checkpoint runs/view_wrist/1.0.3/.../50000_chkpt \
    --num_episodes 10 \
    --max_ik_attempts 5
```

### PoseVLM Evaluation
```bash
# Skill-aware evaluation
python vla-evaluation/pose_vlm_eval/evaluate_pose_vlm_skill_aware.py \
    --checkpoint_path runs/pose_vlm/1.0.0/.../vla--90000_checkpoint.pt \
    --data_root datasets/local_pairs_datasets \
    --split val \
    --sample_size 1000
```

### OpenVLA Evaluation
```bash
# LIBERO evaluation
python vla-evaluation/openvla_eval/run_libero_eval_local.py \
    --pretrained_checkpoint runs/view_wrist/1.0.3/.../50000_chkpt \
    --task_suite_name boss_44 \
    --wrist_only True
```

## Metrics

### Combined Evaluation Metrics
- **IK Success Rate**: Percentage of episodes where IK solved successfully
- **VLA Success Rate**: Percentage of episodes where OpenVLA-OFT completed task (given IK success)
- **Overall Success Rate**: Percentage of episodes where both IK and VLA succeeded
- **IK Failure Rate**: Percentage of episodes where IK failed after all attempts
- **VLA Failure Rate**: Percentage of episodes where VLA failed given IK success

### Debug Images
The combined evaluation saves debug images with the following naming convention:
- `{task_name}_ep{episode}_reset.png` - Reset image from LIBERO environment
- `{task_name}_ep{episode}_ik_result.png` - Image after IK solution is applied
- `{task_name}_ep{episode}_final_success.png` - Final image for successful episodes
- `{task_name}_ep{episode}_final_failure.png` - Final image for failed episodes

## Configuration

### Combined Evaluation Config (`combined_eval/configs/combined_eval_config.yaml`)
- Model checkpoint paths
- Evaluation parameters (episodes per task, IK attempts)
- Debug settings
- Device settings
- PoseVLM and OpenVLA specific settings
- IK settings
- LIBERO environment settings

## File Migration

All existing evaluation files have been copied to the new structure:
- Original files remain in their current locations
- New structure provides better organization
- See `file_migration_plan.md` for detailed mapping

## Notes

1. **No file deletion**: All original files remain in their current locations
2. **Import paths**: New files may need updated import paths
3. **Shell scripts**: Reference existing `shells/` folder structure
4. **Configuration**: New config files standardize evaluation parameters
5. **Shared utilities**: Common functionality extracted to `shared/` folder 