# Atomic Skills Scripts

This directory contains scripts for generating and visualizing localized robot demonstrations from atomic skills for VLA (Visual Language Action) model training.

## 📂 Scripts Overview

### 1. `generate_local_demos.py` - Main Data Generation Script
**Purpose**: Processes atomic skill BDDL files and extracts "local demos" - short, contact-rich sub-trajectories from long demonstrations.

**Key Features**:
- ✅ Processes three skill types: pick, place, atomic
- ✅ Maps atomic skills to source demos using `cat_split_map.json`  
- ✅ Simulates and replays in libero environments using `atomic_skills` benchmark
- ✅ Trigger-based data extraction with skill-specific slicing logic
- ✅ Adaptive collection windows to ensure data capture
- ✅ Success validation with BDDL goal completion detection
- ✅ Debug video generation for successful and failed trials
- ✅ Multiple demo aggregation into single HDF5 files
- ✅ Configurable offset parameters for all skill types

### 2. `capture_initial_images.py` - Environment Visualization
**Purpose**: Captures initial state images from libero environments for debugging and validation. Uses random initialization when specific initial states aren't available.

### 3. `generate_demo_video.py` - Video Generation Utility  
**Purpose**: Creates MP4 videos from HDF5 demonstration files for visual inspection.

## 🚀 Command Pipeline

### Complete Processing Pipeline

```bash
# 1. Generate local demonstrations (main command)
python scripts/atomic_skills_scripts/generate_local_demos.py

# 2. Debug mode - process only one demo per skill type with videos
python scripts/atomic_skills_scripts/generate_local_demos.py --debug

# 3. Custom configuration with all parameters
python scripts/atomic_skills_scripts/generate_local_demos.py \
    --pick_offset 7 \
    --place_offset 12 \
    --atomic_offset 8 \
    --min_steps_fallback 5 \
    --output_dir ./my_local_demos

# 4. Generate visualization videos for inspection
python scripts/atomic_skills_scripts/generate_demo_video.py \
    datasets/hdf5_datasets/atomic_local_demos/SKILL_NAME_demo.hdf5

# 5. Capture initial scene images for reference
python scripts/atomic_skills_scripts/capture_initial_images.py --debug
```

### Quick Start Commands

```bash
# Basic usage - process all atomic skills
python scripts/atomic_skills_scripts/generate_local_demos.py

# Debug mode with video generation
python scripts/atomic_skills_scripts/generate_local_demos.py --debug

# Custom paths
python scripts/atomic_skills_scripts/generate_local_demos.py \
    --raw_demo_dir /path/to/demos \
    --output_dir /path/to/output
```

## 📋 Detailed Usage

### `generate_local_demos.py` - Main Script

#### Command Line Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--atomic_skills_dir` | Path | `externals/boss/libero/libero/bddl_files/atomic_skills` | Directory containing atomic skill BDDL files |
| `--cat_split_map_file` | Path | `externals/boss/libero/libero/bddl_files/atomic_skills/cat_split_map.json` | Mapping file from original to atomic skills |
| `--raw_demo_dir` | Path | `datasets/hdf5_datasets/libero_90_no_noops/` | Directory containing source HDF5 demonstrations |
| `--output_dir` | Path | `datasets/hdf5_datasets/atomic_local_demos/` | Output directory for generated local demos |
| `--init_offset` | Integer | `3` | Steps before trigger for initial state extraction |
| `--pick_offset` | Integer | `5` | Steps before trigger for pick skill data collection |
| `--place_offset` | Integer | `10` | Steps from end for place skill data collection |
| `--atomic_offset` | Integer | `6` | Steps before trigger for atomic skill data collection |
| `--min_steps_fallback` | Integer | `3` | Minimum steps to collect as fallback when collection window is too late |
| `--debug` | Flag | `False` | Debug mode: process only one demo per skill type with video generation |

#### Processing Logic by Skill Type

**Pick Skills** (`*_pick.bddl`):
- **Trigger Detection**: Gripper closing (negative → positive) OR contact detection fallback
- **Data Collection**: From `trigger_timestep - pick_offset` to end of demonstration  
- **Expected Steps**: ~25-30 steps depending on demonstration length
- **Environment**: Uses corresponding atomic task from `atomic_skills` benchmark

**Place Skills** (`*_place.bddl`):
- **Trigger Detection**: None (time-based with adaptive collection)
- **Data Collection**: From `actual_completion - place_offset + 1` to completion
- **Adaptive Logic**: Pre-simulates to find completion point and adjusts collection window
- **Expected Steps**: ~9-10 steps (close to place_offset value)
- **Environment**: Uses corresponding atomic task from `atomic_skills` benchmark

**Atomic Skills** (no suffix):
- **Trigger Detection**: Contact detection only (end-effector with environment)
- **Data Collection**: From `trigger_timestep - atomic_offset` to end
- **Adaptive Logic**: Uses simulation-based completion detection for robust success validation
- **Expected Steps**: ~20-30 steps depending on task complexity
- **Environment**: Uses corresponding atomic task from `atomic_skills` benchmark

#### Success Detection
- **BDDL Goal Completion**: Tasks are marked successful when `done == True` (BDDL goal achieved)
- **Adaptive Fallback**: For atomic skills, uses simulation-based validation when replay is non-deterministic
- **Data Validation**: Ensures at least minimum steps are collected for meaningful training data

#### Debug Mode Features
When using `--debug`, the script:
- ✅ Processes only one demonstration per skill type
- ✅ Generates MP4 videos for both successful and failed trials
- ✅ Creates 6 videos total: 2 per skill type (agentview + wrist camera)
- ✅ Saves videos to `./debug_videos/` directory
- ✅ Provides detailed logging for troubleshooting

#### Output Files
For each successfully processed skill:
- `{skill_name}_demo.hdf5` - Aggregated demonstration data with proper structure
- `{skill_name}.init` - Initial states for evaluation (pickle format)
- `{skill_name}_success_rate.npy` - Success rate statistics (float value 0.0-1.0)

Debug mode additionally generates:
- `./debug_videos/{skill_name}_{success/failed}_agentview.mp4` - Third-person view videos
- `./debug_videos/{skill_name}_{success/failed}_wrist.mp4` - Wrist camera view videos

### `capture_initial_images.py` - Scene Image Capture

#### Command Line Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--debug` | Flag | `False` | Process only first 10 tasks for debugging |
| `--scene` | String | `None` | Only process tasks containing specific scene (e.g., "SCENE7") |

#### Usage Examples
```bash
# Capture images for all 76 atomic skills tasks
python scripts/atomic_skills_scripts/capture_initial_images.py

# Debug mode: only first 10 tasks
python scripts/atomic_skills_scripts/capture_initial_images.py --debug

# Filter by scene
python scripts/atomic_skills_scripts/capture_initial_images.py --scene SCENE7

# Combined filtering
python scripts/atomic_skills_scripts/capture_initial_images.py --debug --scene SCENE7
```

#### Output
- **Directory**: `imgs/atomic_skills_scene_images/` (or `_debug` suffix in debug mode)
- **Format**: PNG images named by task language description
- **Purpose**: Visual reference for debugging and validation

### `generate_demo_video.py` - Video Generation

#### Command Line Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `input_file` | Required | - | Path to input HDF5 demonstration file |
| `--output_dir` | Path | `./demo_videos` | Output directory for generated videos |
| `--fps` | Integer | `30` | Frames per second for output videos |

#### Usage Examples
```bash
# Basic video generation
python scripts/atomic_skills_scripts/generate_demo_video.py \
    datasets/hdf5_datasets/atomic_local_demos/SKILL_NAME_demo.hdf5

# Custom output and frame rate
python scripts/atomic_skills_scripts/generate_demo_video.py \
    datasets/hdf5_datasets/atomic_local_demos/SKILL_NAME_demo.hdf5 \
    --output_dir ./inspection_videos --fps 24
```

## 📁 File Structure

```
scripts/atomic_skills_scripts/
├── README.md                       # This documentation
├── generate_local_demos.py         # Main data generation script  
├── capture_initial_images.py       # Environment visualization utility
├── generate_demo_video.py          # Video generation utility
└── PIPELINE_DOCUMENTATION.md       # Detailed technical documentation
```

## 🔧 Dependencies

### Required Python Packages
```bash
pip install h5py numpy scipy tqdm opencv-python pillow
```

### External Dependencies
- **libero**: Robot simulation environment
- **BOSS framework**: Task definitions and benchmarks
- **OpenVLA constants**: Environment configuration

### Environment Setup
The scripts expect the following directory structure:
```
openvla-oft/
├── externals/boss/libero/          # Libero framework with atomic_skills benchmark
├── datasets/hdf5_datasets/         # Source and output datasets
│   ├── libero_90_no_noops/         # Source demonstration files
│   └── atomic_local_demos/         # Generated local demonstration files
└── scripts/atomic_skills_scripts/  # These scripts
```

## 📊 Data Format

### Input HDF5 Structure (Source Demos)
```
demo_file.hdf5
├── data/
│   ├── demo_0/
│   │   ├── actions         # Shape: (T, 7) - robot actions
│   │   ├── states          # Shape: (T, N) - full sim states
│   │   ├── obs/
│   │   │   ├── robot0_joint_pos          # Joint positions
│   │   │   ├── robot0_gripper_qpos       # Gripper positions  
│   │   │   ├── robot0_eef_pos            # End-effector position
│   │   │   ├── robot0_eef_quat           # End-effector orientation
│   │   │   ├── agentview_image           # Third-person camera
│   │   │   ├── robot0_eye_in_hand_image  # Wrist camera
│   │   │   └── ...
│   │   └── ...
│   └── demo_1/, demo_2/, ...
```

### Output HDF5 Structure (Local Demos)  
```
local_demo.hdf5
├── data/
│   ├── demo_0/
│   │   ├── actions         # Shape: (local_T, 7) - extracted actions
│   │   ├── dones           # Shape: (local_T,) - episode termination flags
│   │   ├── rewards         # Shape: (local_T,) - reward signals
│   │   ├── states          # Shape: (local_T, 84) - simulation states
│   │   ├── robot_states    # Shape: (local_T, 9) - robot proprioception
│   │   ├── obs/
│   │   │   ├── joint_states      # Shape: (local_T, 7) - joint positions
│   │   │   ├── gripper_states    # Shape: (local_T, 2) - gripper positions
│   │   │   ├── ee_pos            # Shape: (local_T, 3) - end-effector position
│   │   │   ├── ee_ori            # Shape: (local_T, 3) - end-effector euler angles
│   │   │   ├── ee_states         # Shape: (local_T, 6) - combined pose
│   │   │   ├── agentview_rgb     # Shape: (local_T, H, W, 3) - third-person images
│   │   │   └── eye_in_hand_rgb   # Shape: (local_T, H, W, 3) - wrist camera images
│   │   └── ...
│   └── demo_1/, demo_2/, ... (if multiple successful demos)
```

## 🐛 Debugging and Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   # Install required packages
   pip install h5py numpy scipy tqdm opencv-python pillow
   
   # Verify libero accessibility
   python -c "from libero.libero import benchmark; print('✅ Libero accessible')"
   ```

2. **Environment Setup**
   ```bash
   # Ensure correct working directory
   cd /path/to/openvla-oft
   
   # Check required directories exist
   ls externals/boss/libero/libero/bddl_files/atomic_skills/
   ls datasets/hdf5_datasets/libero_90_no_noops/
   ```

3. **File Permissions**
   ```bash
   # Check and create output directory
   mkdir -p datasets/hdf5_datasets/atomic_local_demos
   chmod 755 datasets/hdf5_datasets/atomic_local_demos
   ```

4. **Low Success Rates**
   - Use `--debug` mode to generate videos and inspect what's happening
   - Check if BDDL files are properly formatted
   - Verify source demonstration quality
   - Adjust offset parameters if collection windows are inappropriate

### Debug Workflow

1. **Initial Testing**
   ```bash
   # Run debug mode to test pipeline
   python scripts/atomic_skills_scripts/generate_local_demos.py --debug
   ```

2. **Video Inspection**
   ```bash
   # Check generated debug videos in ./debug_videos/
   ls -la debug_videos/
   
   # Videos show actual task execution for verification
   ```

3. **Parameter Tuning**
   ```bash
   # Adjust collection windows if needed
   python scripts/atomic_skills_scripts/generate_local_demos.py \
       --debug --pick_offset 7 --place_offset 15 --atomic_offset 8
   ```

4. **Validation**
   ```bash
   # Generate inspection videos for successful outputs
   python scripts/atomic_skills_scripts/generate_demo_video.py \
       datasets/hdf5_datasets/atomic_local_demos/SKILL_NAME_demo.hdf5
   ```

### Performance Monitoring

The script provides detailed progress information:
- ⏱️ Processing time per skill and demo
- 📊 Success/failure rates with detailed failure reasons
- 📁 Output file paths and sizes
- 🎯 Trigger detection results and collection statistics
- 🎬 Debug video generation status

### Expected Processing Times
- **Debug mode**: ~2-3 minutes (3 skills, 1 demo each, with videos)
- **Full processing**: ~30-60 minutes (76 skills, all demos)
- **Video generation**: ~30 seconds per demo file

## 📈 Success Metrics and Validation

### Expected Output Statistics
- **Pick Skills**: ~32 tasks, success rates typically 80-95%
- **Place Skills**: ~32 tasks, success rates typically 85-98% 
- **Atomic Skills**: ~12 tasks, success rates typically 70-90%

### Data Quality Indicators
- **Step Counts**: Pick (~25 steps), Place (~9 steps), Atomic (~20-30 steps)
- **Success Rates**: >80% overall success rate indicates good pipeline health
- **Video Validation**: Debug videos should show reasonable task execution

### Troubleshooting Low Success Rates
If success rates are significantly lower than expected:

1. **Check Source Data Quality**
   ```bash
   # Verify source demos exist and are valid
   python -c "import h5py; f=h5py.File('datasets/hdf5_datasets/libero_90_no_noops/TASK_demo.hdf5'); print(list(f['data'].keys()))"
   ```

2. **Inspect Debug Videos**
   ```bash
   # Generate debug videos to see what's happening
   python scripts/atomic_skills_scripts/generate_local_demos.py --debug
   # Check videos in ./debug_videos/ directory
   ```

3. **Adjust Collection Parameters**
   ```bash
   # Try different offset values
   python scripts/atomic_skills_scripts/generate_local_demos.py \
       --pick_offset 8 --place_offset 15 --atomic_offset 10
   ```

## 📚 References and Documentation

- **Technical Details**: `PIPELINE_DOCUMENTATION.md` - Comprehensive technical documentation
- **Original Requirements**: `cc_prompts/process_demos.md` - Project requirements
- **Unit Tests**: `unit_tests/atomic_skill_tests/test_data_generation.py` - Validation tests
- **BOSS Framework**: External framework for task definitions and environments
- **Libero Documentation**: Robot simulation environment documentation

## 🔄 Integration with VLA Training Pipeline

The generated local demonstrations are designed to integrate with VLA model training:

1. **Data Format**: Compatible with OpenVLA training pipeline
2. **Structure**: Follows established HDF5 format conventions
3. **Validation**: Initial states provided for evaluation metrics
4. **Statistics**: Success rates for data quality assessment

### Next Steps After Generation
1. **Quality Check**: Review success rates and video outputs
2. **Data Integration**: Copy output files to VLA training data directory
3. **Model Training**: Use local demos for fine-tuning or training
4. **Evaluation**: Use initial states for model evaluation on atomic skills