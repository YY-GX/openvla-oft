#!/usr/bin/env python3
"""
more_real_simplified_pose_shift_eval.py

More realistic atomic skills evaluation with motion planning robustness testing.
Compares VLA performance between:
1. Original pose -> VLA execution (baseline)
2. Large shift (15cm) -> Motion planning back to original -> VLA execution
3. [DISABLED] Direct shifted pose evaluation

The third method tests VLA robustness to motion planning residual errors.

python scripts/phase2/simplified_pose_shift_eval.py \
      --init_files_path datasets/hdf5_datasets/atomic_local_demos/augmented_atomic_skills \
      --num_trials_per_skill 5 \
      --save_videos true \
      --save_failure_videos true \
      --exp_name "baseline_vla_pose_shift_test" \
      --shift_position_std 0.05 \
      --shift_orientation_std 1.047 

python scripts/phase2/simplified_pose_shift_eval.py \
    --pretrained_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/atomic_skills_augmented_debug/1.0.0/openvla-7b+libero_atomic_skills_augmented_debug+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_debug--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--70000_chkpt \
    --init_files_path datasets/hdf5_datasets/atomic_local_demos_augmented/combined/debug \
    --focused_eval_mode True \
    --shift_position_std 0.07 \
    --shift_orientation_std 1.047 

python scripts/phase2/simplified_pose_shift_eval.py \
    --pretrained_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/atomic_skills_augmented_debug/1.0.1/openvla-7b+libero_atomic_skills+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--340000_chkpt \
    --init_files_path datasets/hdf5_datasets/atomic_local_demos_augmented/combined/debug \
    --focused_eval_mode True

python scripts/phase2/more_real_simplified_pose_shift_eval.py \
      --pretrained_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/atomic_skills_augmented_debug/1.0.1/openvla-7b+libero_atomic_skills+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--340000_chkpt \
      --init_files_path datasets/hdf5_datasets/atomic_local_demos_augmented/combined/debug \
      --focused_eval_mode True


python scripts/phase2/more_real_simplified_pose_shift_eval.py \
      --pretrained_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/atomic_skills_augmented_debug/1.0.0/openvla-7b+libero_atomic_skills_augmented_debug+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_debug--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--70000_chkpt \
      --init_files_path datasets/hdf5_datasets/atomic_local_demos_augmented/combined/debug \
      --focused_eval_mode True


python scripts/phase2/more_real_simplified_pose_shift_eval.py \
      --pretrained_checkpoint runs/libero_atomic_skills_augmented_closer/1.0.0/openvla-7b+libero_atomic_skills_augmented_closer+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_closer--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--70000_chkpt \
      --init_files_path datasets/hdf5_datasets/atomic_local_demos_augmented/combined/debug \
      --focused_eval_mode True

python scripts/phase2/more_real_simplified_pose_shift_eval.py \
      --pretrained_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/libero_atomic_skills_augmented_farther/1.0.0/openvla-7b+libero_atomic_skills_augmented_farther+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_farther--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--30000_chkpt \
      --init_files_path datasets/hdf5_datasets/atomic_local_demos_augmented/combined/debug \
      --focused_eval_mode True


python scripts/phase2/more_real_simplified_pose_shift_eval.py \
      --mp_robustness_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/libero_atomic_skills_augmented_farther/1.0.0/openvla-7b+libero_atomic_skills_augmented_farther+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_farther--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
      --init_files_path datasets/hdf5_datasets/atomic_local_demos_augmented/combined/debug \
      --focused_eval_mode True

python scripts/phase2/more_real_simplified_pose_shift_eval.py \
    --pretrained_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/libero_atomic_skills_augmented_closer/1.0.0/openvla-7b+libero_atomic_skills_augmented_closer+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_closer--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --mp_robustness_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/libero_atomic_skills_augmented_closer_original/1.0.0/openvla-7b+libero_atomic_skills_augmented_closer_original+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_closer_original--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --focused_eval_mode True

=====================================================

FULL EVALUATION

farther-augmented
CUDA_VISIBLE_DEVICES=0 python scripts/phase2/more_real_simplified_pose_shift_eval.py \
    --pretrained_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/libero_atomic_skills_augmented_farther/1.0.0/openvla-7b+libero_atomic_skills_augmented_farther+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_farther--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --focused_eval_mode True \
    --init_files_path datasets/hdf5_datasets/atomic_local_demos_augmented/combined/full_by_language

farther-original
CUDA_VISIBLE_DEVICES=1 python scripts/phase2/more_real_simplified_pose_shift_eval.py \
    --pretrained_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/atomic_skills/1.0.1/openvla-7b+libero_atomic_skills+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --init_files_path datasets/hdf5_datasets/atomic_local_demos_augmented/combined/full_by_language

closer-augmented
CUDA_VISIBLE_DEVICES=2 python scripts/phase2/more_real_simplified_pose_shift_eval.py \
      --pretrained_checkpoint runs/libero_atomic_skills_augmented_closer/1.0.0/openvla-7b+libero_atomic_skills_augmented_closer+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_closer--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt

closer-original
CUDA_VISIBLE_DEVICES=6 python scripts/phase2/more_real_simplified_pose_shift_eval.py \
    --pretrained_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/libero_atomic_skills_augmented_closer_original/1.0.0/openvla-7b+libero_atomic_skills_augmented_closer_original+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_closer_original--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt
    
=====================================================

=====================================================

FOCUSED EVALUATION

farther-augmented
CUDA_VISIBLE_DEVICES=3 python scripts/phase2/more_real_simplified_pose_shift_eval.py \
    --pretrained_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/libero_atomic_skills_augmented_farther/1.0.0/openvla-7b+libero_atomic_skills_augmented_farther+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_farther--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --focused_eval_mode True \
    --init_files_path datasets/hdf5_datasets/atomic_local_demos_augmented/combined/full_by_language \
    --focused_eval_mode True

farther-original
CUDA_VISIBLE_DEVICES=4 python scripts/phase2/more_real_simplified_pose_shift_eval.py \
    --pretrained_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/atomic_skills/1.0.1/openvla-7b+libero_atomic_skills+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --init_files_path datasets/hdf5_datasets/atomic_local_demos_augmented/combined/full_by_language \
    --focused_eval_mode True

closer-augmented
CUDA_VISIBLE_DEVICES=5 python scripts/phase2/more_real_simplified_pose_shift_eval.py \
      --pretrained_checkpoint runs/libero_atomic_skills_augmented_closer/1.0.0/openvla-7b+libero_atomic_skills_augmented_closer+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_closer--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
      --focused_eval_mode True

closer-original
CUDA_VISIBLE_DEVICES=7 python scripts/phase2/more_real_simplified_pose_shift_eval.py \
    --pretrained_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/libero_atomic_skills_augmented_closer_original/1.0.0/openvla-7b+libero_atomic_skills_augmented_closer_original+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_closer_original--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --focused_eval_mode True
    
=====================================================

=====================================================

CUDA_VISIBLE_DEVICES=1 python scripts/phase2/more_real_simplified_pose_shift_eval.py \
      --pretrained_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/libero_atomic_skills_augmented_long_id3/1.0.0/openvla-7b+libero_atomic_skills_augmented_long_id3+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_long_id3--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt \
      --init_files_path datasets/hdf5_datasets/atomic_local_demos_augmented/combined/full_by_language

=====================================================

ID EVALUATION MODE

# ID1 evaluation (Cooking Preparation Setup skills)
CUDA_VISIBLE_DEVICES=1 python scripts/phase2/more_real_simplified_pose_shift_eval.py \
    --pretrained_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/libero_atomic_skills_augmented_long_id1/1.0.0/openvla-7b+libero_atomic_skills_augmented_long_id1+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_long_id1--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt \
    --ID_eval_mode True \
    --init_files_path datasets/hdf5_datasets/atomic_local_demos_augmented/combined/full_by_language \
    --shift_position_std 0.02 \
    --shift_orientation_std 0.5235

# ID2 evaluation (Complete Kitchen Organization skills)
CUDA_VISIBLE_DEVICES=2 python scripts/phase2/more_real_simplified_pose_shift_eval.py \
    --pretrained_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/libero_atomic_skills_augmented_long_id2/1.0.0/openvla-7b+libero_atomic_skills_augmented_long_id2+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_long_id2--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt \
    --ID_eval_mode True \
    --init_files_path datasets/hdf5_datasets/atomic_local_demos_augmented/combined/full_by_language \
    --shift_position_std 0.02 \
    --shift_orientation_std 0.5235

# ID3 evaluation (Switch Table Objects skills)
CUDA_VISIBLE_DEVICES=3 python scripts/phase2/more_real_simplified_pose_shift_eval.py \
    --pretrained_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/libero_atomic_skills_augmented_long_id3/1.0.0/openvla-7b+libero_atomic_skills_augmented_long_id3+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_long_id3--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt \
    --ID_eval_mode True \
    --init_files_path datasets/hdf5_datasets/atomic_local_demos_augmented/combined/full_by_language \
    --shift_position_std 0.02 \
    --shift_orientation_std 0.5235

=====================================================

"""
import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
import h5py
import pickle
import traceback
import imageio
from datetime import datetime

# Add project paths (insert at beginning to prioritize)
sys.path.insert(0, '/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.insert(0, '/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.robot_utils import (
    normalize_gripper_action,
    invert_gripper_action,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    get_model,
    get_image_resize_size,
    get_action,
)

# Phase 2 utilities
from scripts.phase2.utils.motion_planner import MotionPlanner as StandardMotionPlanner
from scripts.phase2.utils.simple_pose_shift_utils import (
    generate_shifted_pose,
    find_family_pose,
    load_initial_states_for_skill,
)

# Initialize logging
logging.basicConfig(level=logging.INFO)


@dataclass
class SimplifiedConfig:
    """Simplified configuration for pose shift evaluation."""

    # Model configuration
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/libero_atomic_skills_augmented_closer_original/1.0.0/openvla-7b+libero_atomic_skills_augmented_closer_original+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_closer_original--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt"

    # Model-specific parameters
    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_proprio: bool = True                         # Whether to include proprio state in input
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy
    load_in_8bit: bool = False                       # Load with 8-bit quantization
    load_in_4bit: bool = False                       # Load with 4-bit quantization
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (wrist cam only)

    # Environment configuration
    task_suite_name: str = "atomic_skills"           # Task suite
    env_img_res: int = 256                           # Environment image resolution

    # Simplified pose shifting parameters
    shift_position_std: float = 0.05                 # Position shift range (±5cm for each axis) - used for MP convergence threshold
    shift_orientation_std: float = 1.047             # Orientation shift range (±60° total) - used for MP convergence threshold

    # Large shift parameters for method 3 (motion planning robustness test)
    large_shift_distance: float = 0.15               # 15cm position distance for method 3
    large_shift_orientation_range: float = 1.047     # 60° orientation shift range for method 3 (in radians)

    # Evaluation parameters
    num_trials_per_skill: int = 10                   # Number of trials per skill
    max_steps: int = 220                             # Maximum steps per episode

    # Motion planner parameters
    mp_num_steps: int = 400                          # Number of interpolation steps for motion planner
    mp_pos_gain: float = 5.0                         # Position gain for cartesian linear motion planner
    mp_ori_gain: float = 5.0                         # Orientation gain for cartesian linear motion planner

    # Data paths
    init_files_path: str = "datasets/hdf5_datasets/atomic_local_demos_augmented_pick_10_place_15_atomic_15/combined_by_language"

    # Output configuration
    save_videos: bool = True                         # Save rollout videos
    save_failure_videos: bool = True                 # Save videos even for failed episodes
    exp_name: str = "simplified_pose_shift_eval"     # Experiment name for output directory

    # Test mode
    test_mode: bool = False                          # Test mode: only test setup without loading full model

    # Specific skill for testing (empty = all skills)
    target_skill: str = ""                           # Test specific skill (e.g., "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet")

    # Focused evaluation mode for 3 specific skills
    focused_eval_mode: bool = False                  # If True, evaluates only 3 specific skills with 10 trials each

    # ID-based evaluation mode for different skill sets
    ID_eval_mode: bool = False                       # If True, evaluates skills based on ID in pretrained_checkpoint path

    is_depth: bool = False                           # Whether to use depth images (permanently set to False)

    # Action un-normalization key
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key


def load_bddl_to_init_mapping() -> Dict[str, str]:
    """Load the mapping from BDDL task names to init file names."""
    mapping_file = "scripts/phase2/annotations/bddl_to_init_mapping.json"
    try:
        with open(mapping_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading BDDL to init mapping from {mapping_file}: {e}")
        return {}


def get_init_file_name(task_name: str) -> str:
    """Get the corresponding init file name for a BDDL task name."""
    mapping = load_bddl_to_init_mapping()
    return mapping.get(task_name, task_name)


def get_language_instruction(task_name: str) -> str:
    """Get the language instruction for a task by mapping to init file name and converting format."""
    init_name = get_init_file_name(task_name)
    # Replace underscores with spaces to create natural language instruction
    instruction = init_name.replace("_", " ")
    return instruction


def extract_unnorm_key_from_checkpoint(checkpoint_path: Union[str, Path]) -> str:
    """Extract unnorm_key from checkpoint path by looking for libero_atomic_skills_*** pattern."""
    checkpoint_str = str(checkpoint_path)
    
    # Look for patterns like "libero_atomic_skills_augmented_closer_original" in the checkpoint path
    # Stop at the first '+' character to avoid capturing version numbers and other parts
    import re
    match = re.search(r'libero_atomic_skills_[^+/]+', checkpoint_str)
    if match:
        unnorm_key = match.group(0)
        print(f"📋 Extracted unnorm_key from checkpoint: {unnorm_key}")
        return unnorm_key
    
    # Fallback patterns - more specific matching
    if "atomic_skills_augmented_closer_original" in checkpoint_str:
        unnorm_key = "libero_atomic_skills_augmented_closer_original"
        print(f"📋 Extracted unnorm_key from checkpoint: {unnorm_key}")
        return unnorm_key
    elif "atomic_skills_augmented_closer" in checkpoint_str:
        unnorm_key = "libero_atomic_skills_augmented_closer"
        print(f"📋 Extracted unnorm_key from checkpoint: {unnorm_key}")
        return unnorm_key
    elif "atomic_skills" in checkpoint_str:
        unnorm_key = "libero_atomic_skills"
        print(f"📋 Extracted unnorm_key from checkpoint: {unnorm_key}")
        return unnorm_key
    else:
        print(f"⚠️  Could not extract unnorm_key from checkpoint path: {checkpoint_path}")
        return "libero_atomic_skills"  # Default fallback


def extract_id_from_checkpoint(checkpoint_path: Union[str, Path]) -> str:
    """Extract ID (id1/id2/id3) from checkpoint path."""
    checkpoint_str = str(checkpoint_path)
    
    # Look for patterns like "long_id1", "long_id2", "long_id3" in the checkpoint path
    import re
    match = re.search(r'long_id([123])', checkpoint_str)
    if match:
        id_num = match.group(1)
        print(f"📋 Extracted ID from checkpoint: id{id_num}")
        return f"id{id_num}"
    
    print(f"⚠️  Could not extract ID from checkpoint path: {checkpoint_path}")
    return "id1"  # Default fallback


def get_skills_for_id(id_str: str) -> list:
    """Get the list of skills for a given ID based on the dataset builder files."""
    
    if id_str == "id1":
        # Skills from LIBERO_Atomic_Skills_Augmented_Long_ID_1
        skills = [
            "pick_moka_pot",
            "place_moka_pot_on_the_stove", 
            "turn_on_the_stove",
            "pick_frying_pan",
            "place_frying_pan_on_the_stove",
            "open_the_microwave"
        ]
    elif id_str == "id2":
        # Skills from LIBERO_Atomic_Skills_Augmented_Long_ID_2
        skills = [
            "pick_black_bowl",
            "place_black_bowl_on_the_plate",
            "open_the_top_drawer_of_the_cabinet",
            "pick_ketchup",
            "place_ketchup_in_top_drawer_of_the_cabinet",
            "close_the_top_drawer_of_the_cabinet"
        ]
    elif id_str == "id3":
        # Skills from LIBERO_Atomic_Skills_Augmented_Long_ID_3
        skills = [
            "open_the_bottom_drawer_of_the_cabinet",
            "pick_black_bowl",
            "place_black_bowl_on_the_plate",
            "pick_wine_bottle",
            "place_wine_bottle_in_bottom_drawer_of_the_cabinet",
            "close_the_bottom_drawer_of_the_cabinet"
        ]
    else:
        print(f"⚠️  Unknown ID: {id_str}, using default id1 skills")
        skills = [
            "pick_moka_pot",
            "place_moka_pot_on_the_stove", 
            "turn_on_the_stove",
            "pick_frying_pan",
            "place_frying_pan_on_the_stove",
            "open_the_microwave"
        ]
    
    print(f"📋 Skills for {id_str}: {skills}")
    return skills


def load_vla_model_and_processor(cfg: SimplifiedConfig, checkpoint_path: Union[str, Path] = None):
    """Load VLA model and processor with specified checkpoint."""
    if cfg.test_mode:
        print("🧪 Test mode: Skipping model loading")
        return None, None, None, None

    # Use provided checkpoint or default
    if checkpoint_path is None:
        checkpoint_path = cfg.pretrained_checkpoint

    print(f"🤖 Loading VLA model: {cfg.model_family}")
    print(f"   Checkpoint: {checkpoint_path}")

    # Create a copy of cfg with the specified checkpoint
    cfg_copy = SimplifiedConfig(**cfg.__dict__)
    cfg_copy.pretrained_checkpoint = checkpoint_path

    # Extract and set the unnorm_key from the checkpoint path
    unnorm_key = extract_unnorm_key_from_checkpoint(checkpoint_path)
    cfg_copy.unnorm_key = unnorm_key

    # Also set the unnorm_key in the original cfg to ensure it's available later
    cfg.unnorm_key = unnorm_key

    # Load VLA model using get_model like the original script
    vla = get_model(cfg_copy)

    # Get VLA components - following original script pattern
    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg_copy, vla.llm_dim, proprio_dim=8)

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg_copy, vla.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg_copy)
        check_unnorm_key(cfg_copy, vla)

    print(f"✅ VLA model loaded successfully")
    return vla, processor, action_head, proprio_projector


def check_unnorm_key(cfg: SimplifiedConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Use the unnorm_key that was extracted from the checkpoint path
    unnorm_key = cfg.unnorm_key
    
    if not unnorm_key:
        print("⚠️  No unnorm_key provided, attempting to extract from checkpoint path")
        unnorm_key = extract_unnorm_key_from_checkpoint(cfg.pretrained_checkpoint)
        cfg.unnorm_key = unnorm_key

    # Check if the key exists in the model's norm_stats
    if unnorm_key not in model.norm_stats:
        available_keys = list(model.norm_stats.keys())
        print(f"⚠️  Expected key '{unnorm_key}' not found in model. Available keys: {available_keys}")

        # Try to find atomic skills related key as fallback
        atomic_keys = [k for k in available_keys if "atomic" in k.lower() or "libero" in k.lower()]
        if atomic_keys:
            fallback_key = atomic_keys[0]
            print(f"🔄 Using fallback atomic-related key: {fallback_key}")
            unnorm_key = fallback_key
            cfg.unnorm_key = unnorm_key
        else:
            # Final fallback to bridge_orig
            fallback_key = "bridge_orig"
            print(f"🔄 Using final fallback key: {fallback_key}")
            unnorm_key = fallback_key
            cfg.unnorm_key = unnorm_key

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
    print(f"✅ Using unnorm_key: {unnorm_key}")


def prepare_observation_for_policy(obs, resize_size):
    """Prepare observation for policy input - wrist cam only (from original script)."""
    # Get preprocessed images - wrist cam only
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict - use end-effector pose like working script
    observation = {
        "full_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation


def process_action(action, model_family):
    """Process action before sending to environment (from working script)."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def create_config_with_unnorm_key(cfg: SimplifiedConfig, checkpoint_path: Union[str, Path]) -> SimplifiedConfig:
    """Create a config copy with the correct unnorm_key for the given checkpoint."""
    cfg_copy = SimplifiedConfig(**cfg.__dict__)
    cfg_copy.pretrained_checkpoint = checkpoint_path
    cfg_copy.unnorm_key = extract_unnorm_key_from_checkpoint(checkpoint_path)
    return cfg_copy


def generate_vla_action(cfg, vla, observation, task_description, processor, action_head, proprio_projector):
    """Generate VLA action using original script pattern."""
    if cfg.test_mode:
        # Return dummy action in test mode
        return get_libero_dummy_action("libero_spatial")

    # Get actions using the original get_action function
    actions = get_action(
        cfg,
        vla,
        observation,
        task_description,
        processor=processor,
        action_head=action_head,
        proprio_projector=proprio_projector,
        noisy_action_projector=None,  # Not using diffusion
        use_film=cfg.use_film,
    )

    # Return the first action (original script uses action queue)
    return actions[0] if len(actions) > 0 else get_libero_dummy_action("libero_spatial")


def load_atomic_skill_metadata(cfg: SimplifiedConfig):
    """Load atomic skills metadata by dynamically discovering tasks with .init files."""
    print("📊 Loading atomic skills metadata...")

    # Initialize task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()

    # Get tasks with initial state files
    atomic_skills = []
    for task_id in range(task_suite.n_tasks):
        task_name = task_suite.get_task(task_id).name
        # Get the correct init file name using the mapping
        init_file_name = get_init_file_name(task_name)
        # Check for .init files - only use plain .init (not _original.init or _augmented.init)
        plain_init_path = os.path.join(cfg.init_files_path, f"{init_file_name}.init")

        if os.path.exists(plain_init_path):
            atomic_skills.append(task_name)
        else:
            print(f"[WARNING] {task_name} does not have corresponding {init_file_name}.init file.")

    print(f"✅ Loaded {len(atomic_skills)} atomic skills with .init files")
    return atomic_skills


def evaluate_skill_with_pose_shift(skill_name: str, cfg: SimplifiedConfig, vla, processor, action_head, proprio_projector, video_dir: Path, resize_size):
    """
    Evaluate a single skill with pose shifting using multiple trials.

    Workflow:
    1. Evaluate: Initial state -> Direct VLA execution (baseline) - multiple trials
    2. [DISABLED] Initial state -> Move to shifted pose -> VLA execution
    3. Evaluate: Initial state -> Large shift -> Motion plan back to original -> VLA execution - multiple trials

    The third method tests VLA robustness to motion planning residual errors.
    """
    print(f"\n{'='*80}")
    print(f"🎯 Evaluating skill with pose shifts: {skill_name}")
    print(f"   Trials per evaluation: {cfg.num_trials_per_skill}")
    print(f"{'='*80}")

    try:
        # Load initial states for this skill
        # Get the correct init file name using the mapping
        init_file_name = get_init_file_name(skill_name)
        initial_states = load_initial_states_for_skill(init_file_name, cfg.init_files_path)
        if len(initial_states) == 0:
            print(f"❌ No initial states found for skill: {skill_name}")
            return None

        # Extract skill instruction using the mapping
        instruction = get_language_instruction(skill_name)
        print(f"💬 Instruction: {instruction}")

        results = {
            "baseline": {"success_rate": 0.0, "successes": 0, "total_trials": cfg.num_trials_per_skill},
            "mp_robustness": {"success_rate": 0.0, "successes": 0, "total_trials": cfg.num_trials_per_skill},
            "shifted": {"success_rate": 0.0, "successes": 0, "total_trials": cfg.num_trials_per_skill, "disabled": True}
        }

        # Evaluation 1: Initial state -> Direct VLA execution (baseline) - multiple trials
        print(f"\n🔄 Evaluation 1: Baseline evaluation - {cfg.num_trials_per_skill} trials")
        baseline_successes = 0

        for trial_idx in range(cfg.num_trials_per_skill):
            print(f"  Trial {trial_idx + 1}/{cfg.num_trials_per_skill}...")

            # Use different initial states for variety
            reference_initial_state = initial_states[trial_idx % len(initial_states)]

            baseline_result = evaluate_single_trial_direct(
                skill_name, reference_initial_state, instruction,
                cfg, vla, processor, action_head, proprio_projector,
                video_dir, video_suffix=None, resize_size=resize_size  # Don't save videos for individual trials
            )

            if baseline_result and baseline_result.get("success", False):
                baseline_successes += 1

        results["baseline"]["successes"] = baseline_successes
        results["baseline"]["success_rate"] = baseline_successes / cfg.num_trials_per_skill

        # Evaluation 2: Shifted pose evaluation - multiple trials
        print(f"\n🔄 Evaluation 2: Shifted pose evaluation - {cfg.num_trials_per_skill} trials")
        shifted_successes = 0

        for trial_idx in range(cfg.num_trials_per_skill):
            print(f"  Trial {trial_idx + 1}/{cfg.num_trials_per_skill}...")

            # Use different initial states for variety
            reference_initial_state = initial_states[trial_idx % len(initial_states)]

            shifted_result = evaluate_single_trial_with_shift(
                skill_name, reference_initial_state, instruction,
                cfg, vla, processor, action_head, proprio_projector,
                video_dir, video_suffix=None, initial_states=initial_states, resize_size=resize_size  # Don't save videos for individual trials
            )

            if shifted_result and shifted_result.get("success", False):
                shifted_successes += 1

        results["shifted"]["successes"] = shifted_successes
        results["shifted"]["success_rate"] = shifted_successes / cfg.num_trials_per_skill
        results["shifted"]["disabled"] = False

        # Evaluation 3: Initial state -> Large shift -> Motion plan back to original -> VLA execution - multiple trials
        print(f"\n🔄 Evaluation 3: MP robustness evaluation - {cfg.num_trials_per_skill} VLA trials (retrying MP failures)")
        mp_robustness_successes = 0
        vla_trials_completed = 0
        total_attempts = 0

        while vla_trials_completed < cfg.num_trials_per_skill:
            total_attempts += 1
            print(f"  VLA Trial {vla_trials_completed + 1}/{cfg.num_trials_per_skill} (Attempt {total_attempts})...")

            # Use different initial states for variety
            reference_initial_state = initial_states[total_attempts % len(initial_states)]

            mp_robustness_result = evaluate_single_trial_with_mp_robustness(
                skill_name, reference_initial_state, instruction, cfg,
                vla, processor, action_head, proprio_projector,
                video_dir, video_suffix=None, initial_states=initial_states, resize_size=resize_size  # Don't save videos for individual trials
            )

            # Check if motion planning failed - if so, retry without counting as VLA trial
            if mp_robustness_result and mp_robustness_result.get("reason", "").startswith("motion_planner_failed"):
                print(f"   🔄 Motion planning failed, retrying with different initial state...")
                continue  # Don't count this as a VLA trial

            # Motion planning succeeded, count this as a VLA trial
            vla_trials_completed += 1

            if mp_robustness_result and mp_robustness_result.get("success", False):
                mp_robustness_successes += 1

        print(f"   📊 MP Robustness: {mp_robustness_successes}/{cfg.num_trials_per_skill} VLA successes ({total_attempts} total attempts including MP failures)")

        results["mp_robustness"]["successes"] = mp_robustness_successes
        results["mp_robustness"]["success_rate"] = mp_robustness_successes / cfg.num_trials_per_skill

        # Summary
        print(f"\n📊 Results Summary for {skill_name}:")
        print(f"   Baseline: {baseline_successes}/{cfg.num_trials_per_skill} ({results['baseline']['success_rate']*100:.1f}%)")
        print(f"   Shifted: {shifted_successes}/{cfg.num_trials_per_skill} ({results['shifted']['success_rate']*100:.1f}%)")
        print(f"   MP Robustness: {mp_robustness_successes}/{cfg.num_trials_per_skill} ({results['mp_robustness']['success_rate']*100:.1f}%)")

        return results

    except Exception as e:
        print(f"❌ Error evaluating skill {skill_name}: {e}")
        traceback.print_exc()
        return None


def print_cumulative_success_rates(all_results: dict, skill_name: str):
    """Print cumulative success rates for all evaluated skills so far."""
    print(f"\n{'='*60}")
    print(f"📊 CUMULATIVE SUCCESS RATES (after {skill_name})")
    print(f"{'='*60}")
    
    # Calculate cumulative statistics
    total_skills = len([r for r in all_results.values() if r])
    if total_skills == 0:
        print("No skills evaluated yet.")
        return
    
    total_baseline_successes = 0
    total_baseline_trials = 0
    total_shifted_successes = 0
    total_shifted_trials = 0
    total_mp_robustness_successes = 0
    total_mp_robustness_trials = 0
    
    for skill, results in all_results.items():
        if results:
            # Baseline stats
            baseline_successes = results.get("baseline", {}).get("successes", 0)
            baseline_trials = results.get("baseline", {}).get("total_trials", 0)
            total_baseline_successes += baseline_successes
            total_baseline_trials += baseline_trials
            
            # Shifted stats
            shifted_successes = results.get("shifted", {}).get("successes", 0)
            shifted_trials = results.get("shifted", {}).get("total_trials", 0)
            total_shifted_successes += shifted_successes
            total_shifted_trials += shifted_trials
            
            # MP robustness stats
            mp_successes = results.get("mp_robustness", {}).get("successes", 0)
            mp_trials = results.get("mp_robustness", {}).get("total_trials", 0)
            total_mp_robustness_successes += mp_successes
            total_mp_robustness_trials += mp_trials
    
    # Calculate overall rates
    overall_baseline_rate = (total_baseline_successes / total_baseline_trials * 100) if total_baseline_trials > 0 else 0
    overall_shifted_rate = (total_shifted_successes / total_shifted_trials * 100) if total_shifted_trials > 0 else 0
    overall_mp_robustness_rate = (total_mp_robustness_successes / total_mp_robustness_trials * 100) if total_mp_robustness_trials > 0 else 0
    
    print(f"Skills evaluated: {total_skills}")
    print(f"Baseline:      {total_baseline_successes}/{total_baseline_trials} ({overall_baseline_rate:.1f}%)")
    print(f"Shifted:       {total_shifted_successes}/{total_shifted_trials} ({overall_shifted_rate:.1f}%)")
    print(f"MP Robustness: {total_mp_robustness_successes}/{total_mp_robustness_trials} ({overall_mp_robustness_rate:.1f}%)")
    print(f"Shift Gap:     {overall_baseline_rate - overall_shifted_rate:.1f}%")
    print(f"Robustness Gap: {overall_baseline_rate - overall_mp_robustness_rate:.1f}%")
    
    # Print individual skill breakdown
    print(f"\nIndividual Skills:")
    for skill, results in all_results.items():
        if results:
            baseline_rate = results.get("baseline", {}).get("success_rate", 0.0) * 100
            shifted_rate = results.get("shifted", {}).get("success_rate", 0.0) * 100
            mp_rate = results.get("mp_robustness", {}).get("success_rate", 0.0) * 100
            baseline_successes = results.get("baseline", {}).get("successes", 0)
            baseline_trials = results.get("baseline", {}).get("total_trials", 0)
            shifted_successes = results.get("shifted", {}).get("successes", 0)
            shifted_trials = results.get("shifted", {}).get("total_trials", 0)
            mp_successes = results.get("mp_robustness", {}).get("successes", 0)
            mp_trials = results.get("mp_robustness", {}).get("total_trials", 0)
            
            print(f"  {skill}:")
            print(f"    Baseline:      {baseline_successes}/{baseline_trials} ({baseline_rate:.1f}%)")
            print(f"    Shifted:       {shifted_successes}/{shifted_trials} ({shifted_rate:.1f}%)")
            print(f"    MP Robustness: {mp_successes}/{mp_trials} ({mp_rate:.1f}%)")
            print(f"    Shift Gap:     {baseline_rate - shifted_rate:.1f}%")
            print(f"    Robustness Gap: {baseline_rate - mp_rate:.1f}%")
    
    print(f"{'='*60}")


def evaluate_single_trial_with_shift(skill_name: str, initial_state: np.ndarray, instruction: str, cfg: SimplifiedConfig,
                                    vla, processor, action_head, proprio_projector, video_dir: Path, video_suffix: str, initial_states: list, resize_size):
    """
    Evaluate a single trial with shift generation: Initial state -> Generate shifted pose from current -> VLA execution.
    """
    try:
        # Find task by name (following original script pattern)
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict["atomic_skills"]()
        task = None
        for task_id in range(task_suite.n_tasks):
            if task_suite.get_task(task_id).name == skill_name:
                task = task_suite.get_task(task_id)
                break

        if task is None:
            print(f"   ❌ Task not found: {skill_name}")
            return {"success": False, "reason": "task_not_found"}

        # Create environment using the task object
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res, horizon=10000)

        # Reset environment first (critical for proper initialization)
        env.reset()

        # Set initial state using proper Libero method
        print(f"   🔄 Setting initial simulation state...")
        # Use the full initial state format: [joint(7) + gripper(2) + sim_state]
        # The sim_state part (from index 9:) is used for complete environment reset
        sim_state = initial_state[9:]  # Extract simulation state

        # Use env.set_init_state() for proper environment reset (same as generation script)
        env.set_init_state(sim_state)

        # Get observations after setting state
        obs = env.env._get_observations()

        # Get actual current EE pose from observation
        current_ee_pos = obs['robot0_eef_pos'].copy()
        current_ee_quat = obs['robot0_eef_quat'].copy()

        current_ee_axisangle = quat2axisangle(current_ee_quat)
        print(f"   📍 Current EE: pos=[{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}] ori=[{np.degrees(current_ee_axisangle[0]):.1f}°, {np.degrees(current_ee_axisangle[1]):.1f}°, {np.degrees(current_ee_axisangle[2]):.1f}°]")

        # Generate shifted pose from actual current pose
        current_pose = (current_ee_pos, current_ee_quat)
        shifted_pose = generate_shifted_pose(
            current_pose,
            position_shift_range=cfg.shift_position_std,
            orientation_shift_range=cfg.shift_orientation_std
        )
        shifted_pos, shifted_quat = shifted_pose

        shifted_axisangle = quat2axisangle(shifted_quat)
        print(f"   🎯 Shifted EE: pos=[{shifted_pos[0]:.3f}, {shifted_pos[1]:.3f}, {shifted_pos[2]:.3f}] ori=[{np.degrees(shifted_axisangle[0]):.1f}°, {np.degrees(shifted_axisangle[1]):.1f}°, {np.degrees(shifted_axisangle[2]):.1f}°]")
        pos_diff = np.linalg.norm(shifted_pos - current_ee_pos)
        print(f"   📏 Position difference: {pos_diff*100:.1f}cm")

        # Create adaptive motion planner
        motion_planner = StandardMotionPlanner(
            env,
            method="cartesian_linear",
            num_steps=cfg.mp_num_steps,
            pos_gain=cfg.mp_pos_gain,
            ori_gain=cfg.mp_ori_gain
        )

        # Determine skill type for gripper control
        skill_lower = skill_name.lower()
        if 'pick' in skill_lower:
            skill_type = 'pick'
        elif 'place' in skill_lower:
            skill_type = 'place'
        else:
            skill_type = 'other'

        # Move to shifted pose with tight thresholds
        print(f"   🚀 Moving to shifted pose using motion planner...")
        position_threshold = 0.01  # 1cm
        orientation_threshold = np.radians(5)  # 5 degrees
        move_success, final_obs = motion_planner.move_to_pose(
            shifted_pos, shifted_quat,
            position_threshold=position_threshold,
            orientation_threshold=orientation_threshold,
            skill_type=skill_type
        )

        if not move_success:
            print(f"   ❌ Motion planner failed to reach shifted pose")
            return {"success": False, "reason": "motion_planner_failed"}

        print(f"   ✅ Motion planner successfully reached shifted pose")

        # Start VLA evaluation from shifted pose
        print(f"   🤖 Starting VLA evaluation from shifted pose...")

        # Debug: Print EE pose and gripper state before VLA execution
        current_ee_pos = final_obs['robot0_eef_pos'].copy()
        current_ee_quat = final_obs['robot0_eef_quat'].copy()
        current_ee_axisangle = quat2axisangle(current_ee_quat)
        gripper_state = final_obs['robot0_gripper_qpos'].copy()
        print(f"   📍 EVAL 1 - EE pose: pos=[{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}] ori=[{np.degrees(current_ee_axisangle[0]):.1f}°, {np.degrees(current_ee_axisangle[1]):.1f}°, {np.degrees(current_ee_axisangle[2]):.1f}°] gripper=[{gripper_state[0]:.3f}, {gripper_state[1]:.3f}]")

        # Initialize video recording
        frames = []
        if cfg.save_videos:
            # Record initial frame
            img = get_libero_image(final_obs)
            frames.append(img)

        # Initialize action queue (like working script)
        action_queue = deque(maxlen=cfg.num_open_loop_steps)

        # VLA execution loop
        success = False
        obs = final_obs

        for step in range(cfg.max_steps):
            # If action queue is empty, requery model (like working script)
            if len(action_queue) == 0:
                # Prepare observation for policy
                observation = prepare_observation_for_policy(obs, resize_size)
                # Get actions using VLA
                actions = get_action(
                    cfg,
                    vla,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=None,  # Not using diffusion
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action (like working script)
            action = process_action(action, cfg.model_family)


            # Execute action
            obs, reward, done, info = env.step(action.tolist())

            # Record frame
            if cfg.save_videos:
                img = get_libero_image(obs)
                frames.append(img)

            # Check for task completion
            if done:
                success = True  # done=True means task completed successfully
                print(f"   ✅ Episode finished at step {step+1}: Success")
                break

        if not done:
            print(f"   ⏰ Episode timed out after {cfg.max_steps} steps")

        # Save video (only if video_suffix is provided)
        video_path = None
        if cfg.save_videos and video_suffix and (success or cfg.save_failure_videos):
            video_name = f"{skill_name}_{video_suffix}_{'success' if success else 'failure'}.mp4"
            video_path = video_dir / video_name
            video_writer = imageio.get_writer(str(video_path), fps=30)
            for img in frames:
                video_writer.append_data(img)
            video_writer.close()
            print(f"   🎥 Video saved: {video_path}")

        return {
            "success": success,
            "steps": step + 1 if done else cfg.max_steps,
            "video_path": video_path,
            "shifted_pose": shifted_pose,  # Return shifted pose for family pose generation
            "original_pose": current_pose   # Return original pose for family pose generation
        }

    except Exception as e:
        print(f"   ❌ Trial evaluation failed: {e}")
        traceback.print_exc()
        return {"success": False, "reason": f"evaluation_error: {e}"}


def evaluate_single_trial(skill_name: str, initial_state: np.ndarray, target_pose: tuple, instruction: str,
                         cfg: SimplifiedConfig, vla, processor, action_head, proprio_projector,
                         video_dir: Path, video_suffix: str, resize_size):
    """
    Evaluate a single trial: Initial state -> Move to target pose -> VLA execution.
    """
    target_pos, target_quat = target_pose
    print(f"   🎯 Target pose: pos=[{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")

    try:
        # Find task by name (following original script pattern)
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict["atomic_skills"]()
        task = None
        for task_id in range(task_suite.n_tasks):
            if task_suite.get_task(task_id).name == skill_name:
                task = task_suite.get_task(task_id)
                break

        if task is None:
            print(f"   ❌ Task not found: {skill_name}")
            return {"success": False, "reason": "task_not_found"}

        # Create environment using the task object
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res, horizon=10000)

        # Reset environment first (critical for proper initialization)
        env.reset()

        # Set initial state using proper Libero method
        print(f"   🔄 Setting initial simulation state...")
        # Use the full initial state format: [joint(7) + gripper(2) + sim_state]
        # The sim_state part (from index 9:) is used for complete environment reset
        sim_state = initial_state[9:]  # Extract simulation state

        # Use env.set_init_state() for proper environment reset (same as generation script)
        env.set_init_state(sim_state)

        # Get observations after setting state
        obs = env.env._get_observations()

        # Get actual current EE pose from observation
        current_ee_pos = obs['robot0_eef_pos'].copy()
        current_ee_quat = obs['robot0_eef_quat'].copy()

        current_ee_axisangle = quat2axisangle(current_ee_quat)
        target_axisangle = quat2axisangle(target_quat)
        
        print(f"   📍 Actual current EE: pos=[{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}] ori=[{np.degrees(current_ee_axisangle[0]):.1f}°, {np.degrees(current_ee_axisangle[1]):.1f}°, {np.degrees(current_ee_axisangle[2]):.1f}°]")
        print(f"   🎯 Target EE (input):  pos=[{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] ori=[{np.degrees(target_axisangle[0]):.1f}°, {np.degrees(target_axisangle[1]):.1f}°, {np.degrees(target_axisangle[2]):.1f}°]")

        # Create adaptive motion planner
        motion_planner = StandardMotionPlanner(
            env,
            method="cartesian_linear",
            num_steps=cfg.mp_num_steps,
            pos_gain=cfg.mp_pos_gain,
            ori_gain=cfg.mp_ori_gain
        )

        # Determine skill type for gripper control
        skill_lower = skill_name.lower()
        if 'pick' in skill_lower:
            skill_type = 'pick'
        elif 'place' in skill_lower:
            skill_type = 'place'
        else:
            skill_type = 'other'

        # Move to target pose with tight thresholds
        print(f"   🚀 Moving to target pose using motion planner...")
        position_threshold = 0.01  # 1cm
        orientation_threshold = np.radians(5)  # 5 degrees
        move_success, final_obs = motion_planner.move_to_pose(
            target_pos, target_quat,
            position_threshold=position_threshold,
            orientation_threshold=orientation_threshold,
            skill_type=skill_type
        )

        if not move_success:
            print(f"   ❌ Motion planner failed to reach target pose")
            return {"success": False, "reason": "motion_planner_failed"}

        print(f"   ✅ Motion planner successfully reached target pose")

        # Start VLA evaluation from this pose
        print(f"   🤖 Starting VLA evaluation from target pose...")

        # Initialize video recording
        frames = []
        if cfg.save_videos:
            # Record initial frame
            img = get_libero_image(final_obs)
            frames.append(img)

        # Initialize action queue (like working script)
        action_queue = deque(maxlen=cfg.num_open_loop_steps)

        # VLA execution loop
        success = False
        obs = final_obs

        for step in range(cfg.max_steps):
            # If action queue is empty, requery model (like working script)
            if len(action_queue) == 0:
                # Prepare observation for policy
                observation = prepare_observation_for_policy(obs, resize_size)
                # Get actions using VLA
                actions = get_action(
                    cfg,
                    vla,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=None,  # Not using diffusion
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action (like working script)
            action = process_action(action, cfg.model_family)


            # Execute action
            obs, reward, done, info = env.step(action.tolist())

            # Record frame
            if cfg.save_videos:
                img = get_libero_image(obs)
                frames.append(img)

            # Check for task completion
            if done:
                success = True  # done=True means task completed successfully
                print(f"   ✅ Episode finished at step {step+1}: Success")
                break

        if not done:
            print(f"   ⏰ Episode timed out after {cfg.max_steps} steps")

        # Save video (only if video_suffix is provided)
        video_path = None
        if cfg.save_videos and video_suffix and (success or cfg.save_failure_videos):
            video_name = f"{skill_name}_{video_suffix}_{'success' if success else 'failure'}.mp4"
            video_path = video_dir / video_name
            video_writer = imageio.get_writer(str(video_path), fps=30)
            for img in frames:
                video_writer.append_data(img)
            video_writer.close()
            print(f"   🎥 Video saved: {video_path}")

        return {
            "success": success,
            "steps": step + 1 if done else cfg.max_steps,
            "video_path": video_path
        }

    except Exception as e:
        print(f"   ❌ Trial evaluation failed: {e}")
        traceback.print_exc()
        return {"success": False, "reason": f"evaluation_error: {e}"}


def evaluate_single_trial_with_mp_robustness(skill_name: str, initial_state: np.ndarray, instruction: str, cfg: SimplifiedConfig,
                                            vla, processor, action_head, proprio_projector, video_dir: Path, video_suffix: str, initial_states: list, resize_size):
    """
    Evaluate a single trial with motion planning robustness test:
    Initial state -> Move to large shift (15cm) -> Motion plan back to original -> VLA execution.
    """
    try:
        # Find task by name (following original script pattern)
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict["atomic_skills"]()
        task = None
        for task_id in range(task_suite.n_tasks):
            if task_suite.get_task(task_id).name == skill_name:
                task = task_suite.get_task(task_id)
                break

        if task is None:
            print(f"   ❌ Task not found: {skill_name}")
            return {"success": False, "reason": "task_not_found"}

        # Create environment using the task object
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res, horizon=10000)

        # Reset environment first (critical for proper initialization)
        env.reset()

        # Set initial state using proper Libero method
        print(f"   🔄 Setting initial simulation state...")
        # Use the full initial state format: [joint(7) + gripper(2) + sim_state]
        # The sim_state part (from index 9:) is used for complete environment reset
        sim_state = initial_state[9:]  # Extract simulation state

        # Use env.set_init_state() for proper environment reset (same as generation script)
        env.set_init_state(sim_state)

        # Get observations after setting state
        obs = env.env._get_observations()

        # Get actual current EE pose from observation (this is our target to return to)
        original_ee_pos = obs['robot0_eef_pos'].copy()
        original_ee_quat = obs['robot0_eef_quat'].copy()

        original_ee_axisangle = quat2axisangle(original_ee_quat)
        print(f"   📍 Original EE: pos=[{original_ee_pos[0]:.3f}, {original_ee_pos[1]:.3f}, {original_ee_pos[2]:.3f}] ori=[{np.degrees(original_ee_axisangle[0]):.1f}°, {np.degrees(original_ee_axisangle[1]):.1f}°, {np.degrees(original_ee_axisangle[2]):.1f}°]")

        # Generate large shifted pose using the same function as regular shifts but with larger parameters
        original_pose = (original_ee_pos, original_ee_quat)
        large_shifted_pose = generate_shifted_pose(
            original_pose,
            position_shift_range=cfg.large_shift_distance,
            orientation_shift_range=cfg.large_shift_orientation_range
        )
        large_shifted_pos, large_shifted_quat = large_shifted_pose

        large_shifted_axisangle = quat2axisangle(large_shifted_quat)
        print(f"   🎯 Large shifted EE: pos=[{large_shifted_pos[0]:.3f}, {large_shifted_pos[1]:.3f}, {large_shifted_pos[2]:.3f}] ori=[{np.degrees(large_shifted_axisangle[0]):.1f}°, {np.degrees(large_shifted_axisangle[1]):.1f}°, {np.degrees(large_shifted_axisangle[2]):.1f}°]")
        pos_diff = np.linalg.norm(large_shifted_pos - original_ee_pos)
        print(f"   📏 Position difference: {pos_diff*100:.1f}cm")

        # Create adaptive motion planner
        motion_planner = StandardMotionPlanner(
            env,
            method="cartesian_linear",
            num_steps=cfg.mp_num_steps,
            pos_gain=cfg.mp_pos_gain,
            ori_gain=cfg.mp_ori_gain
        )

        # Determine skill type for gripper control
        skill_lower = skill_name.lower()
        if 'pick' in skill_lower:
            skill_type = 'pick'
        elif 'place' in skill_lower:
            skill_type = 'place'
        else:
            skill_type = 'other'

        # Move to large shifted pose first
        print(f"   🚀 Step 1: Moving to large shifted pose...")
        position_threshold = 0.02  # 2cm threshold for large move
        orientation_threshold = np.radians(10)  # 10 degrees for large move
        move_success_1, obs_at_shift = motion_planner.move_to_pose(
            large_shifted_pos, large_shifted_quat,
            position_threshold=position_threshold,
            orientation_threshold=orientation_threshold,
            skill_type=skill_type
        )

        if not move_success_1:
            print(f"   ❌ Motion planner failed to reach large shifted pose")
            return {"success": False, "reason": "motion_planner_failed_large_shift"}

        print(f"   ✅ Motion planner successfully reached large shifted pose")

        # Now move back to original pose with tight thresholds (using cfg parameters)
        print(f"   🚀 Step 2: Motion planning back to original pose...")
        position_threshold_back = cfg.shift_position_std  # Use cfg.shift_position_std as threshold
        orientation_threshold_back = cfg.shift_orientation_std  # Use cfg.shift_orientation_std as threshold
        move_success_2, final_obs = motion_planner.move_to_pose(
            original_ee_pos, original_ee_quat,
            position_threshold=position_threshold_back,
            orientation_threshold=orientation_threshold_back,
            skill_type=skill_type
        )

        if not move_success_2:
            print(f"   ❌ Motion planner failed to return to original pose")
            return {"success": False, "reason": "motion_planner_failed_return"}

        print(f"   ✅ Motion planner successfully returned to original pose")

        # Check final pose accuracy
        final_ee_pos = final_obs['robot0_eef_pos'].copy()
        final_ee_quat = final_obs['robot0_eef_quat'].copy()
        final_pos_error = np.linalg.norm(final_ee_pos - original_ee_pos)
        # Calculate orientation error
        from scipy.spatial.transform import Rotation as R
        original_rot = R.from_quat(original_ee_quat)
        final_rot = R.from_quat(final_ee_quat)
        orientation_error = (original_rot.inv() * final_rot).magnitude()

        print(f"   📏 Final pose error: pos={final_pos_error*100:.2f}cm, ori={np.degrees(orientation_error):.2f}°")

        # Start VLA evaluation from this motion-planned pose
        print(f"   🤖 Starting VLA evaluation from motion-planned pose...")

        # Debug: Print EE pose and gripper state before VLA execution
        current_ee_pos = final_obs['robot0_eef_pos'].copy()
        current_ee_quat = final_obs['robot0_eef_quat'].copy()
        current_ee_axisangle = quat2axisangle(current_ee_quat)
        gripper_state = final_obs['robot0_gripper_qpos'].copy()
        print(f"   📍 EVAL 3 - EE pose: pos=[{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}] ori=[{np.degrees(current_ee_axisangle[0]):.1f}°, {np.degrees(current_ee_axisangle[1]):.1f}°, {np.degrees(current_ee_axisangle[2]):.1f}°] gripper=[{gripper_state[0]:.3f}, {gripper_state[1]:.3f}]")

        # Initialize video recording
        frames = []
        if cfg.save_videos:
            # Record initial frame
            img = get_libero_image(final_obs)
            frames.append(img)

        # Initialize action queue (like working script)
        action_queue = deque(maxlen=cfg.num_open_loop_steps)

        # VLA execution loop
        success = False
        obs = final_obs

        for step in range(cfg.max_steps):
            # If action queue is empty, requery model (like working script)
            if len(action_queue) == 0:
                # Prepare observation for policy
                observation = prepare_observation_for_policy(obs, resize_size)
                # Get actions using VLA
                actions = get_action(
                    cfg,
                    vla,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=None,  # Not using diffusion
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action (like working script)
            action = process_action(action, cfg.model_family)

            # Execute action
            obs, reward, done, info = env.step(action.tolist())

            # Record frame
            if cfg.save_videos:
                img = get_libero_image(obs)
                frames.append(img)

            # Check for task completion
            if done:
                success = True  # done=True means task completed successfully
                print(f"   ✅ Episode finished at step {step+1}: Success")
                break

        if not done:
            print(f"   ⏰ Episode timed out after {cfg.max_steps} steps")

        # Save video (only if video_suffix is provided)
        video_path = None
        if cfg.save_videos and video_suffix and (success or cfg.save_failure_videos):
            video_name = f"{skill_name}_{video_suffix}_{'success' if success else 'failure'}.mp4"
            video_path = video_dir / video_name
            video_writer = imageio.get_writer(str(video_path), fps=30)
            for img in frames:
                video_writer.append_data(img)
            video_writer.close()
            print(f"   🎥 Video saved: {video_path}")

        return {
            "success": success,
            "steps": step + 1 if done else cfg.max_steps,
            "video_path": video_path,
            "final_pose_error": {
                "position": final_pos_error,
                "orientation": orientation_error
            },
            "large_shift_distance": pos_diff
        }

    except Exception as e:
        print(f"   ❌ MP robustness trial evaluation failed: {e}")
        traceback.print_exc()
        return {"success": False, "reason": f"evaluation_error: {e}"}


def evaluate_single_trial_direct(skill_name: str, initial_state: np.ndarray, instruction: str,
                                cfg: SimplifiedConfig, vla, processor, action_head, proprio_projector,
                                video_dir: Path, video_suffix: str, resize_size):
    """
    Evaluate a single trial: Initial state -> Direct VLA execution (no motion planning).
    """
    print(f"   🎯 Direct VLA execution from initial state")

    try:
        # Find task by name (following original script pattern)
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict["atomic_skills"]()
        task = None
        for task_id in range(task_suite.n_tasks):
            if task_suite.get_task(task_id).name == skill_name:
                task = task_suite.get_task(task_id)
                break

        if task is None:
            print(f"   ❌ Task not found: {skill_name}")
            return {"success": False, "reason": "task_not_found"}

        # Create environment using the task object
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res, horizon=10000)

        # Reset environment first (critical for proper initialization)
        env.reset()

        # Set initial state using proper Libero method
        print(f"   🔄 Setting initial simulation state...")
        # Use the full initial state format: [joint(7) + gripper(2) + sim_state]
        # The sim_state part (from index 9:) is used for complete environment reset
        sim_state = initial_state[9:]  # Extract simulation state

        # Use env.set_init_state() for proper environment reset (same as generation script)
        env.set_init_state(sim_state)

        # Get observations after setting state
        obs = env.env._get_observations()

        # Get current EE pose for logging
        current_ee_pos = obs['robot0_eef_pos'].copy()
        current_ee_quat = obs['robot0_eef_quat'].copy()
        current_ee_axisangle = quat2axisangle(current_ee_quat)
        print(f"   📍 Current EE: pos=[{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}] ori=[{np.degrees(current_ee_axisangle[0]):.1f}°, {np.degrees(current_ee_axisangle[1]):.1f}°, {np.degrees(current_ee_axisangle[2]):.1f}°]")

        # Start VLA evaluation directly from initial state
        print(f"   🤖 Starting VLA evaluation from initial state...")

        # Debug: Print EE pose and gripper state before VLA execution
        current_ee_pos = obs['robot0_eef_pos'].copy()
        current_ee_quat = obs['robot0_eef_quat'].copy()
        current_ee_axisangle = quat2axisangle(current_ee_quat)
        gripper_state = obs['robot0_gripper_qpos'].copy()
        print(f"   📍 EVAL 2 - EE pose: pos=[{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}] ori=[{np.degrees(current_ee_axisangle[0]):.1f}°, {np.degrees(current_ee_axisangle[1]):.1f}°, {np.degrees(current_ee_axisangle[2]):.1f}°] gripper=[{gripper_state[0]:.3f}, {gripper_state[1]:.3f}]")

        # Initialize video recording
        frames = []
        if cfg.save_videos:
            # Record initial frame
            img = get_libero_image(obs)
            frames.append(img)

        # Initialize action queue (like working script)
        action_queue = deque(maxlen=cfg.num_open_loop_steps)

        # VLA execution loop
        success = False

        for step in range(cfg.max_steps):
            # If action queue is empty, requery model (like working script)
            if len(action_queue) == 0:
                # Prepare observation for policy
                observation = prepare_observation_for_policy(obs, resize_size)
                # Get actions using VLA
                actions = get_action(
                    cfg,
                    vla,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=None,  # Not using diffusion
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action (like working script)
            action = process_action(action, cfg.model_family)


            # Execute action
            obs, reward, done, info = env.step(action.tolist())

            # Record frame
            if cfg.save_videos:
                img = get_libero_image(obs)
                frames.append(img)

            # Check for task completion
            if done:
                success = True  # done=True means task completed successfully
                print(f"   ✅ Episode finished at step {step+1}: Success")
                break

        if not done:
            print(f"   ⏰ Episode timed out after {cfg.max_steps} steps")

        # Save video (only if video_suffix is provided)
        video_path = None
        if cfg.save_videos and video_suffix and (success or cfg.save_failure_videos):
            video_name = f"{skill_name}_{video_suffix}_{'success' if success else 'failure'}.mp4"
            video_path = video_dir / video_name
            video_writer = imageio.get_writer(str(video_path), fps=30)
            for img in frames:
                video_writer.append_data(img)
            video_writer.close()
            print(f"   🎥 Video saved: {video_path}")

        return {
            "success": success,
            "steps": step + 1 if done else cfg.max_steps,
            "video_path": video_path
        }

    except Exception as e:
        print(f"   ❌ Trial evaluation failed: {e}")
        traceback.print_exc()
        return {"success": False, "reason": f"evaluation_error: {e}"}


def evaluate_skill_focused_mode(skill_name: str, cfg: SimplifiedConfig, vla, processor, action_head, proprio_projector, video_dir: Path, resize_size):
    """
    Evaluate a single skill in focused mode with multiple trials.

    Runs 10 trials for both shifted and original evaluations, saves 1 success + 1 failure
    video per evaluation type (4 videos total per skill), and returns success rates for both.

    Expected videos per skill:
    - {skill_name}_shifted_success.mp4 (if any success found)
    - {skill_name}_shifted_failure.mp4 (if any failure found)
    - {skill_name}_original_success.mp4 (if any success found)
    - {skill_name}_original_failure.mp4 (if any failure found)
    """
    print(f"\n{'='*80}")
    print(f"🎯 FOCUSED EVALUATION: {skill_name}")
    print(f"{'='*80}")

    try:
        # Load initial states for this skill
        # Get the correct init file name using the mapping
        init_file_name = get_init_file_name(skill_name)
        initial_states = load_initial_states_for_skill(init_file_name, cfg.init_files_path)
        if len(initial_states) == 0:
            print(f"❌ No initial states found for skill: {skill_name}")
            return None

        # Extract skill instruction using the mapping
        instruction = get_language_instruction(skill_name)
        print(f"💬 Instruction: {instruction}")

        results = {
            "baseline": {"success_rate": 0.0, "successes": 0, "total_trials": cfg.num_trials_per_skill},
            "mp_robustness": {"success_rate": 0.0, "successes": 0, "total_trials": cfg.num_trials_per_skill},
            "shifted": {"success_rate": 0.0, "successes": 0, "total_trials": cfg.num_trials_per_skill, "disabled": True}
        }

        # Run baseline evaluation trials (method 1)
        print(f"\n🔄 BASELINE EVALUATION: {cfg.num_trials_per_skill} trials")
        baseline_successes = 0
        baseline_success_video_saved = False
        baseline_failure_video_saved = False

        for trial_idx in range(cfg.num_trials_per_skill):
            print(f"  Trial {trial_idx + 1}/{cfg.num_trials_per_skill}...")

            # Use different initial states for variety
            reference_initial_state = initial_states[trial_idx % len(initial_states)]

            # Determine if we should save video for this trial
            save_video_for_trial = False
            video_suffix = None

            trial_result = evaluate_single_trial_direct(
                skill_name, reference_initial_state, instruction,
                cfg, vla, processor, action_head, proprio_projector,
                video_dir, video_suffix=None,  # Don't save yet, decide after
                resize_size=resize_size
            )

            trial_success = trial_result and trial_result.get("success", False)
            if trial_success:
                baseline_successes += 1
                # Save first success video
                if not baseline_success_video_saved:
                    video_suffix = f"baseline_success"
                    save_video_for_trial = True
                    baseline_success_video_saved = True
            else:
                # Save first failure video
                if not baseline_failure_video_saved:
                    video_suffix = f"baseline_failure"
                    save_video_for_trial = True
                    baseline_failure_video_saved = True

            # Re-run trial with video saving if needed
            if save_video_for_trial:
                print(f"   🎥 Re-running trial to save {video_suffix} video...")
                trial_result = evaluate_single_trial_direct(
                    skill_name, reference_initial_state, instruction,
                    cfg, vla, processor, action_head, proprio_projector,
                    video_dir, video_suffix=video_suffix,
                    resize_size=resize_size
                )

        results["baseline"]["successes"] = baseline_successes
        results["baseline"]["success_rate"] = baseline_successes / cfg.num_trials_per_skill

        # Run shifted evaluation trials (method 2)
        print(f"\n🔄 METHOD 2: Shifted pose evaluation - {cfg.num_trials_per_skill} trials")
        shifted_successes = 0
        shifted_success_video_saved = False
        shifted_failure_video_saved = False

        for trial_idx in range(cfg.num_trials_per_skill):
            print(f"  Trial {trial_idx + 1}/{cfg.num_trials_per_skill}...")

            # Use different initial states for variety
            reference_initial_state = initial_states[trial_idx % len(initial_states)]

            # Determine if we should save video for this trial
            save_video_for_trial = False
            video_suffix = None

            trial_result = evaluate_single_trial_with_shift(
                skill_name, reference_initial_state, instruction,
                cfg, vla, processor, action_head, proprio_projector,
                video_dir, video_suffix=None,  # Don't save yet, decide after
                initial_states=initial_states, resize_size=resize_size
            )

            trial_success = trial_result and trial_result.get("success", False)
            if trial_success:
                shifted_successes += 1
                # Save first success video
                if not shifted_success_video_saved:
                    video_suffix = f"shifted_success"
                    save_video_for_trial = True
                    shifted_success_video_saved = True
            else:
                # Save first failure video
                if not shifted_failure_video_saved:
                    video_suffix = f"shifted_failure"
                    save_video_for_trial = True
                    shifted_failure_video_saved = True

            # Re-run trial with video saving if needed
            if save_video_for_trial:
                print(f"   🎥 Re-running trial to save {video_suffix} video...")
                trial_result = evaluate_single_trial_with_shift(
                    skill_name, reference_initial_state, instruction,
                    cfg, vla, processor, action_head, proprio_projector,
                    video_dir, video_suffix=video_suffix,
                    initial_states=initial_states, resize_size=resize_size
                )

        results["shifted"]["successes"] = shifted_successes
        results["shifted"]["success_rate"] = shifted_successes / cfg.num_trials_per_skill
        results["shifted"]["disabled"] = False

        # Run MP robustness evaluation trials (method 3)
        print(f"\n🔄 MP ROBUSTNESS EVALUATION: {cfg.num_trials_per_skill} VLA trials (retrying MP failures)")
        mp_robustness_successes = 0
        mp_robustness_success_video_saved = False
        mp_robustness_failure_video_saved = False
        vla_trials_completed = 0
        total_attempts = 0

        while vla_trials_completed < cfg.num_trials_per_skill:
            total_attempts += 1
            print(f"  VLA Trial {vla_trials_completed + 1}/{cfg.num_trials_per_skill} (Attempt {total_attempts})...")

            # Use different initial states for variety
            reference_initial_state = initial_states[total_attempts % len(initial_states)]

            # Determine if we should save video for this trial
            save_video_for_trial = False
            video_suffix = None

            trial_result = evaluate_single_trial_with_mp_robustness(
                skill_name, reference_initial_state, instruction, cfg,
                vla, processor, action_head, proprio_projector,
                video_dir, video_suffix=None,  # Don't save yet, decide after
                initial_states=initial_states, resize_size=resize_size
            )

            # Check if motion planning failed - if so, retry without counting as VLA trial
            if trial_result and trial_result.get("reason", "").startswith("motion_planner_failed"):
                print(f"   🔄 Motion planning failed, retrying with different initial state...")
                continue  # Don't count this as a VLA trial

            # Motion planning succeeded, count this as a VLA trial
            vla_trials_completed += 1

            trial_success = trial_result and trial_result.get("success", False)
            if trial_success:
                mp_robustness_successes += 1
                # Save first success video
                if not mp_robustness_success_video_saved:
                    video_suffix = f"mp_robustness_success"
                    save_video_for_trial = True
                    mp_robustness_success_video_saved = True
            else:
                # Save first failure video (VLA failed, not MP)
                if not mp_robustness_failure_video_saved:
                    video_suffix = f"mp_robustness_failure"
                    save_video_for_trial = True
                    mp_robustness_failure_video_saved = True

            # Re-run trial with video saving if needed
            if save_video_for_trial:
                print(f"   🎥 Re-running trial to save {video_suffix} video...")
                trial_result = evaluate_single_trial_with_mp_robustness(
                    skill_name, reference_initial_state, instruction, cfg,
                    vla, processor, action_head, proprio_projector,
                    video_dir, video_suffix=video_suffix,
                    initial_states=initial_states, resize_size=resize_size
                )

        print(f"   📊 MP Robustness: {mp_robustness_successes}/{cfg.num_trials_per_skill} VLA successes ({total_attempts} total attempts including MP failures)")

        results["mp_robustness"]["successes"] = mp_robustness_successes
        results["mp_robustness"]["success_rate"] = mp_robustness_successes / cfg.num_trials_per_skill

        # Print trial summary
        print(f"\n📊 TRIAL SUMMARY for {skill_name}:")
        print(f"   Baseline: {baseline_successes}/{cfg.num_trials_per_skill} ({results['baseline']['success_rate']*100:.1f}%)")
        print(f"   Shifted: {shifted_successes}/{cfg.num_trials_per_skill} ({results['shifted']['success_rate']*100:.1f}%)")
        print(f"   MP Robustness: {mp_robustness_successes}/{cfg.num_trials_per_skill} ({results['mp_robustness']['success_rate']*100:.1f}%)")

        return results

    except Exception as e:
        print(f"❌ Error evaluating skill {skill_name}: {e}")
        traceback.print_exc()
        return None


def run_simplified_evaluation(cfg: SimplifiedConfig):
    """Run simplified pose shift evaluation."""
    print(f"🚀 Starting Simplified Pose Shift Evaluation")
    print(f"   Model: {cfg.model_family}")
    print(f"   Position shift: ±{cfg.shift_position_std*100:.1f}cm")
    print(f"   Orientation shift: ±{np.degrees(cfg.shift_orientation_std):.1f}°")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/{cfg.exp_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Create video directories with shift parameters in name
    video_base_dir = Path("images_videos/atomic_local_vla_eval_videos_with_shifts")
    pos_std_cm = cfg.shift_position_std * 100
    ori_std_deg = np.degrees(cfg.shift_orientation_std)
    video_dir = video_base_dir / f"simplified_eval_{timestamp}_pos{pos_std_cm:.1f}cm_ori{ori_std_deg:.1f}deg"
    video_dir.mkdir(parents=True, exist_ok=True)
    print(f"🎥 Video directory: {video_dir}")

    # Load VLA model and processor
    print("🔄 Loading VLA model...")
    vla, processor, action_head, proprio_projector = load_vla_model_and_processor(cfg, cfg.pretrained_checkpoint)

    # Make sure cfg has the correct unnorm_key extracted from the checkpoint
    if not cfg.unnorm_key:
        cfg.unnorm_key = extract_unnorm_key_from_checkpoint(cfg.pretrained_checkpoint)
        print(f"📋 Set unnorm_key in main config: {cfg.unnorm_key}")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Load atomic skills
    atomic_skills = load_atomic_skill_metadata(cfg)

    # Filter to target skill if specified
    if cfg.target_skill:
        if cfg.target_skill in atomic_skills:
            atomic_skills = [cfg.target_skill]
            print(f"🎯 Targeting specific skill: {cfg.target_skill}")
        else:
            print(f"❌ Target skill not found: {cfg.target_skill}")
            return

    # Handle focused evaluation mode for 3 specific skills
    if cfg.focused_eval_mode:
        focused_skills = [
            "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
            "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_pick",
            "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet"
        ]

        # Check which focused skills are available
        available_focused_skills = [skill for skill in focused_skills if skill in atomic_skills]
        if not available_focused_skills:
            print(f"❌ None of the focused skills found in available atomic skills")
            return

        atomic_skills = available_focused_skills
        print(f"🎯 Focused evaluation mode: {len(atomic_skills)} skills with 10 trials each")
        print(f"   Skills: {atomic_skills}")

        # Override num_trials_per_skill for focused mode
        cfg.num_trials_per_skill = 10

    # Handle ID-based evaluation mode
    elif cfg.ID_eval_mode:
        # Extract ID from checkpoint path
        id_str = extract_id_from_checkpoint(cfg.pretrained_checkpoint)
        
        # Get skills for this ID
        id_skills = get_skills_for_id(id_str)
        
        # Load BDDL to init mapping to find corresponding BDDL task names
        bddl_mapping = load_bddl_to_init_mapping()
        
        # Find BDDL task names that map to the ID skills
        bddl_skills = []
        for skill in id_skills:
            # Find BDDL task names that map to this skill
            for bddl_name, init_name in bddl_mapping.items():
                if init_name == skill and bddl_name in atomic_skills:
                    bddl_skills.append(bddl_name)
                    break
        
        if not bddl_skills:
            print(f"❌ None of the ID skills found in available atomic skills")
            return

        atomic_skills = bddl_skills
        print(f"🎯 ID evaluation mode ({id_str}): {len(atomic_skills)} skills")
        print(f"   Skills: {atomic_skills}")

        # Override num_trials_per_skill for ID mode (same as focused mode)
        cfg.num_trials_per_skill = 10

    # Evaluate each skill
    all_results = {}

    if cfg.focused_eval_mode or cfg.ID_eval_mode:
        # Use focused evaluation with multiple trials and success rate tracking
        for skill_name in tqdm.tqdm(atomic_skills, desc="Evaluating skills"):
            skill_results = evaluate_skill_focused_mode(
                skill_name, cfg, vla, processor, action_head, proprio_projector, video_dir, resize_size
            )
            all_results[skill_name] = skill_results
    else:
        # Use standard multi-trial evaluation
        for skill_name in tqdm.tqdm(atomic_skills, desc="Evaluating skills"):
            skill_results = evaluate_skill_with_pose_shift(
                skill_name, cfg, vla, processor, action_head, proprio_projector, video_dir, resize_size
            )
            all_results[skill_name] = skill_results

            # Print cumulative success rates after each skill
            print_cumulative_success_rates(all_results, skill_name)

    # Save results summary
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print final summary
    print(f"\n{'='*80}")
    print(f"📊 FINAL EVALUATION SUMMARY")
    print(f"{'='*80}")

    if cfg.focused_eval_mode or cfg.ID_eval_mode:
        # Focused/ID mode: show detailed success rates for each skill
        mode_name = "FOCUSED" if cfg.focused_eval_mode else "ID"
        print(f"🎯 {mode_name} EVALUATION RESULTS (10 trials per skill)")
        print(f"{'='*60}")

        for skill_name, results in all_results.items():
            if results:
                baseline_rate = results.get("baseline", {}).get("success_rate", 0.0) * 100
                shifted_rate = results.get("shifted", {}).get("success_rate", 0.0) * 100
                mp_robustness_rate = results.get("mp_robustness", {}).get("success_rate", 0.0) * 100

                print(f"\n🔸 {skill_name}:")
                print(f"   Baseline:      {results['baseline']['successes']}/{results['baseline']['total_trials']} ({baseline_rate:.1f}%)")
                print(f"   Shifted:       {results['shifted']['successes']}/{results['shifted']['total_trials']} ({shifted_rate:.1f}%)")
                print(f"   MP Robustness: {results['mp_robustness']['successes']}/{results['mp_robustness']['total_trials']} ({mp_robustness_rate:.1f}%)")
                print(f"   Shift Gap:     {baseline_rate - shifted_rate:.1f}%")
                print(f"   Robustness Gap: {baseline_rate - mp_robustness_rate:.1f}%")

        # Calculate overall averages
        total_skills = len([r for r in all_results.values() if r])
        if total_skills > 0:
            avg_baseline_rate = sum(r.get("baseline", {}).get("success_rate", 0.0) for r in all_results.values() if r) / total_skills * 100
            avg_shifted_rate = sum(r.get("shifted", {}).get("success_rate", 0.0) for r in all_results.values() if r) / total_skills * 100
            avg_mp_robustness_rate = sum(r.get("mp_robustness", {}).get("success_rate", 0.0) for r in all_results.values() if r) / total_skills * 100

            print(f"\n📊 OVERALL AVERAGES:")
            print(f"   Baseline:      {avg_baseline_rate:.1f}%")
            print(f"   Shifted:       {avg_shifted_rate:.1f}%")
            print(f"   MP Robustness: {avg_mp_robustness_rate:.1f}%")
            print(f"   Shift Gap:     {avg_baseline_rate - avg_shifted_rate:.1f}%")
            print(f"   Robustness Gap: {avg_baseline_rate - avg_mp_robustness_rate:.1f}%")

    else:
        # Standard mode: multi-trial results
        total_skills = len([r for r in all_results.values() if r])
        if total_skills > 0:
            # Calculate overall statistics
            total_baseline_successes = 0
            total_baseline_trials = 0
            total_shifted_successes = 0
            total_shifted_trials = 0
            total_mp_robustness_successes = 0
            total_mp_robustness_trials = 0

            for skill_name, results in all_results.items():
                if results:
                    # Baseline stats
                    baseline_successes = results.get("baseline", {}).get("successes", 0)
                    baseline_trials = results.get("baseline", {}).get("total_trials", 0)
                    total_baseline_successes += baseline_successes
                    total_baseline_trials += baseline_trials
                    
                    # Shifted stats
                    shifted_successes = results.get("shifted", {}).get("successes", 0)
                    shifted_trials = results.get("shifted", {}).get("total_trials", 0)
                    total_shifted_successes += shifted_successes
                    total_shifted_trials += shifted_trials
                    
                    # MP robustness stats
                    mp_successes = results.get("mp_robustness", {}).get("successes", 0)
                    mp_trials = results.get("mp_robustness", {}).get("total_trials", 0)
                    total_mp_robustness_successes += mp_successes
                    total_mp_robustness_trials += mp_trials

            # Calculate overall rates
            overall_baseline_rate = (total_baseline_successes / total_baseline_trials * 100) if total_baseline_trials > 0 else 0
            overall_shifted_rate = (total_shifted_successes / total_shifted_trials * 100) if total_shifted_trials > 0 else 0
            overall_mp_robustness_rate = (total_mp_robustness_successes / total_mp_robustness_trials * 100) if total_mp_robustness_trials > 0 else 0

            print(f"🎯 STANDARD EVALUATION RESULTS ({cfg.num_trials_per_skill} trials per skill)")
            print(f"{'='*60}")
            print(f"Total skills evaluated: {total_skills}")
            print(f"Baseline:      {total_baseline_successes}/{total_baseline_trials} ({overall_baseline_rate:.1f}%)")
            print(f"Shifted:       {total_shifted_successes}/{total_shifted_trials} ({overall_shifted_rate:.1f}%)")
            print(f"MP Robustness: {total_mp_robustness_successes}/{total_mp_robustness_trials} ({overall_mp_robustness_rate:.1f}%)")
            print(f"Shift Gap:     {overall_baseline_rate - overall_shifted_rate:.1f}%")
            print(f"Robustness Gap: {overall_baseline_rate - overall_mp_robustness_rate:.1f}%")

    print(f"📁 Results saved to: {results_path}")
    print(f"🎥 Videos saved to: {video_dir}")


if __name__ == "__main__":
    cfg = draccus.parse(SimplifiedConfig)
    run_simplified_evaluation(cfg)