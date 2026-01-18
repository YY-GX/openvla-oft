#!/usr/bin/env python3
"""
ckpts:
- id 8:
    - h100 for 60k steps (50k): runs/libero_above_atomic_long_id8/1.0.1/openvla-7b+libero_above_atomic_long_id8+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_8--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt
    - A6000 for 100k steps: 
    - A6000 for 100k steps (relative proprio + data augmentation - random erasing): runs/libero_above_atomic_long_id8/1.0.3/openvla-7b+libero_above_atomic_long_id8+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_8--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt
- id 10:
    - h100 for 100k steps (50k): runs/libero_above_atomic_long_id10/1.0.1/openvla-7b+libero_above_atomic_long_id10+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_10--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt
    - A6000 for 100k steps (relative proprio + data augmentation - random erasing): runs/libero_above_atomic_long_id10/1.0.2/openvla-7b+libero_above_atomic_long_id10+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_10--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt
"""

"""
Evaluate trained OpenVLA-OFT checkpoint on atomic skills using above-pose strategy.

Calculates above-object poses and uses MPlib to move there before VLA execution.

Usage:
  # Debug mode (3 skills, 3 trials each)
  python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_real_moving.py \
    --checkpoint runs/libero_above_atomic_long_id8/1.0.0/openvla-7b+libero_above_atomic_long_id8+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_8--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt \
    --eval_mode debug

  # ID mode
  python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_real_moving.py \
    --checkpoint runs/libero_above_atomic_long_id8/1.0.0/openvla-7b+libero_above_atomic_long_id8+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_8--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--40000_chkpt \    --eval_mode ID \
    --ID 2

  # Normal mode (all skills)
  python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_real_moving.py \
    --checkpoint runs/path/to/checkpoint \
    --eval_mode normal

  # With custom above height
  python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_real_moving.py \
    --checkpoint runs/path/to/checkpoint \
    --eval_mode ID \
    --ID 1 \
    --above_height 0.05

Evaluation Variants
--------------------
  # Debug mode (3 skills, 3 trials each)
  python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_real_moving.py \
    --checkpoint runs/libero_above_atomic_long_id10/1.0.1/openvla-7b+libero_above_atomic_long_id10+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_10--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --eval_mode debug \
    --eval_variant object_shift

  # # ID 8  

  # Clean evaluation (no perturbation) - ID mode
  CUDA_VISIBLE_DEVICES=3 python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_real_moving.py \
    --checkpoint runs/libero_above_atomic_long_id10/1.0.1/openvla-7b+libero_above_atomic_long_id10+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_10--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt \
    --eval_mode ID --ID 8 \
    --eval_variant clean

  CUDA_VISIBLE_DEVICES=6 python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_real_moving.py \
    --checkpoint runs/libero_above_atomic_long_id8/1.0.3/openvla-7b+libero_above_atomic_long_id8+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_8--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --eval_mode ID --ID 8 \
    --eval_variant clean

  # Above-pose shift evaluation - ID mode
  CUDA_VISIBLE_DEVICES=4 python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_real_moving.py \
    --checkpoint runs/libero_above_atomic_long_id8/1.0.1/openvla-7b+libero_above_atomic_long_id8+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_8--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt \
    --eval_mode ID --ID 8 \
    --eval_variant above_shift

  CUDA_VISIBLE_DEVICES=7 python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_real_moving.py \
    --checkpoint runs/libero_above_atomic_long_id8/1.0.3/openvla-7b+libero_above_atomic_long_id8+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_8--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --eval_mode ID --ID 8 \
    --eval_variant above_shift

  # Object shift evaluation - ID mode
  CUDA_VISIBLE_DEVICES=5 python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_real_moving.py \
    --checkpoint runs/libero_above_atomic_long_id8/1.0.1/openvla-7b+libero_above_atomic_long_id8+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_8--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt \
    --eval_mode ID --ID 8 \
    --eval_variant object_shift \
    --shift_xy_range 0.1

  CUDA_VISIBLE_DEVICES=0 python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_real_moving.py \
    --checkpoint runs/libero_above_atomic_long_id8/1.0.3/openvla-7b+libero_above_atomic_long_id8+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_8--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --eval_mode ID --ID 8 \
    --eval_variant object_shift \
    --shift_xy_range 0.1

  # # ID 10

  # Clean evaluation (no perturbation) - ID mode
  CUDA_VISIBLE_DEVICES=3 python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_real_moving.py \
    --checkpoint runs/libero_above_atomic_long_id10/1.0.1/openvla-7b+libero_above_atomic_long_id8+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_8--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt \
    --eval_mode ID --ID 10 \
    --eval_variant clean

  CUDA_VISIBLE_DEVICES=3 python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_real_moving.py \
    --checkpoint runs/libero_above_atomic_long_id10/1.0.1/openvla-7b+libero_above_atomic_long_id10+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_10--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --eval_mode ID --ID 10 \
    --eval_variant clean

  CUDA_VISIBLE_DEVICES=3 python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_real_moving.py \
    --checkpoint runs/libero_above_atomic_long_id10/1.0.2/openvla-7b+libero_above_atomic_long_id10+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_10--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --eval_mode ID --ID 10 \
    --eval_variant clean

  # Above-pose shift evaluation - ID mode
  CUDA_VISIBLE_DEVICES=4 python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_real_moving.py \
    --checkpoint runs/libero_above_atomic_long_id10/1.0.1/openvla-7b+libero_above_atomic_long_id10+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_10--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --eval_mode ID --ID 10 \
    --eval_variant above_shift \
    --shift_xy_range 0.02 --shift_z_range 0.02 --shift_ori_range 10.0

  CUDA_VISIBLE_DEVICES=4 python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_real_moving.py \
    --checkpoint runs/libero_above_atomic_long_id10/1.0.2/openvla-7b+libero_above_atomic_long_id10+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_10--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --eval_mode ID --ID 10 \
    --eval_variant above_shift \
    --shift_xy_range 0.02 --shift_z_range 0.02 --shift_ori_range 10.0

  # Object shift evaluation - ID mode
  CUDA_VISIBLE_DEVICES=5 python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_real_moving.py \
    --checkpoint runs/libero_above_atomic_long_id10/1.0.1/openvla-7b+libero_above_atomic_long_id10+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_10--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --eval_mode ID --ID 10 \
    --eval_variant object_shift \
    --shift_xy_range 0.1

  CUDA_VISIBLE_DEVICES=5 python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_real_moving.py \
    --checkpoint runs/libero_above_atomic_long_id10/1.0.2/openvla-7b+libero_above_atomic_long_id10+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_10--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --eval_mode ID --ID 10 \
    --eval_variant object_shift \
    --shift_xy_range 0.1
    
"""

"""
Real moving object poses (fixed debug):


Long horizon tasks v1:
   📏 skill_type: pick, grasped_object_name: None
   📏 after special handling: obj_pos: [-0.372 -0.248 0.928], above_height: 0.03, bbox_z: 0.028277621484617654
  ✓ Object pose: [-0.372, -0.248, 0.928] (h=0.030m)
  ✓ Above pose: [-0.372, -0.248, 0.986] (h=0.030m)



========================================================
   📏 Initial obj_pos: [0.115 0.240 0.487], above_height: 0.03, bbox_z: 0.03711814965388327
========================================================
   📏 skill_type: place, grasped_object_name: wine_bottle_1_main
   📏 Added grasped object (wine_bottle_1_main) bbox_z (0.062m) to above_height
   📏 after special handling: obj_pos: [0.065 0.140 0.487], above_height: 0.1595367121019185, bbox_z: 0.03711814965388327


========================================================
   📏 Initial obj_pos: [0.006 0.142 0.951], above_height: 0.03, bbox_z: 0.037433316241989756
========================================================
   📏 skill_type: place, grasped_object_name: wine_bottle_1_main
   📏 Added grasped object (wine_bottle_1_main) bbox_z (0.051m) to above_height
   📏 after special handling: obj_pos: [-0.044 0.042 0.951], above_height: 0.13617898955271474, bbox_z: 0.037433316241989756
      Object pose: [-0.044, 0.042, 0.951] (h=0.136m)
      Above pose: [-0.044, 0.042, 1.124] (h=0.136m)   
"""


import argparse
import json
import os
import pickle
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add paths
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')

from libero.libero.envs import OffScreenRenderEnv

# Import VLA utilities
from experiments.robot.libero.libero_utils import get_libero_wrist_image, quat2axisangle
from experiments.robot.openvla_utils import (
    get_action_head, get_processor, get_proprio_projector, resize_image_for_policy
)
from experiments.robot.robot_utils import (
    get_model, get_image_resize_size, get_action,
    normalize_gripper_action, invert_gripper_action
)

# Import motion planner
from scripts.phase3.pipeline.motion_planning.mplib.planner import MPlibMotionPlanner

# Import pose calculation functions
from scripts.phase3.pipeline.utils.local_pose_tools import (
    get_object_pose,
    get_special_object_handling,
    calculate_above_pose
)

# Import random erasing utility
from scripts.phase3.pipeline.utils.random_erasing_mask_tools import apply_random_erasing_augmentation


def shift_object_position(env, target_object: str, shift_range: float = 0.03, eval_mode: str = 'normal'):
    """
    Shift movable object position by random XY offset.

    Args:
        env: LIBERO environment
        target_object: Object name to shift
        shift_range: Max XY shift in meters (default ±3cm)
        eval_mode: Evaluation mode. If 'debug', uses fixed shift [-0.373, -0.245]

    Returns:
        (success: bool, shift_xy: np.ndarray or None)
    """
    try:
        sim = env.sim
        body_id = sim.model.body_name2id(target_object)

        # Check if object has free joint (movable)
        if sim.model.body_jntnum[body_id] == 0:
            return False, None  # Fixture, can't move

        joint_id = sim.model.body_jntadr[body_id]
        if sim.model.jnt_type[joint_id] != 0:  # 0 = free joint
            return False, None

        qpos_start = sim.model.jnt_qposadr[joint_id]

        # Get original position before shift (for debug output)
        original_xy = sim.data.qpos[qpos_start:qpos_start+2].copy()

        # Apply fixed shift in debug mode, otherwise random XY shift (keep Z unchanged)
        if eval_mode == 'debug':
            shift_xy = np.array([-0.32, -0.245])
        else:
            shift_xy = np.random.uniform(-shift_range, shift_range, size=2)
        sim.data.qpos[qpos_start:qpos_start+2] += shift_xy
        sim.forward()

        # Debug: Print actual position after shift
        if eval_mode == 'debug':
            final_xy = sim.data.qpos[qpos_start:qpos_start+2].copy()
            print(f"      [DEBUG] Object '{target_object}' shift: original={original_xy}, shift={shift_xy}, final={final_xy}")

        return True, shift_xy
    except:
        return False, None


def load_bddl_to_init_mapping():
    """Load BDDL to init mapping."""
    mapping_path = "scripts/phase3/pipeline/config/bddl_to_init_mapping.json"
    with open(mapping_path, 'r') as f:
        return json.load(f)


def load_skill_config():
    """Load skill config."""
    config_path = "scripts/phase3/pipeline/config/skill_config.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def get_target_object_and_bddl(skill_name: str, skill_config: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Get target object and BDDL file from skill config."""
    for skill_key, skill_info in skill_config.items():
        if skill_key.lower() == skill_name.lower():
            # Get target object (take first one if list)
            target_objects = skill_info.get('target_object', [])
            target_object = target_objects[0] if target_objects else None

            # Get BDDL file (take first one if list)
            bddl_files = skill_info.get('bddl_files', [])
            bddl_file = bddl_files[0] if bddl_files else None

            return target_object, bddl_file

    return None, None


def find_bddl_from_skill(skill_name: str, mapping: Dict) -> Optional[str]:
    """Find BDDL file from skill name (returns first match)."""
    for bddl_name, init_name in mapping.items():
        if init_name == skill_name:
            return f"{bddl_name}.bddl"
    return None


def infer_skill_type(skill_name: str) -> str:
    """Infer skill type from name."""
    if 'pick' in skill_name.lower():
        return 'pick'
    elif 'place' in skill_name.lower():
        return 'place'
    else:
        return 'other'


def get_grasped_object_candidates(skill_name: str) -> List[str]:
    """Get possible grasped object names for place skills."""
    object_map = {
        'black_bowl': ['akita_black_bowl_1_main', 'akita_black_bowl_2_main', 'akita_black_bowl_3_main'],
        'white_bowl': ['white_bowl_1_main', 'white_bowl_2_main'],
        'ketchup': ['ketchup_1_main'],
        'wine_bottle': ['wine_bottle_1_main'],
        'frying_pan': ['frying_pan_1_main'],
        'moka_pot': ['moka_pot_1_main', 'moka_pot_2_main'],
        'red_mug': ['red_mug_1_main'],
        'white_mug': ['white_mug_1_main'],
        'chocolate_pudding': ['chocolate_pudding_1_main'],
    }

    for keyword, candidates in object_map.items():
        if keyword in skill_name.lower():
            return candidates

    return []


def find_grasped_object_in_env(env, candidates: List[str]) -> Optional[str]:
    """Find which grasped object exists in environment."""
    for obj_name in candidates:
        try:
            obj_id = env.sim.model.body_name2id(obj_name)
            if obj_id >= 0:
                return obj_name
        except:
            continue
    return None


def combine_camera_views(obs: dict) -> np.ndarray:
    """Combine agentview and wrist camera into side-by-side frame.

    Args:
        obs: Observation dict containing 'agentview_image' and 'robot0_eye_in_hand_image'

    Returns:
        Combined image with agentview on left, wrist on right (both rotated 180°)
    """
    # Get both camera views
    agentview = obs['agentview_image'].copy()
    wrist = obs['robot0_eye_in_hand_image'].copy()

    # Ensure uint8 format
    if agentview.dtype != np.uint8:
        agentview = (agentview * 255).astype(np.uint8)
    if wrist.dtype != np.uint8:
        wrist = (wrist * 255).astype(np.uint8)

    # Rotate both 180 degrees
    agentview = cv2.rotate(agentview, cv2.ROTATE_180)
    wrist = cv2.rotate(wrist, cv2.ROTATE_180)

    # Concatenate side-by-side (agentview left, wrist right)
    combined = np.concatenate([agentview, wrist], axis=1)

    return combined


def evaluate_with_above_pose(env, init_state, skill_name: str, skill_type: str,
                            bddl_file: str, target_object: str,
                            cfg, vla, processor, action_head, proprio_projector,
                            resize_size: int, above_height: float = 0.1,
                            max_steps: int = 300,
                            eval_variant: str = 'clean',
                            shift_xy_range: float = 0.05,
                            shift_z_range: float = 0.02,
                            shift_ori_range: float = 0.524,
                            eval_mode: str = 'normal',
                            use_relative_pose: bool = True,
                            mask_as_agentview: bool = False,
                            collision_aware: bool = False,
                            save_pointcloud: bool = False,
                            pointcloud_output_dir: Optional[str] = None,
                            velocity_factor: float = 1.0,
                            time_step: float = 0.025,
                            safety_margin: float = 1.1,
                            apply_background_erasing: bool = False,
                            apply_distractor_masking: bool = False) -> Tuple[bool, List]:
    """Evaluate by calculating above pose and moving there with mplib before VLA."""

    is_place = (skill_type == 'place')
    grasped_object = None

    # Handle place skills: disable gravity, set state, close gripper, enable gravity
    if is_place:
        old_gravity = env.sim.model.opt.gravity.copy()
        env.sim.model.opt.gravity[:] = 0
        env.sim.forward()

    # Set init state (full sim state)
    if skill_type != 'place':    
        for _ in range(5):
            obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
    else:
        env.set_init_state(init_state)
    obs = env.env._get_observations()

    frames = []
    frames.append(combine_camera_views(obs))

    # Object shift variant: Perturb target object position
    if eval_variant == 'object_shift':
        success, shift = shift_object_position(env, target_object, shift_range=shift_xy_range, eval_mode=eval_mode)
        if success:
            print(f"      Object shifted: [{shift[0]:.3f}, {shift[1]:.3f}]m")
        else:
            print(f"      ⚠️  Cannot shift {target_object} (fixture), using clean eval")

    if is_place:
        # Close gripper (5 steps)
        for _ in range(5):
            obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
            frames.append(combine_camera_views(obs))

        # Re-enable gravity
        env.sim.model.opt.gravity[:] = old_gravity
        env.sim.forward()
        obs = env.env._get_observations()
        frames.append(combine_camera_views(obs))

        # Find grasped object for collision avoidance
        candidates = get_grasped_object_candidates(skill_name)
        grasped_object = find_grasped_object_in_env(env, candidates)
        if grasped_object:
            print(f"      Grasped object: {grasped_object}")

    # Calculate above pose using same method as evaluate_above.py
    try:
        obj_pos, _, calculated_above_height, bbox_z = get_special_object_handling(
            env, target_object, bddl_file, above_height,
            skill_type=skill_type, grasped_object_name=grasped_object
        )

        print(f"      Object pose: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}] (h={calculated_above_height:.3f}m)")

        # Calculate above pose with optional shift
        if eval_variant == 'above_shift':
            above_pos, above_quat, shift_info = calculate_above_pose(
                obj_pos,
                above_height=calculated_above_height,
                shift=True,
                xy_range=shift_xy_range,
                z_range=shift_z_range,
                ori_range=shift_ori_range,
                bbox_z=bbox_z
            )
            print(f"      Above shift: XY=[{shift_info['xy_shift'][0]:.3f}, {shift_info['xy_shift'][1]:.3f}], Z={shift_info['z_shift']:.3f}, Ori={shift_info['ori_shift']:.1f}°")
        else:
            above_pos, above_quat, _ = calculate_above_pose(
                obj_pos,
                above_height=calculated_above_height,
                shift=False,
                bbox_z=bbox_z
            )

        print(f"      Above pose: [{above_pos[0]:.3f}, {above_pos[1]:.3f}, {above_pos[2]:.3f}] (h={calculated_above_height:.3f}m)")

    except Exception as e:
        print(f"      ✗ Above pose calculation error: {e}")
        return False, frames

    # Move to above pose using mplib with retry mechanism
    motion_planner = MPlibMotionPlanner(env, collision_aware=collision_aware,
                                       velocity_factor=velocity_factor,
                                       time_step=time_step,
                                       safety_margin=safety_margin,
                                       verbose=False)

    move_success = False
    max_mp_retries = 3
    obs_list = []

    for mp_retry in range(max_mp_retries):
        if mp_retry > 0:
            print(f"      Retrying motion planning (attempt {mp_retry + 1}/{max_mp_retries})...")

        move_success, results = motion_planner.move_to_pose(
            above_pos, above_quat,
            position_threshold=0.02,
            orientation_threshold=0.524,
            skill_type=skill_type,
            grasped_object_name=grasped_object,
            save_pointcloud=save_pointcloud,
            pointcloud_output_dir=pointcloud_output_dir
        )
        obs_list = results[0]

        # Add motion planning frames (even if failed, for debugging)
        if isinstance(obs_list, list):
            for obs_mp in obs_list:
                frames.append(combine_camera_views(obs_mp))

        if move_success:
            if mp_retry > 0:
                print(f"      ✓ Motion planning succeeded on attempt {mp_retry + 1}")
            else:
                print(f"      ✓ Motion planning succeeded")
            break

    if not move_success:
        print(f"      ✗ Motion planning failed after {max_mp_retries} attempts")
        return False, frames

    # Get fresh observation at VLA start (matches evaluate_above.py behavior)
    obs = env.env._get_observations()

    # Run VLA from above pose
    language = skill_name.replace('_', ' ')
    success, vla_frames = run_vla(env, obs, language, cfg, vla, processor,
                             action_head, proprio_projector, resize_size, max_steps,
                             target_object, use_relative_pose, mask_as_agentview, skill_name,
                             apply_background_erasing, apply_distractor_masking)

    return success, frames + vla_frames


def run_vla(env, obs, language: str, cfg, vla, processor, action_head,
           proprio_projector, resize_size: int, max_steps: int,
           target_object: str, use_relative_pose: bool = True, mask_as_agentview: bool = False,
           skill_name: str = "", apply_background_erasing: bool = False,
           apply_distractor_masking: bool = False) -> Tuple[bool, List]:
    """Run VLA until success or max steps."""

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    frames = []

    # Capture initial state frame before any actions
    frames.append(combine_camera_views(obs))

    for t in range(max_steps):
        # Get action if queue is empty
        if len(action_queue) == 0:
            # Prepare observation with AXIS-ANGLE (not euler)
            wrist_img = get_libero_wrist_image(obs)

            # Apply background erasing if enabled
            wrist_img_masked = wrist_img
            if apply_background_erasing and target_object:
                try:
                    from scripts.phase3.pipeline.utils.segmentation_utils import create_wrist_segmentation_mask
                    seg_mask = create_wrist_segmentation_mask(env, target_object, resolution=256)

                    # Fixed parameters (matching dataset builder intent)
                    erasing_area_ratio = 30  # 30% of background
                    num_rectangles = 3       # 3 rectangles

                    # Apply random erasing to segmentation mask
                    erased_mask, erased_pixels = apply_random_erasing_augmentation(
                        seg_mask,
                        erasing_area_ratio=erasing_area_ratio,
                        num_rectangles=num_rectangles,
                        random_state=None  # Use random seed each time
                    )

                    # Black out erased regions in wrist image
                    wrist_img_masked = wrist_img.copy()
                    wrist_img_masked[erased_pixels] = 0
                except Exception as e:
                    pass  # Use original wrist_img if erasing fails

            # Apply distractor masking if enabled
            if apply_distractor_masking and target_object:
                try:
                    from scripts.phase3.pipeline.utils.segmentation_utils import apply_distractor_masking

                    # Determine grasped object for place skills
                    is_place_skill = "place" in skill_name.lower()
                    grasped_object_name = None
                    if is_place_skill:
                        candidates = get_grasped_object_candidates(skill_name)
                        grasped_object_name = find_grasped_object_in_env(env, candidates)

                    wrist_img_masked = apply_distractor_masking(
                        env,
                        wrist_img_masked,
                        target_object,
                        grasped_object_name=grasped_object_name,
                        resolution=256
                    )
                except Exception as e:
                    pass  # Use original wrist_img if masking fails

            wrist_img_resized = resize_image_for_policy(wrist_img_masked, resize_size)

            # Generate mask as separate image if mask_as_agentview is enabled
            mask_img_resized = None
            if mask_as_agentview:
                if target_object:
                    try:
                        from scripts.phase3.pipeline.utils.segmentation_utils import (
                            create_wrist_segmentation_mask, create_wrist_segmentation_mask_with_grasped
                        )
                        is_place_skill = "place" in skill_name.lower()
                        if is_place_skill:
                            # For place skills, need to find grasped object
                            candidates = get_grasped_object_candidates(skill_name)
                            grasped_object = find_grasped_object_in_env(env, candidates)
                            if grasped_object:
                                mask_img = create_wrist_segmentation_mask_with_grasped(
                                    env, target_object, grasped_object, resolution=256
                                )
                            else:
                                # Fallback to normal mask if can't find grasped object
                                mask_img = create_wrist_segmentation_mask(env, target_object, resolution=256)
                        else:
                            mask_img = create_wrist_segmentation_mask(env, target_object, resolution=256)

                        # Convert 2D mask to 3-channel if needed
                        if len(mask_img.shape) == 2:
                            mask_img = np.stack([mask_img] * 3, axis=-1)

                        mask_img_resized = resize_image_for_policy(mask_img, resize_size)
                    except Exception as e:
                        print(f"Warning: Failed to generate mask as agentview: {e}")
                        mask_img_resized = None
                else:
                    # No target object - create blank white mask
                    print(f"  ⚠️  No target object for {skill_name}, using blank mask")
                    mask_img = np.ones((256, 256, 3), dtype=np.uint8) * 255
                    mask_img_resized = resize_image_for_policy(mask_img, resize_size)

            if use_relative_pose:
                # Get object pose
                obj_pos, obj_quat_wxyz = get_object_pose(env, target_object)

                # Calculate relative position
                ee_pos = obs["robot0_eef_pos"]
                relative_pos = ee_pos - obj_pos

                # Calculate relative orientation
                # Convert quaternions to scipy format [x,y,z,w]
                ee_quat_xyzw = obs["robot0_eef_quat"]  # Already [x,y,z,w]
                obj_quat_xyzw = np.array([obj_quat_wxyz[1], obj_quat_wxyz[2], obj_quat_wxyz[3], obj_quat_wxyz[0]])

                # Calculate relative rotation
                obj_rot = R.from_quat(obj_quat_xyzw)
                ee_rot = R.from_quat(ee_quat_xyzw)
                relative_rot = obj_rot.inv() * ee_rot
                relative_quat_xyzw = relative_rot.as_quat()

                # Convert to axis-angle
                relative_ori = quat2axisangle(relative_quat_xyzw)

                observation = {
                    "full_image": wrist_img_resized,
                    "state": np.concatenate((
                        relative_pos,
                        relative_ori,
                        obs["robot0_gripper_qpos"]
                    ))
                }
            else:
                # Use absolute pose (backward compatibility)
                observation = {
                    "full_image": wrist_img_resized,
                    "state": np.concatenate((
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),  # AXIS, not euler
                        obs["robot0_gripper_qpos"]
                    ))
                }

            # Add mask as second view if mask_as_agentview is enabled (must contain "wrist" in key name)
            if mask_as_agentview and mask_img_resized is not None:
                observation["wrist_mask"] = mask_img_resized

            actions = get_action(
                cfg, vla, observation, language,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=None,
                use_film=False
            )
            action_queue.extend(actions)

        # Get action and process
        action = action_queue.popleft()
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)

        # Execute action
        obs, reward, done, info = env.step(action.tolist())

        # Apply background erasing and/or distractor masking to observation for video recording
        if (apply_background_erasing or apply_distractor_masking) and target_object:
            try:
                import cv2
                wrist_img_for_video = get_libero_wrist_image(obs)

                # Apply background erasing if enabled
                if apply_background_erasing:
                    from scripts.phase3.pipeline.utils.segmentation_utils import create_wrist_segmentation_mask
                    seg_mask = create_wrist_segmentation_mask(env, target_object, resolution=256)

                    erased_mask, erased_pixels = apply_random_erasing_augmentation(
                        seg_mask,
                        erasing_area_ratio=30,
                        num_rectangles=3,
                        random_state=None
                    )

                    wrist_img_for_video = wrist_img_for_video.copy()
                    wrist_img_for_video[erased_pixels] = 0

                # Apply distractor masking if enabled
                if apply_distractor_masking:
                    from scripts.phase3.pipeline.utils.segmentation_utils import apply_distractor_masking

                    # Determine grasped object for place skills
                    is_place_skill = "place" in skill_name.lower()
                    grasped_object_name = None
                    if is_place_skill:
                        candidates = get_grasped_object_candidates(skill_name)
                        grasped_object_name = find_grasped_object_in_env(env, candidates)

                    wrist_img_for_video = apply_distractor_masking(
                        env,
                        wrist_img_for_video,
                        target_object,
                        grasped_object_name=grasped_object_name,
                        resolution=256
                    )

                # Rotate back to original orientation for obs (combine_camera_views will rotate again)
                wrist_img_for_obs = cv2.rotate(wrist_img_for_video, cv2.ROTATE_180)
                obs['robot0_eye_in_hand_image'] = wrist_img_for_obs
            except Exception as e:
                pass

        frames.append(combine_camera_views(obs))

        if done:
            return True, frames

    return False, frames


def save_video(frames: List, output_path: str):
    """Save video from frames."""
    if len(frames) == 0:
        return

    frames_array = np.array(frames)
    if frames_array.dtype != np.uint8:
        frames_array = (frames_array * 255).astype(np.uint8)

    imageio.mimwrite(output_path, frames_array, fps=30, quality=8)
    print(f"      💾 Saved video: {Path(output_path).name}")


def parse_checkpoint_info(checkpoint_path: str) -> Tuple[str, str]:
    """
    Parse version and steps from checkpoint path.

    Example: runs/libero_above_atomic_long_id8/1.0.1/openvla-7b+...+50000_chkpt
    Returns: ('1.0.1', '50000')
    """
    path = Path(checkpoint_path)
    parts = path.parts

    # Extract version (e.g., '1.0.1') - typically the 3rd component after 'runs'
    version = "unknown"
    if 'runs' in parts:
        runs_idx = parts.index('runs')
        if runs_idx + 2 < len(parts):
            version = parts[runs_idx + 2]

    # Extract steps from checkpoint name (e.g., '50000_chkpt' -> '50000')
    steps = "unknown"
    checkpoint_name = path.name
    if '_chkpt' in checkpoint_name or 'chkpt' in checkpoint_name:
        # Extract number before _chkpt or chkpt
        import re
        match = re.search(r'--(\d+)_?chkpt', checkpoint_name)
        if match:
            steps = match.group(1)

    return version, steps


def main():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--checkpoint', type=str, required=True)

    # Eval mode
    parser.add_argument('--eval_mode', type=str, default='normal',
                       choices=['normal', 'ID', 'debug'])
    parser.add_argument('--ID', type=int, default=None,
                       help='ID for ID mode (1, 2, 3, 8, 9, 10, or 17)')

    # Eval params
    parser.add_argument('--num_trials', type=int, default=10,
                       help='Number of trials per skill')
    parser.add_argument('--above_height', type=float, default=0.1,
                       help='Default height above object (meters) - original 0.03m')

    # Eval variant (works with all modes: debug/ID/normal)
    parser.add_argument('--eval_variant', type=str, default='clean',
                       choices=['clean', 'above_shift', 'object_shift'],
                       help='Evaluation variant: clean (no perturbation), above_shift (perturb above pose), object_shift (perturb object position)')

    # Shift parameters for eval variants (should match training defaults)
    parser.add_argument('--shift_xy_range', type=float, default=0.05,
                       help='XY shift range in meters (default: 0.04m = ±4cm, matches training)')
    parser.add_argument('--shift_z_range', type=float, default=0.05,
                       help='Z shift range in meters (default: 0.02m = ±2cm)')
    parser.add_argument('--shift_ori_range', type=float, default=15.0,
                       help='Orientation shift range in degrees (default: 15.0° = ±15°, matches training)')

    # Paths
    parser.add_argument('--init_dir', type=str,
                       default='datasets/hdf5_datasets/atomic_above_27_skills/all_downsampled')
    parser.add_argument('--bddl_dir', type=str,
                       default='externals/boss/libero/libero/bddl_files/atomic_skills')

    parser.add_argument('--use_relative_pose', type=lambda x: x.lower() == 'true', default=True,
                       help='Use relative EE position and orientation (relative to object) for VLA input (default: True)')
    parser.add_argument('--mask_as_agentview', action='store_true', default=False,
                       help='Use segmentation mask as second image input (for dual-view models trained with wrist+mask, default: False)')
    parser.add_argument('--apply_background_erasing', action='store_true', default=False,
                       help='Apply random rectangular erasing to wrist camera background (default: False)')
    parser.add_argument('--apply_distractor_masking', action='store_true', default=False,
                       help='Mask out distractor objects in wrist camera by blacking out their bounding boxes (default: False)')
    parser.add_argument('--mplib_save_pointcloud', action='store_true', default=False,
                       help='Save scene pointcloud and camera images for each MPlib planning call (default: False)')
    parser.add_argument('--collision_aware', type=lambda x: x.lower() == 'true', default=True,
                       help='Enable MPlib collision avoidance (default: False, must be True to save pointclouds)')

    # MPlib motion planner parameters (should match demo generation defaults)
    parser.add_argument('--velocity_factor', type=float, default=1.0,
                       help='MPLib execution speed factor (0.0-1.0, default: 1.0, matches training)')
    parser.add_argument('--time_step', type=float, default=0.025,
                       help='MPLib trajectory planning time step (default: 0.025, matches training)')
    parser.add_argument('--safety_margin', type=float, default=1.1,
                       help='Safety margin multiplier for bounding box size (default: 1.1, matches training)')

    args = parser.parse_args()

    # Parse checkpoint info
    version, steps = parse_checkpoint_info(args.checkpoint)

    # Create descriptive output directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build folder name components
    folder_parts = [timestamp]

    # Add eval variant (for real_moving)
    folder_parts.append(args.eval_variant)

    # Add mode and mode attribute
    if args.eval_mode == 'ID':
        folder_parts.append(f"ID_{args.ID}")
    elif args.eval_mode == 'debug':
        folder_parts.append("debug")
    else:
        folder_parts.append("normal")

    # Add version and steps
    folder_parts.append(f"v{version}")
    folder_parts.append(f"step{steps}")

    folder_name = "_".join(folder_parts)

    results_dir = Path(f"scripts/phase3/pipeline/outputs/results/atomic_skills_eval/{folder_name}")
    videos_dir = Path(f"scripts/phase3/pipeline/outputs/videos/atomic_skills_eval/{folder_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Results: {results_dir}")
    print(f"🎥 Videos: {videos_dir}")

    # MPlib pointcloud debug directory setup
    mplib_pointcloud_dir = None
    if args.mplib_save_pointcloud:
        pointcloud_base = Path("/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase3/pipeline/outputs/pointclouds/atomic_skills_eval_debug")
        mplib_pointcloud_dir = pointcloud_base / folder_name
        mplib_pointcloud_dir.mkdir(parents=True, exist_ok=True)
        print(f"📊 MPlib debug pointclouds will be saved to: {mplib_pointcloud_dir}")

    # Load VLA model
    from dataclasses import dataclass
    from typing import Union

    # @dataclass
    # class VLAConfig:
    #     model_family: str = "openvla"
    #     pretrained_checkpoint: Union[str, Path] = ""
    #     use_l1_regression: bool = True
    #     use_diffusion: bool = False
    #     use_film: bool = False
    #     num_images_in_input: int = 1
    #     use_proprio: bool = True
    #     center_crop: bool = True
    #     num_open_loop_steps: int = 8
    #     unnorm_key: str = ""
    #     load_in_8bit: bool = False
    #     load_in_4bit: bool = False
    #     is_depth: bool = False


    @dataclass
    class VLAConfig:
        model_family: str = "openvla"
        pretrained_checkpoint: Union[str, Path] = ""
        use_l1_regression: bool = True
        use_diffusion: bool = False
        num_diffusion_steps: int = 50
        use_film: bool = False
        num_images_in_input: int = 2
        use_proprio: bool = True
        center_crop: bool = True
        num_open_loop_steps: int = 8
        unnorm_key: Union[str, Path] = ""
        load_in_8bit: bool = False
        load_in_4bit: bool = False
        task_suite_name: str = "atomic_skills"
        wrist_only: bool = False
        agent_only: bool = False
        pro_only: bool = False
        is_oss: bool = False
        is_depth: bool = False

    cfg = VLAConfig()
    cfg.pretrained_checkpoint = args.checkpoint
    cfg.num_images_in_input = 2 if args.mask_as_agentview else 1

    # Auto-detect unnorm_key from checkpoint path
    # First try to extract from checkpoint filename (contains full dataset name with version)
    # e.g., .../openvla-7b+libero_above_atomic_long_id10:1.0.1+b16+... -> unnorm_key = "libero_above_atomic_long_id10:1.0.1"
    checkpoint_filename = Path(args.checkpoint).name
    cfg.unnorm_key = None

    if "+" in checkpoint_filename:
        # Checkpoint filename format: openvla-7b+dataset_name+other_params
        parts = checkpoint_filename.split("+")
        if len(parts) >= 2:
            # Extract dataset name (second part after first +)
            cfg.unnorm_key = parts[1]

    # Fallback to folder name extraction if filename extraction didn't work
    if not cfg.unnorm_key:
        checkpoint_parts = Path(args.checkpoint).parts
        if "runs" in checkpoint_parts:
            runs_idx = checkpoint_parts.index("runs")
            if runs_idx + 1 < len(checkpoint_parts):
                cfg.unnorm_key = checkpoint_parts[runs_idx + 1]
            else:
                cfg.unnorm_key = "bridge_orig"
        else:
            cfg.unnorm_key = "bridge_orig"

    print(f"🤖 Loading VLA model...")
    vla = get_model(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    resize_size = get_image_resize_size(cfg)

    # Verify the key exists in norm_stats, if not try to find a matching key
    if hasattr(vla, 'norm_stats') and vla.norm_stats:
        if cfg.unnorm_key not in vla.norm_stats:
            # Try to find a key that starts with the base name
            base_key = cfg.unnorm_key.split(":")[0] if ":" in cfg.unnorm_key else cfg.unnorm_key
            matching_keys = [k for k in vla.norm_stats.keys() if k.startswith(base_key)]
            if matching_keys:
                original_key = cfg.unnorm_key
                cfg.unnorm_key = matching_keys[0]
                print(f"   ⚠️  Key '{original_key}' not found in norm_stats, using '{cfg.unnorm_key}' instead")
            else:
                print(f"   ⚠️  Warning: Key '{cfg.unnorm_key}' not found in norm_stats. Available keys: {list(vla.norm_stats.keys())}")

    print(f"   Detected unnorm_key: {cfg.unnorm_key}")
    print(f"✅ Model loaded")

    # Load skill config
    skill_config = load_skill_config()

    # Get skills to evaluate
    if args.eval_mode == 'debug':
        skills = [
            # 'open_the_bottom_drawer_of_the_cabinet',
            # 'pick_ketchup',
            # 'place_ketchup_in_top_drawer_of_the_cabinet',
            # "pick_black_bowl",
            # 'place_frying_pan_on_the_stove',
            'pick_frying_pan'
        ]
        num_trials = 10
    elif args.eval_mode == 'ID':
        if args.ID == 1:
            skills = ['pick_moka_pot', 'place_moka_pot_on_the_stove', 'turn_on_the_stove',
                     'pick_frying_pan', 'place_frying_pan_on_the_stove', 'open_the_microwave']
        elif args.ID == 2:
            skills = ['pick_black_bowl', 'place_black_bowl_on_the_plate', 'open_the_top_drawer_of_the_cabinet',
                     'pick_ketchup', 'place_ketchup_in_top_drawer_of_the_cabinet', 'close_the_top_drawer_of_the_cabinet']
        elif args.ID == 3:
            skills = ['open_the_bottom_drawer_of_the_cabinet', 'pick_black_bowl', 'place_black_bowl_on_the_plate',
                     'pick_wine_bottle', 'place_wine_bottle_in_bottom_drawer_of_the_cabinet',
                     'close_the_bottom_drawer_of_the_cabinet']
        elif args.ID == 8:
            skills = [
                'pick_black_bowl', 'place_black_bowl_in_bottom_drawer_of_the_cabinet', 
                     'close_the_bottom_drawer_of_the_cabinet', 'place_black_bowl_on_the_plate',
                     'place_black_bowl_on_the_black_bowl', 'pick_cream_cheese', 'place_cream_cheese_in_basket'
                     ]
        elif args.ID == 9:
            skills = [
                    # 'place_wine_bottle_in_bottom_drawer_of_the_cabinet',
                     'pick_alphabet_soup', 'place_alphabet_soup_in_basket', 'pick_cream_cheese',
                     'place_cream_cheese_in_basket', 'pick_tomato_sauce', 'place_tomato_sauce_in_basket',
                     'pick_wine_bottle', 'place_wine_bottle_in_bottom_drawer_of_the_cabinet',
                     'close_the_bottom_drawer_of_the_cabinet'
                     ]
        elif args.ID == 10:
            skills = ['turn_on_the_stove', 'pick_frying_pan', 'place_frying_pan_on_the_stove',
                     'pick_moka_pot', 'place_moka_pot_on_the_stove', 'open_the_microwave']
        elif args.ID == 17:
            skills = [
                # 'place_white_mug_on_the_plate', 'place_yellow_and_white_mug_on_right_plate'
                     'pick_alphabet_soup', 'pick_butter', 'pick_chocolate_pudding',
                     'pick_cream_cheese', 'pick_moka_pot', 'pick_tomato_sauce',
                     'pick_white_mug', 'pick_yellow_and_white_mug',
                     'place_alphabet_soup_in_basket', 'place_butter_in_basket',
                     'place_chocolate_pudding_to_right_of_plate', 'place_cream_cheese_in_basket',
                     'place_moka_pot_on_the_stove', 'place_tomato_sauce_in_basket',
                     'place_white_mug_on_the_plate', 'place_yellow_and_white_mug_on_right_plate',
                     'turn_on_the_stove'
                     ]
        elif args.ID == 8910:
            # Combine all skills from ID 8, 9, and 10, keeping them unique
            skills_id8 = ['pick_black_bowl', 'place_black_bowl_in_bottom_drawer_of_the_cabinet', 
                         'close_the_bottom_drawer_of_the_cabinet', 'place_black_bowl_on_the_plate',
                         'place_black_bowl_on_the_black_bowl', 'pick_cream_cheese', 'place_cream_cheese_in_basket']
            skills_id9 = ['pick_alphabet_soup', 'place_alphabet_soup_in_basket', 'pick_cream_cheese',
                         'place_cream_cheese_in_basket', 'pick_tomato_sauce', 'place_tomato_sauce_in_basket',
                         'pick_wine_bottle', 'place_wine_bottle_in_bottom_drawer_of_the_cabinet',
                         'close_the_bottom_drawer_of_the_cabinet']
            skills_id10 = ['turn_on_the_stove', 'pick_frying_pan', 'place_frying_pan_on_the_stove',
                          'pick_moka_pot', 'place_moka_pot_on_the_stove', 'open_the_microwave']
            skills = list(set(skills_id8 + skills_id9 + skills_id10))
        else:
            raise ValueError(f"Invalid ID: {args.ID}. Must be 1, 2, 3, 8, 9, 10, 17, or 8910")
        num_trials = args.num_trials
    else:  # normal mode
        init_dir = Path(args.init_dir)
        skills = sorted([f.stem for f in init_dir.glob('*.init')])
        num_trials = args.num_trials

    print(f"\n🎯 Evaluating {len(skills)} skills × {num_trials} trials")
    print(f"   Mode: {args.eval_mode}")
    if args.eval_mode == 'ID':
        print(f"   ID: {args.ID}")
    print(f"   Variant: {args.eval_variant}")
    if args.eval_variant != 'clean':
        print(f"   Shift ranges: XY=±{args.shift_xy_range*100:.1f}cm, Z=±{args.shift_z_range*100:.1f}cm, Ori=±{args.shift_ori_range:.1f}°")

    # Results storage
    all_results = {}

    # Evaluate each skill
    for skill_idx, skill_name in enumerate(skills):
        print(f"\n{'='*80}")
        print(f"[{skill_idx+1}/{len(skills)}] {skill_name}")
        print(f"{'='*80}")

        # Load init states
        init_path = Path(args.init_dir) / f"{skill_name}.init"
        if not init_path.exists():
            print(f"  ❌ Init file not found: {init_path}")
            continue

        with open(init_path, 'rb') as f:
            init_states = pickle.load(f)

        # Get target object and BDDL file from skill config
        target_object, bddl_file = get_target_object_and_bddl(skill_name, skill_config)
        if target_object is None or bddl_file is None:
            print(f"  ❌ Skill not found in skill_config.json: {skill_name}")
            continue

        bddl_path = Path(args.bddl_dir) / bddl_file
        if not bddl_path.exists():
            print(f"  ❌ BDDL file not found: {bddl_path}")
            continue

        # Infer skill type
        skill_type = infer_skill_type(skill_name)
        print(f"  Type: {skill_type} | BDDL: {bddl_file} | Init states: {len(init_states)}")

        # Create environment
        env_args = {
            'bddl_file_name': str(bddl_path),
            'camera_heights': 256,
            'camera_widths': 256,
            'has_renderer': False,
            'has_offscreen_renderer': True,
            'ignore_done': True,
            'use_camera_obs': True,
            'control_freq': 20,
            'camera_names': ['agentview', 'robot0_eye_in_hand']
        }



        # Track results
        results = {
            'successes': 0,
            'failures': 0,
            'success_video': None,
            'success_videos': [],
            'failure_video': None
        }

        env = OffScreenRenderEnv(**env_args)
        # Run trials
        for trial_idx in range(num_trials):            
            env.reset()  # Only reset once after creating env
            print(f"\n  Trial {trial_idx+1}/{num_trials}")

            # Get init state
            init_state = init_states[trial_idx % len(init_states)]

            # Evaluate with above pose
            success, frames = evaluate_with_above_pose(
                env, init_state, skill_name, skill_type,
                bddl_file, target_object,
                cfg, vla, processor, action_head, proprio_projector,
                resize_size, args.above_height,
                eval_variant=args.eval_variant,
                shift_xy_range=args.shift_xy_range,
                shift_z_range=args.shift_z_range,
                shift_ori_range=args.shift_ori_range,
                eval_mode=args.eval_mode,
                use_relative_pose=args.use_relative_pose,
                mask_as_agentview=args.mask_as_agentview,
                collision_aware=args.collision_aware,
                save_pointcloud=args.mplib_save_pointcloud,
                pointcloud_output_dir=str(mplib_pointcloud_dir) if mplib_pointcloud_dir else None,
                velocity_factor=args.velocity_factor,
                time_step=args.time_step,
                safety_margin=args.safety_margin,
                apply_background_erasing=args.apply_background_erasing,
                apply_distractor_masking=args.apply_distractor_masking
            )

            if success:
                results['successes'] += 1
                if skill_type == 'place':
                    if len(results['success_videos']) < 3:
                        results['success_videos'].append(frames)
                elif results['success_video'] is None:
                    results['success_video'] = frames
                print(f"    ✅ Success")
            else:
                results['failures'] += 1
                if results['failure_video'] is None:
                    results['failure_video'] = frames
                print(f"    ❌ Failed")

        # Save videos
        if skill_type == 'place':
            if len(results['success_videos']) > 0:
                for idx, frames in enumerate(results['success_videos']):
                    video_path = videos_dir / f"{skill_name}_success_{idx+1}.mp4"
                    save_video(frames, str(video_path))
        elif results['success_video'] is not None:
            video_path = videos_dir / f"{skill_name}_success.mp4"
            save_video(results['success_video'], str(video_path))

        if results['failure_video'] is not None:
            video_path = videos_dir / f"{skill_name}_failure.mp4"
            save_video(results['failure_video'], str(video_path))

        # Calculate success rate
        total = results['successes'] + results['failures']
        success_rate = results['successes'] / total if total > 0 else 0.0

        all_results[skill_name] = {
            'skill_type': skill_type,
            'successes': results['successes'],
            'failures': results['failures'],
            'success_rate': success_rate
        }

        print(f"\n  📊 Results: {results['successes']}/{num_trials} ({success_rate*100:.1f}%)")

        env.close()

    # Save results
    variant_suffix = f"_{args.eval_variant}" if args.eval_variant != 'clean' else ""
    results_path = results_dir / f"results{variant_suffix}.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"✅ Evaluation Complete!")
    print(f"{'='*80}")
    print(f"Variant: {args.eval_variant}")

    # Calculate overall stats
    total_skills = len(all_results)
    if total_skills > 0:
        avg_success_rate = sum(r['success_rate'] for r in all_results.values()) / total_skills
        print(f"Overall Success Rate: {avg_success_rate*100:.1f}%")

    print(f"\n📄 Results: {results_path}")
    print(f"🎥 Videos: {videos_dir}")


if __name__ == "__main__":
    main()
