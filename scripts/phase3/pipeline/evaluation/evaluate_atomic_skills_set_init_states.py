#!/usr/bin/env python3
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
Evaluate trained OpenVLA-OFT checkpoint on atomic skills from above-region demos.

Usage:
  # Debug mode (3 skills, 3 trials each)
  python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills.py \
    --checkpoint runs/libero_above_atomic/1.0.0/openvla-7b+libero_above_atomic+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --eval_mode debug

    python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills.py \
    --checkpoint runs/libero_above_atomic/1.0.0/openvla-7b+libero_above_atomic+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --eval_mode debug

  # ID mode

  CUDA_VISIBLE_DEVICES=1 python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_set_init_states.py \
    --checkpoint runs/libero_above_atomic_long_id8/1.0.1/openvla-7b+libero_above_atomic_long_id8+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_8--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt \
    --eval_mode ID \
    --ID 8

  CUDA_VISIBLE_DEVICES=1 python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_set_init_states.py \
    --checkpoint runs/libero_above_atomic_long_id8/1.0.3/openvla-7b+libero_above_atomic_long_id8+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_8--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --eval_mode ID \
    --ID 8 \
    --enable_shift

  CUDA_VISIBLE_DEVICES=2 python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_set_init_states.py \
    --checkpoint runs/libero_above_atomic_long_id10/1.0.1/openvla-7b+libero_above_atomic_long_id10+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_10--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt \
    --eval_mode ID \
    --ID 10

  CUDA_VISIBLE_DEVICES=2 python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_set_init_states.py \
    --checkpoint runs/libero_above_atomic_long_id10/1.0.1/openvla-7b+libero_above_atomic_long_id10+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_10--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --eval_mode ID \
    --ID 10 \
    --enable_shift

  CUDA_VISIBLE_DEVICES=2 python scripts/phase3/pipeline/evaluation/evaluate_atomic_skills_set_init_states.py \
    --checkpoint runs/libero_above_atomic_long_id10/1.0.2/openvla-7b+libero_above_atomic_long_id10+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_10--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --eval_mode ID \
    --ID 10 \
    --enable_shift
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
from scripts.phase3.pipeline.utils.local_pose_tools import get_object_pose


def load_bddl_to_init_mapping():
    """Load BDDL to init mapping."""
    mapping_path = "scripts/phase3/pipeline/config/bddl_to_init_mapping.json"
    with open(mapping_path, 'r') as f:
        return json.load(f)


def find_bddl_from_skill(skill_name: str, mapping: Dict) -> Optional[str]:
    """Find BDDL file from skill name (returns first match)."""
    for bddl_name, init_name in mapping.items():
        if init_name == skill_name:
            return f"{bddl_name}.bddl"
    return None


def load_skill_config():
    """Load skill config."""
    config_path = "scripts/phase3/pipeline/config/skill_config.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def get_target_object_from_skill_config(skill_name: str, skill_config: Dict) -> Optional[str]:
    """Get target object from skill config (returns first target object)."""
    if skill_name in skill_config:
        target_objects = skill_config[skill_name].get("target_object", [])
        if target_objects and len(target_objects) > 0:
            return target_objects[0]  # Return first target object
    return None


def get_bddl_from_skill_config(skill_name: str, skill_config: Dict) -> Optional[str]:
    """Get BDDL file from skill config (returns first BDDL file)."""
    if skill_name in skill_config:
        bddl_files = skill_config[skill_name].get("bddl_files", [])
        if bddl_files and len(bddl_files) > 0:
            return bddl_files[0]  # Return first BDDL file
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


def evaluate_direct(env, init_state, skill_name: str, skill_type: str,
                   cfg, vla, processor, action_head, proprio_projector,
                   resize_size: int, target_object: Optional[str] = None,
                   use_relative_pose: bool = True, mask_as_agentview: bool = False,
                   max_steps: int = 300) -> Tuple[bool, List]:
    """Evaluate directly from init state without shift."""

    is_place = (skill_type == 'place')

    # Handle place skills: disable gravity, set state, close gripper, enable gravity
    if is_place:
        old_gravity = env.sim.model.opt.gravity.copy()
        env.sim.model.opt.gravity[:] = 0
        env.sim.forward()

    # Set init state (full sim state)
    env.set_init_state(init_state)
    obs = env.env._get_observations()
    frames = []
    frames.append(combine_camera_views(obs))

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
    # Run VLA
    language = skill_name.replace('_', ' ')
    success, vla_frames = run_vla(env, obs, language, cfg, vla, processor,
                             action_head, proprio_projector, resize_size, max_steps,
                             target_object, use_relative_pose, mask_as_agentview, skill_name)

    return success, frames + vla_frames


def evaluate_with_shift(env, init_state, skill_name: str, skill_type: str,
                       cfg, vla, processor, action_head, proprio_projector,
                       resize_size: int, xy_range: float, z_range: float,
                       ori_range: float, target_object: Optional[str] = None,
                       use_relative_pose: bool = True, mask_as_agentview: bool = False,
                       max_steps: int = 300) -> Tuple[bool, List]:
    """Evaluate with random EE shift using mplib."""

    is_place = (skill_type == 'place')
    grasped_object = None

    # Handle place skills: disable gravity, set state, close gripper, enable gravity
    if is_place:
        old_gravity = env.sim.model.opt.gravity.copy()
        env.sim.model.opt.gravity[:] = 0
        env.sim.forward()

    # Set init state (full sim state)
    env.set_init_state(init_state)
    obs = env.env._get_observations()
    frames = []
    frames.append(combine_camera_views(obs))

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

    # Get current EE pose
    current_ee_pos = obs['robot0_eef_pos'].copy()
    current_ee_quat = obs['robot0_eef_quat'].copy()  # [x,y,z,w]

    # Retry loop: keep resampling shifted poses until motion planning succeeds
    # MP failures should NOT count as VLA failures
    max_mp_retries = 50
    obs_list = None
    for mp_attempt in range(max_mp_retries):
        # Generate random shift
        shifted_pos = current_ee_pos.copy()
        shifted_pos[0] += np.random.uniform(-xy_range, xy_range)
        shifted_pos[1] += np.random.uniform(-xy_range, xy_range)
        shifted_pos[2] += np.random.uniform(-z_range, z_range)

        # Orientation shift (in radians)
        ori_range_rad = np.radians(ori_range)
        delta_rot = R.from_rotvec(np.random.uniform(-ori_range_rad, ori_range_rad, 3))
        current_rot = R.from_quat(current_ee_quat)  # [x,y,z,w]
        shifted_rot = delta_rot * current_rot
        shifted_quat = shifted_rot.as_quat()  # [x,y,z,w]

        # Convert to [w,x,y,z] for mplib
        shifted_quat_wxyz = np.array([shifted_quat[3], shifted_quat[0], shifted_quat[1], shifted_quat[2]])

        # Use mplib to move to shifted pose
        motion_planner = MPlibMotionPlanner(env, collision_aware=True, verbose=False)

        move_success, results = motion_planner.move_to_pose(
            shifted_pos, shifted_quat_wxyz,
            position_threshold=0.02,
            orientation_threshold=0.524,
            skill_type=skill_type,
            grasped_object_name=grasped_object
        )
        obs_list = results[0]

        if move_success:
            if mp_attempt > 0:
                print(f"      ✓ Motion planning succeeded (attempt {mp_attempt + 1})")
            break
        else:
            print(f"      ↻ Motion planning failed, resampling... (attempt {mp_attempt + 1}/{max_mp_retries})")
            # Reset to initial state before next attempt
            if is_place:
                env.sim.model.opt.gravity[:] = 0
                env.sim.forward()

            env.set_init_state(init_state)
            obs = env.env._get_observations()

            if is_place:
                # Close gripper (5 steps)
                for _ in range(5):
                    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
                # Re-enable gravity
                env.sim.model.opt.gravity[:] = old_gravity
                env.sim.forward()
                obs = env.env._get_observations()
    else:
        # All retries exhausted - this is an MP failure, NOT a VLA failure
        print(f"      ⚠️  MOTION PLANNING FAILURE (not VLA failure): Could not reach shifted pose after {max_mp_retries} attempts")
        print(f"      ⚠️  This trial should NOT be counted in VLA success/failure statistics")
        return False, frames

    # Add motion planning frames
    if isinstance(obs_list, list):
        for obs_mp in obs_list:
            frames.append(combine_camera_views(obs_mp))

    # Get final observation
    final_obs = obs_list[-1] if isinstance(obs_list, list) and len(obs_list) > 0 else obs_list

    # Run VLA from shifted pose
    language = skill_name.replace('_', ' ')
    success, vla_frames = run_vla(env, final_obs, language, cfg, vla, processor,
                             action_head, proprio_projector, resize_size, max_steps,
                             target_object, use_relative_pose, mask_as_agentview, skill_name)

    return success, frames + vla_frames


def run_vla(env, obs, language: str, cfg, vla, processor, action_head,
           proprio_projector, resize_size: int, max_steps: int,
           target_object: Optional[str] = None, use_relative_pose: bool = True,
           mask_as_agentview: bool = False, skill_name: str = "") -> Tuple[bool, List]:
    """Run VLA until success or max steps."""

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    frames = []

    # Capture initial state frame before any actions
    frames.append(combine_camera_views(obs))

    for t in range(max_steps):
        # Prepare observation with AXIS-ANGLE (not euler)
        wrist_img = get_libero_wrist_image(obs)
        wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

        # Generate segmentation mask as agentview if enabled
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

        if use_relative_pose and target_object:
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

        # Add mask as second image if enabled (must contain "wrist" in key name)
        if mask_as_agentview and mask_img_resized is not None:
            observation["wrist_mask"] = mask_img_resized

        # Get action if queue is empty
        if len(action_queue) == 0:
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
                       help='ID for ID mode (1, 2, 3, 8, 9, or 10)')

    # Eval params
    parser.add_argument('--num_trials', type=int, default=10,
                       help='Number of trials per skill')
    parser.add_argument('--enable_shift', action='store_true', default=False,
                       help='Enable method 2 (with shift)')
    parser.add_argument('--xy_range', type=float, default=0.04,
                       help='XY shift range in meters')
    parser.add_argument('--z_range', type=float, default=0.02,
                       help='Z shift range in meters')
    parser.add_argument('--ori_range', type=float, default=15.0,
                       help='Orientation shift range in degrees')

    # Paths
    parser.add_argument('--init_dir', type=str,
                       default='datasets/hdf5_datasets/atomic_above_27_skills/all_downsampled')
    parser.add_argument('--bddl_dir', type=str,
                       default='externals/boss/libero/libero/bddl_files/atomic_skills')

    parser.add_argument('--use_relative_pose', type=lambda x: x.lower() == 'true', default=True,
                       help='Use relative EE position and orientation (relative to object) for VLA input (default: True)')
    parser.add_argument('--mask_as_agentview', action='store_true', default=False,
                       help='Use segmentation mask as second image input instead of agentview (requires num_images_in_input=2)')

    args = parser.parse_args()

    # Parse checkpoint info
    version, steps = parse_checkpoint_info(args.checkpoint)

    # Create descriptive output directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build folder name components
    folder_parts = [timestamp]

    # Add "set_init" identifier to distinguish from real_moving evals
    folder_parts.append("set_init")

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

    # Load VLA model
    from dataclasses import dataclass
    from typing import Union

    @dataclass
    class VLAConfig:
        model_family: str = "openvla"
        pretrained_checkpoint: Union[str, Path] = ""
        use_l1_regression: bool = True
        use_diffusion: bool = False
        use_film: bool = False
        num_images_in_input: int = 1
        use_proprio: bool = True
        center_crop: bool = True
        num_open_loop_steps: int = 8
        unnorm_key: str = ""
        load_in_8bit: bool = False
        load_in_4bit: bool = False
        is_depth: bool = False

    cfg = VLAConfig()
    cfg.pretrained_checkpoint = args.checkpoint
    cfg.num_images_in_input = 2 if args.mask_as_agentview else 1

    # Auto-detect unnorm_key from checkpoint path folder name
    # e.g., runs/libero_above_atomic/1.0.0/... -> unnorm_key = "libero_above_atomic"
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
    print(f"✅ Model loaded (unnorm_key: {cfg.unnorm_key})")

    # Load skill config
    skill_config = load_skill_config()

    # Get skills to evaluate
    if args.eval_mode == 'debug':
        skills = [
            'open_the_bottom_drawer_of_the_cabinet',
            'pick_ketchup',
            'place_ketchup_in_top_drawer_of_the_cabinet'
        ]
        num_trials = 3
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
            skills = ['pick_black_bowl'
            # , 'place_black_bowl_on_the_plate', 'place_black_bowl_on_the_black_bowl',
            #          'open_the_top_drawer_of_the_cabinet', 'place_black_bowl_in_top_drawer_of_the_cabinet',
            #          'close_the_top_drawer_of_the_cabinet', 'pick_cream_cheese', 'place_cream_cheese_in_basket'
                     ]                     
        elif args.ID == 9:
            skills = ['pick_alphabet_soup', 'place_alphabet_soup_in_basket', 'pick_cream_cheese',
                     'place_cream_cheese_in_basket', 'pick_tomato_sauce', 'place_tomato_sauce_in_basket',
                     'pick_wine_bottle', 'place_wine_bottle_in_bottom_drawer_of_the_cabinet',
                     'close_the_bottom_drawer_of_the_cabinet']
        elif args.ID == 10:
            skills = ['turn_on_the_stove', 'pick_frying_pan', 'place_frying_pan_on_the_stove',
                     'pick_moka_pot', 'place_moka_pot_on_the_stove', 'open_the_microwave']
        else:
            raise ValueError(f"Invalid ID: {args.ID}. Must be 1, 2, 3, 8, 9, or 10")
        num_trials = args.num_trials
    else:  # normal mode
        init_dir = Path(args.init_dir)
        skills = sorted([f.stem for f in init_dir.glob('*.init')])
        num_trials = args.num_trials

    print(f"\n🎯 Evaluating {len(skills)} skills × {num_trials} trials")
    print(f"   Mode: {args.eval_mode}")
    if args.eval_mode == 'ID':
        print(f"   ID: {args.ID}")

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

        # Get BDDL file and target object from skill config
        bddl_file = get_bddl_from_skill_config(skill_name, skill_config)
        if bddl_file is None:
            print(f"  ❌ Skill not found in skill_config.json: {skill_name}")
            continue

        bddl_path = Path(args.bddl_dir) / bddl_file
        if not bddl_path.exists():
            print(f"  ❌ BDDL file not found: {bddl_path}")
            continue

        # Infer skill type and get target object
        skill_type = infer_skill_type(skill_name)
        target_object = get_target_object_from_skill_config(skill_name, skill_config)
        if target_object:
            print(f"  Type: {skill_type} | Target: {target_object} | BDDL: {bddl_file} | Init states: {len(init_states)}")
        else:
            print(f"  Type: {skill_type} | BDDL: {bddl_file} | Init states: {len(init_states)}")
            print(f"  ⚠️  Warning: No target object found for {skill_name}, will use absolute pose")

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

        env = OffScreenRenderEnv(**env_args)
        env.reset()  # Only reset once after creating env

        # Track results
        results = {
            'direct': {
                'successes': 0,
                'failures': 0,
                'success_video': None,
                'success_videos': [],
                'failure_video': None
            },
            'shifted': {
                'successes': 0,
                'failures': 0,
                'success_video': None,
                'success_videos': [],
                'failure_video': None
            }
        }

        # Run trials
        for trial_idx in range(num_trials):
            print(f"\n  Trial {trial_idx+1}/{num_trials}")

            # Get init state
            init_state = init_states[trial_idx % len(init_states)]

            # Method 1: Direct evaluation
            print(f"    Method 1 (direct)...")
            success_1, frames_1 = evaluate_direct(
                env, init_state, skill_name, skill_type,
                cfg, vla, processor, action_head, proprio_projector,
                resize_size, target_object, args.use_relative_pose, args.mask_as_agentview
            )

            if success_1:
                results['direct']['successes'] += 1
                if skill_type == 'place':
                    if len(results['direct']['success_videos']) < 3:
                        results['direct']['success_videos'].append(frames_1)
                elif results['direct']['success_video'] is None:
                    results['direct']['success_video'] = frames_1
                print(f"      ✅ Success")
            else:
                results['direct']['failures'] += 1
                if results['direct']['failure_video'] is None:
                    results['direct']['failure_video'] = frames_1
                print(f"      ❌ Failed")

            # Method 2: With shift (if enabled)
            if args.enable_shift:
                print(f"    Method 2 (with shift)...")
                success_2, frames_2 = evaluate_with_shift(
                    env, init_state, skill_name, skill_type,
                    cfg, vla, processor, action_head, proprio_projector,
                    resize_size, args.xy_range, args.z_range, args.ori_range,
                    target_object, args.use_relative_pose, args.mask_as_agentview
                )

                if success_2:
                    results['shifted']['successes'] += 1
                    if skill_type == 'place':
                        if len(results['shifted']['success_videos']) < 3:
                            results['shifted']['success_videos'].append(frames_2)
                    elif results['shifted']['success_video'] is None:
                        results['shifted']['success_video'] = frames_2
                    print(f"      ✅ Success")
                else:
                    results['shifted']['failures'] += 1
                    if results['shifted']['failure_video'] is None:
                        results['shifted']['failure_video'] = frames_2
                    print(f"      ❌ Failed")

        # Save videos
        if skill_type == 'place':
            if len(results['direct']['success_videos']) > 0:
                for idx, frames in enumerate(results['direct']['success_videos']):
                    video_path = videos_dir / f"{skill_name}_direct_success_{idx+1}.mp4"
                    save_video(frames, str(video_path))
        elif results['direct']['success_video'] is not None:
            video_path = videos_dir / f"{skill_name}_direct_success.mp4"
            save_video(results['direct']['success_video'], str(video_path))

        if results['direct']['failure_video'] is not None:
            video_path = videos_dir / f"{skill_name}_direct_failure.mp4"
            save_video(results['direct']['failure_video'], str(video_path))

        if args.enable_shift:
            if skill_type == 'place':
                if len(results['shifted']['success_videos']) > 0:
                    for idx, frames in enumerate(results['shifted']['success_videos']):
                        video_path = videos_dir / f"{skill_name}_shifted_success_{idx+1}.mp4"
                        save_video(frames, str(video_path))
            elif results['shifted']['success_video'] is not None:
                video_path = videos_dir / f"{skill_name}_shifted_success.mp4"
                save_video(results['shifted']['success_video'], str(video_path))

            if results['shifted']['failure_video'] is not None:
                video_path = videos_dir / f"{skill_name}_shifted_failure.mp4"
                save_video(results['shifted']['failure_video'], str(video_path))

        # Calculate success rates
        total_direct = results['direct']['successes'] + results['direct']['failures']
        success_rate_direct = results['direct']['successes'] / total_direct if total_direct > 0 else 0.0

        if args.enable_shift:
            total_shifted = results['shifted']['successes'] + results['shifted']['failures']
            success_rate_shifted = results['shifted']['successes'] / total_shifted if total_shifted > 0 else 0.0
        else:
            success_rate_shifted = None

        all_results[skill_name] = {
            'skill_type': skill_type,
            'direct': {
                'successes': results['direct']['successes'],
                'failures': results['direct']['failures'],
                'success_rate': success_rate_direct
            },
            'shifted': {
                'successes': results['shifted']['successes'],
                'failures': results['shifted']['failures'],
                'success_rate': success_rate_shifted
            } if args.enable_shift else None
        }

        print(f"\n  📊 Results:")
        print(f"     Direct:  {results['direct']['successes']}/{num_trials} ({success_rate_direct*100:.1f}%)")
        if args.enable_shift:
            print(f"     Shifted: {results['shifted']['successes']}/{num_trials} ({success_rate_shifted*100:.1f}%)")

        env.close()

    # Save results
    results_path = results_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"✅ Evaluation Complete!")
    print(f"{'='*80}")

    # Calculate overall stats
    total_skills = len(all_results)
    if total_skills > 0:
        avg_direct = sum(r['direct']['success_rate'] for r in all_results.values()) / total_skills
        print(f"Overall Direct Success Rate: {avg_direct*100:.1f}%")

        if args.enable_shift:
            avg_shifted = sum(r['shifted']['success_rate'] for r in all_results.values() if r['shifted']) / total_skills
            print(f"Overall Shifted Success Rate: {avg_shifted*100:.1f}%")

    print(f"\n📄 Results: {results_path}")
    print(f"🎥 Videos: {videos_dir}")


if __name__ == "__main__":
    main()
