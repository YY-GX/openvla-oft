#!/usr/bin/env python3
"""
Generate recovery demonstration data for atomic skills.

Only saves demos where:
- First attempt fails
- Subsequent retry (within max 3 attempts) succeeds

Each saved demo contains: failure trajectory + recovery trajectory concatenated.

For place skills: If VLA fails, recovers by re-executing pick + place.

The script runs until collecting target_demos (default 20) recovery demos per skill,
or reaching max_trials (default 1000) attempts. Timing is tracked and saved to JSON.

Usage:
  # Debug mode (stops after 1 successful recovery per skill)
  CUDA_VISIBLE_DEVICES=2 python scripts/phase3/pipeline/data_generation/generate_recovery_demos.py \
    --checkpoint runs/libero_above_atomic_long_id10/1.0.1/openvla-7b+libero_above_atomic_long_id10+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_10--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --eval_mode debug

  # ID mode (specific skill sets)
  CUDA_VISIBLE_DEVICES=2 python scripts/phase3/pipeline/data_generation/generate_recovery_demos.py \
    --checkpoint runs/libero_above_atomic_long_id10/1.0.1/openvla-7b+libero_above_atomic_long_id10+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_above_id_10--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt \
    --eval_mode ID \
    --ID 10

  # Normal mode (collect 20 demos per skill, max 1000 trials)
  python scripts/phase3/pipeline/data_generation/generate_recovery_demos.py \
    --checkpoint runs/path/to/checkpoint \
    --eval_mode normal \
    --target_demos 20 \
    --max_trials 1000
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

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

# Import above pose calculation
from scripts.phase3.pipeline.utils.above_pose_calculator import calculate_above_pose
from scripts.phase3.pipeline.data_generation.generate_above_augmented_demos import (
    get_special_object_handling
)


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
            target_objects = skill_info.get('target_object', [])
            target_object = target_objects[0] if target_objects else None

            bddl_files = skill_info.get('bddl_files', [])
            bddl_file = bddl_files[0] if bddl_files else None

            return target_object, bddl_file

    return None, None


def find_bddl_from_skill(skill_name: str, mapping: Dict) -> Optional[str]:
    """Find BDDL file from skill name."""
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


def collect_step_data(obs: Dict, action: np.ndarray, reward: float, done: bool,
                     env, state: np.ndarray = None) -> Dict:
    """
    Collect data for a single step (matches generate_above_augmented_demos.py structure).

    Returns:
        Dict with obs (agentview_rgb, eye_in_hand_rgb), action, reward, done, state (84-dim sim state)
    """
    step_data = {
        'obs': {
            'agentview_rgb': obs['agentview_image'].copy(),
            'eye_in_hand_rgb': obs['robot0_eye_in_hand_image'].copy()
        },
        'action': action.copy() if action is not None else np.zeros(7),
        'reward': reward,
        'done': done,
        'state': state if state is not None else env.sim.get_state().flatten()  # 84-dim full sim state
    }

    return step_data


def get_corresponding_pick_skill(place_skill_name: str) -> Optional[str]:
    """
    Get corresponding pick skill name from place skill name.
    E.g., 'place_black_bowl_on_the_plate' -> 'pick_black_bowl'
    """
    place_lower = place_skill_name.lower()

    # Extract object name from place skill
    object_keywords = ['black_bowl', 'white_bowl', 'ketchup', 'wine_bottle',
                      'frying_pan', 'moka_pot', 'red_mug', 'white_mug',
                      'chocolate_pudding']

    for obj in object_keywords:
        if obj in place_lower:
            return f'pick_{obj}'

    return None


def execute_skill_once(env, init_state, skill_name: str, skill_type: str,
                      bddl_file: str, target_object: str,
                      cfg, vla, processor, action_head, proprio_projector,
                      resize_size: int, above_height: float = 0.03,
                      max_vla_steps: int = 200,
                      grasped_object: str = None,
                      is_first_attempt: bool = True,
                      pos_shift_range: float = 0.0,
                      ori_shift_range: float = 0.0) -> Tuple[bool, List[Dict]]:
    """
    Execute skill once: stabilize → MP to above → VLA from above.

    NOTE: For place skills, gravity handling (disable/enable) should be done
    OUTSIDE before calling this function (only once, not per retry).

    Args:
        grasped_object: For place skills, the object being held (for collision avoidance)
        is_first_attempt: If True, stabilize before execution and apply random shifts. If False (retry), skip both.
        pos_shift_range: Max position shift (meters) to randomly sample from [-range, +range] for first attempt
        ori_shift_range: Max orientation shift (degrees) to randomly apply for first attempt

    Returns:
        (success, step_data_list) where step_data_list contains all steps
    """
    step_data_list = []

    # Stabilize environment (5 dummy actions) - only on first attempt
    if is_first_attempt:
        for _ in range(5):
            action = np.array([0, 0, 0, 0, 0, 0, -1])
            obs, _, _, _ = env.step(action)
            step_data_list.append(collect_step_data(obs, action, 0, False, env))

    # Calculate above pose
    try:
        obj_pos, _, calculated_above_height, bbox_z = get_special_object_handling(
            env, target_object, bddl_file, above_height,
            skill_type=skill_type, grasped_object_name=grasped_object
        )

        above_pos, above_quat, _ = calculate_above_pose(
            obj_pos,
            above_height=calculated_above_height,
            shift=False,
            bbox_z=bbox_z
        )

        # Apply random shifts for first attempt to increase failure probability
        if is_first_attempt and (pos_shift_range > 0 or ori_shift_range > 0):
            # Random position shift: sample from [-range, +range] for x, y, z
            if pos_shift_range > 0:
                pos_shift = np.array([
                    random.uniform(-pos_shift_range, pos_shift_range),
                    random.uniform(-pos_shift_range, pos_shift_range),
                    random.uniform(-pos_shift_range, pos_shift_range)
                ])
                above_pos = above_pos + pos_shift
                print(f"      → Applied position shift: [{pos_shift[0]:.4f}, {pos_shift[1]:.4f}, {pos_shift[2]:.4f}]m")

            # Random orientation shift: apply random rotation up to ori_shift_range degrees
            if ori_shift_range > 0:
                # Generate random axis
                random_axis = np.random.randn(3)
                random_axis /= np.linalg.norm(random_axis)

                # Random angle up to ori_shift_range (in radians)
                random_angle = random.uniform(-np.deg2rad(ori_shift_range), np.deg2rad(ori_shift_range))

                # Apply rotation
                random_rotation = R.from_rotvec(random_axis * random_angle)
                current_rotation = R.from_quat(above_quat)
                new_rotation = random_rotation * current_rotation
                above_quat = new_rotation.as_quat()
                print(f"      → Applied orientation shift: {np.rad2deg(random_angle):.2f}°")

    except Exception as e:
        print(f"      ✗ Above pose calculation error: {e}")
        return False, step_data_list

    # Move to above pose using mplib
    motion_planner = MPlibMotionPlanner(env, collision_aware=True, verbose=False)

    move_success, results = motion_planner.move_to_pose(
        above_pos, above_quat,
        position_threshold=0.02,
        orientation_threshold=0.524,
        skill_type=skill_type,
        grasped_object_name=grasped_object
    )
    obs_list = results[0]

    if not move_success:
        # Motion planning failed - still save the trajectory
        if isinstance(obs_list, list):
            for obs_mp in obs_list:
                step_data_list.append(collect_step_data(obs_mp, None, 0, False, env))
        return False, step_data_list

    # Add motion planning frames
    if isinstance(obs_list, list):
        for obs_mp in obs_list:
            step_data_list.append(collect_step_data(obs_mp, None, 0, False, env))

    # Get final observation after motion planning
    final_obs = obs_list[-1] if isinstance(obs_list, list) and len(obs_list) > 0 else obs_list

    # Run VLA from above pose
    language = skill_name.replace('_', ' ')
    success = run_vla(env, final_obs, language, cfg, vla, processor,
                     action_head, proprio_projector, resize_size,
                     max_vla_steps, step_data_list)

    # Post-actions if succeeded
    if success:
        # Close gripper for pick, open for place
        if skill_type == 'pick':
            gripper_action = 1.0  # Close
        elif skill_type == 'place':
            gripper_action = -1.0  # Open
        else:
            gripper_action = 0.0  # No change

        for _ in range(5):
            action = np.array([0, 0, 0, 0, 0, 0, gripper_action])
            obs, _, _, _ = env.step(action)
            step_data_list.append(collect_step_data(obs, action, 0, False, env))

    return success, step_data_list


def run_vla(env, obs, language: str, cfg, vla, processor, action_head,
           proprio_projector, resize_size: int, max_steps: int,
           step_data_list: List[Dict]) -> bool:
    """Run VLA until success or max steps, appending to step_data_list."""

    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    for t in range(max_steps):
        # Prepare observation
        wrist_img = get_libero_wrist_image(obs)
        wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

        observation = {
            "full_image": wrist_img_resized,
            "state": np.concatenate((
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"]
            ))
        }

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
        step_data_list.append(collect_step_data(obs, action, reward, done, env))

        if done:
            return True

    return False


def save_demo_to_hdf5(h5file: h5py.File, demo_idx: int, step_data_list: List[Dict],
                      skill_name: str, trial_idx: int, num_failures: int,
                      failure_end: int, recovery_end: int):
    """Save concatenated failure + recovery trajectory to HDF5."""

    demo_key = f'demo_{demo_idx}'
    demo_group = h5file['data'].create_group(demo_key)

    # Stack all observations
    obs_group = demo_group.create_group('obs')
    obs_group.create_dataset('agentview_rgb',
        data=np.stack([s['obs']['agentview_rgb'] for s in step_data_list]),
        compression='gzip')
    obs_group.create_dataset('eye_in_hand_rgb',
        data=np.stack([s['obs']['eye_in_hand_rgb'] for s in step_data_list]),
        compression='gzip')

    # Stack actions, rewards, dones, states
    demo_group.create_dataset('actions',
        data=np.stack([s['action'] for s in step_data_list]))
    demo_group.create_dataset('rewards',
        data=np.array([s['reward'] for s in step_data_list]))
    demo_group.create_dataset('dones',
        data=np.array([s['done'] for s in step_data_list]))
    demo_group.create_dataset('states',
        data=np.stack([s['state'] for s in step_data_list]))

    # Save metadata
    meta_group = demo_group.create_group('metadata')
    meta_group.attrs['skill_name'] = skill_name
    meta_group.attrs['trial_idx'] = trial_idx
    meta_group.attrs['num_failures'] = num_failures
    meta_group.attrs['total_attempts'] = num_failures + 1
    meta_group.attrs['failure_end_step'] = failure_end
    meta_group.attrs['recovery_end_step'] = recovery_end
    meta_group.attrs['total_steps'] = len(step_data_list)


def save_metadata_json(output_dir: Path, skill_name: str, metadata: Dict):
    """Save metadata JSON for a skill."""
    json_path = output_dir / f"{skill_name}_metadata.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--checkpoint', type=str, required=True)

    # Eval mode
    parser.add_argument('--eval_mode', type=str, default='normal',
                       choices=['normal', 'ID', 'debug'])
    parser.add_argument('--ID', type=int, default=None,
                       help='ID for ID mode (1, 2, 3, 8, 9, or 10)')

    # Data generation params
    parser.add_argument('--target_demos', type=int, default=30,
                       help='Target number of recovery demos to collect per skill')
    parser.add_argument('--max_trials', type=int, default=1000,
                       help='Maximum number of trials per skill before stopping')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='Maximum number of retry attempts per trial (e.g., 3 = first attempt + 2 retries)')
    parser.add_argument('--above_height', type=float, default=0.03,
                       help='Default height above object (meters)')
    parser.add_argument('--first_attempt_pos_shift', type=float, default=0.04,
                       help='Random position shift (meters) for first attempt to increase failure probability')
    parser.add_argument('--first_attempt_ori_shift', type=float, default=20.0,
                       help='Random orientation shift (degrees) for first attempt to increase failure probability')

    # Paths
    parser.add_argument('--init_dir', type=str,
                       default='datasets/hdf5_datasets/atomic_above_fewer/all')
    parser.add_argument('--bddl_dir', type=str,
                       default='externals/boss/libero/libero/bddl_files/atomic_skills')
    parser.add_argument('--output_dir', type=str,
                       default='datasets/hdf5_datasets/atomic_recovery_demos')
    parser.add_argument('--skills', nargs='+', default=None,
                       help='Specific skills to generate (default: all or based on mode)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*80}")
    print(f"Recovery Demo Generation")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Target demos per skill: {args.target_demos}")
    print(f"Max trials per skill: {args.max_trials}")
    print(f"Max retries per trial: {args.max_retries}")

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

    # Auto-detect unnorm_key
    checkpoint_parts = Path(args.checkpoint).parts
    if "runs" in checkpoint_parts:
        runs_idx = checkpoint_parts.index("runs")
        if runs_idx + 1 < len(checkpoint_parts):
            cfg.unnorm_key = checkpoint_parts[runs_idx + 1]
        else:
            cfg.unnorm_key = "bridge_orig"
    else:
        cfg.unnorm_key = "bridge_orig"

    print(f"\n🤖 Loading VLA model...")
    vla = get_model(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    resize_size = get_image_resize_size(cfg)
    print(f"✅ Model loaded (unnorm_key: {cfg.unnorm_key})")

    # Load configs
    bddl_mapping = load_bddl_to_init_mapping()
    skill_config = load_skill_config()

    # Get skills to process based on mode
    if args.skills:
        skills = args.skills
    elif args.eval_mode == 'debug':
        skills = ['pick_moka_pot']
        print(f"Debug mode: Using {skills[0]} and will stop after 1 recovery")
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
            skills = ['pick_black_bowl', 'place_black_bowl_on_the_plate', 'place_black_bowl_on_the_black_bowl',
                     'open_the_top_drawer_of_the_cabinet', 'place_black_bowl_in_top_drawer_of_the_cabinet',
                     'close_the_top_drawer_of_the_cabinet', 'pick_cream_cheese', 'place_cream_cheese_in_basket']
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
    else:  # normal mode
        init_dir = Path(args.init_dir)
        skills = sorted([f.stem for f in init_dir.glob('*.init')])

    print(f"\n🎯 Processing {len(skills)} skills (target: {args.target_demos} demos each, max: {args.max_trials} trials)")

    # Process each skill
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

        # Find BDDL file
        bddl_file = find_bddl_from_skill(skill_name, bddl_mapping)
        if bddl_file is None:
            print(f"  ❌ BDDL mapping not found")
            continue

        bddl_path = Path(args.bddl_dir) / bddl_file
        if not bddl_path.exists():
            print(f"  ❌ BDDL file not found: {bddl_path}")
            continue

        # Infer skill type
        skill_type = infer_skill_type(skill_name)

        # Get target object
        target_object, bddl_file_from_config = get_target_object_and_bddl(skill_name, skill_config)
        if target_object is None or bddl_file_from_config is None:
            print(f"  ❌ Could not find target_object in skill_config")
            continue

        print(f"  Type: {skill_type} | Target: {target_object} | Init states: {len(init_states)}")

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
        env.reset()

        # Create HDF5 file for this skill
        hdf5_path = output_dir / f"{skill_name}.hdf5"
        h5file = h5py.File(hdf5_path, 'w')
        h5file.create_group('data')

        # Metadata tracking
        metadata = {
            'skill_name': skill_name,
            'target_demos': args.target_demos,
            'max_trials': args.max_trials,
            'saved_demos': 0,
            'direct_successes': 0,
            'permanent_failures': 0,
            'total_trials_run': 0,
            'time_seconds': 0.0,
            'demos': []
        }

        demo_count = 0
        is_place = (skill_type == 'place')

        # Get corresponding pick skill for place recovery
        pick_skill_name = None
        pick_target_object = None
        pick_bddl_file = None
        if is_place:
            pick_skill_name = get_corresponding_pick_skill(skill_name)
            if pick_skill_name:
                pick_target_object, pick_bddl_file = get_target_object_and_bddl(pick_skill_name, skill_config)

        # Start timing
        start_time = time.time()

        # Adaptive shift difficulty: increase when direct successes are too frequent
        current_pos_shift = args.first_attempt_pos_shift
        current_ori_shift = args.first_attempt_ori_shift
        last_direct_success_threshold = 0

        # Run trials until target_demos reached or max_trials exceeded
        trial_idx = 0
        pbar = tqdm(total=args.target_demos, desc=f"  {skill_name}")
        while demo_count < args.target_demos and trial_idx < args.max_trials:
            # Get init state (cycle through available states)
            init_state = init_states[trial_idx % len(init_states)]

            # Place skill: Handle gravity ONCE at beginning of trial
            grasped_object = None
            if is_place:
                old_gravity = env.sim.model.opt.gravity.copy()
                env.sim.model.opt.gravity[:] = 0
                env.sim.forward()

                # Set init state and close gripper
                env.set_init_state(init_state)
                obs = env.env._get_observations()

                for _ in range(5):
                    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))

                # Re-enable gravity
                env.sim.model.opt.gravity[:] = old_gravity
                env.sim.forward()

                # Find grasped object
                candidates = get_grasped_object_candidates(skill_name)
                grasped_object = find_grasped_object_in_env(env, candidates)
            else:
                # Non-place: just set init state
                env.set_init_state(init_state)

            # Retry loop - continue from current state for each retry
            attempt = 0
            first_failed = False
            all_attempts = []  # Track all attempts with their trajectories

            while attempt < args.max_retries:
                # Execute skill once
                success, step_data_list = execute_skill_once(
                    env, init_state, skill_name, skill_type,
                    bddl_file_from_config, target_object,
                    cfg, vla, processor, action_head, proprio_projector,
                    resize_size, args.above_height,
                    grasped_object=grasped_object,
                    is_first_attempt=(attempt == 0),
                    pos_shift_range=current_pos_shift,
                    ori_shift_range=current_ori_shift
                )

                # Track this attempt
                all_attempts.append({
                    'attempt_num': attempt,
                    'success': success,
                    'trajectory': step_data_list,
                    'num_steps': len(step_data_list)
                })

                if attempt == 0:
                    if not success:
                        # First attempt failed - continue to retry
                        first_failed = True
                    else:
                        # Direct success - don't save
                        print(f"    ✅ DIRECT SUCCESS (trial {trial_idx}): Succeeded on first attempt")
                        metadata['direct_successes'] += 1

                        # Adaptive difficulty: increase shift every 10 direct successes
                        current_threshold = (metadata['direct_successes'] // 10) * 10
                        if current_threshold > last_direct_success_threshold and metadata['direct_successes'] % 10 == 0:
                            current_pos_shift *= 1.5
                            current_ori_shift *= 1.5
                            last_direct_success_threshold = current_threshold
                            print(f"    🔼 DIFFICULTY INCREASED: pos_shift={current_pos_shift:.4f}m, ori_shift={current_ori_shift:.1f}° (after {metadata['direct_successes']} direct successes)")

                        break
                else:
                    # Retry attempt
                    if success:
                        # Recovery success! Save all attempts
                        print(f"    ✅ RETRY SUCCESS (trial {trial_idx}): Failed {attempt} time(s), then succeeded on attempt {attempt+1}")

                        # Concatenate all trajectories
                        combined_trajectory = []
                        for att in all_attempts:
                            combined_trajectory.extend(att['trajectory'])

                        # Calculate frame ranges for each attempt
                        attempt_details = []
                        current_frame = 0
                        for att in all_attempts:
                            start_frame = current_frame
                            end_frame = current_frame + att['num_steps']
                            attempt_details.append({
                                'attempt_num': att['attempt_num'],
                                'success': att['success'],
                                'start_frame': start_frame,
                                'end_frame': end_frame,
                                'num_steps': att['num_steps']
                            })
                            current_frame = end_frame

                        save_demo_to_hdf5(
                            h5file, demo_count, combined_trajectory,
                            skill_name, trial_idx, attempt,
                            attempt_details[-2]['end_frame'],  # End of last failure
                            attempt_details[-1]['end_frame']   # End of recovery
                        )

                        metadata['demos'].append({
                            'demo_id': f'demo_{demo_count}',
                            'trial_idx': trial_idx,
                            'num_retries': attempt,  # More intuitive: number of retries (not including first attempt)
                            'total_attempts': attempt + 1,
                            'attempts': attempt_details,  # Detailed per-attempt info
                            'total_steps': len(combined_trajectory)
                        })

                        demo_count += 1
                        metadata['saved_demos'] += 1
                        pbar.update(1)  # Update progress bar

                        # Debug mode: stop after first recovery
                        if args.eval_mode == 'debug':
                            print(f"\n  ✅ Debug mode: Found 1 recovery, stopping")
                            break

                        break

                attempt += 1

            # All attempts failed
            if first_failed and attempt >= args.max_retries:
                print(f"    ❌ PERMANENT FAILURE (trial {trial_idx}): Failed on all {args.max_retries} attempts")
                metadata['permanent_failures'] += 1

            # Increment trial counter
            trial_idx += 1

            # Debug mode: exit after first recovery found
            if args.eval_mode == 'debug' and demo_count > 0:
                break

        # Close progress bar
        pbar.close()

        # Record timing
        end_time = time.time()
        metadata['time_seconds'] = end_time - start_time
        metadata['total_trials_run'] = trial_idx

        # Record final adaptive shift values
        metadata['initial_pos_shift'] = args.first_attempt_pos_shift
        metadata['initial_ori_shift'] = args.first_attempt_ori_shift
        metadata['final_pos_shift'] = current_pos_shift
        metadata['final_ori_shift'] = current_ori_shift
        metadata['difficulty_increases'] = (metadata['direct_successes'] // 10)

        # Close HDF5 file
        h5file.close()

        # Calculate success ratios
        total_trials_run = metadata['direct_successes'] + metadata['saved_demos'] + metadata['permanent_failures']
        if total_trials_run > 0:
            metadata['direct_success_ratio'] = metadata['direct_successes'] / total_trials_run
            metadata['retry_success_ratio'] = metadata['saved_demos'] / total_trials_run
            metadata['failure_ratio'] = metadata['permanent_failures'] / total_trials_run
        else:
            metadata['direct_success_ratio'] = 0.0
            metadata['retry_success_ratio'] = 0.0
            metadata['failure_ratio'] = 0.0

        # Save metadata JSON
        save_metadata_json(output_dir, skill_name, metadata)

        # Print summary
        print(f"\n  📊 Summary:")
        print(f"     Saved demos (retry successes): {metadata['saved_demos']} / {args.target_demos} target ({metadata['retry_success_ratio']*100:.1f}%)")
        print(f"     Direct successes: {metadata['direct_successes']} ({metadata['direct_success_ratio']*100:.1f}%) (not saved)")
        print(f"     Permanent failures: {metadata['permanent_failures']} ({metadata['failure_ratio']*100:.1f}%) (not saved)")
        print(f"     Total trials run: {metadata['total_trials_run']} / {args.max_trials} max")
        print(f"     Time elapsed: {metadata['time_seconds']:.1f}s ({metadata['time_seconds']/60:.1f}min)")
        if metadata['saved_demos'] > 0:
            print(f"     Avg time per demo: {metadata['time_seconds']/metadata['saved_demos']:.1f}s")
        print(f"     Adaptive difficulty: {metadata['difficulty_increases']} increases (pos: {metadata['initial_pos_shift']:.4f}→{metadata['final_pos_shift']:.4f}m, ori: {metadata['initial_ori_shift']:.1f}→{metadata['final_ori_shift']:.1f}°)")
        print(f"  💾 Saved: {hdf5_path}")
        print(f"  💾 Saved: {output_dir / f'{skill_name}_metadata.json'}")

        env.close()

    print(f"\n{'='*80}")
    print(f"✅ Recovery Demo Generation Complete!")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
