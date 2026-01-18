#!/usr/bin/env python3
"""
Test above height calculation by verifying MPlib can reach calculated poses.

Usage:
  # Debug mode (3 skills)
  python scripts/phase3/pipeline/utils/test_above_height.py --eval_mode debug

  # ID mode (6 skills for ID 2)
  python scripts/phase3/pipeline/utils/test_above_height.py --eval_mode ID --ID 2

  # All skills
  python scripts/phase3/pipeline/utils/test_above_height.py --eval_mode all
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Add project paths
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')

from libero.libero.envs import OffScreenRenderEnv
from scripts.phase3.pipeline.motion_planning.mplib.planner import MPlibMotionPlanner
from scripts.phase3.pipeline.utils.above_pose_calculator import calculate_above_pose

# Import get_special_object_handling from data generation script
sys.path.insert(0, '/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase3/pipeline/data_generation')
from generate_above_augmented_demos import get_special_object_handling


def combine_camera_views(obs: dict) -> np.ndarray:
    """Combine agentview and wrist camera into side-by-side frame."""
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


def load_bddl_to_init_mapping() -> Dict:
    """Load BDDL to init mapping."""
    mapping_path = "scripts/phase3/pipeline/config/bddl_to_init_mapping.json"
    with open(mapping_path, 'r') as f:
        return json.load(f)


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
        'frying_pan': ['chefmate_8_frypan_1_main'],
        'moka_pot': ['moka_pot_1_main', 'moka_pot_2_main'],
        'red_mug': ['red_coffee_mug_1_main'],
        'white_mug': ['porcelain_mug_1_main'],
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


def load_target_object_from_config(skill_name: str) -> Optional[str]:
    """Load target object from skill_config.json."""
    config_path = Path("scripts/phase3/pipeline/config/skill_config.json")
    with open(config_path, 'r') as f:
        skill_configs = json.load(f)

    if skill_name in skill_configs:
        target_objects = skill_configs[skill_name].get('target_object', [])
        if isinstance(target_objects, list) and len(target_objects) > 0:
            return target_objects[0]  # Use first target object
        elif isinstance(target_objects, str):
            return target_objects

    return None


def test_skill_above_pose(skill_name: str, init_state, bddl_path: Path,
                          skill_type: str, max_retries: int = 3) -> Tuple[bool, int, Optional[dict]]:
    """
    Test if above pose is reachable for a skill.

    Returns:
        (success, attempts, final_obs) where:
        - attempts is number of tries before success (0 if failed)
        - final_obs is the observation at above pose (None if failed)
    """
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

    is_place = (skill_type == 'place')
    grasped_object = None

    # Set initial state with gravity handling
    try:
        if is_place:
            old_gravity = env.sim.model.opt.gravity.copy()
            env.sim.model.opt.gravity[:] = 0
            env.sim.forward()

        env.set_init_state(init_state)
        obs = env.env._get_observations()
    except ValueError as e:
        if "could not broadcast" in str(e):
            print(f"  ⚠️  Init state dimension mismatch (incompatible scene), skipping skill")
            env.close()
            return False, -1, None  # -1 indicates skip, not failure
        else:
            raise

    if is_place:
        # Close gripper (5 steps)
        for _ in range(5):
            obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))

        # Re-enable gravity
        env.sim.model.opt.gravity[:] = old_gravity
        env.sim.forward()
        obs = env.env._get_observations()

        # Find grasped object
        candidates = get_grasped_object_candidates(skill_name)
        grasped_object = find_grasped_object_in_env(env, candidates)

    # Get target object from skill config
    target_object = load_target_object_from_config(skill_name)
    if target_object is None:
        print(f"  ⚠️  Could not find target object in skill_config.json: {skill_name}")
        env.close()
        return False, 0, None

    # Get special object handling
    obj_pos, _, above_height, bbox_z = get_special_object_handling(
        env, target_object, str(bddl_path),
        default_above_height=0.03,
        skill_type=skill_type,
        grasped_object_name=grasped_object,
        save_debug_files=True  # Enable debug files to see segmentation
    )

    # Calculate above pose (NO shift)
    above_pos, above_quat, _ = calculate_above_pose(
        obj_pos,
        above_height=above_height,
        shift=False,
        bbox_z=bbox_z
    )

    # Print detailed info
    print(f"  📊 Target object: {target_object}")
    if grasped_object:
        print(f"  📊 Grasped object: {grasped_object}")
    print(f"  📊 Object pos: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
    print(f"  📊 Bbox Z (half-extent): {bbox_z:.3f}m")
    print(f"  📊 Above height: {above_height:.3f}m")
    print(f"  📊 Above pos: [{above_pos[0]:.3f}, {above_pos[1]:.3f}, {above_pos[2]:.3f}]")
    print(f"  📊 Vertical offset: {above_pos[2] - obj_pos[2]:.3f}m")

    # Try MPlib with retries
    motion_planner = MPlibMotionPlanner(env, collision_aware=True, verbose=False)
    final_obs = None

    for attempt in range(1, max_retries + 1):
        move_success, results = motion_planner.move_to_pose(
            above_pos, above_quat,
            position_threshold=0.02,
            orientation_threshold=0.348,
            skill_type=skill_type,
            grasped_object_name=grasped_object
        )

        if move_success:
            # Get final observation for image saving
            obs_list = results[0]
            final_obs = obs_list[-1] if isinstance(obs_list, list) and len(obs_list) > 0 else obs_list

            if attempt > 1:
                print(f"  ✓ Success on attempt {attempt}")
            env.close()
            return True, attempt, final_obs
        else:
            if attempt < max_retries:
                print(f"  ↻ Attempt {attempt} failed, retrying...")

    print(f"  ✗ Failed all {max_retries} attempts")
    env.close()
    return False, 0, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_mode', type=str, default='debug',
                       choices=['debug', 'ID', 'all'],
                       help='Evaluation mode')
    parser.add_argument('--ID', type=int, default=2,
                       help='Task suite ID (for ID mode)')
    parser.add_argument('--init_dir', type=str,
                       default='datasets/hdf5_datasets/atomic_above_fewer/all')
    parser.add_argument('--bddl_dir', type=str,
                       default='externals/boss/libero/libero/bddl_files/atomic_skills')
    args = parser.parse_args()

    # Load mapping
    bddl_mapping = load_bddl_to_init_mapping()

    # Get all init files
    init_dir = Path(args.init_dir)
    init_files = sorted(list(init_dir.glob("*.init")))

    # Filter based on eval mode
    if args.eval_mode == 'debug':
        # Debug mode: test moka pot on stove (to visualize stove pointcloud)
        
        debug_skill = "open_the_bottom_drawer_of_the_cabinet.init"
        debug_skill = "open_the_top_drawer_of_the_cabinet.init"
        debug_skill = "place_moka_pot_on_the_stove.init"
        debug_skill = "close_the_bottom_drawer_of_the_cabinet.init"
        debug_skill = "place_black_bowl_in_bottom_drawer_of_the_cabinet.init"
        debug_skill = "place_wine_bottle_in_bottom_drawer_of_the_cabinet.init"
        debug_skill = "place_white_mug_on_the_plate.init"

        # debug_skill = "place_red_mug_on_the_plate.init"
        debug_skill = "place_ketchup_in_top_drawer_of_the_cabinet.init"
        # debug_skill = "place_frying_pan_on_the_stove.init"

        debug_skill = "turn_on_the_stove.init"
        # debug_skill = "turn_off_the_stove.init"

        debug_file = init_dir / debug_skill
        if debug_file.exists():
            init_files = [debug_file]
        else:
            print(f"⚠️  Debug skill not found: {debug_skill}, using first 3 skills")
            init_files = init_files[:3]
    elif args.eval_mode == 'ID':
        # For ID mode, just test first 6 skills (simplified for testing)
        # TODO: Load ID-specific skills from proper config
        init_files = init_files[:6]

    # Create output directory
    output_dir = Path("scripts/phase3/pipeline/outputs/images/test_above_height")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🧪 Testing Above Height Calculation")
    print(f"Mode: {args.eval_mode}")
    if args.eval_mode == 'ID':
        print(f"ID: {args.ID}")
    print(f"Skills: {len(init_files)}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # Track results
    results = {
        'first_attempt': 0,
        'after_retry': 0,
        'failed': 0,
        'skipped': 0,
        'failed_skills': [],
        'skipped_skills': []
    }

    # Test each skill
    for idx, init_file in enumerate(init_files, 1):
        skill_name = init_file.stem
        print(f"[{idx}/{len(init_files)}] {skill_name}")

        # Load init state
        with open(init_file, 'rb') as f:
            init_states = pickle.load(f)

        if len(init_states) == 0:
            print(f"  ⚠️  No init states found")
            results['failed'] += 1
            results['failed_skills'].append(f"{skill_name} (no init states)")
            continue

        init_state = init_states[0]

        # Find BDDL file
        bddl_file = find_bddl_from_skill(skill_name, bddl_mapping)
        if bddl_file is None:
            print(f"  ⚠️  BDDL mapping not found")
            results['failed'] += 1
            results['failed_skills'].append(f"{skill_name} (no BDDL)")
            continue

        bddl_path = Path(args.bddl_dir) / bddl_file
        if not bddl_path.exists():
            print(f"  ⚠️  BDDL file not found: {bddl_path}")
            results['failed'] += 1
            results['failed_skills'].append(f"{skill_name} (BDDL not found)")
            continue

        skill_type = infer_skill_type(skill_name)
        print(f"  Type: {skill_type}")

        # Test the skill
        success, attempts, final_obs = test_skill_above_pose(
            skill_name, init_state, bddl_path, skill_type, max_retries=3
        )

        if attempts == -1:
            # Skipped due to incompatible scene
            results['skipped'] += 1
            results['skipped_skills'].append(skill_name)
        elif success:
            if attempts == 1:
                results['first_attempt'] += 1
                print(f"  ✅ Success (1st attempt)")
            else:
                results['after_retry'] += 1
                print(f"  ✅ Success (after {attempts} attempts)")

            # Save side-by-side image for successful attempts
            if final_obs is not None:
                combined_img = combine_camera_views(final_obs)
                img_path = output_dir / f"{skill_name}_above_pose.png"
                Image.fromarray(combined_img).save(img_path)
                print(f"  💾 Saved: {img_path.name}")
        else:
            results['failed'] += 1
            results['failed_skills'].append(skill_name)
            print(f"  ❌ Failed")

        print()

    # Print summary
    total = len(init_files)
    success_total = results['first_attempt'] + results['after_retry']
    print(f"{'='*80}")
    print(f"ABOVE HEIGHT TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total skills tested: {total}")
    print(f"Success (1st attempt): {results['first_attempt']} ({results['first_attempt']/total*100:.1f}%)")
    print(f"Success (after retry): {results['after_retry']} ({results['after_retry']/total*100:.1f}%)")
    print(f"Failed (all 3 attempts): {results['failed']} ({results['failed']/total*100:.1f}%)")
    print(f"Skipped (incompatible): {results['skipped']} ({results['skipped']/total*100:.1f}%)")
    print(f"\nImages saved: {success_total}")
    print(f"Output directory: {output_dir}")

    if results['failed_skills']:
        print(f"\nFailed skills:")
        for skill in results['failed_skills']:
            print(f"  - {skill}")

    if results['skipped_skills']:
        print(f"\nSkipped skills:")
        for skill in results['skipped_skills']:
            print(f"  - {skill}")

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
