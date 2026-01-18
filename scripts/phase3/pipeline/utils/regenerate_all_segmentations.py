#!/usr/bin/env python3
"""
Regenerate eye_in_hand_segmentation for all skills in the dataset.

This script reads HDF5 files for all skill types (pick, place, open, close, turn_on),
sets the state for each step, and regenerates correct segmentation masks:
- Pick/Open/Close/TurnOn skills: gripper + target object
- Place skills: gripper + target object + grasped object

IMPORTANT: Gripper mask includes ALL gripper geoms (visual + collision geoms).
- Visual geoms: Visible parts of gripper (hand, fingers)
- Collision geoms: Invisible physics geoms (includes line/capsule between finger pads)
- The collision geoms are INTENTIONALLY included to prevent random erasing of gripper structure
- This ensures the gripper line is preserved even when it occludes target objects

The script uses functions from segmentation_utils.py which:
- Apply get_related_bodies() to handle hierarchical objects (stoves, cabinets, etc.)
- Include all geoms matching keywords 'gripper', 'finger', 'hand' (both visual and collision)
"""

import sys
import os
import h5py
import numpy as np
import json
from typing import Dict, List, Optional
from tqdm import tqdm

# Add project paths
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')

from libero.libero.envs import OffScreenRenderEnv
from scripts.phase3.pipeline.utils.segmentation_utils import (
    create_wrist_segmentation_mask,
    create_wrist_segmentation_mask_with_grasped
)

# Import get_object_pose
import importlib.util
spec = importlib.util.spec_from_file_location("contact_detector",
    "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase2/3_phase2_contact_object_detector.py")
contact_detector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contact_detector)
get_object_pose = contact_detector.get_object_pose

# Paths
SKILL_CONFIG_PATH = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase3/pipeline/config/skill_config.json"
INPUT_DIR = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_above_fewer/all_downsampled"
OUTPUT_DIR = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_above_fewer/all_downsampled_fixed_seg"
BDDL_BASE_PATH = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss/libero/libero/bddl_files/atomic_skills"


def load_skill_config() -> Dict:
    """Load skill config from JSON."""
    with open(SKILL_CONFIG_PATH, 'r') as f:
        return json.load(f)


def find_object_in_env(env, object_list: List[str]) -> Optional[str]:
    """Find which object from the list actually exists in the environment."""
    if not object_list:
        return None

    for obj_name in object_list:
        try:
            pos, quat = get_object_pose(env, obj_name)
            if pos is not None:
                return obj_name
        except:
            continue

    return None


def get_skill_info_from_name(skill_name: str, skill_config: Dict) -> Optional[Dict]:
    """Get skill info from skill name."""
    if skill_name not in skill_config:
        return None

    skill_info = skill_config[skill_name]
    return {
        'target_object': skill_info.get('target_object', []),
        'grasped_object_name': skill_info.get('grasped_object_name', None),
        'bddl_files': skill_info.get('bddl_files', []),
        'skill_type': skill_info.get('skill_type', 'other')
    }


def process_single_demo(
    demo_group,
    env,
    skill_type: str,
    target_object_list: List[str],
    grasped_object_list: Optional[List[str]]
) -> Dict[str, int]:
    """
    Process a single demo and update segmentation masks.

    Returns:
        Dict with statistics: {'steps_processed', 'steps_failed', 'steps_skipped'}
    """
    stats = {'steps_processed': 0, 'steps_failed': 0, 'steps_skipped': 0}

    # Load states
    if 'states' not in demo_group:
        return stats

    states = demo_group['states'][:]
    obs_group = demo_group['obs']

    if 'eye_in_hand_segmentation' not in obs_group:
        return stats

    num_steps = len(states)
    if num_steps == 0:
        return stats

    seg_dataset = obs_group['eye_in_hand_segmentation']

    # Process each step
    for step_idx in range(num_steps):
        try:
            # Set state
            state = states[step_idx]
            env.sim.set_state_from_flattened(state)
            env.sim.forward()

            # Find target object at this state (may vary across demos)
            actual_target = find_object_in_env(env, target_object_list)
            if actual_target is None:
                stats['steps_skipped'] += 1
                continue

            # Generate segmentation based on skill type
            if skill_type == 'place':
                # Place skills: include grasped object
                grasped_object_name = find_object_in_env(env, grasped_object_list)
                seg_mask = create_wrist_segmentation_mask_with_grasped(
                    env, actual_target, grasped_object_name, resolution=256
                )
            else:
                # Pick/Open/Close/TurnOn skills: only gripper + target
                seg_mask = create_wrist_segmentation_mask(
                    env, actual_target, resolution=256
                )

            # Update segmentation
            if seg_dataset.shape[0] > step_idx:
                seg_dataset[step_idx] = seg_mask
                stats['steps_processed'] += 1
            else:
                stats['steps_skipped'] += 1

        except Exception as e:
            stats['steps_failed'] += 1
            continue

    return stats


def process_skill_hdf5(
    hdf5_path: str,
    skill_name: str,
    skill_config: Dict,
    output_path: str
) -> Dict[str, any]:
    """
    Process a single skill HDF5 file and regenerate all segmentations.

    Returns:
        Dict with statistics: {'success', 'skill_type', 'demos_processed', 'total_stats'}
    """
    result = {
        'success': False,
        'skill_type': None,
        'demos_processed': 0,
        'total_stats': {'steps_processed': 0, 'steps_failed': 0, 'steps_skipped': 0}
    }

    # Get skill info
    skill_info = get_skill_info_from_name(skill_name, skill_config)
    if skill_info is None:
        return result

    skill_type = skill_info['skill_type']
    target_object_list = skill_info['target_object']
    grasped_object_list = skill_info['grasped_object_name']
    bddl_files = skill_info['bddl_files']

    result['skill_type'] = skill_type

    if not bddl_files or not target_object_list:
        return result

    # Initialize environment with first BDDL file
    bddl_file = os.path.join(BDDL_BASE_PATH, bddl_files[0])
    if not os.path.exists(bddl_file):
        return result

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=256,
        camera_widths=256,
        horizon=10000
    )
    env.reset()

    try:
        # Open HDF5 files
        with h5py.File(hdf5_path, 'r') as f_in:
            if 'data' not in f_in:
                return result

            demo_keys = [k for k in f_in['data'].keys() if k.startswith('demo_')]
            if not demo_keys:
                return result

            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Copy file and update segmentations
            with h5py.File(output_path, 'w') as f_out:
                # Copy top-level groups (except data)
                for key in f_in.keys():
                    if key != 'data':
                        f_in.copy(key, f_out, name=key)

                # Process data group
                data_in = f_in['data']
                data_out = f_out.create_group('data')

                # Process each demo
                demo_pbar = tqdm(demo_keys, desc=f"  {skill_name} ({skill_type})", leave=False, unit="demo")
                for demo_key in demo_pbar:
                    demo_group_in = data_in[demo_key]
                    demo_group_out = data_out.create_group(demo_key)

                    # Copy non-obs datasets
                    for key in demo_group_in.keys():
                        if key != 'obs':
                            demo_group_in.copy(key, demo_group_out, name=key)

                    # Copy obs group
                    obs_group_in = demo_group_in['obs']
                    obs_group_out = demo_group_out.create_group('obs')

                    for obs_key in obs_group_in.keys():
                        if obs_key == 'eye_in_hand_segmentation':
                            # Create dataset, will update later
                            seg_data = obs_group_in[obs_key]
                            seg_dataset = obs_group_out.create_dataset(
                                obs_key,
                                shape=seg_data.shape,
                                dtype=seg_data.dtype
                            )
                            seg_dataset[:] = seg_data[:]
                        else:
                            obs_group_in.copy(obs_key, obs_group_out, name=obs_key)

                    # Regenerate segmentations for this demo
                    demo_stats = process_single_demo(
                        demo_group_out,
                        env,
                        skill_type,
                        target_object_list,
                        grasped_object_list
                    )

                    # Accumulate statistics
                    result['demos_processed'] += 1
                    for key in demo_stats:
                        result['total_stats'][key] += demo_stats[key]

                demo_pbar.close()

        result['success'] = True
        return result

    except Exception as e:
        print(f"\n  ❌ Error processing {skill_name}: {e}")
        return result

    finally:
        env.close()


def verify_segmentation(hdf5_path: str, skill_name: str) -> bool:
    """Verify that segmentation was regenerated correctly."""
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'data' not in f:
                return False

            for demo_key in f['data'].keys():
                if not demo_key.startswith('demo_'):
                    continue

                demo = f['data'][demo_key]
                if 'obs' not in demo or 'eye_in_hand_segmentation' not in demo['obs']:
                    return False

                seg = demo['obs']['eye_in_hand_segmentation']

                # Check shape
                if len(seg.shape) != 3 or seg.shape[-2:] != (256, 256):
                    print(f"  ⚠️  Invalid shape for {skill_name}/{demo_key}: {seg.shape}")
                    return False

                # Check dtype
                if seg.dtype != np.uint8:
                    print(f"  ⚠️  Invalid dtype for {skill_name}/{demo_key}: {seg.dtype}")
                    return False

                # Check value range
                if seg.max() > 255 or seg.min() < 0:
                    print(f"  ⚠️  Invalid value range for {skill_name}/{demo_key}: [{seg.min()}, {seg.max()}]")
                    return False

        return True

    except Exception as e:
        print(f"  ⚠️  Verification failed for {skill_name}: {e}")
        return False


def main():
    """Main function."""
    print("=" * 80)
    print("Regenerating Segmentation Masks for All Skills")
    print("=" * 80)
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    print(f"Loading skill config...")
    skill_config = load_skill_config()

    # Find all HDF5 files
    all_skill_files = []
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('.hdf5'):
            skill_name = filename.replace('.hdf5', '')
            if skill_name in skill_config:
                hdf5_path = os.path.join(INPUT_DIR, filename)
                output_path = os.path.join(OUTPUT_DIR, filename)
                all_skill_files.append((skill_name, hdf5_path, output_path))

    print(f"Found {len(all_skill_files)} skill files to process")
    print()

    # Process each skill with progress bar
    results_by_type = {
        'pick': {'success': 0, 'failed': 0, 'demos': 0, 'steps': 0},
        'place': {'success': 0, 'failed': 0, 'demos': 0, 'steps': 0},
        'open': {'success': 0, 'failed': 0, 'demos': 0, 'steps': 0},
        'close': {'success': 0, 'failed': 0, 'demos': 0, 'steps': 0},
        'turn_on': {'success': 0, 'failed': 0, 'demos': 0, 'steps': 0},
        'other': {'success': 0, 'failed': 0, 'demos': 0, 'steps': 0},
    }

    for skill_name, hdf5_path, output_path in tqdm(all_skill_files, desc="Processing skills"):
        try:
            result = process_skill_hdf5(hdf5_path, skill_name, skill_config, output_path)

            skill_type = result['skill_type'] if result['skill_type'] else 'other'

            if result['success']:
                # Verify output
                if verify_segmentation(output_path, skill_name):
                    results_by_type[skill_type]['success'] += 1
                    results_by_type[skill_type]['demos'] += result['demos_processed']
                    results_by_type[skill_type]['steps'] += result['total_stats']['steps_processed']
                else:
                    results_by_type[skill_type]['failed'] += 1
                    print(f"  ⚠️  Verification failed for {skill_name}")
            else:
                results_by_type[skill_type]['failed'] += 1

        except Exception as e:
            print(f"\n❌ Error processing {skill_name}: {e}")
            skill_type = 'other'
            results_by_type[skill_type]['failed'] += 1

    # Print summary
    print()
    print("=" * 80)
    print("Segmentation Regeneration Summary")
    print("=" * 80)

    total_success = 0
    total_failed = 0
    total_demos = 0
    total_steps = 0

    for skill_type, stats in results_by_type.items():
        if stats['success'] + stats['failed'] > 0:
            success_rate = stats['success'] / (stats['success'] + stats['failed']) * 100
            print(f"{skill_type.capitalize():12s}: {stats['success']:3d} success, {stats['failed']:3d} failed "
                  f"({success_rate:5.1f}%) | Demos: {stats['demos']:5d} | Steps: {stats['steps']:7d}")
            total_success += stats['success']
            total_failed += stats['failed']
            total_demos += stats['demos']
            total_steps += stats['steps']

    print("-" * 80)
    total_skills = total_success + total_failed
    overall_rate = total_success / total_skills * 100 if total_skills > 0 else 0
    print(f"{'Total':12s}: {total_success:3d} success, {total_failed:3d} failed "
          f"({overall_rate:5.1f}%) | Demos: {total_demos:5d} | Steps: {total_steps:7d}")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
