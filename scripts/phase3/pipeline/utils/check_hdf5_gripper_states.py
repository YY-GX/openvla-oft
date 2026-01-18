#!/usr/bin/env python3
"""
Check all demos in Task 8 HDF5 files for gripper_states attribute.

This script validates that all demos have the required 'gripper_states' field
in their observation data, which is needed by the RLDS dataset builder.
"""

import os
import h5py
import sys
from pathlib import Path


# Task 8 demos files (from LIBERO_Above_Atomic_Long_ID_8_dataset_builder.py)
atomic_demos_path = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_above_fewer/all"

demos_file = [
    f"{atomic_demos_path}/pick_black_bowl.hdf5",
    f"{atomic_demos_path}/place_black_bowl_on_the_plate.hdf5",
    f"{atomic_demos_path}/place_black_bowl_on_the_black_bowl.hdf5",
    f"{atomic_demos_path}/open_the_top_drawer_of_the_cabinet.hdf5",
    f"{atomic_demos_path}/place_black_bowl_in_top_drawer_of_the_cabinet.hdf5",
    f"{atomic_demos_path}/close_the_top_drawer_of_the_cabinet.hdf5",
    f"{atomic_demos_path}/pick_cream_cheese.hdf5",
    f"{atomic_demos_path}/place_cream_cheese_in_basket.hdf5",
]


def check_demo_structure(hdf5_path: str, demo_id: int) -> dict:
    """
    Check if a specific demo has all required fields.

    Returns:
        dict with 'valid', 'error', and 'obs_keys' fields
    """
    result = {
        'valid': False,
        'error': None,
        'obs_keys': None,
        'has_gripper_states': False
    }

    try:
        with h5py.File(hdf5_path, 'r') as f:
            demo_key = f'demo_{demo_id}'

            # Check if demo exists
            if demo_key not in f['data'].keys():
                result['error'] = 'Demo does not exist'
                return result

            demo_group = f['data'][demo_key]

            # Check if obs group exists
            if 'obs' not in demo_group:
                result['error'] = 'No obs group'
                return result

            obs_group = demo_group['obs']
            obs_keys = list(obs_group.keys())
            result['obs_keys'] = obs_keys

            # Check for gripper_states
            if 'gripper_states' not in obs_keys:
                result['error'] = 'Missing gripper_states'
                result['has_gripper_states'] = False
                return result

            # Try to access gripper_states
            try:
                gripper_data = obs_group['gripper_states'][()]
                result['has_gripper_states'] = True
                result['valid'] = True
            except Exception as e:
                result['error'] = f'Cannot access gripper_states: {e}'
                return result

    except Exception as e:
        result['error'] = f'File error: {e}'

    return result


def check_all_demos():
    """Check all demos in all Task 8 HDF5 files."""
    print("=" * 80)
    print("Checking Task 8 HDF5 Files for gripper_states")
    print("=" * 80)
    print()

    total_files = 0
    total_demos = 0
    total_valid = 0
    total_invalid = 0

    all_issues = []

    for hdf5_path in demos_file:
        skill_name = os.path.basename(hdf5_path).replace('.hdf5', '')

        # Check if file exists
        if not os.path.exists(hdf5_path):
            print(f"✗ {skill_name}")
            print(f"  ERROR: File does not exist - {hdf5_path}")
            print()
            all_issues.append((skill_name, -1, 'File does not exist'))
            continue

        print(f"📁 {skill_name}")
        print(f"   Path: {hdf5_path}")

        total_files += 1
        file_valid = 0
        file_invalid = 0

        try:
            with h5py.File(hdf5_path, 'r') as f:
                n_demos = len(f['data'])
                print(f"   Demos: {n_demos}")

                # Check each demo
                invalid_demos = []
                for demo_id in range(n_demos):
                    total_demos += 1
                    result = check_demo_structure(hdf5_path, demo_id)

                    if result['valid']:
                        file_valid += 1
                        total_valid += 1
                    else:
                        file_invalid += 1
                        total_invalid += 1
                        invalid_demos.append({
                            'demo_id': demo_id,
                            'error': result['error'],
                            'obs_keys': result['obs_keys']
                        })
                        all_issues.append((skill_name, demo_id, result['error']))

                # Print file summary
                if file_invalid == 0:
                    print(f"   ✓ All {n_demos} demos valid")
                else:
                    print(f"   ✗ {file_invalid}/{n_demos} demos invalid")

                    # Show first 5 invalid demos
                    for invalid in invalid_demos[:5]:
                        print(f"      demo_{invalid['demo_id']}: {invalid['error']}")
                        if invalid['obs_keys']:
                            print(f"         Available keys: {invalid['obs_keys']}")

                    if len(invalid_demos) > 5:
                        print(f"      ... and {len(invalid_demos) - 5} more")

        except Exception as e:
            print(f"   ✗ ERROR: Cannot read file - {e}")
            all_issues.append((skill_name, -1, f'Cannot read file: {e}'))

        print()

    # Print overall summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total files checked: {total_files}/{len(demos_file)}")
    print(f"Total demos checked: {total_demos}")
    print(f"Valid demos: {total_valid}")
    print(f"Invalid demos: {total_invalid}")
    print()

    if total_invalid > 0:
        print("❌ VALIDATION FAILED")
        print()
        print("Issues found:")
        for skill, demo_id, error in all_issues:
            if demo_id >= 0:
                print(f"  - {skill} demo_{demo_id}: {error}")
            else:
                print(f"  - {skill}: {error}")
        return 1
    else:
        print("✅ All demos valid!")
        return 0


if __name__ == "__main__":
    exit_code = check_all_demos()
    sys.exit(exit_code)
