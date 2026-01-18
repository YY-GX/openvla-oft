#!/usr/bin/env python3
"""
Investigate combined HDF5 dataset in datasets/hdf5_datasets/atomic_above_fewer/all

This script analyzes:
1. Shift ratio for each skill (non-shifted, standard shift, z-only shift)
2. Target object availability in observations

Usage:
    python scripts/phase3/pipeline/utils/investigate_combined_dataset.py
    python scripts/phase3/pipeline/utils/investigate_combined_dataset.py --output_file investigation_report.json
"""

import os
import json
import h5py
import glob
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from datetime import datetime


def analyze_shift_types(hdf5_path: str) -> Dict[str, int]:
    """
    Analyze shift types in an HDF5 file.

    Returns:
        dict with counts: {
            'total': int,
            'non_shifted': int,
            'standard_shift': int,
            'z_only_shift': int,
            'unknown': int
        }
    """
    counts = {
        'total': 0,
        'non_shifted': 0,
        'standard_shift': 0,
        'z_only_shift': 0,
        'unknown': 0
    }

    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'data' not in f:
                return counts

            for demo_key in f['data'].keys():
                demo = f['data'][demo_key]
                counts['total'] += 1

                if 'metadata' not in demo:
                    counts['unknown'] += 1
                    continue

                meta = demo['metadata']

                # Read shifted flag
                shifted = False
                if 'shifted' in meta.attrs:
                    shifted = meta.attrs['shifted']
                    if isinstance(shifted, (np.bool_, np.integer)):
                        shifted = bool(shifted)
                elif 'iteration' in meta.attrs:
                    iteration = meta.attrs['iteration']
                    if isinstance(iteration, np.integer):
                        iteration = int(iteration)
                    shifted = (iteration > 0)

                if not shifted:
                    counts['non_shifted'] += 1
                else:
                    # Check if z_only shift
                    z_only = False
                    if 'shift_info' in meta.attrs:
                        shift_info_str = meta.attrs['shift_info']
                        if isinstance(shift_info_str, bytes):
                            shift_info_str = shift_info_str.decode('utf-8')
                        try:
                            shift_info = json.loads(shift_info_str)
                            z_only = shift_info.get('z_only_positive', False)
                        except:
                            pass

                    if z_only:
                        counts['z_only_shift'] += 1
                    else:
                        counts['standard_shift'] += 1

    except Exception as e:
        print(f"  ⚠️  Error analyzing {hdf5_path}: {e}")

    return counts


def check_target_object_availability(hdf5_path: str) -> Dict[str, int]:
    """
    Check if target_object_pos and target_object_quat exist in observations.

    Returns:
        dict with counts: {
            'total_timesteps': int,
            'missing_target_pos': int,
            'missing_target_quat': int,
            'zero_target_pos': int,
            'default_target_quat': int
        }
    """
    counts = {
        'total_timesteps': 0,
        'missing_target_pos': 0,
        'missing_target_quat': 0,
        'zero_target_pos': 0,
        'default_target_quat': 0
    }

    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'data' not in f:
                return counts

            for demo_key in f['data'].keys():
                demo = f['data'][demo_key]

                if 'obs' not in demo:
                    continue

                obs = demo['obs']

                # Check if target_object fields exist
                has_target_pos = 'target_object_pos' in obs
                has_target_quat = 'target_object_quat' in obs

                if not has_target_pos and not has_target_quat:
                    # Fields don't exist at all - skip this demo
                    continue

                # Get number of timesteps
                if has_target_pos:
                    num_steps = obs['target_object_pos'].shape[0]
                elif has_target_quat:
                    num_steps = obs['target_object_quat'].shape[0]
                else:
                    continue

                counts['total_timesteps'] += num_steps

                # Check each timestep
                if has_target_pos:
                    target_pos = obs['target_object_pos'][()]
                    for step_idx in range(num_steps):
                        pos = target_pos[step_idx]
                        # Check if all zeros
                        if np.allclose(pos, 0.0, atol=1e-9):
                            counts['zero_target_pos'] += 1
                else:
                    counts['missing_target_pos'] += num_steps

                if has_target_quat:
                    target_quat = obs['target_object_quat'][()]
                    for step_idx in range(num_steps):
                        quat = target_quat[step_idx]
                        # Check if default quaternion [1, 0, 0, 0]
                        default_quat = np.array([1.0, 0.0, 0.0, 0.0])
                        if np.allclose(quat, default_quat, atol=1e-9):
                            counts['default_target_quat'] += 1
                else:
                    counts['missing_target_quat'] += num_steps

    except Exception as e:
        print(f"  ⚠️  Error checking target object in {hdf5_path}: {e}")

    return counts


def investigate_dataset(dataset_dir: str) -> Dict:
    """
    Investigate all HDF5 files in the dataset directory.

    Returns:
        dict with investigation results for each skill
    """
    print("=" * 80)
    print("Dataset Investigation")
    print("=" * 80)
    print(f"Dataset directory: {dataset_dir}")
    print()

    # Find all success HDF5 files
    hdf5_files = glob.glob(os.path.join(dataset_dir, "*.hdf5"))
    success_files = [f for f in hdf5_files if "_failure" not in os.path.basename(f)]

    print(f"Found {len(success_files)} HDF5 files to investigate")
    print()

    results = {}

    for idx, file_path in enumerate(sorted(success_files), 1):
        skill_name = os.path.basename(file_path).replace('.hdf5', '')
        print(f"[{idx}/{len(success_files)}] Investigating {skill_name}...")

        # Analyze shift types
        shift_counts = analyze_shift_types(file_path)

        # Check target object availability
        target_counts = check_target_object_availability(file_path)

        # Calculate ratios
        total_demos = shift_counts['total']
        shift_ratios = {}
        if total_demos > 0:
            shift_ratios = {
                'non_shifted': shift_counts['non_shifted'] / total_demos,
                'standard_shift': shift_counts['standard_shift'] / total_demos,
                'z_only_shift': shift_counts['z_only_shift'] / total_demos,
                'unknown': shift_counts['unknown'] / total_demos
            }

        total_timesteps = target_counts['total_timesteps']
        target_ratios = {}
        if total_timesteps > 0:
            target_ratios = {
                'missing_pos': target_counts['missing_target_pos'] / total_timesteps,
                'missing_quat': target_counts['missing_target_quat'] / total_timesteps,
                'zero_pos': target_counts['zero_target_pos'] / total_timesteps,
                'default_quat': target_counts['default_target_quat'] / total_timesteps
            }

        results[skill_name] = {
            'shift_analysis': {
                'counts': shift_counts,
                'ratios': shift_ratios
            },
            'target_object_analysis': {
                'counts': target_counts,
                'ratios': target_ratios
            }
        }

        # Print summary
        print(f"  Shift analysis:")
        print(f"    Total demos: {total_demos}")
        if total_demos > 0:
            print(f"    Non-shifted: {shift_counts['non_shifted']} ({shift_ratios['non_shifted']:.2%})")
            print(f"    Standard shift (xy+z): {shift_counts['standard_shift']} ({shift_ratios['standard_shift']:.2%})")
            print(f"    Z-only shift: {shift_counts['z_only_shift']} ({shift_ratios['z_only_shift']:.2%})")
            if shift_counts['unknown'] > 0:
                print(f"    Unknown: {shift_counts['unknown']} ({shift_ratios['unknown']:.2%})")

        print(f"  Target object analysis:")
        print(f"    Total timesteps: {total_timesteps}")
        if total_timesteps > 0:
            if target_counts['zero_target_pos'] > 0:
                print(f"    ⚠️  Zero position: {target_counts['zero_target_pos']} ({target_ratios['zero_pos']:.2%})")
            if target_counts['default_target_quat'] > 0:
                print(f"    ⚠️  Default quaternion: {target_counts['default_target_quat']} ({target_ratios['default_quat']:.2%})")
            if target_counts['missing_target_pos'] > 0:
                print(f"    ⚠️  Missing position: {target_counts['missing_target_pos']} ({target_ratios['missing_pos']:.2%})")
            if target_counts['missing_target_quat'] > 0:
                print(f"    ⚠️  Missing quaternion: {target_counts['missing_target_quat']} ({target_ratios['missing_quat']:.2%})")
            if (target_counts['zero_target_pos'] == 0 and
                target_counts['default_target_quat'] == 0 and
                target_counts['missing_target_pos'] == 0 and
                target_counts['missing_target_quat'] == 0):
                print(f"    ✓ All timesteps have valid target object")

        print()

    return results


def generate_summary(results: Dict) -> Dict:
    """Generate summary statistics across all skills."""

    summary = {
        'total_skills': len(results),
        'skills_with_issues': [],
        'shift_statistics': {
            'total_demos': 0,
            'non_shifted': 0,
            'standard_shift': 0,
            'z_only_shift': 0
        },
        'target_object_issues': {
            'skills_with_missing_target': [],
            'skills_with_zero_pos': [],
            'skills_with_default_quat': []
        }
    }

    for skill_name, skill_data in results.items():
        shift_counts = skill_data['shift_analysis']['counts']
        target_counts = skill_data['target_object_analysis']['counts']

        # Aggregate shift statistics
        summary['shift_statistics']['total_demos'] += shift_counts['total']
        summary['shift_statistics']['non_shifted'] += shift_counts['non_shifted']
        summary['shift_statistics']['standard_shift'] += shift_counts['standard_shift']
        summary['shift_statistics']['z_only_shift'] += shift_counts['z_only_shift']

        # Check for target object issues
        if target_counts['zero_target_pos'] > 0 or target_counts['default_target_quat'] > 0:
            ratio_zero = target_counts['zero_target_pos'] / max(target_counts['total_timesteps'], 1)
            ratio_default = target_counts['default_target_quat'] / max(target_counts['total_timesteps'], 1)

            issue_info = {
                'skill_name': skill_name,
                'zero_pos_ratio': ratio_zero,
                'default_quat_ratio': ratio_default
            }

            summary['skills_with_issues'].append(issue_info)

            if target_counts['zero_target_pos'] > 0:
                summary['target_object_issues']['skills_with_zero_pos'].append(skill_name)
            if target_counts['default_target_quat'] > 0:
                summary['target_object_issues']['skills_with_default_quat'].append(skill_name)

        if target_counts['missing_target_pos'] > 0 or target_counts['missing_target_quat'] > 0:
            summary['target_object_issues']['skills_with_missing_target'].append(skill_name)

    # Calculate overall shift ratios
    total_demos = summary['shift_statistics']['total_demos']
    if total_demos > 0:
        summary['shift_statistics']['ratios'] = {
            'non_shifted': summary['shift_statistics']['non_shifted'] / total_demos,
            'standard_shift': summary['shift_statistics']['standard_shift'] / total_demos,
            'z_only_shift': summary['shift_statistics']['z_only_shift'] / total_demos
        }

    return summary


def print_summary(summary: Dict):
    """Print summary statistics."""
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total skills: {summary['total_skills']}")
    print()

    print("Shift Statistics (Overall):")
    shift_stats = summary['shift_statistics']
    print(f"  Total demos: {shift_stats['total_demos']}")
    if 'ratios' in shift_stats:
        print(f"  Non-shifted: {shift_stats['non_shifted']} ({shift_stats['ratios']['non_shifted']:.2%})")
        print(f"  Standard shift: {shift_stats['standard_shift']} ({shift_stats['ratios']['standard_shift']:.2%})")
        print(f"  Z-only shift: {shift_stats['z_only_shift']} ({shift_stats['ratios']['z_only_shift']:.2%})")
    print()

    print("Target Object Issues:")
    target_issues = summary['target_object_issues']

    if target_issues['skills_with_zero_pos']:
        print(f"  ⚠️  Skills with zero position ({len(target_issues['skills_with_zero_pos'])}):")
        for skill in target_issues['skills_with_zero_pos'][:10]:
            print(f"      - {skill}")
        if len(target_issues['skills_with_zero_pos']) > 10:
            print(f"      ... and {len(target_issues['skills_with_zero_pos']) - 10} more")

    if target_issues['skills_with_default_quat']:
        print(f"  ⚠️  Skills with default quaternion ({len(target_issues['skills_with_default_quat'])}):")
        for skill in target_issues['skills_with_default_quat'][:10]:
            print(f"      - {skill}")
        if len(target_issues['skills_with_default_quat']) > 10:
            print(f"      ... and {len(target_issues['skills_with_default_quat']) - 10} more")

    if target_issues['skills_with_missing_target']:
        print(f"  ⚠️  Skills with missing target fields ({len(target_issues['skills_with_missing_target'])}):")
        for skill in target_issues['skills_with_missing_target'][:10]:
            print(f"      - {skill}")
        if len(target_issues['skills_with_missing_target']) > 10:
            print(f"      ... and {len(target_issues['skills_with_missing_target']) - 10} more")

    if not any([target_issues['skills_with_zero_pos'],
                target_issues['skills_with_default_quat'],
                target_issues['skills_with_missing_target']]):
        print(f"  ✓ No target object issues found!")

    print()

    if summary['skills_with_issues']:
        print(f"Skills with Issues ({len(summary['skills_with_issues'])}):")
        for issue in sorted(summary['skills_with_issues'],
                           key=lambda x: max(x['zero_pos_ratio'], x['default_quat_ratio']),
                           reverse=True)[:10]:
            print(f"  {issue['skill_name']}:")
            if issue['zero_pos_ratio'] > 0:
                print(f"    Zero pos: {issue['zero_pos_ratio']:.2%}")
            if issue['default_quat_ratio'] > 0:
                print(f"    Default quat: {issue['default_quat_ratio']:.2%}")
        if len(summary['skills_with_issues']) > 10:
            print(f"  ... and {len(summary['skills_with_issues']) - 10} more")


def create_basic_skill_stats(results: Dict) -> Dict:
    """
    Create simplified per-skill statistics for easy viewing.

    Returns:
        dict: {skill_name: {basic stats}}
    """
    basic_stats = {}

    for skill_name, skill_data in results.items():
        shift_counts = skill_data['shift_analysis']['counts']
        shift_ratios = skill_data['shift_analysis']['ratios']
        target_counts = skill_data['target_object_analysis']['counts']

        # Check if target object was missing/invalid in any timestep
        has_missing_target = (
            target_counts['zero_target_pos'] > 0 or
            target_counts['default_target_quat'] > 0 or
            target_counts['missing_target_pos'] > 0 or
            target_counts['missing_target_quat'] > 0
        )

        basic_stats[skill_name] = {
            'num_demos': shift_counts['total'],
            'ratio_non_shifted': round(shift_ratios.get('non_shifted', 0.0), 4),
            'ratio_standard_shift': round(shift_ratios.get('standard_shift', 0.0), 4),
            'ratio_z_only_shift': round(shift_ratios.get('z_only_shift', 0.0), 4),
            'has_missing_target_object': has_missing_target
        }

    return basic_stats


def main():
    parser = argparse.ArgumentParser(description='Investigate combined HDF5 dataset')
    parser.add_argument('--dataset_dir', type=str,
                       default='/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_above_fewer/all',
                       help='Path to combined dataset directory')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Additional output file path (optional, will also save to stats.json in dataset_dir)')
    args = parser.parse_args()

    # Check if directory exists
    if not os.path.exists(args.dataset_dir):
        print(f"❌ Error: Dataset directory not found: {args.dataset_dir}")
        return

    # Investigate dataset
    results = investigate_dataset(args.dataset_dir)

    # Generate summary
    summary = generate_summary(results)

    # Create basic skill stats
    basic_skill_stats = create_basic_skill_stats(results)

    # Print summary
    print_summary(summary)

    # Prepare output data with reorganized structure
    output_data = {
        'summary': {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_dir': args.dataset_dir,
            'total_skills': summary['total_skills'],
            'total_demos': summary['shift_statistics']['total_demos'],
            'overall_shift_ratios': summary['shift_statistics'].get('ratios', {}),
            'skills_with_missing_target': len(summary['skills_with_issues'])
        },
        'skills': basic_skill_stats,
        'detailed_analysis': results
    }

    # Always save to stats.json in dataset directory
    stats_file = os.path.join(args.dataset_dir, 'stats.json')
    with open(stats_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"💾 Results saved to: {stats_file}")

    # Save to additional output file if specified
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"💾 Results also saved to: {args.output_file}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
