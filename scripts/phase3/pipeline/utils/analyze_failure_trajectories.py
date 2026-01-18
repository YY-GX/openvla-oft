#!/usr/bin/env python3
"""
analyze_failure_trajectories.py

Script to analyze failure trajectories in atomic_above_fewer dataset.
Counts total failure trajectories across all v* directories and calculates
statistics on trajectory lengths.

Usage:
    python scripts/phase3/pipeline/utils/analyze_failure_trajectories.py
"""

import argparse
import h5py
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Tuple


def get_trajectory_lengths(hdf5_path: str) -> List[int]:
    """
    Extract trajectory lengths from an HDF5 file.
    
    Args:
        hdf5_path: Path to the HDF5 file
        
    Returns:
        List of trajectory lengths, empty list if error
    """
    traj_lengths = []
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'data' not in f:
                return traj_lengths
            
            data_group = f['data']
            # Get all demo keys (e.g., demo_0, demo_1, etc.)
            demo_keys = sorted([key for key in data_group.keys() if key.startswith('demo_')])
            
            for demo_key in demo_keys:
                demo = data_group[demo_key]
                if 'actions' in demo:
                    # Trajectory length is the number of actions
                    length = demo['actions'].shape[0]
                    traj_lengths.append(length)
                elif 'action' in demo:
                    # Alternative key name
                    length = demo['action'].shape[0]
                    traj_lengths.append(length)
                    
    except Exception as e:
        print(f"Error reading {hdf5_path}: {e}", file=sys.stderr)
        return []
    
    return traj_lengths


def analyze_failure_trajectories(base_dir: str) -> Tuple[List[int], dict]:
    """
    Analyze all failure trajectories in v* directories.
    
    Args:
        base_dir: Base directory containing v* subdirectories
        
    Returns:
        Tuple of (all trajectory lengths, statistics dict)
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise ValueError(f"Base directory does not exist: {base_dir}")
    
    # Find all v* directories
    v_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('v')])
    
    if not v_dirs:
        raise ValueError(f"No v* directories found in {base_dir}")
    
    print(f"Found {len(v_dirs)} v* directories: {[d.name for d in v_dirs]}")
    
    all_traj_lengths = []
    file_stats = {}
    
    for v_dir in v_dirs:
        print(f"\nAnalyzing {v_dir.name}...")
        # Find all _failure.hdf5 files
        failure_files = sorted(v_dir.glob("*_failure.hdf5"))
        
        if not failure_files:
            print(f"  No failure files found in {v_dir.name}")
            continue
        
        print(f"  Found {len(failure_files)} failure files")
        
        v_traj_lengths = []
        for failure_file in failure_files:
            traj_lengths = get_trajectory_lengths(str(failure_file))
            v_traj_lengths.extend(traj_lengths)
            file_stats[str(failure_file)] = {
                'num_trajectories': len(traj_lengths),
                'traj_lengths': traj_lengths
            }
            if traj_lengths:
                print(f"    {failure_file.name}: {len(traj_lengths)} trajectories")
        
        all_traj_lengths.extend(v_traj_lengths)
        print(f"  {v_dir.name} total: {len(v_traj_lengths)} trajectories")
    
    return all_traj_lengths, file_stats


def print_statistics(traj_lengths: List[int]) -> None:
    """
    Print statistics about trajectory lengths.
    
    Args:
        traj_lengths: List of trajectory lengths
    """
    if not traj_lengths:
        print("\nNo trajectories found!")
        return
    
    traj_array = np.array(traj_lengths)
    
    print("\n" + "="*80)
    print("FAILURE TRAJECTORY STATISTICS")
    print("="*80)
    print(f"\nTotal failure trajectories: {len(traj_lengths)}")
    print(f"Average length: {np.mean(traj_array):.2f}")
    print(f"Min length: {np.min(traj_array)}")
    print(f"Max length: {np.max(traj_array)}")
    print(f"Std length: {np.std(traj_array):.2f}")
    print(f"Median length: {np.median(traj_array):.2f}")
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze failure trajectories in atomic_above_fewer dataset"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="datasets/hdf5_datasets/atomic_above_fewer",
        help="Base directory containing v* subdirectories (default: datasets/hdf5_datasets/atomic_above_fewer)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed information for each file"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute path if relative
    base_dir = args.base_dir
    if not os.path.isabs(base_dir):
        # Assume relative to project root
        # Script is at: scripts/phase3/pipeline/utils/analyze_failure_trajectories.py
        # Project root is 4 levels up
        project_root = Path(__file__).parent.parent.parent.parent
        base_dir = str(project_root / base_dir)
    
    # Also try current working directory if the above doesn't work
    if not os.path.exists(base_dir):
        # Try relative to current working directory
        cwd_base = Path.cwd() / args.base_dir
        if cwd_base.exists():
            base_dir = str(cwd_base)
    
    try:
        print(f"Analyzing failure trajectories in: {base_dir}")
        traj_lengths, file_stats = analyze_failure_trajectories(base_dir)
        print_statistics(traj_lengths)
        
        if args.verbose:
            print("\n" + "="*80)
            print("DETAILED FILE STATISTICS")
            print("="*80)
            for file_path, stats in sorted(file_stats.items()):
                if stats['num_trajectories'] > 0:
                    print(f"\n{Path(file_path).name}:")
                    print(f"  Trajectories: {stats['num_trajectories']}")
                    if stats['traj_lengths']:
                        lengths = np.array(stats['traj_lengths'])
                        print(f"  Avg length: {np.mean(lengths):.2f}")
                        print(f"  Min/Max: {np.min(lengths)}/{np.max(lengths)}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit(main())

