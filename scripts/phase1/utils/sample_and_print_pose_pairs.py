#!/usr/bin/env python3
"""
Sample and print pose pairs from all .pkl files in the object_ee_poses_pairs directory.

This script loads all .pkl files containing pose pairs, samples 3 pose pairs from each file,
and prints detailed information about each sampled pose pair.
"""

import os
import pickle
import numpy as np
import random
from pathlib import Path
from typing import List, Dict


def sample_pose_pairs(pkl_file_path: str, num_samples: int = 3) -> List[Dict]:
    """
    Load a .pkl file and sample a specified number of pose pairs.
    
    Args:
        pkl_file_path: Path to the .pkl file
        num_samples: Number of pose pairs to sample
        
    Returns:
        List of sampled pose pair dictionaries
    """
    try:
        with open(pkl_file_path, 'rb') as f:
            pose_pairs = pickle.load(f)
        
        # Sample up to num_samples pose pairs (or all if less than num_samples)
        if len(pose_pairs) <= num_samples:
            sampled_pairs = pose_pairs
        else:
            sampled_pairs = random.sample(pose_pairs, num_samples)
        
        return sampled_pairs
    except Exception as e:
        print(f"ERROR: Failed to load {pkl_file_path}: {e}")
        return []


def print_pose_pair_info(pair: Dict, pair_index: int, file_name: str):
    """
    Print detailed information about a single pose pair.
    
    Args:
        pair: Pose pair dictionary
        pair_index: Index of the pose pair within the file
        file_name: Name of the source file
    """
    print(f"File: {file_name}")
    print(f"Pose Pair {pair_index + 1}:")
    print(f"  Init State ID: {pair['init_state_id']}")
    print(f"  Object Name: {pair['object_pose']['object_name']}")
    print(f"  Object Position: {pair['object_pose']['position']}")
    print(f"  Object Quaternion: {pair['object_pose']['quaternion']}")
    print(f"  EE Position: {pair['ee_pose']['position']}")
    print(f"  EE Quaternion: {pair['ee_pose']['quaternion']}")
    print('-' * 60)


def main():
    """Main function to sample and print pose pairs from all .pkl files."""
    
    # Directory containing .pkl files
    pkl_dir = "datasets/hdf5_datasets/atomic_local_demos/full_atomic_skills/object_ee_poses_pairs"
    
    if not os.path.exists(pkl_dir):
        print(f"ERROR: Directory {pkl_dir} does not exist!")
        return
    
    # Find all .pkl files
    pkl_files = [f for f in os.listdir(pkl_dir) if f.endswith('.pkl')]
    pkl_files.sort()  # Sort for consistent ordering
    
    if not pkl_files:
        print(f"No .pkl files found in {pkl_dir}")
        return
    
    print(f"Found {len(pkl_files)} .pkl files")
    print("=" * 80)
    
    total_sampled_pairs = 0
    
    # Process each .pkl file
    for pkl_file in pkl_files:
        pkl_path = os.path.join(pkl_dir, pkl_file)
        
        print(f"\nProcessing: {pkl_file}")
        print("=" * 80)
        
        # Sample 3 pose pairs from this file
        sampled_pairs = sample_pose_pairs(pkl_path, num_samples=3)
        
        if sampled_pairs:
            # Print information for each sampled pair
            for i, pair in enumerate(sampled_pairs):
                print_pose_pair_info(pair, i, pkl_file)
            
            total_sampled_pairs += len(sampled_pairs)
            print(f"✅ Sampled {len(sampled_pairs)} pose pairs from {pkl_file}")
        else:
            print(f"❌ No pose pairs sampled from {pkl_file}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SAMPLING COMPLETE")
    print(f"{'='*80}")
    print(f"Total files processed: {len(pkl_files)}")
    print(f"Total pose pairs sampled: {total_sampled_pairs}")
    print(f"Expected total (3 per file): {len(pkl_files) * 3}")
    print(f"Average pairs per file: {total_sampled_pairs/len(pkl_files):.1f}")


if __name__ == "__main__":
    # Set random seed for reproducible sampling
    random.seed(42)
    main()
