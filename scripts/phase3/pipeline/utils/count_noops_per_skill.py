#!/usr/bin/env python3
"""
Count average noops per skill for all HDF5 files in a directory.

This script:
1. Processes all HDF5 files in the specified directory
2. Counts noop actions per demo using the same logic as regenerate_libero_dataset.py
3. Calculates average noops per skill (per HDF5 file)
4. Outputs results to JSON and console

Usage:
    python scripts/phase3/pipeline/utils/count_noops_per_skill.py \
        --data_dir datasets/hdf5_datasets/atomic_above_fewer/all \
        --output_file scripts/phase3/pipeline/outputs/results/noop_counts.json
"""

import argparse
import os
import json
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
import glob


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.
    
    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.
    
    This matches the logic from regenerate_libero_dataset.py
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    
    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def count_noops_in_demo(actions: np.ndarray, threshold=1e-4, return_samples=False):
    """
    Count the number of noop actions in a single demo.
    
    Args:
        actions: Array of shape (N, 7) where N is number of timesteps
        threshold: Threshold for noop detection
        return_samples: If True, return list of sample noop actions
        
    Returns:
        Number of noop actions (and optionally sample noop actions)
    """
    if len(actions) == 0:
        return (0, []) if return_samples else 0
    
    num_noops = 0
    prev_action = None
    samples = []
    
    for idx, action in enumerate(actions):
        if is_noop(action, prev_action, threshold):
            num_noops += 1
            if return_samples and len(samples) < 3:  # Keep first 3 samples
                samples.append({
                    "timestep": idx,
                    "action": action.tolist(),
                    "prev_action": prev_action.tolist() if prev_action is not None else None,
                    "norm": float(np.linalg.norm(action[:-1])),
                    "gripper": float(action[-1]),
                    "prev_gripper": float(prev_action[-1]) if prev_action is not None else None
                })
        prev_action = action
    
    if return_samples:
        return num_noops, samples
    return num_noops


def process_hdf5_file(file_path: str, threshold=1e-4, show_samples=False) -> Dict:
    """
    Process a single HDF5 file and count noops per demo.
    
    Args:
        file_path: Path to HDF5 file
        threshold: Threshold for noop detection
        show_samples: If True, include sample noop actions in results
        
    Returns:
        Dictionary with skill name, per-demo counts, and statistics
    """
    skill_name = os.path.splitext(os.path.basename(file_path))[0]
    
    result = {
        "skill_name": skill_name,
        "file_path": file_path,
        "demos": [],
        "total_demos": 0,
        "total_actions": 0,
        "total_noops": 0,
        "avg_noops_per_demo": 0.0,
        "avg_noops_per_action": 0.0,
        "error": None
    }
    
    try:
        with h5py.File(file_path, 'r') as h5file:
            if 'data' not in h5file:
                result["error"] = "No 'data' group found"
                return result
            
            data_group = h5file['data']
            demo_keys = sorted(data_group.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
            
            for demo_key in demo_keys:
                demo_group = data_group[demo_key]
                
                if 'actions' not in demo_group:
                    continue
                
                actions = demo_group['actions'][()]
                
                # Ensure actions is 2D
                if len(actions.shape) == 1:
                    actions = actions.reshape(1, -1)
                
                num_actions = len(actions)
                if show_samples:
                    num_noops, samples = count_noops_in_demo(actions, threshold, return_samples=True)
                    if samples:
                        result["demos"].append({
                            "demo_key": demo_key,
                            "num_actions": num_actions,
                            "num_noops": num_noops,
                            "noop_ratio": num_noops / num_actions if num_actions > 0 else 0.0,
                            "noop_samples": samples
                        })
                    else:
                        result["demos"].append({
                            "demo_key": demo_key,
                            "num_actions": num_actions,
                            "num_noops": num_noops,
                            "noop_ratio": num_noops / num_actions if num_actions > 0 else 0.0
                        })
                else:
                    num_noops = count_noops_in_demo(actions, threshold)
                    result["demos"].append({
                        "demo_key": demo_key,
                        "num_actions": num_actions,
                        "num_noops": num_noops,
                        "noop_ratio": num_noops / num_actions if num_actions > 0 else 0.0
                    })
                
                result["total_actions"] += num_actions
                result["total_noops"] += num_noops
            
            result["total_demos"] = len(result["demos"])
            
            if result["total_demos"] > 0:
                result["avg_noops_per_demo"] = result["total_noops"] / result["total_demos"]
                result["avg_noops_per_action"] = result["total_noops"] / result["total_actions"] if result["total_actions"] > 0 else 0.0
    
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
    
    return result


def find_hdf5_files(data_dir: str) -> List[str]:
    """
    Find all HDF5 files in the data directory.
    
    Args:
        data_dir: Directory containing HDF5 files
        
    Returns:
        List of HDF5 file paths
    """
    hdf5_files = glob.glob(os.path.join(data_dir, "*.hdf5"))
    return sorted(hdf5_files)


def main():
    parser = argparse.ArgumentParser(description="Count average noops per skill in HDF5 files")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/hdf5_datasets/atomic_above_fewer/all",
        help="Directory containing HDF5 files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="scripts/phase3/pipeline/outputs/results/noop_counts.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-4,
        help="Threshold for noop detection (default: 1e-4)"
    )
    parser.add_argument(
        "--show_samples",
        action="store_true",
        help="Show sample noop actions for verification"
    )
    
    args = parser.parse_args()
    
    # Update global threshold (would need to modify is_noop to use it, but keeping original for now)
    
    print("=" * 80)
    print("Noop Counter per Skill")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output file: {args.output_file}")
    print()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all HDF5 files
    print("Finding HDF5 files...")
    hdf5_files = find_hdf5_files(args.data_dir)
    print(f"Found {len(hdf5_files)} HDF5 files")
    print()
    
    if len(hdf5_files) == 0:
        print("No HDF5 files found!")
        return
    
    # Process each file
    print("Processing files...")
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_directory": args.data_dir,
        "threshold": args.threshold,
        "total_files": len(hdf5_files),
        "skills": [],
        "summary": {
            "total_skills": 0,
            "total_demos": 0,
            "total_actions": 0,
            "total_noops": 0,
            "avg_noops_per_skill": 0.0,
            "avg_noops_per_demo": 0.0,
            "avg_noops_per_action": 0.0
        }
    }
    
    for file_path in tqdm(hdf5_files, desc="Processing"):
        skill_result = process_hdf5_file(file_path, args.threshold, args.show_samples)
        results["skills"].append(skill_result)
        
        if skill_result["error"] is None:
            results["summary"]["total_skills"] += 1
            results["summary"]["total_demos"] += skill_result["total_demos"]
            results["summary"]["total_actions"] += skill_result["total_actions"]
            results["summary"]["total_noops"] += skill_result["total_noops"]
    
    # Calculate summary statistics
    if results["summary"]["total_skills"] > 0:
        results["summary"]["avg_noops_per_skill"] = results["summary"]["total_noops"] / results["summary"]["total_skills"]
    
    if results["summary"]["total_demos"] > 0:
        results["summary"]["avg_noops_per_demo"] = results["summary"]["total_noops"] / results["summary"]["total_demos"]
    
    if results["summary"]["total_actions"] > 0:
        results["summary"]["avg_noops_per_action"] = results["summary"]["total_noops"] / results["summary"]["total_actions"]
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total skills processed: {results['summary']['total_skills']}")
    print(f"Total demos: {results['summary']['total_demos']}")
    print(f"Total actions: {results['summary']['total_actions']:,}")
    print(f"Total noops: {results['summary']['total_noops']:,}")
    print(f"Average noops per skill: {results['summary']['avg_noops_per_skill']:.2f}")
    print(f"Average noops per demo: {results['summary']['avg_noops_per_demo']:.2f}")
    print(f"Average noops per action: {results['summary']['avg_noops_per_action']:.6f}")
    print(f"Noop ratio: {results['summary']['avg_noops_per_action']*100:.4f}%")
    print()
    
    # Print per-skill statistics
    print("=" * 80)
    print("Per-Skill Statistics (Top 10 by noop count)")
    print("=" * 80)
    
    # Sort skills by total noops
    valid_skills = [s for s in results["skills"] if s["error"] is None]
    sorted_skills = sorted(valid_skills, key=lambda x: x["total_noops"], reverse=True)
    
    print(f"{'Skill Name':<50} {'Demos':<8} {'Actions':<12} {'Noops':<10} {'Avg/Demo':<10} {'Ratio':<10}")
    print("-" * 100)
    
    for skill in sorted_skills[:10]:
        ratio = skill["avg_noops_per_action"] * 100
        print(f"{skill['skill_name']:<50} {skill['total_demos']:<8} {skill['total_actions']:<12} "
              f"{skill['total_noops']:<10} {skill['avg_noops_per_demo']:<10.2f} {ratio:<10.4f}%")
    
    # Check for errors
    error_skills = [s for s in results["skills"] if s["error"] is not None]
    if error_skills:
        print()
        print("=" * 80)
        print("Files with Errors:")
        print("=" * 80)
        for skill in error_skills:
            print(f"  {skill['skill_name']}: {skill['error']}")
    
    # Show sample noops if requested
    if args.show_samples:
        print()
        print("=" * 80)
        print("Sample Noop Actions (for verification)")
        print("=" * 80)
        sample_count = 0
        for skill in valid_skills:
            if sample_count >= 5:  # Limit to 5 samples total
                break
            for demo in skill["demos"]:
                if "noop_samples" in demo and demo["noop_samples"]:
                    print(f"\nSkill: {skill['skill_name']}, Demo: {demo['demo_key']}")
                    for sample in demo["noop_samples"]:
                        print(f"  Timestep {sample['timestep']}:")
                        print(f"    Action: {sample['action']}")
                        print(f"    Norm(action[:-1]): {sample['norm']:.6f} (threshold: {args.threshold})")
                        print(f"    Gripper: {sample['gripper']}, Prev gripper: {sample['prev_gripper']}")
                        sample_count += 1
                        if sample_count >= 5:
                            break
                if sample_count >= 5:
                    break
    
    print()
    print(f"Results saved to: {args.output_file}")
    print("Done!")


if __name__ == "__main__":
    main()

