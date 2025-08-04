#!/usr/bin/env python3
"""
compute_target_object_diversity.py

Compute target object diversity (average pairwise distance) per skill.
Saves diversity statistics to the evaluation results directory.

Usage:
    python utils/compute_target_object_diversity.py
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm


def compute_target_object_diversity(data_root: str = "datasets/local_pairs_datasets"):
    """
    Compute target object diversity per skill.
    
    Args:
        data_root: Root directory containing the dataset
    """
    # Prepare to capture output for saving
    from io import StringIO
    import sys
    output_capture = StringIO()
    orig_stdout = sys.stdout
    sys.stdout = output_capture
    
    print("Computing target object diversity...")
    
    # Load target object poses and annotations
    poses_path = os.path.join(data_root, "target_object_poses.npy")
    csv_path = os.path.join(data_root, "target_object_poses.csv")
    
    if not os.path.exists(poses_path) or not os.path.exists(csv_path):
        print("Error: Target object poses not found. Run extract_target_object_poses.py first.")
        return
    
    target_object_poses = np.load(poses_path)
    target_object_df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(target_object_poses)} target object poses")
    
    # Group poses by skill and by object within each skill
    skill_groups = defaultdict(list)
    skill_object_groups = defaultdict(lambda: defaultdict(list))  # skill -> object -> list of pose indices
    for idx, row in target_object_df.iterrows():
        skill_name = row['language_description']
        pose_idx = row['pose_index']
        obj_name = row['contact_object_name']
        skill_groups[skill_name].append(pose_idx)
        skill_object_groups[skill_name][obj_name].append(pose_idx)
    
    # For collecting object-wise means for total
    all_obj_mean_pos_dists = []
    all_obj_mean_ori_dists = []
    # For collecting mean position vectors of each object
    object_mean_positions = []
    # Compute diversity for each skill
    diversity_stats = {}
    
    print(f"Computing diversity for {len(skill_groups)} skills...")
    for skill_name, pose_indices in tqdm(skill_groups.items(), desc="Computing skill diversity"):
        if len(pose_indices) < 2:
            print(f"Warning: Skill {skill_name} has only {len(pose_indices)} target object pose(s), skipping diversity computation")
            continue
        
        # Get poses for this skill
        skill_poses = target_object_poses[pose_indices]
        
        # Compute pairwise distances
        distances = pdist(skill_poses, metric='euclidean')
        
        # Compute statistics
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        
        # Compute position-only distances (first 3 dimensions)
        pos_distances = pdist(skill_poses[:, :3], metric='euclidean')
        mean_pos_distance = np.mean(pos_distances)
        std_pos_distance = np.std(pos_distances)
        
        # Compute orientation-only distances (last 3 dimensions)
        ori_distances = pdist(skill_poses[:, 3:], metric='euclidean')
        mean_ori_distance = np.mean(ori_distances)
        std_ori_distance = np.std(ori_distances)
        
        # Per-object diversity within this skill
        per_object_stats = {}
        for obj_name, obj_pose_indices in skill_object_groups[skill_name].items():
            if len(obj_pose_indices) < 2:
                # Still want to include mean position for single-pose objects
                obj_poses = target_object_poses[obj_pose_indices]
                obj_mean_position = np.mean(obj_poses[:, :3], axis=0)
                object_mean_positions.append(obj_mean_position)
                continue
            obj_poses = target_object_poses[obj_pose_indices]
            obj_distances = pdist(obj_poses, metric='euclidean')
            obj_mean_distance = np.mean(obj_distances)
            obj_std_distance = np.std(obj_distances)
            obj_mean_pos_distance = np.mean(pdist(obj_poses[:, :3], metric='euclidean'))
            obj_mean_ori_distance = np.mean(pdist(obj_poses[:, 3:], metric='euclidean'))
            per_object_stats[obj_name] = {
                'num_poses': len(obj_pose_indices),
                'mean_pairwise_distance': float(obj_mean_distance),
                'std_pairwise_distance': float(obj_std_distance),
                'mean_position_distance': float(obj_mean_pos_distance),
                'mean_orientation_distance': float(obj_mean_ori_distance),
            }
            all_obj_mean_pos_dists.append(obj_mean_pos_distance)
            all_obj_mean_ori_dists.append(obj_mean_ori_distance)
            # Also collect mean position vector
            obj_mean_position = np.mean(obj_poses[:, :3], axis=0)
            object_mean_positions.append(obj_mean_position)
        diversity_stats[skill_name] = {
            'num_target_objects': len(pose_indices),
            'mean_pairwise_distance': float(mean_distance),
            'std_pairwise_distance': float(std_distance),
            'min_pairwise_distance': float(min_distance),
            'max_pairwise_distance': float(max_distance),
            'mean_position_distance': float(mean_pos_distance),
            'std_position_distance': float(std_pos_distance),
            'mean_orientation_distance': float(mean_ori_distance),
            'std_orientation_distance': float(std_ori_distance),
            'pose_indices': pose_indices,
            'per_object_stats': per_object_stats
        }
        
        print(f"Skill: {skill_name}")
        print(f"  Num target objects: {len(pose_indices)}")
        print(f"  Mean pairwise distance: {mean_distance:.4f} ± {std_distance:.4f}")
        print(f"  Mean position distance: {mean_pos_distance:.4f} ± {std_pos_distance:.4f}")
        print(f"  Mean orientation distance: {mean_ori_distance:.4f} ± {std_ori_distance:.4f}")
        if per_object_stats:
            print(f"  Per-object diversity:")
            for obj_name, obj_stats in per_object_stats.items():
                print(f"    Object: {obj_name}")
                print(f"      Num poses: {obj_stats['num_poses']}")
                print(f"      Mean pairwise distance: {obj_stats['mean_pairwise_distance']:.4f} ± {obj_stats['std_pairwise_distance']:.4f}")
                print(f"      Mean position distance: {obj_stats['mean_position_distance']:.4f}")
                print(f"      Mean orientation distance: {obj_stats['mean_orientation_distance']:.4f}")
        print()
    
    # Compute overall statistics
    all_distances = []
    all_pos_distances = []
    all_ori_distances = []
    
    for skill_name, stats in diversity_stats.items():
        pose_indices = stats['pose_indices']
        skill_poses = target_object_poses[pose_indices]
        
        if len(skill_poses) >= 2:
            distances = pdist(skill_poses, metric='euclidean')
            pos_distances = pdist(skill_poses[:, :3], metric='euclidean')
            ori_distances = pdist(skill_poses[:, 3:], metric='euclidean')
            
            all_distances.extend(distances)
            all_pos_distances.extend(pos_distances)
            all_ori_distances.extend(ori_distances)
    
    overall_stats = {
        'total_num_target_objects': len(target_object_poses),
        'total_num_skills': len(diversity_stats),
        'overall_mean_pairwise_distance': float(np.mean(all_distances)) if all_distances else 0.0,
        'overall_std_pairwise_distance': float(np.std(all_distances)) if all_distances else 0.0,
        'overall_mean_position_distance': float(np.mean(all_pos_distances)) if all_pos_distances else 0.0,
        'overall_std_position_distance': float(np.std(all_pos_distances)) if all_pos_distances else 0.0,
        'overall_mean_orientation_distance': float(np.mean(all_ori_distances)) if all_ori_distances else 0.0,
        'overall_std_orientation_distance': float(np.std(all_ori_distances)) if all_ori_distances else 0.0,
    }
    
    # Compute overall average position and orientation (not just pairwise distances)
    overall_mean_position = np.mean(target_object_poses[:, :3], axis=0)
    overall_mean_orientation = np.mean(target_object_poses[:, 3:], axis=0)
    overall_stats['overall_mean_position'] = overall_mean_position.tolist()
    overall_stats['overall_mean_orientation'] = overall_mean_orientation.tolist()
    print(f"  Overall mean position: {overall_mean_position}")
    print(f"  Overall mean orientation: {overall_mean_orientation}")
    # Compute overall object-wise mean position/orientation distances
    if all_obj_mean_pos_dists:
        objwise_mean_pos_dist = np.mean(all_obj_mean_pos_dists)
        objwise_mean_ori_dist = np.mean(all_obj_mean_ori_dists)
    else:
        objwise_mean_pos_dist = 0.0
        objwise_mean_ori_dist = 0.0
    overall_stats['overall_objectwise_mean_position_distance'] = float(objwise_mean_pos_dist)
    overall_stats['overall_objectwise_mean_orientation_distance'] = float(objwise_mean_ori_dist)
    print(f"  Overall object-wise mean position distance: {objwise_mean_pos_dist:.4f}")
    print(f"  Overall object-wise mean orientation distance: {objwise_mean_ori_dist:.4f}")
    
    # Compute mean of object mean positions
    if object_mean_positions:
        object_mean_positions_arr = np.stack(object_mean_positions)
        mean_of_object_mean_positions = np.mean(object_mean_positions_arr, axis=0)
        std_of_object_mean_positions = np.std(object_mean_positions_arr, axis=0)
    else:
        mean_of_object_mean_positions = np.zeros(3)
        std_of_object_mean_positions = np.zeros(3)
    overall_stats['mean_of_object_mean_positions'] = mean_of_object_mean_positions.tolist()
    overall_stats['std_of_object_mean_positions'] = std_of_object_mean_positions.tolist()
    print(f"  Mean of object mean positions: {mean_of_object_mean_positions}")
    print(f"  Std of object mean positions: {std_of_object_mean_positions}")
    
    # Create results directory
    results_dir = "runs/eval_results/vlm_pose_generator/eval_obj_dist_metric"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results
    results = {
        'overall_stats': overall_stats,
        'per_skill_stats': diversity_stats,
        'metadata': {
            'data_root': data_root,
            'poses_path': poses_path,
            'csv_path': csv_path
        }
    }
    
    output_path = os.path.join(results_dir, "target_object_diversity.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save as CSV for easier analysis
    csv_data = []
    for skill_name, stats in diversity_stats.items():
        csv_data.append({
            'skill_name': skill_name,
            'num_target_objects': stats['num_target_objects'],
            'mean_pairwise_distance': stats['mean_pairwise_distance'],
            'std_pairwise_distance': stats['std_pairwise_distance'],
            'min_pairwise_distance': stats['min_pairwise_distance'],
            'max_pairwise_distance': stats['max_pairwise_distance'],
            'mean_position_distance': stats['mean_position_distance'],
            'std_position_distance': stats['std_position_distance'],
            'mean_orientation_distance': stats['mean_orientation_distance'],
            'std_orientation_distance': stats['std_orientation_distance'],
        })
    
    csv_df = pd.DataFrame(csv_data)
    csv_output_path = os.path.join(results_dir, "target_object_diversity.csv")
    csv_df.to_csv(csv_output_path, index=False)
    
    print(f"\nDiversity computation complete!")
    print(f"Results saved to: {output_path}")
    print(f"CSV results saved to: {csv_output_path}")
    print(f"\nOverall statistics:")
    print(f"  Total target objects: {overall_stats['total_num_target_objects']}")
    print(f"  Total skills: {overall_stats['total_num_skills']}")
    print(f"  Overall mean pairwise distance: {overall_stats['overall_mean_pairwise_distance']:.4f} ± {overall_stats['overall_std_pairwise_distance']:.4f}")
    
    # Save printed output to a file
    sys.stdout = orig_stdout
    output_txt_path = os.path.join(results_dir, "target_object_diversity.txt")
    with open(output_txt_path, 'w') as f:
        f.write(output_capture.getvalue())
    print(f"\nPrinted output also saved to: {output_txt_path}")
    
    return results


def main():
    """Main function."""
    compute_target_object_diversity()


if __name__ == "__main__":
    main() 