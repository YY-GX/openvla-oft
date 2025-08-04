import os
import json
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import argparse
import random
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist

# LIBERO env creation (as in run_libero_eval.py)
from libero.libero import benchmark
import cv2
import pandas as pd
from experiments.robot.libero.libero_utils import get_libero_env

# IK tools
from utils.tracik_tools import solve_ik, pose6d_to_matrix

def save_env_image(env, save_path):
    # Try to use the public get_observation method if available
    if hasattr(env, 'get_observation'):
        obs = env.get_observation()
    else:
        obs = env.env._get_observations()
    img = obs["agentview_image"][::-1, ::-1]  # Rotate 180 degrees as in get_libero_image

    # Debug: print min, max, dtype, shape
    print(f"[DEBUG] Saving image: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")

    # If grayscale, convert to 3-channel
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    # Ensure image is uint8 for OpenCV
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    import cv2
    cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def calculate_pose_distance(pose1, pose2):
    """
    Calculate distance between two 6D poses (xyz + axis-angle).
    Returns position distance and orientation distance separately.
    """
    # Position distance (Euclidean)
    pos1, pos2 = pose1[:3], pose2[:3]
    position_distance = np.linalg.norm(pos1 - pos2)
    
    # Orientation distance (geodesic distance between rotations)
    rotvec1, rotvec2 = pose1[3:], pose2[3:]
    rot1 = R.from_rotvec(rotvec1)
    rot2 = R.from_rotvec(rotvec2)
    
    # Calculate geodesic distance between rotations
    rotation_diff = rot1.inv() * rot2
    orientation_distance = np.linalg.norm(rotation_diff.as_rotvec())
    
    # Total pose distance (weighted combination)
    total_distance = position_distance + 0.1 * orientation_distance  # Weight position more heavily
    
    return {
        'position_distance': position_distance,
        'orientation_distance': orientation_distance,
        'total_distance': total_distance
    }

def get_skill_poses(annotation_df, target_skill):
    """
    Get unique poses for a specific skill, organized by demo.
    Returns unique poses for visualization and first poses for statistics.
    """
    skill_df = annotation_df[annotation_df['task_name'] == target_skill]
    
    if len(skill_df) == 0:
        return None  # Return None if skill not found
    
    print(f"Found skill: {target_skill}")
    print(f"Total rows: {len(skill_df)}")
    print(f"Unique demos: {skill_df['source_demo_idx'].nunique()}")
    print(f"Unique poses: {skill_df['ee_pose_idx'].nunique()}")
    
    # Get unique poses for visualization (one per unique ee_pose_idx)
    unique_poses = []
    unique_pose_indices = []
    unique_demo_names = []
    
    # Track which poses we've already seen
    seen_pose_indices = set()
    
    for idx, row in skill_df.iterrows():
        ee_pose_idx = row["ee_pose_idx"]
        demo_name = row["source_demo_idx"]
        
        # Only add if we haven't seen this pose index before
        if ee_pose_idx not in seen_pose_indices:
            unique_poses.append((idx, ee_pose_idx, demo_name))
            unique_pose_indices.append(ee_pose_idx)
            unique_demo_names.append(demo_name)
            seen_pose_indices.add(ee_pose_idx)
    
    # Get first pose per demo for statistics
    first_poses = []
    first_pose_indices = []
    
    demo_to_first_pose = {}
    for idx, row in skill_df.iterrows():
        demo_name = row["source_demo_idx"]
        ee_pose_idx = row["ee_pose_idx"]
        
        if demo_name not in demo_to_first_pose:
            demo_to_first_pose[demo_name] = (idx, ee_pose_idx)
    
    for demo_name, (idx, ee_pose_idx) in demo_to_first_pose.items():
        first_poses.append((idx, ee_pose_idx, demo_name))
        first_pose_indices.append(ee_pose_idx)
    
    print(f"Unique poses for visualization: {len(unique_poses)}")
    print(f"First poses for statistics: {len(first_poses)}")
    
    return {
        'unique_poses': unique_poses,  # Unique poses for visualization
        'first_poses': first_poses,  # First pose per demo for statistics
        'unique_pose_indices': unique_pose_indices,
        'first_pose_indices': first_pose_indices
    }

def calculate_pose_statistics(poses):
    """
    Calculate statistics for a set of poses.
    """
    n_poses = len(poses)
    if n_poses < 2:
        return None
    
    # Calculate all pairwise distances
    position_distances = []
    orientation_distances = []
    total_distances = []
    
    for i in range(n_poses):
        for j in range(i + 1, n_poses):
            dist = calculate_pose_distance(poses[i], poses[j])
            position_distances.append(dist['position_distance'])
            orientation_distances.append(dist['orientation_distance'])
            total_distances.append(dist['total_distance'])
    
    # Calculate statistics
    stats = {
        'num_poses': n_poses,
        'num_pairwise_comparisons': len(position_distances),
        'position_distance': {
            'mean': np.mean(position_distances),
            'std': np.std(position_distances),
            'min': np.min(position_distances),
            'max': np.max(position_distances),
            'median': np.median(position_distances)
        },
        'orientation_distance': {
            'mean': np.mean(orientation_distances),
            'std': np.std(orientation_distances),
            'min': np.min(orientation_distances),
            'max': np.max(orientation_distances),
            'median': np.median(orientation_distances)
        },
        'total_distance': {
            'mean': np.mean(total_distances),
            'std': np.std(total_distances),
            'min': np.min(total_distances),
            'max': np.max(total_distances),
            'median': np.median(total_distances)
        }
    }
    
    return stats

def process_single_skill(annotation_df, poses, target_skill, save_dir, benchmark_dict):
    """
    Process a single skill: calculate statistics and visualize poses.
    """
    # Get poses for the specific skill
    print(f"Getting poses for skill: {target_skill}")
    skill_data = get_skill_poses(annotation_df, target_skill)
    
    if skill_data is None:
        print(f"Skill '{target_skill}' not found in annotation, skipping...")
        return False
    
    # Extract poses for statistics (first pose per demo)
    print("Extracting first poses per demo for statistics...")
    first_skill_poses = []
    first_pose_indices = []
    
    for idx, ee_pose_idx, demo_name in skill_data['first_poses']:
        try:
            pose = poses[ee_pose_idx]
            first_skill_poses.append(pose)
            first_pose_indices.append(ee_pose_idx)
        except Exception as e:
            print(f"[Warning] Could not get pose for idx {idx}: {e}")
            continue
    
    print(f"Extracted {len(first_skill_poses)} first poses per demo")
    
    # Calculate pose statistics using first poses only
    print("Calculating pose statistics using first poses per demo...")
    pose_stats = calculate_pose_statistics(first_skill_poses)
    
    if pose_stats is None:
        print("Not enough poses to calculate statistics")
        return False
    
    # Print statistics
    print("\n=== POSE DIVERSITY STATISTICS ===")
    print(f"Number of poses: {pose_stats['num_poses']}")
    print(f"Number of pairwise comparisons: {pose_stats['num_pairwise_comparisons']}")
    print("\nPosition Distance (meters):")
    print(f"  Mean: {pose_stats['position_distance']['mean']:.4f}")
    print(f"  Std:  {pose_stats['position_distance']['std']:.4f}")
    print(f"  Min:  {pose_stats['position_distance']['min']:.4f}")
    print(f"  Max:  {pose_stats['position_distance']['max']:.4f}")
    print(f"  Median: {pose_stats['position_distance']['median']:.4f}")
    print("\nOrientation Distance (radians):")
    print(f"  Mean: {pose_stats['orientation_distance']['mean']:.4f}")
    print(f"  Std:  {pose_stats['orientation_distance']['std']:.4f}")
    print(f"  Min:  {pose_stats['orientation_distance']['min']:.4f}")
    print(f"  Max:  {pose_stats['orientation_distance']['max']:.4f}")
    print(f"  Median: {pose_stats['orientation_distance']['median']:.4f}")
    print("\nTotal Distance:")
    print(f"  Mean: {pose_stats['total_distance']['mean']:.4f}")
    print(f"  Std:  {pose_stats['total_distance']['std']:.4f}")
    print(f"  Min:  {pose_stats['total_distance']['min']:.4f}")
    print(f"  Max:  {pose_stats['total_distance']['max']:.4f}")
    print(f"  Median: {pose_stats['total_distance']['median']:.4f}")

    # Find the task id by name
    task_suite = benchmark_dict["libero_90"]()
    task_id = None
    for i in range(task_suite.n_tasks):
        if task_suite.get_task(i).name == target_skill:
            task_id = i
            break
    
    # Define the manual offset to apply to all poses (same as in visualize_vlm_pose_predictions.py)
    pose_offset = np.array([0.667, 0.0, -0.81, 0.0, 0.0, 0.0])
    
    if task_id is None:
        print(f"[Warning] Task {target_skill} not found in libero_90 suite.")
        print("Skipping visualization, but statistics are saved.")
        ik_success_count = 0
        ik_fail_count = 0
    else:
        task = task_suite.get_task(task_id)
        env, _ = get_libero_env(task, model_family="openvla", resolution=256)
        
        # Visualize all poses
        print("Visualizing all poses...")
        skill_name_for_file = target_skill.replace(" ", "_")
        ik_success_count = 0
        ik_fail_count = 0
        
        for i, (idx, pose_idx, demo_name) in enumerate(tqdm(skill_data['unique_poses'], desc="Visualizing poses")):
            try:
                pose = poses[pose_idx]
            except Exception as e:
                print(f"[Warning] Could not get pose for pose_idx {pose_idx}: {e}")
                ik_fail_count += 1
                continue
                
            # Apply offset to pose
            pose_offset_applied = pose + pose_offset
            
            # Convert to 4x4 matrix
            pose_mat = pose6d_to_matrix(pose_offset_applied)
            
            # Run IK
            joints = solve_ik(pose_mat)
            
            if joints is not None:
                # Set robot joints
                env.env.robots[0].set_robot_joint_positions(joints)
                env.env.sim.forward()  # Ensure simulation state is updated
                if hasattr(env.env, '_update_observables'):
                    env.env._update_observables(force=True)
                
                # Save image with skill_name + demo_idx + pose_idx naming
                img_name = f"{skill_name_for_file}_{demo_name}_pose_{pose_idx}.png"
                save_env_image(env, save_dir / img_name)
                ik_success_count += 1
            else:
                print(f"[Warning] IK failed for pose {i} (demo {demo_name}, pose_idx {pose_idx})")
                ik_fail_count += 1
        
        env.close()
        print(f"IK success: {ik_success_count}, IK fail: {ik_fail_count}")

    # Save statistics and metadata
    skill_name_for_file = target_skill.replace(" ", "_")
    metadata = {
        'skill_name': target_skill,
        'num_unique_poses': len(skill_data['unique_poses']),
        'num_first_poses': len(first_skill_poses),
        'unique_pose_indices': skill_data['unique_pose_indices'],
        'first_pose_indices': first_pose_indices,
        'ik_success_count': ik_success_count,
        'ik_fail_count': ik_fail_count,
        'pose_offset': pose_offset.tolist(),
        'statistics': pose_stats
    }
    
    # Save metadata
    metadata_path = save_dir / f"{skill_name_for_file}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save raw poses (both unique poses and first poses)
    unique_poses_path = save_dir / f"{skill_name_for_file}_unique_poses.npy"
    first_poses_path = save_dir / f"{skill_name_for_file}_first_poses.npy"
    
    # Extract actual pose arrays
    unique_pose_arrays = []
    for idx, pose_idx, demo_name in skill_data['unique_poses']:
        unique_pose_arrays.append(poses[pose_idx])
    
    first_pose_arrays = []
    for idx, pose_idx, demo_name in skill_data['first_poses']:
        first_pose_arrays.append(poses[pose_idx])
    
    np.save(unique_poses_path, np.array(unique_pose_arrays))
    np.save(first_poses_path, np.array(first_pose_arrays))
    
    print(f"\nResults saved to: {save_dir}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Unique poses saved to: {unique_poses_path}")
    print(f"First poses saved to: {first_poses_path}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and visualize pose diversity for all skills.")
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of dataset')
    parser.add_argument('--split', type=str, default='val', help='Dataset split (train/val/test)')
    parser.add_argument('--save_dir', type=str, default='runs/eval_results/skill_pose_diversity', help='Directory to save images and stats')
    parser.add_argument('--splits_folder', type=str, default='smaller_splits', help='Folder name for annotation splits')
    args = parser.parse_args()

    data_root = args.data_root
    split = args.split
    save_dir = Path(args.save_dir)
    splits_folder = args.splits_folder
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load annotation and poses
    print("Loading annotation and poses...")
    annotation_path = os.path.join(data_root, f"{split}_annotation.csv")
    if not os.path.exists(annotation_path):
        annotation_path = os.path.join(data_root, splits_folder, f"{split}_annotation.csv")
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found in {data_root} or {data_root}/{splits_folder}")
    
    annotation_df = pd.read_csv(annotation_path)
    poses = np.load(os.path.join(data_root, "local_poses.npy"))

    # 2. Get all unique skills in the dataset
    print("Getting all unique skills...")
    unique_skills = annotation_df['task_name'].unique()
    print(f"Found {len(unique_skills)} unique skills:")
    for i, skill in enumerate(unique_skills):
        print(f"  {i+1}. {skill}")

    # 3. Prepare LIBERO benchmark
    print("Preparing LIBERO environment...")
    benchmark_dict = benchmark.get_benchmark_dict()

    # 4. Process each skill
    print("\n" + "="*50)
    print("PROCESSING ALL SKILLS")
    print("="*50)
    
    successful_skills = []
    failed_skills = []
    
    for i, skill in enumerate(unique_skills):
        print(f"\n{'='*20} Processing Skill {i+1}/{len(unique_skills)}: {skill} {'='*20}")
        
        # Create subfolder for this skill
        skill_name_for_file = skill.replace(" ", "_")
        skill_save_dir = save_dir / skill_name_for_file
        skill_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Process the skill
        success = process_single_skill(annotation_df, poses, skill, skill_save_dir, benchmark_dict)
        
        if success:
            successful_skills.append(skill)
        else:
            failed_skills.append(skill)
        
        print(f"Completed skill {i+1}/{len(unique_skills)}: {skill}")
    
    # 5. Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total skills: {len(unique_skills)}")
    print(f"Successful: {len(successful_skills)}")
    print(f"Failed: {len(failed_skills)}")
    
    if successful_skills:
        print("\nSuccessful skills:")
        for skill in successful_skills:
            print(f"  ✓ {skill}")
    
    if failed_skills:
        print("\nFailed skills:")
        for skill in failed_skills:
            print(f"  ✗ {skill}")
    
    print(f"\nAll results saved to: {save_dir}")
    print("Analysis complete!") 