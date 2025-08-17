import sys
sys.path.append('../../..')

import h5py
import numpy as np
import os
import json
import argparse
import glob
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

"""
Pre-Analysis Script for Pose Similarity
This script directly extracts poses from demonstrations using contact_timesteps.json
and calculates pose similarity statistics within each skill.

Key Features:
- Loads contact timesteps from contact_timesteps.json
- Extracts poses at fixed window relative to contact (default: [-4, -3, -2])
- Calculates pose similarity statistics within each skill
- Generates visualizations and reports
- No need for pre-extracted pose files
"""

def load_hdf5_to_dict(file_path):
    """Load HDF5 file into a dictionary recursively."""
    def recursively_extract(group):
        result = {}
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Dataset):
                result[key] = item[()]
            elif isinstance(item, h5py.Group):
                result[key] = recursively_extract(item)
        return result

    with h5py.File(file_path, 'r') as file:
        return recursively_extract(file)

def load_contact_timesteps(contact_file):
    """Load pre-computed contact timesteps from JSON file."""
    with open(contact_file, 'r') as f:
        return json.load(f)

def load_single_skill_tasks(tasks_file):
    """Load list of single skill task names."""
    with open(tasks_file, 'r') as f:
        return json.load(f)

def match_task_name_to_file(task_name, demo_files):
    """
    Match base task name to actual demo file.
    
    Args:
        task_name: Base task name from single_skill_tasks_44.json
        demo_files: List of demo file paths
        
    Returns:
        str: Matched demo file path, or None if no match found
    """
    # Try exact match with _demo.hdf5 suffix
    expected_filename = f"{task_name}_demo.hdf5"
    for file_path in demo_files:
        if os.path.basename(file_path) == expected_filename:
            return file_path
    
    # Try partial match
    for file_path in demo_files:
        if task_name in os.path.basename(file_path):
            return file_path
    
    return None

def extract_contact_based_poses(demo, contact_timestep, window_offsets=[-4, -3, -2]):
    """
    Extract poses at fixed window relative to contact timestep.
    
    Args:
        demo: Dictionary containing demonstration data
        contact_timestep: Timestep when contact occurs
        window_offsets: List of offsets relative to contact (e.g., [-4, -3, -2])
        
    Returns:
        List of poses (N x 6 arrays)
    """
    ee_pos = demo['obs']['ee_pos']
    ee_ori = demo['obs']['ee_ori']
    
    poses = []
    for offset in window_offsets:
        actual_timestep = contact_timestep + offset  # offset is negative, so this goes back in time
        
        # Ensure timestep is within bounds
        if actual_timestep < 0:
            # Use first available pose if going before start
            actual_timestep = 0
        elif actual_timestep >= len(ee_pos):
            # Use last available pose if going beyond end
            actual_timestep = len(ee_pos) - 1
        
        pose = np.concatenate([ee_pos[actual_timestep], ee_ori[actual_timestep]])
        poses.append(pose)
    
    return poses

def parse_language_description(task_name):
    """
    Parse task name to extract language description.
    
    Args:
        task_name: Task name like "KITCHEN_SCENE1_open_the_drawer"
        
    Returns:
        str: Language description
    """
    # Remove scene prefix
    if task_name.startswith("KITCHEN_SCENE"):
        parts = task_name.split("_", 2)  # Split into ["KITCHEN", "SCENE1", "rest..."]
        if len(parts) >= 3:
            description_parts = parts[2:]
        else:
            description_parts = parts
    elif task_name.startswith("LIVING_ROOM_SCENE"):
        parts = task_name.split("_", 3)  # Split into ["LIVING", "ROOM", "SCENE1", "rest..."]
        if len(parts) >= 4:
            description_parts = parts[3:]
        else:
            description_parts = parts
    else:
        description_parts = parts
    
    return ' '.join(description_parts)

def compute_pose_distances(poses):
    """
    Compute pairwise distances between poses.
    
    Args:
        poses: Array of poses (N x 6)
        
    Returns:
        dict: Contains position distances, orientation distances, and combined distances
    """
    n_poses = len(poses)
    if n_poses < 2:
        return {
            'position_distances': np.array([]),
            'orientation_distances': np.array([]),
            'combined_distances': np.array([])
        }
    
    position_distances = np.zeros((n_poses, n_poses))
    orientation_distances = np.zeros((n_poses, n_poses))
    combined_distances = np.zeros((n_poses, n_poses))
    
    for i in range(n_poses):
        for j in range(i+1, n_poses):
            # Position distance (first 3 elements)
            pos_dist = np.linalg.norm(poses[i][:3] - poses[j][:3])
            position_distances[i, j] = position_distances[j, i] = pos_dist
            
            # Orientation distance (last 3 elements)
            ori_dist = np.linalg.norm(poses[i][3:] - poses[j][3:])
            orientation_distances[i, j] = orientation_distances[j, i] = ori_dist
            
            # Combined distance (weighted sum)
            combined_dist = pos_dist + 0.1 * ori_dist  # Weight orientation less
            combined_distances[i, j] = combined_distances[j, i] = combined_dist
    
    return {
        'position_distances': position_distances,
        'orientation_distances': orientation_distances,
        'combined_distances': combined_distances
    }

def analyze_skill_poses(skill_poses):
    """
    Analyze poses for a specific skill.
    
    Args:
        skill_poses: List of poses for this skill
        
    Returns:
        dict: Statistics for this skill
    """
    if len(skill_poses) < 2:
        return {
            'n_poses': len(skill_poses),
            'error': 'Too few poses for analysis'
        }
    
    # Convert to numpy array
    poses_array = np.array(skill_poses)
    
    # Compute distances
    distances = compute_pose_distances(poses_array)
    
    # Extract upper triangular part (exclude diagonal)
    def get_upper_tri(matrix):
        if matrix.size == 0:
            return np.array([])
        return matrix[np.triu_indices_from(matrix, k=1)]
    
    pos_dists = get_upper_tri(distances['position_distances'])
    ori_dists = get_upper_tri(distances['orientation_distances'])
    combined_dists = get_upper_tri(distances['combined_distances'])
    
    if len(pos_dists) == 0:
        return {
            'n_poses': len(skill_poses),
            'error': 'No valid distance pairs'
        }
    
    # Compute statistics
    stats = {
        'n_poses': len(skill_poses),
        
        # Position statistics
        'position_mean': float(np.mean(pos_dists)),
        'position_std': float(np.std(pos_dists)),
        'position_min': float(np.min(pos_dists)),
        'position_max': float(np.max(pos_dists)),
        'position_median': float(np.median(pos_dists)),
        
        # Orientation statistics
        'orientation_mean': float(np.mean(ori_dists)),
        'orientation_std': float(np.std(ori_dists)),
        'orientation_min': float(np.min(ori_dists)),
        'orientation_max': float(np.max(ori_dists)),
        'orientation_median': float(np.median(ori_dists)),
        
        # Combined statistics
        'combined_mean': float(np.mean(combined_dists)),
        'combined_std': float(np.std(combined_dists)),
        'combined_min': float(np.min(combined_dists)),
        'combined_max': float(np.max(combined_dists)),
        'combined_median': float(np.median(combined_dists)),
    }
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Pre-analysis: Extract poses and calculate similarity statistics.")
    parser.add_argument("--raw_demo_dir", type=str, 
                       default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/libero_90_no_noops/",
                       help="Directory containing raw demo HDF5 files")
    parser.add_argument("--contact_timesteps_file", type=str,
                       default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/vlm-pose/data_preparation/v2_pipeline/contact_timesteps.json",
                       help="JSON file containing contact timesteps")
    parser.add_argument("--single_skill_tasks_file", type=str,
                       default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/vlm-pose/data_preparation/v2_pipeline/single_skill_tasks_44.json",
                       help="JSON file containing single skill task names")
    parser.add_argument("--output_dir", type=str,
                       default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/vlm-pose/data_preparation/v2_pipeline/pre_analysis_results",
                       help="Output directory for results")
    parser.add_argument("--window_offsets", nargs='+', type=int, default=[-4, -3, -2],
                       help="Timestep offsets relative to contact (e.g., -4 -3 -2)")
    args = parser.parse_args()

    # Validate window offsets (should all be negative for pre-contact)
    if any(offset >= 0 for offset in args.window_offsets):
        print("Warning: Some window offsets are non-negative. This may extract poses at/after contact.")

    # Load required data
    contact_timesteps = load_contact_timesteps(args.contact_timesteps_file)
    single_skill_tasks = load_single_skill_tasks(args.single_skill_tasks_file)
    print(f"Loaded contact timesteps for {len(contact_timesteps)} tasks")
    print(f"Loaded {len(single_skill_tasks)} single skill tasks")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get all demo files
    demo_files = sorted(glob.glob(os.path.join(args.raw_demo_dir, "*.hdf5")))
    print(f"Found {len(demo_files)} demo files")

    # Initialize data structures
    skill_poses = defaultdict(list)  # skill_name -> list of poses
    skill_stats = {}  # skill_name -> statistics
    processed_tasks = 0
    processed_demos = 0
    total_poses = 0

    # Process each task
    for task_name in tqdm(contact_timesteps.keys(), desc="Processing tasks"):
        if task_name not in single_skill_tasks:
            continue
        
        # Find matching demo file
        file_path = match_task_name_to_file(task_name, demo_files)
        if file_path is None:
            print(f"No demo file found for task: {task_name}")
            continue
        
        print(f"Processing task: {task_name} -> {os.path.basename(file_path)}")
        
        # Load demo data
        try:
            data_dict = load_hdf5_to_dict(file_path)
        except Exception as e:
            print(f"Failed to load {file_path}: {str(e)}")
            continue
        
        # Get language description for this task
        language_description = parse_language_description(task_name)
        
        # Get contact timesteps for this task
        task_contact_timesteps = contact_timesteps[task_name]
        
        # Process each demo that has a contact timestep
        for demo_key in tqdm(task_contact_timesteps.keys(), 
                           desc=f"Processing {os.path.basename(file_path)}", 
                           leave=False):
            
            demo = data_dict['data'][demo_key]
            contact_timestep = task_contact_timesteps[demo_key]
            
            # Extract poses at contact-based window
            poses = extract_contact_based_poses(demo, contact_timestep, args.window_offsets)
            
            # Add poses to skill collection
            skill_poses[language_description].extend(poses)
            total_poses += len(poses)
            
            processed_demos += 1
        
        processed_tasks += 1

    print(f"\nExtracted {total_poses} poses from {processed_demos} demos across {processed_tasks} tasks")

    # Analyze poses for each skill
    print(f"\nAnalyzing pose similarity for {len(skill_poses)} skills...")
    
    all_stats = []
    for skill_name, poses in tqdm(skill_poses.items(), desc="Analyzing skills"):
        stats = analyze_skill_poses(poses)
        stats['skill_name'] = skill_name
        stats['n_demos'] = len(poses) // len(args.window_offsets)  # Each demo contributes len(window_offsets) poses
        
        if 'error' not in stats:
            all_stats.append(stats)
            skill_stats[skill_name] = stats
        else:
            print(f"Warning: {skill_name} - {stats['error']}")

    # Create summary statistics
    if all_stats:
        summary = {
            'total_skills': len(skill_poses),
            'valid_skills': len(all_stats),
            'total_poses': int(np.sum([s['n_poses'] for s in all_stats])),
            'total_demos': int(np.sum([s['n_demos'] for s in all_stats])),
            
            # Overall position statistics
            'overall_position_mean': float(np.mean([s['position_mean'] for s in all_stats])),
            'overall_position_std': float(np.mean([s['position_std'] for s in all_stats])),
            
            # Overall orientation statistics
            'overall_orientation_mean': float(np.mean([s['orientation_mean'] for s in all_stats])),
            'overall_orientation_std': float(np.mean([s['orientation_std'] for s in all_stats])),
            
            # Overall combined statistics
            'overall_combined_mean': float(np.mean([s['combined_mean'] for s in all_stats])),
            'overall_combined_std': float(np.mean([s['combined_std'] for s in all_stats])),
        }
        
        # Save detailed statistics
        stats_file = os.path.join(args.output_dir, 'pose_similarity_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump({
                'summary': summary,
                'per_skill_stats': all_stats,
                'extraction_params': {
                    'window_offsets': args.window_offsets,
                    'processed_tasks': processed_tasks,
                    'processed_demos': processed_demos,
                    'total_poses': total_poses
                }
            }, f, indent=2)
        
        # Save CSV for easy analysis
        stats_df = pd.DataFrame(all_stats)
        csv_file = os.path.join(args.output_dir, 'pose_similarity_statistics.csv')
        stats_df.to_csv(csv_file, index=False)
        
        # Generate plots
        generate_plots(all_stats, args.output_dir)
        
        # Print summary
        print(f"\n=== Pose Similarity Analysis Summary ===")
        print(f"Total skills analyzed: {summary['total_skills']}")
        print(f"Valid skills: {summary['valid_skills']}")
        print(f"Total poses: {summary['total_poses']}")
        print(f"Total demos: {summary['total_demos']}")
        print(f"\nPosition distances:")
        print(f"  Mean: {summary['overall_position_mean']:.4f}")
        print(f"  Std:  {summary['overall_position_std']:.4f}")
        print(f"\nOrientation distances:")
        print(f"  Mean: {summary['overall_orientation_mean']:.4f}")
        print(f"  Std:  {summary['overall_orientation_std']:.4f}")
        print(f"\nCombined distances:")
        print(f"  Mean: {summary['overall_combined_mean']:.4f}")
        print(f"  Std:  {summary['overall_combined_std']:.4f}")
        print(f"\nReports saved to: {args.output_dir}")
    else:
        print("No valid statistics generated!")

def generate_plots(stats, output_dir):
    """Generate visualization plots."""
    plt.style.use('default')
    
    # Plot 1: Distribution of position distances
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    pos_means = [s['position_mean'] for s in stats]
    plt.hist(pos_means, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Mean Position Distance')
    plt.ylabel('Number of Skills')
    plt.title('Distribution of Mean Position Distances Across Skills')
    
    plt.subplot(2, 2, 2)
    ori_means = [s['orientation_mean'] for s in stats]
    plt.hist(ori_means, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Mean Orientation Distance')
    plt.ylabel('Number of Skills')
    plt.title('Distribution of Mean Orientation Distances Across Skills')
    
    plt.subplot(2, 2, 3)
    combined_means = [s['combined_mean'] for s in stats]
    plt.hist(combined_means, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Mean Combined Distance')
    plt.ylabel('Number of Skills')
    plt.title('Distribution of Mean Combined Distances Across Skills')
    
    plt.subplot(2, 2, 4)
    n_poses = [s['n_poses'] for s in stats]
    plt.hist(n_poses, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Poses per Skill')
    plt.ylabel('Number of Skills')
    plt.title('Distribution of Poses per Skill')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pose_similarity_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Scatter plot of position vs orientation distances
    plt.figure(figsize=(10, 6))
    plt.scatter(pos_means, ori_means, alpha=0.6)
    plt.xlabel('Mean Position Distance')
    plt.ylabel('Mean Orientation Distance')
    plt.title('Position vs Orientation Distance Relationship')
    plt.savefig(os.path.join(output_dir, 'position_vs_orientation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Top skills by consistency (lowest mean distances)
    plt.figure(figsize=(12, 8))
    
    # Sort by combined mean distance (lower is better)
    sorted_stats = sorted(stats, key=lambda x: x['combined_mean'])
    top_skills = sorted_stats[:10]  # Top 10 most consistent skills
    
    skill_names = [s['skill_name'][:30] + '...' if len(s['skill_name']) > 30 else s['skill_name'] 
                   for s in top_skills]
    combined_means = [s['combined_mean'] for s in top_skills]
    
    plt.barh(range(len(skill_names)), combined_means)
    plt.yticks(range(len(skill_names)), skill_names)
    plt.xlabel('Mean Combined Distance (Lower = More Consistent)')
    plt.title('Top 10 Most Consistent Skills')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_consistent_skills.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 