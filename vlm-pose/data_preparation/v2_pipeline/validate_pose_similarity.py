import sys
sys.path.append('../../..')

import numpy as np
import pandas as pd
import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

"""
Pose Similarity Validation
This script validates pose similarity within skills and optionally filters out inconsistent poses.

Two modes:
1. Statistics Mode: Calculate and report detailed statistics about pose similarity within skills
2. Filtering Mode: Filter out poses that are too different from skill average
"""

class PoseStatisticsAnalyzer:
    """Analyzes pose statistics within skills without filtering."""
    
    def __init__(self, poses_file, annotation_file):
        """
        Initialize analyzer with pose and annotation data.
        
        Args:
            poses_file: Path to local_poses.npy
            annotation_file: Path to annotation.csv
        """
        self.poses = np.load(poses_file)
        self.annotations = pd.read_csv(annotation_file)
        self.pose_stats = {}
        
    def compute_pose_distances(self, poses):
        """
        Compute pairwise distances between poses.
        
        Args:
            poses: Array of poses (N x 6)
            
        Returns:
            dict: Contains position distances, orientation distances, and combined distances
        """
        n_poses = len(poses)
        position_distances = np.zeros((n_poses, n_poses))
        orientation_distances = np.zeros((n_poses, n_poses))
        combined_distances = np.zeros((n_poses, n_poses))
        
        for i in range(n_poses):
            for j in range(i+1, n_poses):
                # Position distance (first 3 elements)
                pos_dist = np.linalg.norm(poses[i][:3] - poses[j][:3])
                position_distances[i, j] = position_distances[j, i] = pos_dist
                
                # Orientation distance (last 3 elements, using angle between quaternions)
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
    
    def analyze_skill_poses(self, skill_name, skill_annotations):
        """
        Analyze poses for a specific skill.
        
        Args:
            skill_name: Name of the skill
            skill_annotations: Annotations for this skill
            
        Returns:
            dict: Statistics for this skill
        """
        # Get unique pose indices for this skill
        pose_indices = skill_annotations['ee_pose_idx'].unique()
        skill_poses = self.poses[pose_indices]
        
        if len(skill_poses) < 2:
            return {
                'n_poses': len(skill_poses),
                'error': 'Too few poses for analysis'
            }
        
        # Compute distances
        distances = self.compute_pose_distances(skill_poses)
        
        # Extract upper triangular part (exclude diagonal)
        def get_upper_tri(matrix):
            return matrix[np.triu_indices_from(matrix, k=1)]
        
        pos_dists = get_upper_tri(distances['position_distances'])
        ori_dists = get_upper_tri(distances['orientation_distances'])
        combined_dists = get_upper_tri(distances['combined_distances'])
        
        # Compute statistics
        stats = {
            'skill_name': skill_name,
            'n_poses': len(skill_poses),
            'n_pairs': len(skill_annotations),
            'n_demos': skill_annotations['source_demo_idx'].nunique(),
            
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
    
    def analyze_all_skills(self):
        """Analyze poses for all skills."""
        # Group by language description (skill)
        grouped = self.annotations.groupby('language_description')
        
        print(f"Analyzing {len(grouped)} skills...")
        
        all_stats = []
        for skill_name, skill_annotations in tqdm(grouped, desc="Analyzing skills"):
            stats = self.analyze_skill_poses(skill_name, skill_annotations)
            all_stats.append(stats)
            self.pose_stats[skill_name] = stats
        
        return all_stats
    
    def generate_report(self, output_dir):
        """Generate detailed statistics report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze all skills
        all_stats = self.analyze_all_skills()
        
        # Filter out error cases
        valid_stats = [s for s in all_stats if 'error' not in s]
        
        # Create summary statistics
        summary = {
            'total_skills': len(all_stats),
            'valid_skills': len(valid_stats),
            'total_poses': int(np.sum([s['n_poses'] for s in valid_stats])),
            'total_pairs': int(np.sum([s['n_pairs'] for s in valid_stats])),
            
            # Overall position statistics
            'overall_position_mean': float(np.mean([s['position_mean'] for s in valid_stats])),
            'overall_position_std': float(np.mean([s['position_std'] for s in valid_stats])),
            
            # Overall orientation statistics
            'overall_orientation_mean': float(np.mean([s['orientation_mean'] for s in valid_stats])),
            'overall_orientation_std': float(np.mean([s['orientation_std'] for s in valid_stats])),
            
            # Overall combined statistics
            'overall_combined_mean': float(np.mean([s['combined_mean'] for s in valid_stats])),
            'overall_combined_std': float(np.mean([s['combined_std'] for s in valid_stats])),
        }
        
        # Save detailed statistics
        stats_file = os.path.join(output_dir, 'pose_similarity_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump({
                'summary': summary,
                'per_skill_stats': valid_stats
            }, f, indent=2)
        
        # Save CSV for easy analysis
        stats_df = pd.DataFrame(valid_stats)
        csv_file = os.path.join(output_dir, 'pose_similarity_statistics.csv')
        stats_df.to_csv(csv_file, index=False)
        
        # Generate plots
        self._generate_plots(valid_stats, output_dir)
        
        # Print summary
        print(f"\n=== Pose Similarity Statistics Report ===")
        print(f"Total skills analyzed: {summary['total_skills']}")
        print(f"Valid skills: {summary['valid_skills']}")
        print(f"Total poses: {summary['total_poses']}")
        print(f"Total pairs: {summary['total_pairs']}")
        print(f"\nPosition distances:")
        print(f"  Mean: {summary['overall_position_mean']:.4f}")
        print(f"  Std:  {summary['overall_position_std']:.4f}")
        print(f"\nOrientation distances:")
        print(f"  Mean: {summary['overall_orientation_mean']:.4f}")
        print(f"  Std:  {summary['overall_orientation_std']:.4f}")
        print(f"\nCombined distances:")
        print(f"  Mean: {summary['overall_combined_mean']:.4f}")
        print(f"  Std:  {summary['overall_combined_std']:.4f}")
        print(f"\nReports saved to: {output_dir}")
    
    def _generate_plots(self, stats, output_dir):
        """Generate visualization plots."""
        plt.style.use('seaborn-v0_8')
        
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


class PoseFilteringProcessor:
    """Filters poses based on similarity thresholds."""
    
    def __init__(self, poses_file, annotation_file):
        """
        Initialize processor with pose and annotation data.
        
        Args:
            poses_file: Path to local_poses.npy
            annotation_file: Path to annotation.csv
        """
        self.poses = np.load(poses_file)
        self.annotations = pd.read_csv(annotation_file)
        
    def filter_poses_by_similarity(self, position_threshold=0.1, orientation_threshold=0.5, combined_threshold=0.15):
        """
        Filter out poses that are too different from skill average.
        
        Args:
            position_threshold: Maximum allowed position distance from skill center
            orientation_threshold: Maximum allowed orientation distance from skill center
            combined_threshold: Maximum allowed combined distance from skill center
            
        Returns:
            tuple: (filtered_poses, filtered_annotations, filtering_stats)
        """
        grouped = self.annotations.groupby('language_description')
        
        keep_pose_indices = set()
        keep_pair_indices = []
        filtering_stats = defaultdict(dict)
        
        print(f"Filtering {len(grouped)} skills...")
        
        for skill_name, skill_annotations in tqdm(grouped, desc="Filtering skills"):
            # Get unique pose indices for this skill
            pose_indices = skill_annotations['ee_pose_idx'].unique()
            skill_poses = self.poses[pose_indices]
            
            if len(skill_poses) < 2:
                # Keep all poses if too few for filtering
                keep_pose_indices.update(pose_indices)
                keep_pair_indices.extend(skill_annotations.index.tolist())
                filtering_stats[skill_name] = {
                    'original_poses': len(skill_poses),
                    'kept_poses': len(skill_poses),
                    'filtered_poses': 0,
                    'reason': 'too_few_poses'
                }
                continue
            
            # Compute skill center (mean pose)
            skill_center = np.mean(skill_poses, axis=0)
            
            # Compute distances from center
            position_distances = np.linalg.norm(skill_poses[:, :3] - skill_center[:3], axis=1)
            orientation_distances = np.linalg.norm(skill_poses[:, 3:] - skill_center[3:], axis=1)
            combined_distances = position_distances + 0.1 * orientation_distances
            
            # Apply filtering thresholds
            position_mask = position_distances <= position_threshold
            orientation_mask = orientation_distances <= orientation_threshold
            combined_mask = combined_distances <= combined_threshold
            
            # Combine all masks (pose must satisfy all criteria)
            final_mask = position_mask & orientation_mask & combined_mask
            
            # Keep poses that pass the filter
            kept_pose_indices = pose_indices[final_mask]
            keep_pose_indices.update(kept_pose_indices)
            
            # Keep corresponding pairs
            kept_pairs_mask = skill_annotations['ee_pose_idx'].isin(kept_pose_indices)
            keep_pair_indices.extend(skill_annotations[kept_pairs_mask].index.tolist())
            
            # Record filtering statistics
            filtering_stats[skill_name] = {
                'original_poses': len(skill_poses),
                'kept_poses': len(kept_pose_indices),
                'filtered_poses': len(skill_poses) - len(kept_pose_indices),
                'position_failures': np.sum(~position_mask),
                'orientation_failures': np.sum(~orientation_mask),
                'combined_failures': np.sum(~combined_mask),
                'position_threshold': position_threshold,
                'orientation_threshold': orientation_threshold,
                'combined_threshold': combined_threshold
            }
        
        # Create filtered datasets
        filtered_poses = self.poses[sorted(keep_pose_indices)]
        filtered_annotations = self.annotations.iloc[keep_pair_indices].copy()
        
        # Update pose indices in annotations to account for filtering
        old_to_new_pose_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(keep_pose_indices))}
        filtered_annotations['ee_pose_idx'] = filtered_annotations['ee_pose_idx'].map(old_to_new_pose_idx)
        
        return filtered_poses, filtered_annotations, dict(filtering_stats)
    
    def save_filtered_data(self, filtered_poses, filtered_annotations, output_dir):
        """Save filtered poses and annotations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save poses
        poses_file = os.path.join(output_dir, 'local_poses_filtered.npy')
        np.save(poses_file, filtered_poses)
        
        # Save annotations
        annotations_file = os.path.join(output_dir, 'annotation_filtered.csv')
        filtered_annotations.to_csv(annotations_file, index=False)
        
        print(f"Filtered data saved to: {output_dir}")
        print(f"  Poses: {poses_file} ({len(filtered_poses)} poses)")
        print(f"  Annotations: {annotations_file} ({len(filtered_annotations)} pairs)")


def main():
    parser = argparse.ArgumentParser(description="Validate pose similarity within skills.")
    parser.add_argument("--poses_file", type=str,
                       default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/local_pairs_datasets_v2/local_poses.npy",
                       help="Path to poses file")
    parser.add_argument("--annotation_file", type=str,
                       default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/local_pairs_datasets_v2/annotation.csv",
                       help="Path to annotation file")
    parser.add_argument("--mode", type=str, choices=['statistics', 'filtering'], default='statistics',
                       help="Mode: 'statistics' for analysis only, 'filtering' for filtering poses")
    parser.add_argument("--output_dir", type=str,
                       default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/vlm-pose/data_preparation/v2_pipeline/pose_similarity_results",
                       help="Output directory for results")
    
    # Filtering parameters
    parser.add_argument("--position_threshold", type=float, default=0.1,
                       help="Maximum allowed position distance from skill center")
    parser.add_argument("--orientation_threshold", type=float, default=0.5,
                       help="Maximum allowed orientation distance from skill center")
    parser.add_argument("--combined_threshold", type=float, default=0.15,
                       help="Maximum allowed combined distance from skill center")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.poses_file):
        print(f"Error: Poses file not found: {args.poses_file}")
        return
    
    if not os.path.exists(args.annotation_file):
        print(f"Error: Annotation file not found: {args.annotation_file}")
        return
    
    if args.mode == 'statistics':
        # Statistics mode
        print("Running in Statistics Mode...")
        analyzer = PoseStatisticsAnalyzer(args.poses_file, args.annotation_file)
        analyzer.generate_report(args.output_dir)
        
    else:
        # Filtering mode
        print("Running in Filtering Mode...")
        processor = PoseFilteringProcessor(args.poses_file, args.annotation_file)
        
        # First generate statistics for original data
        print("Analyzing original data...")
        analyzer = PoseStatisticsAnalyzer(args.poses_file, args.annotation_file)
        original_stats_dir = os.path.join(args.output_dir, 'original_statistics')
        analyzer.generate_report(original_stats_dir)
        
        # Apply filtering
        print("Applying pose filtering...")
        filtered_poses, filtered_annotations, filtering_stats = processor.filter_poses_by_similarity(
            args.position_threshold, args.orientation_threshold, args.combined_threshold
        )
        
        # Save filtered data
        filtered_data_dir = os.path.join(args.output_dir, 'filtered_data')
        processor.save_filtered_data(filtered_poses, filtered_annotations, filtered_data_dir)
        
        # Save filtering statistics
        filtering_stats_file = os.path.join(args.output_dir, 'filtering_statistics.json')
        with open(filtering_stats_file, 'w') as f:
            json.dump(filtering_stats, f, indent=2)
        
        # Generate statistics for filtered data
        print("Analyzing filtered data...")
        filtered_poses_file = os.path.join(filtered_data_dir, 'local_poses_filtered.npy')
        filtered_annotations_file = os.path.join(filtered_data_dir, 'annotation_filtered.csv')
        filtered_analyzer = PoseStatisticsAnalyzer(filtered_poses_file, filtered_annotations_file)
        filtered_stats_dir = os.path.join(args.output_dir, 'filtered_statistics')
        filtered_analyzer.generate_report(filtered_stats_dir)
        
        # Print filtering summary
        total_original_poses = len(np.load(args.poses_file))
        total_filtered_poses = len(filtered_poses)
        total_filtered_out = total_original_poses - total_filtered_poses
        
        print(f"\n=== Filtering Summary ===")
        print(f"Original poses: {total_original_poses}")
        print(f"Filtered poses: {total_filtered_poses}")
        print(f"Poses removed: {total_filtered_out} ({100*total_filtered_out/total_original_poses:.1f}%)")
        print(f"Filtering thresholds:")
        print(f"  Position: {args.position_threshold}")
        print(f"  Orientation: {args.orientation_threshold}")
        print(f"  Combined: {args.combined_threshold}")

if __name__ == "__main__":
    main()