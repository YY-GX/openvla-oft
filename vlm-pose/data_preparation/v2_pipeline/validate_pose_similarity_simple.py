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

"""
Simple Pose Similarity Validation (without seaborn)
This script validates pose similarity within skills and generates statistics.
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
        
        # Generate basic plots
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
        """Generate basic visualization plots."""
        try:
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
            
            print("Generated plots successfully!")
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")


def main():
    parser = argparse.ArgumentParser(description="Validate pose similarity within skills.")
    parser.add_argument("--poses_file", type=str,
                       default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/local_pairs_datasets_v2/local_poses.npy",
                       help="Path to poses file")
    parser.add_argument("--annotation_file", type=str,
                       default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/local_pairs_datasets_v2/annotation.csv",
                       help="Path to annotation file")
    parser.add_argument("--output_dir", type=str,
                       default="pose_similarity_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.poses_file):
        print(f"Error: Poses file not found: {args.poses_file}")
        return
    
    if not os.path.exists(args.annotation_file):
        print(f"Error: Annotation file not found: {args.annotation_file}")
        return
    
    # Run statistics analysis
    print("Running Pose Similarity Statistics Analysis...")
    analyzer = PoseStatisticsAnalyzer(args.poses_file, args.annotation_file)
    analyzer.generate_report(args.output_dir)


if __name__ == "__main__":
    main()