#!/usr/bin/env python3
"""
debug_pose_evaluation.py

Debug script to understand what's wrong with the pose evaluation.
"""

import sys
sys.path.append('..')

import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict

# Load annotation data
data_root = "datasets/local_pairs_datasets"
# annotation_path = os.path.join(data_root, "splits", "val_annotation.csv")
annotation_path = os.path.join(data_root, "debug_splits", "train_annotation.csv")
poses_path = os.path.join(data_root, "local_poses.npy")

print("Loading data...")
annotation_df = pd.read_csv(annotation_path)
poses = np.load(poses_path)

print(f"Annotation shape: {annotation_df.shape}")
print(f"Poses shape: {poses.shape}")

# Look at a few examples
print("\n=== SAMPLE DATA ===")
print("First 5 rows of annotation:")
print(annotation_df.head())

print(f"\nPose statistics:")
print(f"  Mean: {poses.mean(axis=0)}")
print(f"  Std: {poses.std(axis=0)}")
print(f"  Min: {poses.min(axis=0)}")
print(f"  Max: {poses.max(axis=0)}")

# Group by demo
demo_groups = defaultdict(list)
for _, row in annotation_df.iterrows():
    demo_key = (row['source_skill_file_name'], row['source_demo_idx'])
    demo_groups[demo_key].append(row.to_dict())

print(f"\nNumber of demos: {len(demo_groups)}")

# Analyze all demos
print(f"\n=== DEMO ANALYSIS ===")
demo_stats = []
for demo_key, demo_pairs in list(demo_groups.items())[:10]:  # Look at first 10 demos
    unique_pose_indices = list(set(pair['ee_pose_idx'] for pair in demo_pairs))
    unique_overview_indices = list(set(pair['overview_image_idx'] for pair in demo_pairs))
    
    demo_stats.append({
        'demo_key': demo_key,
        'num_pairs': len(demo_pairs),
        'num_unique_poses': len(unique_pose_indices),
        'num_unique_overviews': len(unique_overview_indices),
        'pose_indices': unique_pose_indices,
        'overview_indices': unique_overview_indices[:5],  # Just show first 5
    })

# Print stats for first few demos
for i, stats in enumerate(demo_stats):
    print(f"\nDemo {i+1}: {stats['demo_key']}")
    print(f"  Pairs: {stats['num_pairs']}")
    print(f"  Unique poses: {stats['num_unique_poses']} (indices: {stats['pose_indices']})")
    print(f"  Unique overviews: {stats['num_unique_overviews']} (first 5: {stats['overview_indices']})")

# Calculate averages
all_pose_counts = [stats['num_unique_poses'] for stats in demo_stats]
all_overview_counts = [stats['num_unique_overviews'] for stats in demo_stats]

print(f"\n=== SUMMARY STATISTICS ===")
print(f"Average unique poses per demo: {np.mean(all_pose_counts):.2f} ± {np.std(all_pose_counts):.2f}")
print(f"Average unique overviews per demo: {np.mean(all_overview_counts):.2f} ± {np.std(all_overview_counts):.2f}")
print(f"Expected: 6 poses, ~70% of timesteps as overviews")

# Look at one demo in detail
first_demo_key = list(demo_groups.keys())[0]
first_demo_pairs = demo_groups[first_demo_key]

print(f"\n=== FIRST DEMO DETAILED ANALYSIS ===")
print(f"Demo key: {first_demo_key}")
print(f"Number of pairs: {len(first_demo_pairs)}")

# Get unique pose indices for this demo
unique_pose_indices = list(set(pair['ee_pose_idx'] for pair in first_demo_pairs))
valid_poses = poses[unique_pose_indices]

print(f"Unique pose indices: {unique_pose_indices}")
print(f"Number of unique poses: {len(unique_pose_indices)}")
print(f"Valid poses shape: {valid_poses.shape}")

print(f"\nValid poses for this demo:")
for i, pose_idx in enumerate(unique_pose_indices):
    pose = valid_poses[i]
    print(f"  Pose {pose_idx}: {pose}")

# Check distances between poses in the same demo
print(f"\n=== POSE DISTANCES WITHIN DEMO ===")
for i in range(len(valid_poses)):
    for j in range(i+1, len(valid_poses)):
        dist = np.linalg.norm(valid_poses[i] - valid_poses[j])
        pos_dist = np.linalg.norm(valid_poses[i][:3] - valid_poses[j][:3])
        ori_dist = np.linalg.norm(valid_poses[i][3:] - valid_poses[j][3:])
        print(f"  Pose {unique_pose_indices[i]} vs {unique_pose_indices[j]}: total={dist:.4f}, pos={pos_dist:.4f}, ori={ori_dist:.4f}")

# Check if poses are in reasonable units
print(f"\n=== POSE UNIT ANALYSIS ===")
print(f"Position range (first 3 dims): {poses[:, :3].min(axis=0)} to {poses[:, :3].max(axis=0)}")
print(f"Orientation range (last 3 dims): {poses[:, 3:].min(axis=0)} to {poses[:, 3:].max(axis=0)}")

# Check if positions are in meters (should be reasonable robot workspace)
pos_magnitudes = np.linalg.norm(poses[:, :3], axis=1)
print(f"Position magnitudes - mean: {pos_magnitudes.mean():.4f}, std: {pos_magnitudes.std():.4f}")
print(f"Position magnitudes - 95th percentile: {np.percentile(pos_magnitudes, 95):.4f}")

# Check if orientations are in radians (should be -π to π)
ori_magnitudes = np.linalg.norm(poses[:, 3:], axis=1)
print(f"Orientation magnitudes - mean: {ori_magnitudes.mean():.4f}, std: {ori_magnitudes.std():.4f}")
print(f"Orientation magnitudes - 95th percentile: {np.percentile(ori_magnitudes, 95):.4f}")

print("\n=== CONCLUSION ===")
if pos_magnitudes.mean() > 10:
    print("❌ Positions seem to be in wrong units (too large for robot workspace)")
else:
    print("✅ Positions seem to be in reasonable units")

if ori_magnitudes.mean() > 10:
    print("❌ Orientations seem to be in wrong units (too large for radians)")
else:
    print("✅ Orientations seem to be in reasonable units")

# Calculate maximal pose distance within each demo
max_dists = []
demo_keys = list(demo_groups.keys())
np.random.seed(42)
if len(demo_keys) > 1000:
    sampled_keys = np.random.choice(demo_keys, 1000, replace=False)
else:
    sampled_keys = demo_keys

for demo_key in sampled_keys:
    demo_pairs = demo_groups[demo_key]
    unique_pose_indices = list(set(pair['ee_pose_idx'] for pair in demo_pairs))
    if len(unique_pose_indices) < 2:
        max_dists.append(0.0)
        continue
    demo_poses = poses[unique_pose_indices]
    # Compute all pairwise distances
    dists = []
    for i in range(len(demo_poses)):
        for j in range(i+1, len(demo_poses)):
            dist = np.linalg.norm(demo_poses[i] - demo_poses[j])
            dists.append(dist)
    max_dists.append(np.max(dists) if dists else 0.0)

print(f"\n=== MAXIMAL POSE DISTANCE ANALYSIS (sampled {len(sampled_keys)} demos) ===")
print(f"Average maximal distance within demo: {np.mean(max_dists):.4f} ± {np.std(max_dists):.4f}")
print(f"Min: {np.min(max_dists):.4f}, Max: {np.max(max_dists):.4f}") 