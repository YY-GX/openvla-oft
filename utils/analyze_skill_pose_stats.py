#!/usr/bin/env python3
"""
analyze_skill_pose_stats.py

Analyze per-skill statistics for overview images and EE poses, and pose distances.
Also prints per-overview statistics: number of demos, number of skills, and avg demos per skill.
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from itertools import combinations

# Config
data_root = "datasets/local_pairs_datasets"
split_names = ["train", "val", "test"]
# split_dir = "splits"  # Change to "smaller_splits" or "debug_splits" as needed
# split_dir = "smaller_splits"  # Change to "smaller_splits" or "debug_splits" as needed
split_dir = "debug_splits"  # Change to "smaller_splits" or "debug_splits" as needed
poses_path = os.path.join(data_root, "local_poses.npy")
poses = np.load(poses_path)

for split_name in split_names:
    print(f"\n===== {split_name.upper()} SPLIT =====")
    annotation_path = os.path.join(data_root, split_dir, f"{split_name}_annotation.csv")
    if not os.path.exists(annotation_path):
        print(f"File not found: {annotation_path}")
        continue
    annotation_df = pd.read_csv(annotation_path)
    # Group by skill
    skill_groups = annotation_df.groupby('language_description')
    skill_stats = []
    all_pose_dists = []
    all_pos_dists = []
    all_ori_dists = []
    for skill, group in skill_groups:
        # Group by demo within skill
        demo_groups = group.groupby(['source_skill_file_name', 'source_demo_idx'])
        overview_counts = []
        pose_counts = []
        all_skill_pose_indices = set()
        for _, demo in demo_groups:
            overview_indices = set(demo['overview_image_idx'])
            pose_indices = set(demo['ee_pose_idx'])
            overview_counts.append(len(overview_indices))
            pose_counts.append(len(pose_indices))
            all_skill_pose_indices.update(pose_indices)
        # Pose distance stats for this skill
        skill_poses = poses[list(all_skill_pose_indices)]
        if len(skill_poses) > 1:
            pose_dists = []
            pos_dists = []
            ori_dists = []
            for a, b in combinations(skill_poses, 2):
                pose_dists.append(np.linalg.norm(a - b))
                pos_dists.append(np.linalg.norm(a[:3] - b[:3]))
                ori_dists.append(np.linalg.norm(a[3:] - b[3:]))
            min_pose_dist = np.min(pose_dists)
            max_pose_dist = np.max(pose_dists)
            avg_pose_dist = np.mean(pose_dists)
            min_pos_dist = np.min(pos_dists)
            max_pos_dist = np.max(pos_dists)
            avg_pos_dist = np.mean(pos_dists)
            min_ori_dist = np.min(ori_dists)
            max_ori_dist = np.max(ori_dists)
            avg_ori_dist = np.mean(ori_dists)
            all_pose_dists.extend(pose_dists)
            all_pos_dists.extend(pos_dists)
            all_ori_dists.extend(ori_dists)
        else:
            min_pose_dist = max_pose_dist = avg_pose_dist = 0.0
            min_pos_dist = max_pos_dist = avg_pos_dist = 0.0
            min_ori_dist = max_ori_dist = avg_ori_dist = 0.0
        skill_stats.append({
            'skill': skill,
            'num_demos': len(demo_groups),
            'avg_overviews_per_demo': np.mean(overview_counts),
            'avg_poses_per_demo': np.mean(pose_counts),
            'min_pose_dist': min_pose_dist,
            'max_pose_dist': max_pose_dist,
            'avg_pose_dist': avg_pose_dist,
            'min_pos_dist': min_pos_dist,
            'max_pos_dist': max_pos_dist,
            'avg_pos_dist': avg_pos_dist,
            'min_ori_dist': min_ori_dist,
            'max_ori_dist': max_ori_dist,
            'avg_ori_dist': avg_ori_dist,
        })
    # Print overall stats
    print("\n=== Overall Statistics (Averaged Across Skills) ===")
    print(f"Avg overview images/demo: {np.mean([s['avg_overviews_per_demo'] for s in skill_stats]):.2f}")
    print(f"Avg local poses/demo: {np.mean([s['avg_poses_per_demo'] for s in skill_stats]):.2f}")
    print(f"Pose distances: min={np.min(all_pose_dists):.4f}, max={np.max(all_pose_dists):.4f}, avg={np.mean(all_pose_dists):.4f}")
    print(f"Position distances: min={np.min(all_pos_dists):.4f}, max={np.max(all_pos_dists):.4f}, avg={np.mean(all_pos_dists):.4f}")
    print(f"Orientation distances: min={np.min(all_ori_dists):.4f}, max={np.max(all_ori_dists):.4f}, avg={np.mean(all_ori_dists):.4f}")

    # # Per-overview stats
    # print("\n=== Per-Overview Image Statistics ===")
    # overview_groups = annotation_df.groupby('overview_image_idx')
    # for overview_idx, group in overview_groups:
    #     demos = set(zip(group['source_skill_file_name'], group['source_demo_idx']))
    #     skills = set(group['language_description'])
    #     skill_demo_counts = Counter(group['language_description'])
    #     avg_demos_per_skill = np.mean(list(skill_demo_counts.values())) if skill_demo_counts else 0
    #     print(f"Overview {overview_idx}: #demos={len(demos)}, #skills={len(skills)}, avg_demos/skill={avg_demos_per_skill:.2f}") 