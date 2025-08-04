import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Extract sampled target object poses for fast evaluation.")
parser.add_argument('--sample_indices', type=str, required=True, help='Path to sample_indices.json')
parser.add_argument('--csv_path', type=str, required=True, help='Path to target_object_poses.csv')
parser.add_argument('--npy_path', type=str, required=True, help='Path to target_object_poses.npy')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory for sampled target object poses')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Load sample indices
def load_sample_indices(path):
    with open(path, 'r') as f:
        return json.load(f)

sample_indices = load_sample_indices(args.sample_indices)

# Load CSV and NPY
print("Loading target object pose CSV and NPY...")
df = pd.read_csv(args.csv_path)
poses = np.load(args.npy_path)

# For each split, extract the sampled target object poses
for split, indices in sample_indices.items():
    print(f"Processing split: {split} ({len(indices)} samples)...")
    # The indices refer to rows in the CSV (and thus in the NPY)
    split_poses = []
    for idx in tqdm(indices, desc=f"{split} samples"):
        pose_idx = df.iloc[idx]['pose_index']
        split_poses.append(poses[int(pose_idx)])
    split_poses = np.stack(split_poses)
    out_path = os.path.join(args.output_dir, f"sampled_target_object_poses_{split}.npy")
    np.save(out_path, split_poses)
    print(f"Saved {split} sampled target object poses to {out_path}") 