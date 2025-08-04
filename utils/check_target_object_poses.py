#!/usr/bin/env python3
"""
check_target_object_poses.py

Checks the integrity and content of target_object_poses.npy and target_object_poses.csv.

Usage:
    python utils/check_target_object_poses.py --data_root datasets/local_pairs_datasets
"""
import os
import numpy as np
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description="Check target object pose extraction results.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="datasets/local_pairs_datasets",
        help="Root directory containing the target_object_poses files"
    )
    args = parser.parse_args()

    npy_path = os.path.join(args.data_root, "target_object_poses.npy")
    csv_path = os.path.join(args.data_root, "target_object_poses.csv")

    # 1. Check file existence
    print(f"Checking files in {args.data_root}...")
    if not os.path.exists(npy_path):
        print(f"ERROR: {npy_path} does not exist!")
        return
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} does not exist!")
        return
    print("Both files exist.")

    # 2. Load files
    poses = np.load(npy_path)
    df = pd.read_csv(csv_path)
    print(f"Loaded {poses.shape[0]} poses from NPY, {len(df)} rows from CSV.")

    # 3. Check shape
    if poses.shape[0] != len(df):
        print(f"ERROR: NPY and CSV row counts do not match!")
        return
    if poses.shape[1] != 6:
        print(f"ERROR: Each pose should have 6 values (got shape {poses.shape})!")
        return
    print("Shape check passed.")

    # 4. Check for all-zeros rows
    zero_rows = np.where(~poses.any(axis=1))[0]
    if len(zero_rows) > 0:
        print(f"WARNING: {len(zero_rows)} poses are all zeros (indices: {zero_rows[:10]})")
    else:
        print("No all-zeros poses found.")

    # 5. Check for None, empty, or gripper_closing in contact_object_name
    bad_obj = df['contact_object_name'].isnull() | (df['contact_object_name'] == '') | (df['contact_object_name'] == 'gripper_closing')
    if bad_obj.any():
        num_bad = bad_obj.sum()
        gripper_closing_only = (df.loc[bad_obj, 'contact_object_name'] == 'gripper_closing').all()
        if gripper_closing_only:
            print(f"INFO: All {num_bad} bad rows are 'gripper_closing' (expected, safe to ignore).")
        else:
            print(f"WARNING: {num_bad} rows have bad contact_object_name (not all are 'gripper_closing').")
            print(df.loc[bad_obj, 'contact_object_name'].value_counts())
    else:
        print("All contact_object_name values look good.")

    # 6. Check for duplicate (skill, demo, object, timestep)
    dup_cols = ['language_description', 'source_demo_idx', 'contact_object_name', 'contact_timestep']
    dups = df.duplicated(subset=dup_cols, keep=False)
    if dups.any():
        print(f"WARNING: {dups.sum()} duplicate rows found for {dup_cols}.")
    else:
        print("No duplicate (skill, demo, object, timestep) rows found.")

    # 7. Spot-check a few random rows
    print("\nSpot-checking 5 random rows:")
    sample = df.sample(n=min(5, len(df)), random_state=42)
    for idx, row in sample.iterrows():
        pose = poses[row['pose_index']]
        print(f"Row {idx}: skill={row['language_description']}, demo={row['source_demo_idx']}, obj={row['contact_object_name']}, timestep={row['contact_timestep']}, pose={pose}")

    # 8. Print the first 5 saved object poses in nice format
    print("\nFirst 5 saved object poses:")
    for i, row in df.head(5).iterrows():
        pose = poses[int(row['pose_index'])]
        print(f"Row {i}: obj={row['contact_object_name']}, timestep={row['contact_timestep']}, pose={pose}")

    print("\nCheck complete.")

if __name__ == "__main__":
    main() 