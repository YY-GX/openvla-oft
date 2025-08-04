#!/usr/bin/env python3
"""
extract_target_object_poses.py

Extract target object poses at contact timesteps for each skill and demo.
Saves both the poses as .npy and annotations as .csv.

Usage:
    python utils/extract_target_object_poses.py
"""

import os
import h5py
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

# Import necessary modules for contact detection
import sys
sys.path.append('.')
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env


def load_hdf5_to_dict(file_path):
    """Load HDF5 file to dictionary."""
    def recursively_extract(group):
        data = {}
        for key in group.keys():
            if isinstance(group[key], h5py.Group):
                data[key] = recursively_extract(group[key])
            else:
                data[key] = group[key][:]
        return data
    with h5py.File(file_path, 'r') as f:
        return recursively_extract(f)


def get_object_pose_at_timestep(env, object_name: str, timestep: int) -> Tuple[np.ndarray, bool]:
    """
    Get the pose of an object at a specific timestep.
    
    Args:
        env: MuJoCo environment
        object_name: Name of the object (could be geometry name or body name)
        timestep: Timestep to get pose at
        
    Returns:
        pose: 6D pose [x, y, z, rx, ry, rz]
        success: Whether pose extraction was successful
    """
    try:
        # First try to get body pose (position + quaternion)
        pos = env.sim.data.get_body_xpos(object_name)
        quat = env.sim.data.get_body_xquat(object_name)
        
        # Convert quaternion to euler angles for consistency with EE poses
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # wxyz to xyzw
        euler = rot.as_euler('xyz')
        
        pose = np.concatenate([pos, euler])
        return pose, True
    except:
        # If that fails, try to convert geometry name to body name
        try:
            # Remove geometry suffix (e.g., "_g34" -> "")
            if "_g" in object_name and object_name.split("_g")[-1].isdigit():
                body_name = object_name.rsplit("_g", 1)[0]
                pos = env.sim.data.get_body_xpos(body_name)
                quat = env.sim.data.get_body_xquat(body_name)
                
                from scipy.spatial.transform import Rotation
                rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # wxyz to xyzw
                euler = rot.as_euler('xyz')
                
                pose = np.concatenate([pos, euler])
                return pose, True
        except:
            pass
        
        # Fallback: try to get geom pose directly
        try:
            geom_id = env.sim.model.geom_name2id(object_name)
            pos = env.sim.data.geom_xpos[geom_id]
            rot = env.sim.data.geom_xmat[geom_id].reshape(3, 3)
            from scipy.spatial.transform import Rotation
            euler = Rotation.from_matrix(rot).as_euler('xyz')
            pose = np.concatenate([pos, euler])
            return pose, True
        except:
            print(f"Warning: Could not get pose for object {object_name}")
            return np.zeros(6), False


def detect_contact_timesteps(demo, file_path, demo_key, task_name, verbose=True):
    """
    Detect first contact timesteps with each unique object in a demonstration.
    
    Args:
        demo: Dictionary containing demonstration data
        file_path: Path to the demo file
        demo_key: Key of the current demo
        task_name: Name of the task
        verbose: If True, print detailed logging
        
    Returns:
        List of tuples: (contact_timestep, contact_object_name)
    """
    actions = demo['actions']
    gripper_commands = actions[:, -1]
    contact_info = []
    
    # 1. Try to detect gripper closing
    for i in range(1, len(gripper_commands)):
        if gripper_commands[i - 1] < 0 and gripper_commands[i] > 0:
            contact_info.append((i, "gripper_closing"))
            if verbose:
                print(f"Found gripper closing at {i} in {file_path} / {demo_key}")
    
    # 2. Fallback to contact detection if no gripper closing OR only gripper_closing found
    if not contact_info or all(obj == "gripper_closing" for _, obj in contact_info):
        if verbose:
            print(f"No real object contacts found via gripper closing in {file_path} / {demo_key}, trying contact-based detection.")
        bm_name = "libero_90"
        task_suite = benchmark.get_benchmark_dict()[bm_name]()
        task = [t for t in task_suite.tasks if t.name == task_name][0]
        env, _ = get_libero_env(task, model_family="openvla", resolution=256)

        EE_GEOM_NAMES = [
            "robot0_eef", "robot0_gripper0_finger0", "robot0_gripper0_finger1",
            "gripper0_hand_collision", "gripper0_finger0_collision", "gripper0_finger1_collision"
        ]
        states = demo['states']
        contacted_objects = set()  # Track unique objects contacted

        for t, sim_state in enumerate(states):
            env.set_init_state(sim_state)
            for j in range(env.sim.data.ncon):
                contact = env.sim.data.contact[j]
                g1 = env.sim.model.geom_id2name(contact.geom1)
                g2 = env.sim.model.geom_id2name(contact.geom2)
                
                # Determine which object the EE is contacting
                if g1 in EE_GEOM_NAMES:
                    contact_object = g2
                elif g2 in EE_GEOM_NAMES:
                    contact_object = g1
                else:
                    continue
                
                # Only record first contact with each unique object
                if contact_object not in contacted_objects:
                    contacted_objects.add(contact_object)
                    contact_info.append((t, contact_object))
                    if verbose:
                        print(f"Found first EE contact with {contact_object} at {t} in {file_path} / {demo_key}")
        
        env.close()
    
    return sorted(contact_info, key=lambda x: x[0])  # Sort by timestep


def extract_target_object_poses(data_root: str = "datasets/local_pairs_datasets", raw_demo_dir: str = None, debug_first_skill: bool = False, debug_demos: int = 10, verbose: bool = True):
    """
    Extract target object poses for all skills and demos, with improved logic for component/whole object and data quality.
    Args:
        data_root: Root directory containing the dataset
        raw_demo_dir: Directory containing the raw demo files (HDF5 files)
        debug_first_skill: If True, only process the first skill for debugging
        debug_demos: Number of demos to process in debug mode (default: 10)
        verbose: If False, suppress detailed logging
    """
    if verbose:
        print("Extracting target object poses...")
    if raw_demo_dir is None:
        raw_demo_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/libero_90_no_noops/"
        if verbose:
            print(f"Using default raw_demo_dir: {raw_demo_dir}")

    annotation_path = os.path.join(data_root, "annotation.csv")
    annotation_df = pd.read_csv(annotation_path)
    unique_pairs = annotation_df[['language_description', 'source_skill_file_name', 'source_demo_idx']].drop_duplicates()

    # 1. Group demos by skill and prepare for processing
    if verbose:
        print("Preparing to process demos...")
    skill_demo_keys = dict()  # skill -> set of (file, demo_idx)
    
    # Group by skill first
    for skill in unique_pairs['language_description'].unique():
        skill_pairs = unique_pairs[unique_pairs['language_description'] == skill]
        skill_demo_keys[skill] = set()
        
        # Limit demos in debug mode
        if debug_first_skill and skill == skill_pairs.iloc[0]['language_description']:
            skill_pairs = skill_pairs.head(debug_demos)
            if verbose:
                print(f"Debug mode: processing only first {debug_demos} demos of skill: {skill}")
        
        for _, row in skill_pairs.iterrows():
            skill_file = row['source_skill_file_name']
            demo_idx = row['source_demo_idx']
            skill_demo_keys[skill].add((skill_file, demo_idx))

    # If debug mode, only process the first skill
    skills_to_process = [list(skill_demo_keys.keys())[0]] if debug_first_skill else list(skill_demo_keys.keys())
    if verbose:
        print(f"Processing skills: {skills_to_process}")

    target_object_data = []
    target_object_poses = []
    bad_rows = 0
    total_rows = 0

    for skill in skills_to_process:
        demo_keys = list(skill_demo_keys[skill])
        
        # Add progress bar
        demo_iter = tqdm(demo_keys, desc=f"Skill: {skill}") if verbose else demo_keys
        
        for skill_file, demo_idx in demo_iter:
            demo_file_path = os.path.join(raw_demo_dir, skill_file)
            if not os.path.exists(demo_file_path):
                continue
            try:
                demo_data = load_hdf5_to_dict(demo_file_path)
                if demo_idx not in demo_data['data']:
                    continue
                demo = demo_data['data'][demo_idx]
                contact_info = detect_contact_timesteps(demo, demo_file_path, demo_idx, skill, verbose=verbose)
                # Filter out gripper_closing and invalid objects, keep all first contacts
                filtered_contacts = []
                seen_objects = set()  # Track objects to ensure only first component per object
                for t, obj in contact_info:
                    if obj == "gripper_closing" or not obj or pd.isna(obj):
                        continue
                    
                    # For components, check if we've already seen this base object
                    if "_g" in obj and obj.split("_g")[-1].isdigit():
                        base_obj = obj.rsplit("_g", 1)[0]
                        if base_obj in seen_objects:
                            continue  # Skip if we already have a component for this object
                        seen_objects.add(base_obj)
                    else:
                        # For non-component objects, check if we've already seen this object
                        if obj in seen_objects:
                            continue  # Skip if we already have this object
                        seen_objects.add(obj)
                    
                    filtered_contacts.append((t, obj))
                
                # Now, for each contact, just use the component name directly
                for t, obj in filtered_contacts:
                    total_rows += 1
                    pose_name = obj  # Use the component name directly
                    # Set environment to contact timestep
                    bm_name = "libero_90"
                    task_suite = benchmark.get_benchmark_dict()[bm_name]()
                    task = [t for t in task_suite.tasks if t.name == skill][0]
                    env, _ = get_libero_env(task, model_family="openvla", resolution=256)
                    env.set_init_state(demo['states'][t])
                    # Get pose
                    target_pose, success = get_object_pose_at_timestep(env, pose_name, t)
                    env.close()
                    # Data quality filtering
                    if not success or not pose_name or pd.isna(pose_name) or np.allclose(target_pose, 0):
                        bad_rows += 1
                        continue
                    target_object_data.append({
                        'language_description': skill,
                        'source_skill_file_name': skill_file,
                        'source_demo_idx': demo_idx,
                        'contact_object_name': pose_name,
                        'contact_timestep': t,
                        'pose_index': len(target_object_poses)
                    })
                    target_object_poses.append(target_pose)
            except Exception as e:
                bad_rows += 1
                continue
        # Assertion: for each skill, number of saved poses == number of demos * contacts per demo
        n_demos = len(demo_keys)
        n_saved = sum(1 for d in target_object_data if d['language_description'] == skill)
        if verbose:
            print(f"Skill {skill}: {n_saved} valid object poses for {n_demos} demos.")
        assert n_saved <= n_demos * 10, f"Too many poses for skill {skill}: {n_saved} > {n_demos}*10"  # Allow up to 10 contacts per demo
        
        # Additional assertion: verify no duplicate objects per demo
        for demo_key in demo_keys:
            demo_objects = [d['contact_object_name'] for d in target_object_data 
                          if d['language_description'] == skill and d['source_demo_idx'] == demo_key[1]]
            # Check for duplicate base objects
            base_objects = set()
            for obj in demo_objects:
                if "_g" in obj and obj.split("_g")[-1].isdigit():
                    base_obj = obj.rsplit("_g", 1)[0]
                    assert base_obj not in base_objects, f"Duplicate base object {base_obj} found in demo {demo_key}"
                    base_objects.add(base_obj)
                else:
                    assert obj not in base_objects, f"Duplicate object {obj} found in demo {demo_key}"
                    base_objects.add(obj)

    # Save files
    poses_path = os.path.join(data_root, "target_object_poses.npy")
    csv_path = os.path.join(data_root, "target_object_poses.csv")
    np.save(poses_path, np.array(target_object_poses))
    pd.DataFrame(target_object_data).to_csv(csv_path, index=False)
    print(f"\nExtraction complete!")
    print(f"Saved {len(target_object_poses)} target object poses (filtered {bad_rows} bad rows out of {total_rows})")
    print(f"Poses saved to: {poses_path}")
    print(f"Annotations saved to: {csv_path}")
    return target_object_poses, pd.DataFrame(target_object_data)


def main():
    parser = argparse.ArgumentParser(description="Extract target object poses from demonstrations")
    parser.add_argument(
        "--data_root",
        type=str,
        default="datasets/local_pairs_datasets",
        help="Root directory containing the dataset"
    )
    parser.add_argument(
        "--raw_demo_dir",
        type=str,
        default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/libero_90_no_noops/",
        help="Directory containing the raw demo files (HDF5 files)"
    )
    parser.add_argument(
        "--debug_first_skill",
        action="store_true",
        help="If set, only process the first skill for debugging"
    )
    parser.add_argument(
        "--debug_demos",
        type=int,
        default=1,
        help="Number of demos to process in debug mode (default: 10)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="If set, print detailed logging (default: True)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="If set, suppress detailed logging"
    )
    args = parser.parse_args()
    
    # Handle verbose/quiet flags
    verbose = args.verbose and not args.quiet
    
    extract_target_object_poses(args.data_root, args.raw_demo_dir, args.debug_first_skill, args.debug_demos, verbose)


if __name__ == "__main__":
    main() 