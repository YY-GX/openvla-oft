import sys
sys.path.append('../../..')

import h5py
import numpy as np
import os
import json
import glob
import argparse
from tqdm import tqdm
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env

"""
Contact Timestep Detection v2
This script detects the FIRST contact between robot components and objects in demonstrations.
Saves contact timesteps as contact_timesteps.json for use in pose extraction.

Key Improvements:
- PRIORITY 1: Detects gripper closing (when gripper command changes from negative to positive)
- PRIORITY 2: Falls back to contact detection if no gripper closing found
- Detects FIRST contact only (ignores subsequent contacts)
- Uses environment simulation for accurate contact detection
- Handles file name matching for single skill tasks
- Saves results in structured JSON format
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

def detect_first_contact_timestep(demo, file_path, demo_key, task_name):
    """
    Detect FIRST contact timestep between any robot component and any object.
    
    Priority:
    1. Gripper closing (when gripper command changes from negative to positive)
    2. Contact detection (fallback)
    
    Args:
        demo: Dictionary containing demonstration data
        file_path: Path to the demo file
        demo_key: Key of the current demo
        task_name: Name of the task
        
    Returns:
        int: First contact timestep, or None if no contact found
    """
    actions = demo['actions']
    gripper_commands = actions[:, -1]
    
    # PRIORITY 1: Try to detect gripper closing
    for i in range(1, len(gripper_commands)):
        if gripper_commands[i - 1] < 0 and gripper_commands[i] > 0:
            print(f"Found gripper closing at timestep {i} in {file_path} / {demo_key}")
            return i
    
    # PRIORITY 2: Fallback to contact detection if no gripper closing
    print(f"No gripper closing found in {file_path} / {demo_key}, trying contact-based detection.")
    
    # End-effector geometry names for contact detection
    EE_GEOM_NAMES = [
        "robot0_eef", "robot0_gripper0_finger0", "robot0_gripper0_finger1",
        "gripper0_hand_collision", "gripper0_finger0_collision", "gripper0_finger1_collision"
    ]
    
    try:
        # Initialize environment for simulation
        bm_name = "libero_90"
        task_suite = benchmark.get_benchmark_dict()[bm_name]()
        task = [t for t in task_suite.tasks if t.name == task_name][0]
        env, _ = get_libero_env(task, model_family="openvla", resolution=256)
        
        states = demo['states']
        
        # Iterate through states to find first contact
        for t, sim_state in enumerate(states):
            env.set_init_state(sim_state)
            
            # Check all contacts at this timestep
            for j in range(env.sim.data.ncon):
                contact = env.sim.data.contact[j]
                g1 = env.sim.model.geom_id2name(contact.geom1)
                g2 = env.sim.model.geom_id2name(contact.geom2)
                
                # Check if any robot component is in contact with any object
                if g1 in EE_GEOM_NAMES or g2 in EE_GEOM_NAMES:
                    print(f"First contact detected at timestep {t} in {file_path} / {demo_key}")
                    print(f"  Contact between: {g1} <-> {g2}")
                    return t
        
        print(f"No contact found in {file_path} / {demo_key}")
        return None
        
    except Exception as e:
        print(f"Error processing {file_path} / {demo_key}: {str(e)}")
        return None

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

def main():
    parser = argparse.ArgumentParser(description="Detect first contact timesteps in demonstrations.")
    parser.add_argument("--raw_demo_dir", type=str, 
                       default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/libero_90_no_noops/",
                       help="Directory containing raw demo HDF5 files")
    parser.add_argument("--single_skill_tasks_file", type=str,
                       default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/vlm-pose/data_preparation/v2_pipeline/single_skill_tasks_44.json",
                       help="JSON file containing single skill task names")
    parser.add_argument("--output_file", type=str,
                       default="contact_timesteps.json",
                       help="Output JSON file for contact timesteps")
    args = parser.parse_args()

    # Ensure output file has a proper path
    if not args.output_file:
        args.output_file = "contact_timesteps.json"
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load single skill task names
    single_skill_tasks = load_single_skill_tasks(args.single_skill_tasks_file)
    print(f"Loaded {len(single_skill_tasks)} single skill tasks")

    # Get all demo files
    demo_files = sorted(glob.glob(os.path.join(args.raw_demo_dir, "*.hdf5")))
    print(f"Found {len(demo_files)} demo files")

    # Initialize results dictionary
    contact_timesteps = {}
    
    # Process each single skill task
    processed_tasks = 0
    for task_name in tqdm(single_skill_tasks, desc="Processing single skill tasks"):
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
        
        # Initialize task entry in results
        contact_timesteps[task_name] = {}
        
        # Process each demo in the file
        for demo_key in tqdm(data_dict['data'], 
                           desc=f"Processing {os.path.basename(file_path)}", 
                           leave=False):
            demo = data_dict['data'][demo_key]
            
            # Detect first contact timestep
            contact_timestep = detect_first_contact_timestep(demo, file_path, demo_key, task_name)
            
            if contact_timestep is not None:
                contact_timesteps[task_name][demo_key] = contact_timestep
            else:
                print(f"No contact found for {task_name} / {demo_key}")
        
        processed_tasks += 1
        
        # Progress report
        total_demos_for_task = len(contact_timesteps[task_name])
        print(f"Task {task_name}: Found contact timesteps for {total_demos_for_task} demos")

    # Save results - FIXED: Ensure proper file saving
    print(f"Saving results to: {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(contact_timesteps, f, indent=2)
    
    # Summary statistics
    total_demos_with_contacts = sum(len(task_demos) for task_demos in contact_timesteps.values())
    total_tasks_processed = len([task for task in contact_timesteps.values() if task])
    
    print(f"\n=== Contact Detection Summary ===")
    print(f"Tasks processed: {processed_tasks}/{len(single_skill_tasks)}")
    print(f"Tasks with contacts: {total_tasks_processed}")
    print(f"Total demos with contacts: {total_demos_with_contacts}")
    print(f"Results saved to: {args.output_file}")
    
    # Verify file was saved
    if os.path.exists(args.output_file):
        print(f"✓ Contact timesteps file successfully saved!")
        file_size = os.path.getsize(args.output_file)
        print(f"  File size: {file_size} bytes")
    else:
        print(f"✗ ERROR: Contact timesteps file was not saved!")

if __name__ == "__main__":
    main()