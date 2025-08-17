import sys
sys.path.append('../../..')

import h5py
import numpy as np
import os
import json
import glob
import argparse
import csv
import random
from tqdm import tqdm
from PIL import Image

"""
Extract Local Pairs v2 - Contact-Based Pose Extraction
This script extracts poses very close to contact points for high-quality dataset creation.

Key Improvements:
- Uses pre-computed contact timesteps from contact_timesteps.json
- Extracts poses at fixed window relative to contact (default: [-4, -3, -2])
- Processes only single skill tasks (44 tasks)
- Ensures pose consistency and quality near contact points
- Optional sampling mode for debugging (100 demos from 10 skills)
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
        List of tuples: (pose, actual_timestep)
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
        
        # Extract 6D pose (position + orientation)
        pose = np.concatenate([ee_pos[actual_timestep], ee_ori[actual_timestep]])
        poses.append((pose, actual_timestep))
    
    return poses

def extract_overview_images(demo, contact_timestep, overview_percentage=70.0):
    """
    Extract overview images from start to a percentage of contact timestep.
    
    Args:
        demo: Dictionary containing demonstration data
        contact_timestep: Timestep when contact occurs
        overview_percentage: Percentage of contact timestep to extract images from
        
    Returns:
        List of overview images
    """
    agentview_rgb = demo['obs']['agentview_rgb']
    end_idx = int(contact_timestep * overview_percentage / 100)
    end_idx = max(1, min(end_idx, len(agentview_rgb)))  # Ensure at least 1 image
    
    images = []
    for i in range(end_idx):
        images.append(agentview_rgb[i])
    
    return images

def save_image(image_array, save_path):
    """Save image array to file."""
    image = Image.fromarray(image_array)
    image.save(save_path)

def parse_language_description(task_name):
    """
    Parse language description from task name.
    
    Args:
        task_name: Task name
        
    Returns:
        str: Language description
    """
    # Remove scene prefix and convert underscores to spaces
    # e.g., "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate" -> "put the black bowl on the plate"
    parts = task_name.split('_')
    if len(parts) > 2 and 'SCENE' in parts[1]:
        # Remove scene part (e.g., "KITCHEN_SCENE1")
        description_parts = parts[2:]
    else:
        description_parts = parts
    
    return ' '.join(description_parts)

def sample_tasks_and_demos(contact_timesteps, num_tasks=10, demos_per_task=10):
    """
    Sample tasks and demos for debugging.
    
    Args:
        contact_timesteps: Contact timesteps data
        num_tasks: Number of tasks to sample
        demos_per_task: Average demos per task
        
    Returns:
        dict: Sampled contact timesteps
    """
    # Get tasks with most demos
    task_demo_counts = {task: len(demos) for task, demos in contact_timesteps.items()}
    sorted_tasks = sorted(task_demo_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Sample top tasks
    sampled_tasks = sorted_tasks[:num_tasks]
    sampled_contact_timesteps = {}
    
    for task_name, demo_count in sampled_tasks:
        task_demos = contact_timesteps[task_name]
        
        # Sample demos for this task
        if len(task_demos) <= demos_per_task:
            sampled_demos = task_demos
        else:
            demo_keys = list(task_demos.keys())
            sampled_keys = random.sample(demo_keys, demos_per_task)
            sampled_demos = {key: task_demos[key] for key in sampled_keys}
        
        sampled_contact_timesteps[task_name] = sampled_demos
    
    return sampled_contact_timesteps

def main():
    parser = argparse.ArgumentParser(description="Extract poses near contact points from demonstrations.")
    parser.add_argument("--raw_demo_dir", type=str, 
                       default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/libero_90_no_noops/",
                       help="Directory containing raw demo HDF5 files")
    parser.add_argument("--contact_timesteps_file", type=str,
                       default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/vlm-pose/data_preparation/v2_pipeline/contact_timesteps.json",
                       help="JSON file containing contact timesteps")
    parser.add_argument("--single_skill_tasks_file", type=str,
                       default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/vlm-pose/data_preparation/v2_pipeline/single_skill_tasks_44.json",
                       help="JSON file containing single skill task names")
    parser.add_argument("--save_dir", type=str,
                       default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/local_pairs_datasets_v2",
                       help="Directory to save extracted pairs")
    parser.add_argument("--window_offsets", nargs='+', type=int, default=[-4, -3, -2],
                       help="Timestep offsets relative to contact (e.g., -4 -3 -2)")
    parser.add_argument("--overview_percentage", type=float, default=70.0,
                       help="Percentage of contact timestep to extract overview images from")
    parser.add_argument("--sampling_mode", action="store_true",
                       help="Enable sampling mode for debugging (100 demos from 10 skills)")
    parser.add_argument("--num_tasks", type=int, default=10,
                       help="Number of tasks to sample (only used in sampling mode)")
    parser.add_argument("--demos_per_task", type=int, default=10,
                       help="Average demos per task to sample (only used in sampling mode)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling (only used in sampling mode)")
    args = parser.parse_args()

    # Validate window offsets (should all be negative for pre-contact)
    if any(offset >= 0 for offset in args.window_offsets):
        print("Warning: Some window offsets are non-negative. This may extract poses at/after contact.")

    # Set random seed if sampling mode
    if args.sampling_mode:
        random.seed(args.seed)
        print(f"Sampling mode enabled with seed {args.seed}")

    # Load required data
    contact_timesteps = load_contact_timesteps(args.contact_timesteps_file)
    single_skill_tasks = load_single_skill_tasks(args.single_skill_tasks_file)
    print(f"Loaded contact timesteps for {len(contact_timesteps)} tasks")
    print(f"Loaded {len(single_skill_tasks)} single skill tasks")

    # Store original contact timesteps for full processing
    original_contact_timesteps = contact_timesteps.copy()
    
    # Apply sampling only for pose debug images if enabled
    if args.sampling_mode:
        sampled_contact_timesteps = sample_tasks_and_demos(contact_timesteps, args.num_tasks, args.demos_per_task)
        print(f"Sampled {len(sampled_contact_timesteps)} tasks for pose debug images only")

    # Create save directories
    os.makedirs(args.save_dir, exist_ok=True)
    images_dir = os.path.join(args.save_dir, "overview_images")
    os.makedirs(images_dir, exist_ok=True)
    debugging_images_dir = os.path.join(args.save_dir, "pose_images")
    os.makedirs(debugging_images_dir, exist_ok=True)

    # Get all demo files
    demo_files = sorted(glob.glob(os.path.join(args.raw_demo_dir, "*.hdf5")))
    print(f"Found {len(demo_files)} demo files")

    # Initialize data structures
    all_poses = []
    all_pairs = []
    image_counter = 0
    pose_counter = 0
    debugging_image_counter = 0
    
    # Statistics
    processed_tasks = 0
    processed_demos = 0
    skipped_demos_no_contact = 0

    # Process each single skill task (always full dataset for overview images and annotations)
    for task_name in tqdm(single_skill_tasks, desc="Processing single skill tasks"):
        if task_name not in contact_timesteps:
            print(f"No contact timesteps found for task: {task_name}")
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
        for demo_key in tqdm(data_dict['data'], 
                           desc=f"Processing {os.path.basename(file_path)}", 
                           leave=False):
            
            if demo_key not in task_contact_timesteps:
                skipped_demos_no_contact += 1
                continue
            
            demo = data_dict['data'][demo_key]
            contact_timestep = task_contact_timesteps[demo_key]
            
            # Extract poses at contact-based window
            poses_with_timesteps = extract_contact_based_poses(demo, contact_timestep, args.window_offsets)
            
            # Extract overview images (from start to 70% of contact timestep)
            overview_images = extract_overview_images(demo, contact_timestep, args.overview_percentage)
            
            # Save overview images
            overview_image_indices = []
            for img in overview_images:
                img_filename = f"{image_counter:06d}.jpg"
                img_path = os.path.join(images_dir, img_filename)
                save_image(img, img_path)
                overview_image_indices.append(image_counter)
                image_counter += 1
            
            # Save poses and corresponding debugging images
            pose_indices = []
            pose_to_debug_image = {}
            
            # Check if this demo should be used for pose debug images (sampling mode)
            should_save_pose_debug = True
            if args.sampling_mode:
                # Only save pose debug images for sampled demos
                should_save_pose_debug = (task_name in sampled_contact_timesteps and 
                                        demo_key in sampled_contact_timesteps[task_name])
            
            for i, (pose, pose_timestep) in enumerate(poses_with_timesteps):
                all_poses.append(pose)
                pose_indices.append(pose_counter)
                
                # Save pose debug image only if selected
                if should_save_pose_debug:
                    pose_image = demo['obs']['agentview_rgb'][pose_timestep]
                    
                    # Use descriptive filename in sampling mode
                    if args.sampling_mode:
                        # Create descriptive filename: skill_name + demo_idx + offset
                        skill_short = task_name.replace("KITCHEN_SCENE", "KS").replace("LIVING_ROOM_SCENE", "LRS")
                        offset = args.window_offsets[i]
                        debug_img_filename = f"{skill_short}_{demo_key}_offset{offset}.jpg"
                    else:
                        debug_img_filename = f"{debugging_image_counter:06d}.jpg"
                    
                    debug_img_path = os.path.join(debugging_images_dir, debug_img_filename)
                    save_image(pose_image, debug_img_path)
                    pose_to_debug_image[pose_counter] = debugging_image_counter
                    debugging_image_counter += 1
                
                pose_counter += 1
            
            # Create pairs
            for pose_idx, (pose, pose_timestep) in zip(pose_indices, poses_with_timesteps):
                # Get debug image idx (use -1 if no debug image was saved)
                debug_image_idx = pose_to_debug_image.get(pose_idx, -1)
                
                for overview_idx in overview_image_indices:
                    pair = {
                        # Core pair information
                        'overview_image_idx': overview_idx,
                        'ee_pose_idx': pose_idx,
                        'language_description': language_description,
                        
                        # Source information
                        'source_task_name': task_name,
                        'source_demo_idx': demo_key,
                        'source_file_name': os.path.basename(file_path),
                        
                        # Contact information
                        'contact_timestep': contact_timestep,
                        'pose_timestep': pose_timestep,
                        'timestep_offset_from_contact': pose_timestep - contact_timestep,
                        
                        # Debug information
                        'pose_debug_image_idx': debug_image_idx,
                        
                        # Extraction parameters
                        'window_offsets': args.window_offsets,
                        'overview_percentage': args.overview_percentage,
                    }
                    all_pairs.append(pair)
            
            processed_demos += 1
        
        processed_tasks += 1

    # Save poses
    poses_path = os.path.join(args.save_dir, "local_poses.npy")
    np.save(poses_path, np.array(all_poses))
    print(f"Saved {len(all_poses)} poses to {poses_path}")

    # Save annotation CSV
    annotation_path = os.path.join(args.save_dir, "annotation.csv")
    if all_pairs:
        fieldnames = all_pairs[0].keys()
        with open(annotation_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_pairs)
        print(f"Saved {len(all_pairs)} pairs to {annotation_path}")
    else:
        print("No pairs generated!")

    # Summary statistics
    mode_str = "Pose Debug Sampled" if args.sampling_mode else "Full"
    print(f"\n=== Extraction Summary ({mode_str}) ===")
    print(f"Tasks processed: {processed_tasks}/{len(single_skill_tasks)}")
    print(f"Demos processed: {processed_demos}")
    print(f"Demos skipped (no contact): {skipped_demos_no_contact}")
    print(f"Total poses extracted: {len(all_poses)}")
    print(f"Total pairs created: {len(all_pairs)}")
    print(f"Overview images saved: {image_counter}")
    print(f"Pose debug images saved: {debugging_image_counter}")
    print(f"Window offsets used: {args.window_offsets}")
    if args.sampling_mode:
        print(f"Pose debug sampling: {args.num_tasks} tasks, ~{args.demos_per_task} demos per task")
        print(f"Note: Full dataset generated for overview images and annotations")
    print(f"Dataset saved to: {args.save_dir}")

if __name__ == "__main__":
    main()