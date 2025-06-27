import h5py
import numpy as np
import os
import glob
import argparse
import csv
import re
from tqdm import tqdm
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env
from PIL import Image

"""
This script extracts (skill language, overview image, pre-contact pose) pairs from raw demonstrations.
It detects contact timesteps and extracts relevant poses and images for skill learning.
"""

# Recursive HDF5 loader
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

def detect_contact_timesteps(demo, file_path, demo_key, task_name):
    """
    Detect first contact timesteps with each unique object in a demonstration.
    
    Args:
        demo: Dictionary containing demonstration data
        file_path: Path to the demo file
        demo_key: Key of the current demo
        task_name: Name of the task
        
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
            print(f"Found gripper closing at {i} in {file_path} / {demo_key}")
    
    # 2. Fallback to contact detection if no gripper closing
    if not contact_info:
        print(f"No gripper closing found in {file_path} / {demo_key}, trying contact-based detection.")
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
                    print(f"Found first EE contact with {contact_object} at {t} in {file_path} / {demo_key}")
    
    return sorted(contact_info, key=lambda x: x[0])  # Sort by timestep

def extract_pre_contact_poses(demo, contact_timestep, pre_contact, pre_contact_range):
    """
    Extract a range of EE poses before contact.
    
    Args:
        demo: Dictionary containing demonstration data
        contact_timestep: Timestep when contact occurs
        pre_contact: Number of steps before contact to start extraction
        pre_contact_range: Number of poses to extract
        
    Returns:
        List of tuples: (pose, pose_timestep) where pose_timestep is the actual timestep of the pose
    """
    ee_pos = demo['obs']['ee_pos']
    ee_ori = demo['obs']['ee_ori']
    
    # Calculate start and end indices for pose extraction
    start_idx = max(0, contact_timestep - pre_contact - pre_contact_range // 2)
    end_idx = min(len(ee_pos), contact_timestep - pre_contact + pre_contact_range // 2)
    
    poses = []
    for i in range(start_idx, end_idx):
        pose = np.concatenate([ee_pos[i], ee_ori[i]])  # 6D pose
        poses.append((pose, i))  # Include the actual timestep
    
    # Pad with last pose if we don't have enough poses
    while len(poses) < pre_contact_range:
        if poses:
            last_pose, last_timestep = poses[-1]
            poses.append((last_pose, last_timestep))
        else:
            poses.append((np.zeros(6), 0))
    
    return poses[:pre_contact_range]

def extract_overview_images(demo, contact_timestep, overview_percentage):
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
    end_idx = min(end_idx, len(agentview_rgb))
    
    images = []
    for i in range(end_idx):
        images.append(agentview_rgb[i])
    
    return images

def parse_language_description(filename):
    """
    Parse language description from filename.
    
    Args:
        filename: Demo filename
        
    Returns:
        List of language descriptions (1 or 2 depending on if there are multiple skills)
    """
    # Remove file extension and demo suffix
    base_name = filename.replace("_demo.hdf5", "")
    
    # Split by "and" to check for multiple skills
    if " and " in base_name:
        parts = base_name.split(" and ")
        # Check if this is actually multiple skills or just one skill with "and"
        # For now, we'll assume it's multiple skills if "and" is present
        return [part.strip() for part in parts]
    else:
        return [base_name]

def save_image(image_array, save_path):
    """Save image array to file."""
    # Convert from uint8 array to PIL Image and save
    image = Image.fromarray(image_array)
    image.save(save_path)

def main():
    parser = argparse.ArgumentParser(description="Extract local pairs from demonstrations.")
    parser.add_argument("--raw_demo_dir", type=str, 
                       default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/libero_90_no_noops/",
                       help="Directory containing raw demo HDF5 files")
    parser.add_argument("--save_dir", type=str,
                       default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/local_pairs_datasets",
                       help="Directory to save extracted pairs")
    parser.add_argument("--pre_contact", type=int, default=5,
                       help="Number of steps before contact to start pose extraction")
    parser.add_argument("--pre_contact_range", type=int, default=6,
                       help="Number of poses to extract before contact")
    parser.add_argument("--overview_percentage", type=float, default=70.0,
                       help="Percentage of contact timestep to extract overview images from")
    args = parser.parse_args()

    # Validate parameters
    assert args.pre_contact_range // 2 < args.pre_contact, \
        f"pre_contact_range//2 ({args.pre_contact_range//2}) must be < pre_contact ({args.pre_contact}) to ensure poses are extracted before contact"

    # Create save directories
    os.makedirs(args.save_dir, exist_ok=True)
    images_dir = os.path.join(args.save_dir, "3rd_imgs")
    os.makedirs(images_dir, exist_ok=True)
    debugging_images_dir = os.path.join(args.save_dir, "debugging_imgs")
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

    # Process each demo file
    for file_path in tqdm(demo_files, desc="Processing demo files"):
        data_dict = load_hdf5_to_dict(file_path)
        task_name = os.path.basename(file_path).split("_demo.hdf5")[0]
        
        # Parse language descriptions
        language_descriptions = parse_language_description(os.path.basename(file_path))
        
        # Process each demo in the file
        for demo_key in tqdm(data_dict['data'], desc=f"Processing {os.path.basename(file_path)}", leave=False):
            demo = data_dict['data'][demo_key]
            
            # Detect contact timesteps and objects
            contact_info = detect_contact_timesteps(demo, file_path, demo_key, task_name)
            
            if not contact_info:
                print(f"Skipping {file_path} / {demo_key} due to no detected contacts.")
                continue
            
            # Extract overview images (shared across all contacts in this demo)
            first_contact_timestep = contact_info[0][0]
            overview_images = extract_overview_images(demo, first_contact_timestep, args.overview_percentage)
            
            # Save overview images
            overview_image_indices = []
            for img in overview_images:
                img_filename = f"{image_counter:05d}.jpg"
                img_path = os.path.join(images_dir, img_filename)
                save_image(img, img_path)
                overview_image_indices.append(image_counter)
                image_counter += 1
            
            # Process each contact timestep
            for contact_idx, (contact_timestep, contact_object) in enumerate(contact_info):
                # Extract pre-contact poses
                poses_with_timesteps = extract_pre_contact_poses(demo, contact_timestep, args.pre_contact, args.pre_contact_range)
                
                # Save poses and corresponding debugging images
                pose_indices = []
                pose_to_debug_image = {}  # Map pose index to debugging image index
                for pose, pose_timestep in poses_with_timesteps:
                    all_poses.append(pose)
                    pose_indices.append(pose_counter)
                    
                    # Save the image at pose_timestep to debugging folder
                    pose_image = demo['obs']['agentview_rgb'][pose_timestep]
                    debug_img_filename = f"{debugging_image_counter:05d}.jpg"
                    debug_img_path = os.path.join(debugging_images_dir, debug_img_filename)
                    save_image(pose_image, debug_img_path)
                    pose_to_debug_image[pose_counter] = debugging_image_counter
                    debugging_image_counter += 1
                    
                    pose_counter += 1
                
                # Determine language description
                if len(language_descriptions) == len(contact_info):
                    # One description per contact
                    language_desc = language_descriptions[contact_idx]
                elif len(language_descriptions) == 1:
                    # Single description for all contacts
                    language_desc = language_descriptions[0]
                else:
                    # Fallback: use first description
                    language_desc = language_descriptions[0]
                
                # Create pairs
                for pose_idx, (pose, pose_timestep) in zip(pose_indices, poses_with_timesteps):
                    # Get the debugging image index for this pose
                    overview_idx_for_pose = pose_to_debug_image[pose_idx]
                    
                    for overview_idx in overview_image_indices:
                        pair = {
                            # Core pair information
                            'overview_image_idx': overview_idx,  # Index of the overview image (0-70% of contact timestep)
                            'ee_pose_idx': pose_idx,  # Index of the end-effector pose in local_poses.npy
                            'language_description': language_desc,  # Skill description (e.g., "pick up the book")
                            
                            # Source information
                            'source_skill_file_name': os.path.basename(file_path),  # Original demo file name
                            'source_demo_idx': demo_key,  # Demo index within the file (e.g., "demo_0")
                            'task_name': task_name,  # Task name extracted from filename
                            
                            # Contact information
                            'pre_contact_number': contact_idx + 1,  # Which contact this is (1st, 2nd, etc.)
                            'total_contacts_in_demo': len(contact_info),  # Total number of contacts in this demo
                            'contact_timestep': contact_timestep,  # When the contact occurred
                            'contact_object_name': contact_object,  # Name of the object being contacted
                            
                            # Pose information
                            'pose_timestep': pose_timestep,  # Actual timestep when this EE pose was recorded
                            'overview_image_idx_for_pose': overview_idx_for_pose,  # Overview image index at pose timestep (for debugging)
                            
                            # Extraction parameters
                            'pre_contact_steps': args.pre_contact,  # How many steps before contact to start extraction
                            'pre_contact_range': args.pre_contact_range,  # How many poses to extract
                            'overview_percentage': args.overview_percentage,  # Percentage of contact timestep for images
                        }
                        all_pairs.append(pair)

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

    print(f"Total images saved: {image_counter}")
    print(f"Total poses saved: {pose_counter}")
    print(f"Total pairs generated: {len(all_pairs)}")

if __name__ == "__main__":
    main() 
