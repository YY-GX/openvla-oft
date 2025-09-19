#!/usr/bin/env python3
"""
post_processing_aug_demos.py

Post-process and combine augmented demonstrations from multiple sources for finetuning preparation.

Features:
- Combine augmented and original demos from v1-v5 parallel runs
- Fix .init files by extracting first timestep states from original demos
- Truncate augmented demos to first 60 timesteps (motion planner + 10 extra steps)
- Support debug mode for 3 specific skills
- Print detailed statistics

Usage:
# Normal mode - combine all skills
python scripts/phase2/utils/post_processing_aug_demos.py

# Debug mode - combine only 3 focus skills
python scripts/phase2/utils/post_processing_aug_demos.py --debug
"""

import argparse
import os
import pickle
import h5py
import json
from pathlib import Path
from typing import Dict, List, Set
import numpy as np
from collections import defaultdict
import traceback
import imageio

def find_all_skills(input_dirs: List[str]) -> Set[str]:
    """Find all unique skills across all input directories."""
    all_skills = set()

    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"⚠️  Directory not found: {input_dir}")
            continue

        for filename in os.listdir(input_dir):
            if filename.endswith('_original.init') or filename.endswith('_augmented.init'):
                # Extract skill name from filename
                skill_name = filename.replace('_original.init', '').replace('_augmented.init', '')
                all_skills.add(skill_name)

    return all_skills

def collect_files_for_skill(skill_name: str, input_dirs: List[str]) -> Dict[str, List[str]]:
    """
    Collect all files for a specific skill across all input directories.

    Returns:
        Dict with keys: 'original_hdf5', 'augmented_hdf5', 'original_init', 'augmented_init'
    """
    files = {
        'original_hdf5': [],
        'augmented_hdf5': [],
        'original_init': [],
        'augmented_init': []
    }

    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            continue

        # Check for HDF5 files
        original_hdf5 = os.path.join(input_dir, f"{skill_name}_original_demo.hdf5")
        augmented_hdf5 = os.path.join(input_dir, f"{skill_name}_augmented_demo.hdf5")

        if os.path.exists(original_hdf5):
            files['original_hdf5'].append(original_hdf5)
        if os.path.exists(augmented_hdf5):
            files['augmented_hdf5'].append(augmented_hdf5)

        # Check for init files
        original_init = os.path.join(input_dir, f"{skill_name}_original.init")
        augmented_init = os.path.join(input_dir, f"{skill_name}_augmented.init")

        if os.path.exists(original_init):
            files['original_init'].append(original_init)
        if os.path.exists(augmented_init):
            files['augmented_init'].append(augmented_init)

    return files

def extract_first_states_from_original_demos(input_files: List[str]) -> List[np.ndarray]:
    """
    Extract first timestep states from original demo HDF5 files.
    Uses the same format as 1_generate_augmented_demos.py:
    initial_state = np.concatenate([joint_states, gripper_states, extra_state])

    This ensures consistency with the augmented demo generation pipeline.

    Args:
        input_files: List of original demo HDF5 file paths

    Returns:
        List of concatenated initial states [joint(7) + gripper(2) + states(variable)]
    """
    first_states = []
    max_demos = 50
    demos_processed = 0

    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"⚠️  File not found: {input_file}")
            continue

        if demos_processed >= max_demos:
            break

        try:
            with h5py.File(input_file, 'r') as f:
                # Check if file has data group structure
                if 'data' in f:
                    data_group = f['data']
                    demo_keys = [k for k in data_group.keys() if k.startswith('demo_')]
                    demo_keys.sort(key=lambda x: int(x.split('_')[1]))  # Sort numerically

                    for demo_key in demo_keys:
                        if demos_processed >= max_demos:
                            break

                        demo_group = data_group[demo_key]

                        # Extract using same format as 1_generate_augmented_demos.py
                        if 'states' in demo_group and 'obs' in demo_group:
                            states = demo_group['states'][:]
                            obs = demo_group['obs']

                            if ('joint_states' in obs and 'gripper_states' in obs and len(states) > 0):
                                joint_states = obs['joint_states'][0]  # First timestep
                                gripper_states = obs['gripper_states'][0]  # First timestep
                                extra_state = states[0]  # First timestep simulation state

                                # Use same concatenation as 1_generate_augmented_demos.py
                                initial_state = np.concatenate([joint_states, gripper_states, extra_state])
                                first_states.append(initial_state)
                                demos_processed += 1

                else:
                    # Handle files with demos at root level (fallback)
                    demo_keys = [k for k in f.keys() if k.startswith('demo_')]
                    demo_keys.sort(key=lambda x: int(x.split('_')[1]))  # Sort numerically

                    for demo_key in demo_keys:
                        if demos_processed >= max_demos:
                            break

                        demo_group = f[demo_key]

                        # Extract using same format as 1_generate_augmented_demos.py
                        if 'states' in demo_group and 'obs' in demo_group:
                            states = demo_group['states'][:]
                            obs = demo_group['obs']

                            if ('joint_states' in obs and 'gripper_states' in obs and len(states) > 0):
                                joint_states = obs['joint_states'][0]  # First timestep
                                gripper_states = obs['gripper_states'][0]  # First timestep
                                extra_state = states[0]  # First timestep simulation state

                                # Use same concatenation as 1_generate_augmented_demos.py
                                initial_state = np.concatenate([joint_states, gripper_states, extra_state])
                                first_states.append(initial_state)
                                demos_processed += 1

        except Exception as e:
            print(f"❌ Error processing {input_file}: {e}")

    if first_states:
        state_dim = len(first_states[0])
        print(f"✅ Extracted {len(first_states)} states with format [joint(7)+gripper(2)+extra_state] = {state_dim}-dim each")
    else:
        print(f"⚠️  No states extracted from input files")

    return first_states

def truncate_augmented_demo(demo_group, max_timesteps: int = 60) -> Dict:
    """
    Truncate an augmented demo to first max_timesteps timesteps.

    Args:
        demo_group: HDF5 demo group
        max_timesteps: Maximum number of timesteps to keep

    Returns:
        Dictionary with truncated demo data
    """
    truncated_demo = {}

    # Get the actual length of the demo
    actions = demo_group['actions'][:]
    actual_length = len(actions)

    # Determine how many timesteps to keep
    timesteps_to_keep = min(actual_length, max_timesteps)

    # Truncate all time-series data
    for key in demo_group.keys():
        item = demo_group[key]

        # Check if this is a dataset (not a group)
        if hasattr(item, 'shape'):
            # This is a dataset
            data = item[:]

            # Check if this is time-series data (first dimension should match action length)
            if hasattr(data, 'shape') and len(data.shape) > 0 and data.shape[0] == actual_length:
                # Truncate time-series data
                truncated_demo[key] = data[:timesteps_to_keep]
            else:
                # Keep non-time-series data as is
                truncated_demo[key] = data
        else:
            # This is a group - handle nested data recursively
            truncated_demo[key] = {}
            for subkey in item.keys():
                subitem = item[subkey]
                if hasattr(subitem, 'shape'):
                    # This is a dataset within the group
                    subdata = subitem[:]

                    # Check if this is time-series data
                    if hasattr(subdata, 'shape') and len(subdata.shape) > 0 and subdata.shape[0] == actual_length:
                        # Truncate time-series data
                        truncated_demo[key][subkey] = subdata[:timesteps_to_keep]
                    else:
                        # Keep non-time-series data as is
                        truncated_demo[key][subkey] = subdata
                else:
                    # Nested group - skip for now (this shouldn't happen in typical robotic datasets)
                    print(f"    ⚠️  Skipping nested group '{key}/{subkey}' in truncation")

    return truncated_demo, timesteps_to_keep, actual_length

def combine_hdf5_files(input_files: List[str], output_file: str, is_original: bool = False,
                      max_demos: int = None, truncate_augmented: bool = False,
                      max_timesteps: int = 60) -> tuple:
    """
    Combine multiple HDF5 files into a single file.

    Args:
        input_files: List of input HDF5 file paths
        output_file: Output file path
        is_original: If True, limit to max_demos (50 for original)
        max_demos: Maximum number of demos to include (only for original demos)
        truncate_augmented: If True, truncate augmented demos to max_timesteps
        max_timesteps: Maximum timesteps for augmented demos

    Returns:
        Tuple of (total_demos_found, demos_actually_saved, truncation_info)
    """
    if not input_files:
        return 0, 0, {}

    total_demos_found = 0
    demos_saved = 0
    demo_idx = 0
    truncation_info = {'truncated_demos': 0, 'avg_original_length': 0, 'avg_final_length': 0}

    # For original demos, limit to 50
    if is_original and max_demos is None:
        max_demos = 50

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with h5py.File(output_file, 'w') as out_f:
        # Create data group in output file
        data_group = out_f.create_group('data')

        total_original_length = 0
        total_final_length = 0

        for input_file in input_files:
            if not os.path.exists(input_file):
                print(f"⚠️  File not found: {input_file}")
                continue

            # Stop if we've reached the demo limit for original demos
            if is_original and max_demos and demos_saved >= max_demos:
                break

            try:
                with h5py.File(input_file, 'r') as in_f:
                    # Check if file has data group structure
                    if 'data' in in_f:
                        # Process demos from data group
                        data_in = in_f['data']
                        demo_keys = [k for k in data_in.keys() if k.startswith('demo_')]
                        demo_keys.sort(key=lambda x: int(x.split('_')[1]))  # Sort numerically

                        for demo_key in demo_keys:
                            total_demos_found += 1

                            # For original demos, check if we've reached the limit
                            if is_original and max_demos and demos_saved >= max_demos:
                                break

                            demo_group = data_in[demo_key]
                            new_demo_key = f"demo_{demo_idx}"

                            if truncate_augmented and not is_original:
                                # Truncate augmented demo
                                truncated_data, kept_timesteps, original_length = truncate_augmented_demo(demo_group, max_timesteps)

                                # Create new demo group in output
                                new_demo_group = data_group.create_group(new_demo_key)

                                # Copy truncated data
                                for key, data in truncated_data.items():
                                    if isinstance(data, dict):
                                        # This is a nested group
                                        subgroup = new_demo_group.create_group(key)
                                        for subkey, subdata in data.items():
                                            subgroup.create_dataset(subkey, data=subdata)
                                    else:
                                        # This is a regular dataset
                                        new_demo_group.create_dataset(key, data=data)

                                # Update truncation stats
                                truncation_info['truncated_demos'] += 1
                                total_original_length += original_length
                                total_final_length += kept_timesteps

                                if kept_timesteps < original_length:
                                    print(f"    📐 Truncated {demo_key}: {original_length} → {kept_timesteps} timesteps")
                            else:
                                # Copy entire demo group without truncation
                                data_in.copy(demo_key, data_group, name=new_demo_key)

                            demo_idx += 1
                            demos_saved += 1
                    else:
                        # Handle files with demos at root level (fallback)
                        demo_keys = [k for k in in_f.keys() if k.startswith('demo_')]
                        demo_keys.sort(key=lambda x: int(x.split('_')[1]))  # Sort numerically

                        for demo_key in demo_keys:
                            total_demos_found += 1

                            # For original demos, check if we've reached the limit
                            if is_original and max_demos and demos_saved >= max_demos:
                                break

                            demo_group = in_f[demo_key]
                            new_demo_key = f"demo_{demo_idx}"

                            if truncate_augmented and not is_original:
                                # Truncate augmented demo
                                truncated_data, kept_timesteps, original_length = truncate_augmented_demo(demo_group, max_timesteps)

                                # Create new demo group in output
                                new_demo_group = data_group.create_group(new_demo_key)

                                # Copy truncated data
                                for key, data in truncated_data.items():
                                    if isinstance(data, dict):
                                        # This is a nested group
                                        subgroup = new_demo_group.create_group(key)
                                        for subkey, subdata in data.items():
                                            subgroup.create_dataset(subkey, data=subdata)
                                    else:
                                        # This is a regular dataset
                                        new_demo_group.create_dataset(key, data=data)

                                # Update truncation stats
                                truncation_info['truncated_demos'] += 1
                                total_original_length += original_length
                                total_final_length += kept_timesteps

                                if kept_timesteps < original_length:
                                    print(f"    📐 Truncated {demo_key}: {original_length} → {kept_timesteps} timesteps")
                            else:
                                # Copy entire demo group without truncation
                                in_f.copy(demo_key, data_group, name=new_demo_key)

                            demo_idx += 1
                            demos_saved += 1

            except Exception as e:
                print(f"❌ Error processing {input_file}: {e}")

        # Calculate averages for truncation stats
        if truncation_info['truncated_demos'] > 0:
            truncation_info['avg_original_length'] = total_original_length / truncation_info['truncated_demos']
            truncation_info['avg_final_length'] = total_final_length / truncation_info['truncated_demos']

    return total_demos_found, demos_saved, truncation_info

def save_corrected_init_file(first_states: List[np.ndarray], output_file: str) -> int:
    """
    Save corrected initial states to a pickle file.

    Args:
        first_states: List of first timestep states
        output_file: Output file path

    Returns:
        Number of states saved
    """
    if not first_states:
        return 0

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save corrected states
    with open(output_file, 'wb') as f:
        pickle.dump(first_states, f)

    return len(first_states)

def combine_init_files(input_files: List[str], output_file: str) -> int:
    """
    Combine multiple pickle init files into a single file.

    Returns:
        Number of initial states combined
    """
    if not input_files:
        return 0

    combined_states = []

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"⚠️  File not found: {input_file}")
            continue

        try:
            with open(input_file, 'rb') as f:
                states = pickle.load(f)
                if isinstance(states, list):
                    combined_states.extend(states)
                else:
                    combined_states.append(states)

        except Exception as e:
            print(f"❌ Error processing {input_file}: {e}")

    # Save combined states
    if combined_states:
        with open(output_file, 'wb') as f:
            pickle.dump(combined_states, f)

    return len(combined_states)

def generate_example_video(hdf5_file: str, output_video_path: str, skill_name: str, demo_type: str) -> bool:
    """
    Generate an example video from the first demo in an HDF5 file.

    Args:
        hdf5_file: Path to HDF5 file
        output_video_path: Output video file path
        skill_name: Name of the skill for logging
        demo_type: Type of demo (original/augmented) for logging

    Returns:
        True if video was generated successfully, False otherwise
    """
    try:
        with h5py.File(hdf5_file, 'r') as f:
            if 'data' not in f:
                print(f"⚠️  No data group in {hdf5_file}")
                return False

            data_group = f['data']
            demo_keys = [k for k in data_group.keys() if k.startswith('demo_')]

            if not demo_keys:
                print(f"⚠️  No demos found in {hdf5_file}")
                return False

            # Use the first demo
            demo_keys.sort(key=lambda x: int(x.split('_')[1]))
            first_demo = demo_keys[0]

            demo_group = data_group[first_demo]

            # Look for observation data with images
            obs_group = demo_group.get('obs')
            if obs_group is None:
                print(f"⚠️  No obs group in {first_demo}")
                return False

            # Try to find camera images (prioritize eye_in_hand/wrist, fall back to agent)
            image_key = None
            for key in ['eye_in_hand_rgb', 'wrist_image', 'agentview_rgb', 'agent_image', 'image']:
                if key in obs_group:
                    image_key = key
                    break

            if image_key is None:
                print(f"⚠️  No image data found in {first_demo}")
                return False

            # Get image data
            images = obs_group[image_key][:]

            if len(images) == 0:
                print(f"⚠️  Empty image sequence in {first_demo}")
                return False

            # Create output directory
            output_dir = os.path.dirname(output_video_path)
            if output_dir:  # Only create if there's a directory part
                os.makedirs(output_dir, exist_ok=True)

            # Generate video
            with imageio.get_writer(output_video_path, fps=10) as writer:
                for img in images:
                    # Handle different image formats
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)

                    # Ensure RGB format
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        writer.append_data(img)
                    else:
                        print(f"⚠️  Unexpected image shape: {img.shape}")
                        return False

            print(f"   🎥 Generated {demo_type} video: {os.path.basename(output_video_path)} ({len(images)} frames)")
            return True

    except Exception as e:
        print(f"❌ Error generating video for {skill_name} ({demo_type}): {e}")
        return False

def combine_skills(skills: List[str], input_dirs: List[str], output_dir: str) -> Dict[str, Dict[str, int]]:
    """
    Combine files for all specified skills with post-processing.

    Returns:
        Statistics dictionary with counts for each skill
    """
    stats = {}

    # Create examples directory for videos
    examples_dir = os.path.join(output_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    print(f"🔄 Processing and combining {len(skills)} skills...")

    for skill_idx, skill_name in enumerate(skills):
        print(f"\n[{skill_idx + 1}/{len(skills)}] Processing: {skill_name}")

        # Collect all files for this skill
        files = collect_files_for_skill(skill_name, input_dirs)

        skill_stats = {
            'original_demos_found': 0,
            'original_demos_saved': 0,
            'augmented_demos_found': 0,
            'augmented_demos_saved': 0,
            'original_inits_corrected': 0,
            'augmented_inits': 0,
            'truncation_stats': {}
        }

        # Extract corrected initial states from original demos
        if files['original_hdf5']:
            print(f"   🔧 Extracting corrected initial states from original demos...")
            first_states = extract_first_states_from_original_demos(files['original_hdf5'])
            if first_states:
                corrected_init_out = os.path.join(output_dir, f"{skill_name}_original.init")
                skill_stats['original_inits_corrected'] = save_corrected_init_file(first_states, corrected_init_out)
                print(f"   ✅ Corrected original init file → {skill_stats['original_inits_corrected']} states from first timesteps")

        # Combine original HDF5 files (limit to 50 demos, no truncation)
        if files['original_hdf5']:
            original_hdf5_out = os.path.join(output_dir, f"{skill_name}_original_demo.hdf5")
            found, saved, _ = combine_hdf5_files(files['original_hdf5'], original_hdf5_out,
                                               is_original=True, truncate_augmented=False)
            skill_stats['original_demos_found'] = found
            skill_stats['original_demos_saved'] = saved
            print(f"   ✅ Combined {len(files['original_hdf5'])} original HDF5 files → {found} found, {saved} saved (limited to 50)")

            # Generate example video for original demos
            if saved > 0:
                video_path = os.path.join(examples_dir, f"{skill_name}_original_example.mp4")
                generate_example_video(original_hdf5_out, video_path, skill_name, "original")

        # Combine augmented HDF5 files with truncation to 60 timesteps
        if files['augmented_hdf5']:
            print(f"   🔧 Combining and truncating augmented demos to 60 timesteps...")
            augmented_hdf5_out = os.path.join(output_dir, f"{skill_name}_augmented_demo.hdf5")
            found, saved, truncation_info = combine_hdf5_files(files['augmented_hdf5'], augmented_hdf5_out,
                                                              is_original=False, truncate_augmented=True,
                                                              max_timesteps=60)
            skill_stats['augmented_demos_found'] = found
            skill_stats['augmented_demos_saved'] = saved
            skill_stats['truncation_stats'] = truncation_info

            print(f"   ✅ Combined {len(files['augmented_hdf5'])} augmented HDF5 files → {found} found, {saved} saved")
            if truncation_info['truncated_demos'] > 0:
                print(f"   📐 Truncated {truncation_info['truncated_demos']} demos: avg {truncation_info['avg_original_length']:.1f} → {truncation_info['avg_final_length']:.1f} timesteps")

            # Generate example video for augmented demos
            if saved > 0:
                video_path = os.path.join(examples_dir, f"{skill_name}_augmented_example.mp4")
                generate_example_video(augmented_hdf5_out, video_path, skill_name, "augmented")

        # Note: Skipping augmented init files - only need original corrected init files

        # Create combined init file with only corrected original states
        if skill_stats['original_inits_corrected'] > 0:
            corrected_init_file = os.path.join(output_dir, f"{skill_name}_original.init")
            combined_init_out = os.path.join(output_dir, f"{skill_name}_combined.init")
            combined_inits = combine_init_files([corrected_init_file], combined_init_out)
            print(f"   ✅ Created combined init file → {combined_inits} corrected original states")

        stats[skill_name] = skill_stats

    return stats

def save_statistics(stats: Dict[str, Dict[str, int]], output_dir: str):
    """Save detailed statistics to JSON file."""
    stats_file = os.path.join(output_dir, "post_processing_statistics.json")

    # Calculate totals
    totals = {
        'total_skills': len(stats),
        'total_original_demos_found': sum(s.get('original_demos_found', 0) for s in stats.values()),
        'total_original_demos_saved': sum(s.get('original_demos_saved', 0) for s in stats.values()),
        'total_augmented_demos_found': sum(s.get('augmented_demos_found', 0) for s in stats.values()),
        'total_augmented_demos_saved': sum(s.get('augmented_demos_saved', 0) for s in stats.values()),
        'total_original_inits_corrected': sum(s.get('original_inits_corrected', 0) for s in stats.values()),
        'total_augmented_inits': sum(s.get('augmented_inits', 0) for s in stats.values()),
        'total_demos_truncated': sum(s.get('truncation_stats', {}).get('truncated_demos', 0) for s in stats.values())
    }

    full_stats = {
        'per_skill_stats': stats,
        'totals': totals
    }

    with open(stats_file, 'w') as f:
        json.dump(full_stats, f, indent=2)

    print(f"\n📊 POST-PROCESSING STATISTICS:")
    print(f"{'='*60}")
    print(f"Total Skills: {totals['total_skills']}")
    print(f"Original Demos: {totals['total_original_demos_found']} found, {totals['total_original_demos_saved']} saved")
    print(f"Augmented Demos: {totals['total_augmented_demos_found']} found, {totals['total_augmented_demos_saved']} saved")
    print(f"Demos Truncated to 60 timesteps: {totals['total_demos_truncated']}")
    print(f"Original Inits Corrected: {totals['total_original_inits_corrected']}")
    print(f"Augmented Inits: {totals['total_augmented_inits']}")

    print(f"\n📝 PER-SKILL BREAKDOWN:")
    print(f"{'='*60}")
    for skill_name, skill_stats in stats.items():
        print(f"{skill_name}:")
        print(f"  Original: {skill_stats.get('original_demos_found', 0)} found, {skill_stats.get('original_demos_saved', 0)} saved, {skill_stats.get('original_inits_corrected', 0)} inits corrected")
        print(f"  Augmented: {skill_stats.get('augmented_demos_found', 0)} found, {skill_stats.get('augmented_demos_saved', 0)} saved, {skill_stats.get('augmented_inits', 0)} inits")

        truncation_stats = skill_stats.get('truncation_stats', {})
        if truncation_stats.get('truncated_demos', 0) > 0:
            print(f"  Truncation: {truncation_stats['truncated_demos']} demos, avg {truncation_stats['avg_original_length']:.1f} → {truncation_stats['avg_final_length']:.1f} timesteps")

    print(f"\n💾 Statistics saved to: {stats_file}")

def get_skill_language_from_bddl(skill_name: str) -> str:
    """
    Extract language description from BDDL file.

    Args:
        skill_name: Name of the skill (e.g., "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet")

    Returns:
        Language description string
    """
    # Try different possible BDDL paths
    possible_paths = [
        f"externals/boss/libero/libero/bddl_files/atomic_skills/{skill_name}.bddl",
        f"/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss/libero/libero/bddl_files/atomic_skills/{skill_name}.bddl"
    ]

    for bddl_file_path in possible_paths:
        if not os.path.exists(bddl_file_path):
            continue

        try:
            with open(bddl_file_path, 'r') as f:
                content = f.read()

            # Look for (:language ...) line
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('(:language'):
                    # Extract the language part after (:language
                    language = line.replace('(:language', '').strip().rstrip(')')
                    return language

        except Exception as e:
            print(f"Warning: Could not read BDDL file {bddl_file_path}: {e}")
            continue

    # Fallback: parse skill name to extract language
    # KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet -> open the bottom drawer of the cabinet
    parts = skill_name.split('_')
    if len(parts) > 2 and parts[0].startswith('KITCHEN_SCENE'):
        language_parts = parts[2:]  # Skip KITCHEN_SCENE1/2/etc
        return ' '.join(language_parts).replace('_', ' ')

    return skill_name.replace('_', ' ')


def combine_skills_by_language(input_dir: str, output_dir: str) -> Dict[str, Dict]:
    """
    Final step: Combine skills by language instruction.
    Creates 3 files per language: *_augmented.hdf5, *_original.hdf5, *.init

    Args:
        input_dir: Directory containing combined skill files
        output_dir: Directory to save language-combined files

    Returns:
        Statistics dictionary
    """
    print(f"\n🔤 FINAL STEP: Combining skills by language instruction")
    print(f"   Input: {input_dir}")
    print(f"   Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Find all skills with their files
    from collections import defaultdict
    language_groups = defaultdict(list)

    # Group skills by language
    for filename in os.listdir(input_dir):
        if filename.endswith('_original_demo.hdf5'):
            skill_name = filename.replace('_original_demo.hdf5', '')
            language = get_skill_language_from_bddl(skill_name)
            language_groups[language].append(skill_name)
            print(f"   {skill_name} -> '{language}'")

    print(f"\n📊 Found {len(language_groups)} unique languages:")
    for language, skills in language_groups.items():
        print(f"   '{language}': {len(skills)} skills")

    stats = {}

    for language, skill_names in language_groups.items():
        print(f"\n🔄 Processing language: '{language}' ({len(skill_names)} skills)")

        # Create safe filename
        safe_language = language.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('.', '')

        # Combine original demos
        original_files = []
        augmented_files = []
        init_files = []

        for skill_name in skill_names:
            orig_file = os.path.join(input_dir, f"{skill_name}_original_demo.hdf5")
            aug_file = os.path.join(input_dir, f"{skill_name}_augmented_demo.hdf5")
            init_file = os.path.join(input_dir, f"{skill_name}_original.init")

            if os.path.exists(orig_file):
                original_files.append(orig_file)
            if os.path.exists(aug_file):
                augmented_files.append(aug_file)
            if os.path.exists(init_file):
                init_files.append(init_file)

        # Combine original demos
        if original_files:
            output_orig = os.path.join(output_dir, f"{safe_language}_original.hdf5")
            orig_found, orig_saved, _ = combine_hdf5_files(original_files, output_orig, is_original=False)
            print(f"   ✅ Combined original: {orig_found} found, {orig_saved} saved -> {os.path.basename(output_orig)}")

        # Combine augmented demos
        if augmented_files:
            output_aug = os.path.join(output_dir, f"{safe_language}_augmented.hdf5")
            aug_found, aug_saved, _ = combine_hdf5_files(augmented_files, output_aug, is_original=False)
            print(f"   ✅ Combined augmented: {aug_found} found, {aug_saved} saved -> {os.path.basename(output_aug)}")

        # Combine init files
        if init_files:
            output_init = os.path.join(output_dir, f"{safe_language}.init")
            combined_states = combine_init_files(init_files, output_init)
            print(f"   ✅ Combined init: {combined_states} states -> {os.path.basename(output_init)}")

        stats[language] = {
            'safe_name': safe_language,
            'skill_count': len(skill_names),
            'original_demos': orig_saved if original_files else 0,
            'augmented_demos': aug_saved if augmented_files else 0,
            'init_states': combined_states if init_files else 0,
            'skills': skill_names
        }

    return stats


def save_language_combination_stats(stats: Dict[str, Dict], output_dir: str):
    """Save statistics for language combination step."""
    stats_file = os.path.join(output_dir, "language_combination_stats.json")

    # Calculate totals
    totals = {
        'total_languages': len(stats),
        'total_skills': sum(s['skill_count'] for s in stats.values()),
        'total_original_demos': sum(s['original_demos'] for s in stats.values()),
        'total_augmented_demos': sum(s['augmented_demos'] for s in stats.values()),
        'total_init_states': sum(s['init_states'] for s in stats.values())
    }

    full_stats = {
        'per_language_stats': stats,
        'totals': totals
    }

    with open(stats_file, 'w') as f:
        json.dump(full_stats, f, indent=2)

    print(f"\n📊 LANGUAGE COMBINATION STATISTICS:")
    print(f"{'='*60}")
    print(f"Total Languages: {totals['total_languages']}")
    print(f"Total Skills: {totals['total_skills']}")
    print(f"Total Original Demos: {totals['total_original_demos']}")
    print(f"Total Augmented Demos: {totals['total_augmented_demos']}")
    print(f"Total Init States: {totals['total_init_states']}")

    print(f"\n📝 PER-LANGUAGE BREAKDOWN:")
    print(f"{'='*60}")
    for language, data in stats.items():
        print(f"{language}:")
        print(f"  Skills: {data['skill_count']} ({', '.join(data['skills'][:3])}{'...' if len(data['skills']) > 3 else ''})")
        print(f"  Original Demos: {data['original_demos']}")
        print(f"  Augmented Demos: {data['augmented_demos']}")
        print(f"  Init States: {data['init_states']}")

    print(f"\n💾 Language combination statistics saved to: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description="Post-process and combine augmented skills data for finetuning")
    parser.add_argument("--debug", action="store_true", help="Debug mode: combine only 3 focus skills")
    parser.add_argument("--language_only", action="store_true", help="Only run final language combination step")
    args = parser.parse_args()

    print("🚀 Augmented Demos Post-Processing Script")
    print("="*50)

    if args.language_only:
        # Only run language combination on existing combined data
        input_dir = "datasets/hdf5_datasets/atomic_local_demos_augmented/combined/debug" if args.debug else "datasets/hdf5_datasets/atomic_local_demos_augmented/combined"
        output_dir = f"{input_dir}_by_language"

        print(f"🔤 LANGUAGE COMBINATION MODE")
        print(f"   Input: {input_dir}")
        print(f"   Output: {output_dir}")

        if not os.path.exists(input_dir):
            print(f"❌ Input directory does not exist: {input_dir}")
            return 1

        stats = combine_skills_by_language(input_dir, output_dir)
        save_language_combination_stats(stats, output_dir)

        print(f"\n✅ Language combination completed successfully!")
        print(f"📁 Combined data saved to: {output_dir}")
        return 0

    # Define input directories (updated path structure)
    input_dirs = [
        "datasets/hdf5_datasets/atomic_local_demos_augmented/v1",
        "datasets/hdf5_datasets/atomic_local_demos_augmented/v2",
        "datasets/hdf5_datasets/atomic_local_demos_augmented/v3",
        "datasets/hdf5_datasets/atomic_local_demos_augmented/v4",
        "datasets/hdf5_datasets/atomic_local_demos_augmented/v5"
    ]

    print(f"📂 Input directories:")
    for input_dir in input_dirs:
        exists = "✅" if os.path.exists(input_dir) else "❌"
        print(f"   {exists} {input_dir}")

    if args.debug:
        # Debug mode: 3 specific skills
        skills = [
            "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
            "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_pick",
            "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet"
        ]
        output_dir = "datasets/hdf5_datasets/atomic_local_demos_augmented/combined/debug"
        print(f"\n🐛 DEBUG MODE: Processing {len(skills)} focus skills")
    else:
        # Normal mode: all available skills
        skills = sorted(list(find_all_skills(input_dirs)))
        output_dir = "datasets/hdf5_datasets/atomic_local_demos_augmented/combined"
        print(f"\n🔄 NORMAL MODE: Processing {len(skills)} available skills")

    print(f"📁 Output directory: {output_dir}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🔧 POST-PROCESSING STEPS:")
    print(f"   1. Extract first timestep states from original demos as corrected .init files")
    print(f"   2. Truncate augmented demos to first 60 timesteps (50 motion planner + 10 extra)")
    print(f"   3. Keep original demos unchanged (limited to 50 per skill)")
    print(f"   4. Combine all data into unified files")
    print(f"   5. Final step: Combine skills by language instruction")

    # Process and combine skills
    try:
        stats = combine_skills(skills, input_dirs, output_dir)

        # Save and display statistics
        save_statistics(stats, output_dir)

        print(f"\n✅ Step 1-4 completed successfully!")
        print(f"📁 Processed data saved to: {output_dir}")

        # Check if videos were generated
        examples_dir = os.path.join(output_dir, "examples")
        if os.path.exists(examples_dir):
            video_files = [f for f in os.listdir(examples_dir) if f.endswith('.mp4')]
            print(f"🎥 Generated {len(video_files)} example videos in: {examples_dir}")

        # Final step: Combine by language
        final_output_dir = f"{output_dir}_by_language"
        language_stats = combine_skills_by_language(output_dir, final_output_dir)
        save_language_combination_stats(language_stats, final_output_dir)

        print(f"\n✅ All post-processing completed successfully!")
        print(f"📁 Final combined data saved to: {final_output_dir}")

    except Exception as e:
        print(f"\n❌ Error during post-processing: {e}")
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())