#!/usr/bin/env python3
"""
combine_skills_for_finetuning.py

Combine augmented demonstrations from multiple sources for finetuning preparation.

Features:
- Combine augmented and original demos from v1-v5 parallel runs
- Merge HDF5 files and initial state files separately
- Support debug mode for 3 specific skills
- Print detailed statistics

Usage:
# Normal mode - combine all skills
python scripts/phase2/combine_skills_for_finetuning.py

# Debug mode - combine only 3 focus skills
python scripts/phase2/combine_skills_for_finetuning.py --debug
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

def combine_hdf5_files(input_files: List[str], output_file: str, is_original: bool = False, max_demos: int = None) -> tuple:
    """
    Combine multiple HDF5 files into a single file.

    Args:
        input_files: List of input HDF5 file paths
        output_file: Output file path
        is_original: If True, limit to max_demos (50 for original)
        max_demos: Maximum number of demos to include (only for original demos)

    Returns:
        Tuple of (total_demos_found, demos_actually_saved)
    """
    if not input_files:
        return 0, 0

    total_demos_found = 0
    demos_saved = 0
    demo_idx = 0

    # For original demos, limit to 50
    if is_original and max_demos is None:
        max_demos = 50

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with h5py.File(output_file, 'w') as out_f:
        # Create data group in output file
        data_group = out_f.create_group('data')

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

                            # Copy entire demo group to output with new index
                            new_demo_key = f"demo_{demo_idx}"
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

                            # Copy entire demo group to output with new index
                            new_demo_key = f"demo_{demo_idx}"
                            in_f.copy(demo_key, data_group, name=new_demo_key)
                            demo_idx += 1
                            demos_saved += 1

            except Exception as e:
                print(f"❌ Error processing {input_file}: {e}")

    return total_demos_found, demos_saved

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
    Combine files for all specified skills.

    Returns:
        Statistics dictionary with counts for each skill
    """
    stats = {}

    # Create examples directory for videos
    examples_dir = os.path.join(output_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    print(f"🔄 Combining {len(skills)} skills...")

    for skill_idx, skill_name in enumerate(skills):
        print(f"\n[{skill_idx + 1}/{len(skills)}] Processing: {skill_name}")

        # Collect all files for this skill
        files = collect_files_for_skill(skill_name, input_dirs)

        skill_stats = {
            'original_demos_found': 0,
            'original_demos_saved': 0,
            'augmented_demos_found': 0,
            'augmented_demos_saved': 0,
            'original_inits': 0,
            'augmented_inits': 0
        }

        # Combine original HDF5 files (limit to 50 demos)
        if files['original_hdf5']:
            original_hdf5_out = os.path.join(output_dir, f"{skill_name}_original_demo.hdf5")
            found, saved = combine_hdf5_files(files['original_hdf5'], original_hdf5_out, is_original=True)
            skill_stats['original_demos_found'] = found
            skill_stats['original_demos_saved'] = saved
            print(f"   ✅ Combined {len(files['original_hdf5'])} original HDF5 files → {found} found, {saved} saved (limited to 50)")

            # Generate example video for original demos
            if saved > 0:
                video_path = os.path.join(examples_dir, f"{skill_name}_original_example.mp4")
                generate_example_video(original_hdf5_out, video_path, skill_name, "original")

        # Combine augmented HDF5 files (no limit)
        if files['augmented_hdf5']:
            augmented_hdf5_out = os.path.join(output_dir, f"{skill_name}_augmented_demo.hdf5")
            found, saved = combine_hdf5_files(files['augmented_hdf5'], augmented_hdf5_out, is_original=False)
            skill_stats['augmented_demos_found'] = found
            skill_stats['augmented_demos_saved'] = saved
            print(f"   ✅ Combined {len(files['augmented_hdf5'])} augmented HDF5 files → {found} found, {saved} saved")

            # Generate example video for augmented demos
            if saved > 0:
                video_path = os.path.join(examples_dir, f"{skill_name}_augmented_example.mp4")
                generate_example_video(augmented_hdf5_out, video_path, skill_name, "augmented")

        # Combine original init files
        if files['original_init']:
            original_init_out = os.path.join(output_dir, f"{skill_name}_original.init")
            skill_stats['original_inits'] = combine_init_files(files['original_init'], original_init_out)
            print(f"   ✅ Combined {len(files['original_init'])} original init files → {skill_stats['original_inits']} states")

        # Combine augmented init files
        if files['augmented_init']:
            augmented_init_out = os.path.join(output_dir, f"{skill_name}_augmented.init")
            skill_stats['augmented_inits'] = combine_init_files(files['augmented_init'], augmented_init_out)
            print(f"   ✅ Combined {len(files['augmented_init'])} augmented init files → {skill_stats['augmented_inits']} states")

        # Combine both original and augmented init files into a single file
        if files['original_init'] or files['augmented_init']:
            combined_init_out = os.path.join(output_dir, f"{skill_name}_combined.init")
            all_init_files = files['original_init'] + files['augmented_init']
            combined_inits = combine_init_files(all_init_files, combined_init_out)
            print(f"   ✅ Combined all init files → {combined_inits} total states")

        stats[skill_name] = skill_stats

    return stats

def save_statistics(stats: Dict[str, Dict[str, int]], output_dir: str):
    """Save detailed statistics to JSON file."""
    stats_file = os.path.join(output_dir, "combination_statistics.json")

    # Calculate totals
    totals = {
        'total_skills': len(stats),
        'total_original_demos_found': sum(s.get('original_demos_found', s.get('original_demos', 0)) for s in stats.values()),
        'total_original_demos_saved': sum(s.get('original_demos_saved', s.get('original_demos', 0)) for s in stats.values()),
        'total_augmented_demos_found': sum(s.get('augmented_demos_found', s.get('augmented_demos', 0)) for s in stats.values()),
        'total_augmented_demos_saved': sum(s.get('augmented_demos_saved', s.get('augmented_demos', 0)) for s in stats.values()),
        'total_original_inits': sum(s['original_inits'] for s in stats.values()),
        'total_augmented_inits': sum(s['augmented_inits'] for s in stats.values())
    }

    full_stats = {
        'per_skill_stats': stats,
        'totals': totals
    }

    with open(stats_file, 'w') as f:
        json.dump(full_stats, f, indent=2)

    print(f"\n📊 COMBINATION STATISTICS:")
    print(f"{'='*60}")
    print(f"Total Skills: {totals['total_skills']}")
    print(f"Original Demos: {totals['total_original_demos_found']} found, {totals['total_original_demos_saved']} saved")
    print(f"Augmented Demos: {totals['total_augmented_demos_found']} found, {totals['total_augmented_demos_saved']} saved")
    print(f"Total Original Inits: {totals['total_original_inits']}")
    print(f"Total Augmented Inits: {totals['total_augmented_inits']}")

    print(f"\n📝 PER-SKILL BREAKDOWN:")
    print(f"{'='*60}")
    for skill_name, skill_stats in stats.items():
        print(f"{skill_name}:")
        if 'original_demos_found' in skill_stats:
            print(f"  Original: {skill_stats['original_demos_found']} found, {skill_stats['original_demos_saved']} saved, {skill_stats['original_inits']} inits")
            print(f"  Augmented: {skill_stats['augmented_demos_found']} found, {skill_stats['augmented_demos_saved']} saved, {skill_stats['augmented_inits']} inits")
        else:
            # Fallback for old format
            print(f"  Original: {skill_stats.get('original_demos', 0)} demos, {skill_stats['original_inits']} inits")
            print(f"  Augmented: {skill_stats.get('augmented_demos', 0)} demos, {skill_stats['augmented_inits']} inits")

    print(f"\n💾 Statistics saved to: {stats_file}")

def main():
    parser = argparse.ArgumentParser(description="Combine augmented skills data for finetuning")
    parser.add_argument("--debug", action="store_true", help="Debug mode: combine only 3 focus skills")
    args = parser.parse_args()

    print("🚀 Skill Data Combination Script")
    print("="*50)

    # Define input directories
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
        print(f"\n🐛 DEBUG MODE: Combining {len(skills)} focus skills")
    else:
        # Normal mode: all available skills
        skills = sorted(list(find_all_skills(input_dirs)))
        output_dir = "datasets/hdf5_datasets/atomic_local_demos_augmented/combined"
        print(f"\n🔄 NORMAL MODE: Combining {len(skills)} available skills")

    print(f"📁 Output directory: {output_dir}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Combine skills
    try:
        stats = combine_skills(skills, input_dirs, output_dir)

        # Save and display statistics
        save_statistics(stats, output_dir)

        print(f"\n✅ Combination completed successfully!")
        print(f"📁 Combined data saved to: {output_dir}")

        # Check if videos were generated
        examples_dir = os.path.join(output_dir, "examples")
        if os.path.exists(examples_dir):
            video_files = [f for f in os.listdir(examples_dir) if f.endswith('.mp4')]
            print(f"🎥 Generated {len(video_files)} example videos in: {examples_dir}")

    except Exception as e:
        print(f"\n❌ Error during combination: {e}")
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())