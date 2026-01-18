#!/usr/bin/env python3
"""
Phase 2 object-EE pose pairs extraction script for augmented datasets.

This script extracts pose pairs from augmented initial states:
1. Load .init file from augmented dataset (contains states where EE is positioned to interact with target object)
2. Set simulation state
3. Extract EE pose from observations (env.step with dummy action)
4. Extract target object pose (known from unified_task_config.json)
5. Calculate orientation analysis (EE facing object ratio)
6. Save pose pairs as .pkl files and preview images as .png files

Uses observations coordinate system instead of MuJoCo world coordinates.
Works with augmented datasets from Phase 2 pipeline.

Modes:
- normal: Process ALL .init files in dataset_path
- lht_oriented: Only process .init files for skills in unified_task_config.json
"""

import os
import sys
import pickle
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

# Add project paths
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')

# LIBERO imports
from libero.libero.envs import OffScreenRenderEnv

# Local imports
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase1')

# Import get_object_pose function
import importlib.util
spec_detector = importlib.util.spec_from_file_location("contact_detector", "scripts/phase2/3_phase2_contact_object_detector.py")
contact_detector = importlib.util.module_from_spec(spec_detector)
spec_detector.loader.exec_module(contact_detector)
get_object_pose = contact_detector.get_object_pose


def load_unified_config() -> Dict[str, Any]:
    """
    Load unified task config from JSON file.

    Returns:
        Dictionary with skill_mappings and other config data
    """
    config_file = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase3/pipeline/config/task_config.json"

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)

        print(f"📋 Loaded unified config with {len(config.get('skill_mappings', {}))} skill mappings")
        return config

    except Exception as e:
        print(f"❌ Error loading unified config: {e}")
        return {}


def build_init_file_mapping(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build mapping from init filename to skill info from unified config.

    Args:
        config: Unified task config dictionary

    Returns:
        Dictionary mapping init filename to skill info (target_object, bddl_files, language)
    """
    mapping = {}
    skill_mappings = config.get('skill_mappings', {})

    for skill_name, skill_data in skill_mappings.items():
        init_files = skill_data.get('init_files', [])
        target_object = skill_data.get('target_object', '')
        bddl_files = skill_data.get('bddl_files', [])
        language = skill_data.get('language', skill_name)

        # Map each init file to this skill's data
        for init_file in init_files:
            mapping[init_file] = {
                'object': target_object,
                'bddl_files': bddl_files if isinstance(bddl_files, list) else [bddl_files],
                'language': language,
                'skill_name': skill_name
            }

    print(f"🗂️  Built mapping for {len(mapping)} init files from unified config")
    return mapping


def calculate_ee_orientation_toward_object(ee_pos: np.ndarray, ee_quat: np.ndarray, 
                                         obj_pos: np.ndarray) -> Tuple[float, bool]:
    """
    Calculate if EE orientation is facing toward the target object.
    
    Args:
        ee_pos: End-effector position (3,)
        ee_quat: End-effector quaternion [w, x, y, z] (4,)
        obj_pos: Object position (3,)
        
    Returns:
        (dot_product, is_facing): dot product and whether EE is facing object (>0.5)
    """
    # Convert quaternion to rotation matrix
    w, x, y, z = ee_quat
    rotation_matrix = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    
    # EE forward direction (assume -Z axis of gripper points forward toward object)
    ee_forward = -rotation_matrix[:, 2]  # -Z axis
    
    # Direction from EE to object
    obj_direction = obj_pos - ee_pos
    obj_direction_norm = obj_direction / (np.linalg.norm(obj_direction) + 1e-8)
    
    # Dot product measures alignment
    dot_product = float(np.dot(ee_forward, obj_direction_norm))
    is_facing = dot_product > 0.5  # Threshold for "facing toward"
    
    return dot_product, is_facing


def save_preview_image(env, skill_name: str, state_idx: int, output_dir: str) -> bool:
    """
    Save a preview image of the current simulation state.
    
    Args:
        env: Environment object
        skill_name: Name of the skill
        state_idx: Index of the state (for filename)
        output_dir: Directory to save the image
        
    Returns:
        True if image saved successfully
    """
    try:
        # Configure matplotlib for headless rendering
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Get image from observation by taking a dummy step
        # Use a zero action (no movement) - 7D action space for robot
        dummy_action = np.zeros(7)
        obs, _, _, _ = env.step(dummy_action)
        
        # Extract agentview image from observation
        if 'agentview_image' in obs:
            img = obs['agentview_image']
        elif 'image' in obs:
            img = obs['image']
        else:
            # Try to find any image key in obs
            img_keys = [k for k in obs.keys() if 'image' in k.lower()]
            if img_keys:
                img = obs[img_keys[0]]
                print(f"    📷 Using image key: {img_keys[0]}")
            else:
                print(f"    ⚠️  No image found in observation keys: {list(obs.keys())}")
                return False
        
        if img is None:
            print(f"    ⚠️  Image is None")
            return False
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save image using matplotlib
        output_file = os.path.join(output_dir, f"{skill_name}_state_{state_idx}_preview.png")
        plt.imsave(output_file, img)
        
        print(f"    💾 Preview image saved: {output_file}")
        return True
        
    except Exception as e:
        print(f"    ⚠️  Failed to save preview image: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_simple_pose_pairs(init_file_path: str, target_object: str, output_dir: str, bddl_files: List[str] = None) -> Tuple[bool, int, float, float]:
    """
    Simple pose pair extraction: load .init state, extract EE and object poses.

    Args:
        init_file_path: Path to the .init file
        target_object: Target object name from unified config
        output_dir: Directory to save extracted pose pairs
        bddl_files: List of BDDL file names (use first valid one)

    Returns:
        (success, num_pairs, avg_distance, facing_ratio) tuple
    """
    try:
        skill_name = Path(init_file_path).stem
        print(f"\n🎯 Processing {skill_name}")
        print(f"  🎯 Target: {target_object}")

        # Load initial states
        with open(init_file_path, 'rb') as f:
            initial_states = pickle.load(f)

        if not initial_states:
            print(f"  ❌ No initial states found")
            return False, 0, 0.0, 0.0

        print(f"  📊 Processing {len(initial_states)} states")

        # Process each initial state
        pose_pairs = []
        distances = []
        facing_results = []

        for state_idx, initial_state in enumerate(initial_states):
            if state_idx >= 3:  # Limit to first 3 states
                break

            print(f"  🔄 State {state_idx + 1}/{min(len(initial_states), 3)}...", end=' ')

            try:
                # Use BDDL file from unified config
                bddl_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss/libero/libero/bddl_files/atomic_skills"
                bddl_file = None

                if bddl_files and isinstance(bddl_files, list) and len(bddl_files) > 0:
                    # Try each BDDL file in order until we find one that exists
                    for bddl_name in bddl_files:
                        test_path = f"{bddl_dir}/{bddl_name}"
                        if os.path.exists(test_path):
                            bddl_file = test_path
                            print(f"    📋 Using BDDL: {bddl_name}")
                            break

                    if not bddl_file:
                        print(f"❌ No valid BDDL file found in {bddl_files}")
                        continue

                if not bddl_file:
                    print(f"❌ No valid BDDL file for {skill_name}")
                    continue
                
                # Initialize environment
                env_args = {
                    'bddl_file_name': bddl_file,
                    'camera_heights': 256,
                    'camera_widths': 256
                }
                
                env = OffScreenRenderEnv(**env_args)
                env.reset()
                
                # Set simulation state (skip first 9 dims: joint+gripper)
                sim_state_only = initial_state[9:]
                env.sim.set_state_from_flattened(sim_state_only)
                env.sim.forward()
                
                # Get target object pose
                obj_pose_result = get_object_pose(env, target_object)
                obj_pos, obj_quat = obj_pose_result

                if obj_pos is None or obj_quat is None:
                    # Try wooden->white substitution for cabinet objects
                    if "cabinet" in target_object and "wooden" in target_object:
                        alternative_name = target_object.replace("wooden", "white")
                        obj_pos, obj_quat = get_object_pose(env, alternative_name)
                        if obj_pos is not None and obj_quat is not None:
                            target_object = alternative_name

                    # If still not found, try fuzzy matching for numbered objects
                    if obj_pos is None or obj_quat is None:
                        import re
                        fuzzy_pattern = re.sub(r'_\d+_', '_.*_', target_object)
                        fuzzy_regex = fuzzy_pattern.replace('*', r'\d+')

                        model = env.sim.model
                        for i in range(model.nbody):
                            body_name = model.body_id2name(i)
                            if body_name and re.match(fuzzy_regex, body_name):
                                obj_pos, obj_quat = get_object_pose(env, body_name)
                                if obj_pos is not None and obj_quat is not None:
                                    target_object = body_name  # Update target name
                                    break

                if obj_pos is None or obj_quat is None:
                    print(f"❌ Object '{target_object}' not found")
                    env.close()
                    continue
                
                # Get end-effector pose from observations
                dummy_action = np.zeros(7)
                obs, _, _, _ = env.step(dummy_action)
                
                # Extract EE pose from observations
                if 'robot0_eef_pos' in obs and 'robot0_eef_quat' in obs:
                    ee_pos = obs['robot0_eef_pos'].copy()
                    ee_quat = obs['robot0_eef_quat'].copy()
                else:
                    print(f"❌ EE pose not found in observations. Available keys: {list(obs.keys())}")
                    env.close()
                    continue
                
                # Calculate distance
                distance = np.linalg.norm(ee_pos - obj_pos)
                distances.append(distance)
                
                # Calculate orientation alignment
                dot_product, is_facing = calculate_ee_orientation_toward_object(ee_pos, ee_quat, obj_pos)
                facing_results.append(is_facing)
                
                # Save preview image for each state (all 3 states)
                save_preview_image(env, skill_name, state_idx, output_dir)
                
                # Create pose pair
                pose_pair = {
                    'init_state_id': state_idx,
                    'object_pose': {
                        'object_name': target_object,
                        'position': obj_pos.tolist(),
                        'quaternion': obj_quat.tolist()
                    },
                    'ee_pose': {
                        'position': ee_pos.tolist(),
                        'quaternion': ee_quat.tolist()
                    },
                    'distance': float(distance),
                    'orientation_dot_product': float(dot_product),
                    'ee_facing_object': bool(is_facing)
                }
                
                pose_pairs.append(pose_pair)
                facing_symbol = "👀" if is_facing else "😵"
                print(f"✅ d={distance:.3f}m, dot={dot_product:.2f} {facing_symbol}")
                
                env.close()
                
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        if not pose_pairs:
            print(f"  ❌ No valid pose pairs extracted")
            return False, 0, 0.0, 0.0
        
        # Calculate statistics
        avg_distance = np.mean(distances)
        num_pairs = len(pose_pairs)
        facing_ratio = np.mean(facing_results) if facing_results else 0.0
        
        # Save pose pairs as pickle file
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{skill_name}_pose_pairs.pkl")
        
        with open(output_file, 'wb') as f:
            pickle.dump(pose_pairs, f)
        
        print(f"  ✅ Saved {num_pairs} pairs, avg_dist: {avg_distance:.3f}m, facing: {facing_ratio*100:.0f}%")
        print(f"  💾 Files: {output_file}")
        
        return True, num_pairs, avg_distance, facing_ratio
        
    except Exception as e:
        print(f"❌ Error processing {init_file_path}: {e}")
        return False, 0, 0.0, 0.0


def main():
    """Main function to process skills with normal or lht_oriented mode."""
    import argparse

    parser = argparse.ArgumentParser(description='Object-EE pose pairs extraction with unified config support')
    parser.add_argument('--mode', type=str, default='lht_oriented', choices=['normal', 'lht_oriented'],
                       help='Extraction mode: normal (all .init files) or lht_oriented (only skills in unified_task_config.json)')
    parser.add_argument('--test_single', type=str, help='Test single init file (e.g., pick_black_bowl.init)')
    parser.add_argument('--dataset_path', type=str,
                       default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_local_demos_augmented_pick_10_place_15_atomic_15/combined_by_language",
                       help='Path to dataset directory containing .init files')
    args = parser.parse_args()

    # Input directory containing .init files
    input_dir = args.dataset_path

    # Output directory for pose pairs (based on input directory)
    dataset_name = os.path.basename(os.path.dirname(input_dir))
    output_dir = os.path.join(os.path.dirname(input_dir), "ee_target_obj_pairs", "simple_extraction_obs_pose_v2_normal")

    print("🚀 Phase 2 Object-EE Pose Pairs Extraction (Unified Config)")
    print(f"📁 Input directory: {input_dir}")
    print(f"💾 Output directory: {output_dir}")
    print(f"🔧 Mode: {args.mode}")

    # Find all .init files in dataset
    all_init_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.init'):
                all_init_files.append(os.path.join(root, file))

    print(f"\n📊 Found {len(all_init_files)} total .init files in dataset")

    # Load unified config and build mapping
    config = load_unified_config()
    if not config:
        print("❌ No unified config loaded, cannot proceed")
        return

    skills_mapping = build_init_file_mapping(config)
    if not skills_mapping:
        print("❌ No skills mapping built from unified config, cannot proceed")
        return

    # Determine which files to process based on mode
    if args.mode == 'normal':
        # Normal mode: process ALL .init files
        focused_init_files = all_init_files
        print(f"\n🔵 Normal mode: Processing ALL {len(focused_init_files)} .init files")

        # For normal mode, we need to infer target objects for files not in mapping
        # This will require BDDL filename parsing or manual specification
        # For now, we'll process files that have mapping and warn about unmapped files
        unmapped_count = 0
        for init_file in all_init_files:
            init_basename = os.path.basename(init_file)
            if init_basename not in skills_mapping:
                unmapped_count += 1

        if unmapped_count > 0:
            print(f"⚠️  Warning: {unmapped_count} .init files have no mapping in unified config")
            print(f"   These files will be skipped. Add them to unified_task_config.json if needed.")

    else:  # lht_oriented mode
        # LHT-oriented mode: only process files in unified config mapping
        focused_init_files = []
        for init_file in all_init_files:
            init_basename = os.path.basename(init_file)
            if init_basename in skills_mapping:
                focused_init_files.append(init_file)

        print(f"\n🎯 LHT-oriented mode: Processing {len(focused_init_files)} files from unified config")

    if not focused_init_files:
        print("❌ No files to process!")
        return

    # Test single file mode
    if args.test_single:
        target_init_file = args.test_single
        if target_init_file not in skills_mapping:
            print(f"❌ Init file '{target_init_file}' not found in unified config mapping")
            print(f"   Available init files in config: {list(skills_mapping.keys())[:10]}...")
            return

        skill_info = skills_mapping[target_init_file]
        target_object = skill_info['object']
        bddl_files = skill_info['bddl_files']

        # Find the actual file path
        found_init_path = None
        for init_file in all_init_files:
            if os.path.basename(init_file) == target_init_file:
                found_init_path = init_file
                break

        if not found_init_path:
            print(f"❌ Init file '{target_init_file}' not found in filesystem")
            return

        print(f"\n🔍 Testing single file: {target_init_file}")
        print(f"  🎯 Target object: {target_object}")
        print(f"  📋 BDDL files: {bddl_files}")

        success, num_pairs, avg_distance, facing_ratio = extract_simple_pose_pairs(
            found_init_path, target_object, output_dir, bddl_files
        )

        print(f"\n🔍 Single file test results:")
        print(f"  ✅ Success: {success}")
        print(f"  📊 Pairs: {num_pairs}")
        print(f"  📏 Avg distance: {avg_distance:.3f}m")
        print(f"  👀 EE facing ratio: {facing_ratio*100:.0f}%")

        return

    # Process all focused files
    skill_stats = {}
    total_pairs = 0
    successful_skills = 0
    failed_skills = 0
    skipped_skills = 0
    total_facing_ratio = []

    for i, init_file in enumerate(focused_init_files):
        init_basename = os.path.basename(init_file)

        # Skip if not in mapping
        if init_basename not in skills_mapping:
            skipped_skills += 1
            print(f"\n⏭️  Skipping {init_basename} (not in unified config)")
            continue

        skill_info = skills_mapping[init_basename]
        target_object = skill_info['object']
        bddl_files = skill_info['bddl_files']

        print(f"\n{'='*80}")
        print(f"🎯 Processing {i+1}/{len(focused_init_files)}: {init_basename}")

        try:
            success, num_pairs, avg_distance, facing_ratio = extract_simple_pose_pairs(
                init_file, target_object, output_dir, bddl_files
            )

            skill_name = Path(init_file).stem
            if success:
                skill_stats[skill_name] = {
                    'pairs': num_pairs,
                    'avg_distance': avg_distance,
                    'facing_ratio': facing_ratio,
                    'target_object': target_object
                }
                total_pairs += num_pairs
                total_facing_ratio.append(facing_ratio)
                successful_skills += 1
            else:
                failed_skills += 1

        except Exception as e:
            print(f"❌ Failed to process {init_file}: {e}")
            failed_skills += 1

    # Print summary statistics
    print(f"\n{'='*80}")
    print("🎯 EXTRACTION SUMMARY")
    print(f"✅ Successful skills: {successful_skills}")
    print(f"❌ Failed skills: {failed_skills}")
    if skipped_skills > 0:
        print(f"⏭️  Skipped skills: {skipped_skills}")
    if successful_skills + failed_skills > 0:
        print(f"📈 Success rate: {successful_skills/(successful_skills+failed_skills)*100:.1f}%")
    print(f"📊 Total pose pairs: {total_pairs}")

    if total_facing_ratio:
        overall_facing_ratio = np.mean(total_facing_ratio)
        print(f"👀 Overall EE facing ratio: {overall_facing_ratio*100:.1f}%")

    print(f"\n📋 PER-SKILL STATISTICS:")
    print(f"{'Skill Name':<60} {'Pairs':<6} {'Dist':<8} {'Facing':<8} {'Target Object'}")
    print("="*110)

    for skill_name, stats in sorted(skill_stats.items()):
        pairs = stats['pairs']
        distance = stats['avg_distance']
        facing = stats['facing_ratio'] * 100
        obj = stats['target_object']
        print(f"{skill_name:<60} {pairs:<6} {distance:<8.3f} {facing:<8.0f}% {obj}")

    if successful_skills > 0:
        avg_pairs_per_skill = total_pairs / successful_skills
        overall_avg_distance = np.mean([stats['avg_distance'] for stats in skill_stats.values()])
        print(f"\n📊 OVERALL AVERAGES:")
        print(f"  Average pairs per skill: {avg_pairs_per_skill:.1f}")
        print(f"  Overall average EE-object distance: {overall_avg_distance:.3f}m")
        print(f"  Overall EE facing object ratio: {overall_facing_ratio*100:.1f}%")
        print(f"\n💾 Results saved to: {output_dir}")
        print("  📁 Files: .pkl (pose pairs) + .png (preview images)")
        print("🎉 Extraction completed!")


if __name__ == "__main__":
    main()