#!/usr/bin/env python3
"""
test_post_processing_results.py

Comprehensive test script to validate outputs from post_processing_aug_demos.py

Features:
- Validate corrected .init files match first states of original demos
- Check augmented demos are truncated to 60 timesteps or less
- Save sample initial images (3 original + 3 augmented per skill)
- Verify HDF5 file structure and data integrity
- Generate detailed validation report

Usage:
# Test debug output
python scripts/phase2/tests/test_post_processing_results.py --input_dir datasets/hdf5_datasets/atomic_local_demos_augmented/combined/debug

# Test full output
python scripts/phase2/tests/test_post_processing_results.py --input_dir datasets/hdf5_datasets/atomic_local_demos_augmented/combined
"""

import argparse
import os
import pickle
import h5py
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import traceback
import matplotlib.pyplot as plt
from datetime import datetime
import cv2

def find_skill_files(input_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Find all skill files in the input directory.

    Returns:
        Dict mapping skill names to their file paths
    """
    skills = {}

    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return skills

    # Find all files and group by skill
    for filename in os.listdir(input_dir):
        if filename.endswith(('.hdf5', '.init')):
            # Extract skill name
            skill_name = None
            if filename.endswith('_original_demo.hdf5'):
                skill_name = filename.replace('_original_demo.hdf5', '')
            elif filename.endswith('_augmented_demo.hdf5'):
                skill_name = filename.replace('_augmented_demo.hdf5', '')
            elif filename.endswith('_original.init'):
                skill_name = filename.replace('_original.init', '')
            elif filename.endswith('_augmented.init'):
                skill_name = filename.replace('_augmented.init', '')
            elif filename.endswith('_combined.init'):
                skill_name = filename.replace('_combined.init', '')

            if skill_name:
                if skill_name not in skills:
                    skills[skill_name] = {}

                # Store file path
                if filename.endswith('_original_demo.hdf5'):
                    skills[skill_name]['original_hdf5'] = os.path.join(input_dir, filename)
                elif filename.endswith('_augmented_demo.hdf5'):
                    skills[skill_name]['augmented_hdf5'] = os.path.join(input_dir, filename)
                elif filename.endswith('_original.init'):
                    skills[skill_name]['original_init'] = os.path.join(input_dir, filename)
                elif filename.endswith('_augmented.init'):
                    skills[skill_name]['augmented_init'] = os.path.join(input_dir, filename)
                elif filename.endswith('_combined.init'):
                    skills[skill_name]['combined_init'] = os.path.join(input_dir, filename)

    return skills

def extract_first_states_from_hdf5(hdf5_file: str) -> List[np.ndarray]:
    """Extract first states from all demos in an HDF5 file."""
    first_states = []

    try:
        with h5py.File(hdf5_file, 'r') as f:
            if 'data' in f:
                data_group = f['data']
                demo_keys = [k for k in data_group.keys() if k.startswith('demo_')]
                demo_keys.sort(key=lambda x: int(x.split('_')[1]))

                for demo_key in demo_keys:
                    demo_group = data_group[demo_key]
                    if 'states' in demo_group:
                        states = demo_group['states'][:]
                        if len(states) > 0:
                            first_states.append(states[0])
            else:
                # Handle files with demos at root level
                demo_keys = [k for k in f.keys() if k.startswith('demo_')]
                demo_keys.sort(key=lambda x: int(x.split('_')[1]))

                for demo_key in demo_keys:
                    demo_group = f[demo_key]
                    if 'states' in demo_group:
                        states = demo_group['states'][:]
                        if len(states) > 0:
                            first_states.append(states[0])

    except Exception as e:
        print(f"❌ Error extracting first states from {hdf5_file}: {e}")

    return first_states

def load_init_file(init_file: str) -> List[np.ndarray]:
    """Load states from an init file."""
    try:
        with open(init_file, 'rb') as f:
            states = pickle.load(f)
            if isinstance(states, list):
                return states
            else:
                return [states]
    except Exception as e:
        print(f"❌ Error loading init file {init_file}: {e}")
        return []

def validate_init_file_correctness(original_hdf5: str, original_init: str) -> Tuple[bool, str, Dict]:
    """
    Validate that the corrected init file matches first states from original demos.

    Returns:
        Tuple of (is_valid, message, stats)
    """
    # Extract first states from HDF5
    hdf5_first_states = extract_first_states_from_hdf5(original_hdf5)

    # Load init file states
    init_states = load_init_file(original_init)

    stats = {
        'hdf5_first_states': len(hdf5_first_states),
        'init_states': len(init_states),
        'matches': 0,
        'mismatches': 0
    }

    if len(hdf5_first_states) != len(init_states):
        return False, f"Count mismatch: HDF5 has {len(hdf5_first_states)} first states, init has {len(init_states)}", stats

    # Compare each state
    for i, (hdf5_state, init_state) in enumerate(zip(hdf5_first_states, init_states)):
        if np.allclose(hdf5_state, init_state, rtol=1e-10, atol=1e-10):
            stats['matches'] += 1
        else:
            stats['mismatches'] += 1
            max_diff = np.max(np.abs(hdf5_state - init_state))
            return False, f"State {i} mismatch: max difference = {max_diff}", stats

    return True, f"All {len(init_states)} states match perfectly", stats

def check_augmented_truncation(augmented_hdf5: str, max_timesteps: int = 60) -> Tuple[bool, str, Dict]:
    """
    Check that augmented demos are properly truncated.

    Returns:
        Tuple of (is_valid, message, stats)
    """
    stats = {
        'total_demos': 0,
        'properly_truncated': 0,
        'over_limit': 0,
        'demo_lengths': [],
        'avg_length': 0,
        'max_length': 0,
        'min_length': float('inf')
    }

    try:
        with h5py.File(augmented_hdf5, 'r') as f:
            if 'data' in f:
                data_group = f['data']
                demo_keys = [k for k in data_group.keys() if k.startswith('demo_')]
                demo_keys.sort(key=lambda x: int(x.split('_')[1]))

                for demo_key in demo_keys:
                    demo_group = data_group[demo_key]
                    if 'actions' in demo_group:
                        actions = demo_group['actions'][:]
                        length = len(actions)

                        stats['total_demos'] += 1
                        stats['demo_lengths'].append(length)
                        stats['max_length'] = max(stats['max_length'], length)
                        stats['min_length'] = min(stats['min_length'], length)

                        if length <= max_timesteps:
                            stats['properly_truncated'] += 1
                        else:
                            stats['over_limit'] += 1

            if stats['total_demos'] > 0:
                stats['avg_length'] = np.mean(stats['demo_lengths'])

                if stats['over_limit'] == 0:
                    return True, f"All {stats['total_demos']} demos properly truncated (max: {stats['max_length']}, avg: {stats['avg_length']:.1f})", stats
                else:
                    return False, f"{stats['over_limit']} demos exceed {max_timesteps} timesteps limit", stats
            else:
                return False, "No demos found in file", stats

    except Exception as e:
        return False, f"Error checking truncation: {e}", stats

def extract_initial_images(hdf5_file: str, num_samples: int = 3) -> List[Tuple[str, np.ndarray]]:
    """
    Extract initial images from first few demos.

    Returns:
        List of (demo_key, initial_image) tuples
    """
    initial_images = []

    try:
        with h5py.File(hdf5_file, 'r') as f:
            if 'data' in f:
                data_group = f['data']
                demo_keys = [k for k in data_group.keys() if k.startswith('demo_')]
                demo_keys.sort(key=lambda x: int(x.split('_')[1]))

                # Sample first num_samples demos
                sampled_demos = demo_keys[:num_samples]

                for demo_key in sampled_demos:
                    demo_group = data_group[demo_key]

                    # Look for observation data with images
                    obs_group = demo_group.get('obs')
                    if obs_group is None:
                        continue

                    # Try to find camera images (prioritize wrist, fall back to agent)
                    image_key = None
                    for key in ['eye_in_hand_rgb', 'wrist_image', 'agentview_rgb', 'agent_image', 'image']:
                        if key in obs_group:
                            image_key = key
                            break

                    if image_key is None:
                        continue

                    # Get first image
                    images = obs_group[image_key][:]
                    if len(images) > 0:
                        initial_image = images[0]

                        # Ensure proper format
                        if initial_image.dtype != np.uint8:
                            initial_image = (initial_image * 255).astype(np.uint8)

                        initial_images.append((demo_key, initial_image))

    except Exception as e:
        print(f"❌ Error extracting initial images from {hdf5_file}: {e}")

    return initial_images

def save_initial_images(skill_name: str, original_images: List[Tuple[str, np.ndarray]],
                       augmented_images: List[Tuple[str, np.ndarray]], output_dir: str):
    """Save initial images for a skill."""
    skill_dir = os.path.join(output_dir, skill_name)
    os.makedirs(skill_dir, exist_ok=True)

    # Save original images
    for i, (demo_key, image) in enumerate(original_images):
        filename = f"original_{demo_key}_initial.png"
        filepath = os.path.join(skill_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Save augmented images
    for i, (demo_key, image) in enumerate(augmented_images):
        filename = f"augmented_{demo_key}_initial.png"
        filepath = os.path.join(skill_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    print(f"   🖼️  Saved {len(original_images)} original + {len(augmented_images)} augmented initial images")

def check_hdf5_structure(hdf5_file: str) -> Tuple[bool, str, Dict]:
    """Check HDF5 file structure and data integrity."""
    stats = {
        'has_data_group': False,
        'demo_count': 0,
        'consistent_structure': True,
        'required_keys': ['actions', 'states', 'obs'],
        'missing_keys': [],
        'data_types_consistent': True
    }

    try:
        with h5py.File(hdf5_file, 'r') as f:
            # Check for data group
            if 'data' in f:
                stats['has_data_group'] = True
                data_group = f['data']
                demo_keys = [k for k in data_group.keys() if k.startswith('demo_')]
                stats['demo_count'] = len(demo_keys)

                # Check structure consistency
                for demo_key in demo_keys[:3]:  # Check first 3 demos
                    demo_group = data_group[demo_key]

                    # Check required keys
                    for req_key in stats['required_keys']:
                        if req_key not in demo_group:
                            stats['missing_keys'].append(f"{demo_key}/{req_key}")
                            stats['consistent_structure'] = False

                if stats['consistent_structure'] and stats['demo_count'] > 0:
                    return True, f"Valid structure with {stats['demo_count']} demos", stats
                else:
                    return False, f"Structure issues: missing keys {stats['missing_keys']}", stats
            else:
                return False, "No 'data' group found", stats

    except Exception as e:
        return False, f"Error checking structure: {e}", stats

def generate_validation_report(results: Dict[str, Any], output_dir: str):
    """Generate a comprehensive validation report."""
    report_file = os.path.join(output_dir, "validation_report.json")
    summary_file = os.path.join(output_dir, "validation_summary.txt")

    # Save detailed JSON report
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Generate human-readable summary
    with open(summary_file, 'w') as f:
        f.write("VALIDATION SUMMARY REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Overall statistics
        total_skills = len(results['skills'])
        init_pass = sum(1 for s in results['skills'].values() if s.get('init_validation', {}).get('passed', False))
        trunc_pass = sum(1 for s in results['skills'].values() if s.get('truncation_validation', {}).get('passed', False))
        struct_pass = sum(1 for s in results['skills'].values()
                         if s.get('structure_validation', {}).get('original_passed', False) and
                            s.get('structure_validation', {}).get('augmented_passed', False))

        f.write(f"OVERALL RESULTS:\n")
        f.write(f"Total Skills Tested: {total_skills}\n")
        f.write(f"Init Validation Passed: {init_pass}/{total_skills} ({init_pass/total_skills*100:.1f}%)\n")
        f.write(f"Truncation Validation Passed: {trunc_pass}/{total_skills} ({trunc_pass/total_skills*100:.1f}%)\n")
        f.write(f"Structure Validation Passed: {struct_pass}/{total_skills} ({struct_pass/total_skills*100:.1f}%)\n\n")

        # Per-skill details
        f.write("PER-SKILL DETAILS:\n")
        f.write("-" * 30 + "\n")

        for skill_name, skill_results in results['skills'].items():
            f.write(f"\n{skill_name}:\n")

            # Init validation
            init_val = skill_results.get('init_validation', {})
            f.write(f"  Init Validation: {'✅ PASS' if init_val.get('passed') else '❌ FAIL'}")
            if 'message' in init_val:
                f.write(f" - {init_val['message']}")
            f.write("\n")

            # Truncation validation
            trunc_val = skill_results.get('truncation_validation', {})
            f.write(f"  Truncation: {'✅ PASS' if trunc_val.get('passed') else '❌ FAIL'}")
            if 'message' in trunc_val:
                f.write(f" - {trunc_val['message']}")
            f.write("\n")

            # Structure validation
            struct_val = skill_results.get('structure_validation', {})
            f.write(f"  Original HDF5 Structure: {'✅ PASS' if struct_val.get('original_passed') else '❌ FAIL'}\n")
            f.write(f"  Augmented HDF5 Structure: {'✅ PASS' if struct_val.get('augmented_passed') else '❌ FAIL'}\n")

            # Images
            images_info = skill_results.get('images', {})
            f.write(f"  Initial Images: {images_info.get('original_count', 0)} original, {images_info.get('augmented_count', 0)} augmented\n")

    print(f"\n📊 VALIDATION REPORT SUMMARY:")
    print(f"{'='*60}")
    print(f"Total Skills Tested: {total_skills}")
    print(f"Init Validation: {init_pass}/{total_skills} passed ({init_pass/total_skills*100:.1f}%)")
    print(f"Truncation Validation: {trunc_pass}/{total_skills} passed ({trunc_pass/total_skills*100:.1f}%)")
    print(f"Structure Validation: {struct_pass}/{total_skills} passed ({struct_pass/total_skills*100:.1f}%)")
    print(f"\n📄 Detailed report saved to: {report_file}")
    print(f"📄 Summary saved to: {summary_file}")

def test_skill(skill_name: str, files: Dict[str, str], images_output_dir: str) -> Dict[str, Any]:
    """Test a single skill's processed files."""
    print(f"\n🔍 Testing skill: {skill_name}")

    results = {
        'skill_name': skill_name,
        'files_found': files,
        'init_validation': {},
        'truncation_validation': {},
        'structure_validation': {},
        'images': {}
    }

    # Test 1: Validate corrected init file
    if 'original_hdf5' in files and 'original_init' in files:
        print(f"   🧪 Testing init file correctness...")
        passed, message, stats = validate_init_file_correctness(files['original_hdf5'], files['original_init'])
        results['init_validation'] = {
            'passed': passed,
            'message': message,
            'stats': stats
        }
        print(f"   {'✅' if passed else '❌'} Init validation: {message}")
    else:
        results['init_validation'] = {'passed': False, 'message': 'Missing required files'}
        print(f"   ❌ Init validation: Missing original HDF5 or init file")

    # Test 2: Check augmented demo truncation
    if 'augmented_hdf5' in files:
        print(f"   🧪 Testing augmented demo truncation...")
        passed, message, stats = check_augmented_truncation(files['augmented_hdf5'])
        results['truncation_validation'] = {
            'passed': passed,
            'message': message,
            'stats': stats
        }
        print(f"   {'✅' if passed else '❌'} Truncation: {message}")
    else:
        results['truncation_validation'] = {'passed': False, 'message': 'Missing augmented HDF5 file'}
        print(f"   ❌ Truncation validation: Missing augmented HDF5 file")

    # Test 3: Check HDF5 structure
    print(f"   🧪 Testing HDF5 file structures...")
    if 'original_hdf5' in files:
        orig_passed, orig_msg, orig_stats = check_hdf5_structure(files['original_hdf5'])
        results['structure_validation']['original_passed'] = orig_passed
        results['structure_validation']['original_message'] = orig_msg
        results['structure_validation']['original_stats'] = orig_stats
        print(f"   {'✅' if orig_passed else '❌'} Original HDF5 structure: {orig_msg}")

    if 'augmented_hdf5' in files:
        aug_passed, aug_msg, aug_stats = check_hdf5_structure(files['augmented_hdf5'])
        results['structure_validation']['augmented_passed'] = aug_passed
        results['structure_validation']['augmented_message'] = aug_msg
        results['structure_validation']['augmented_stats'] = aug_stats
        print(f"   {'✅' if aug_passed else '❌'} Augmented HDF5 structure: {aug_msg}")

    # Test 4: Extract and save initial images
    print(f"   🧪 Extracting initial images...")
    original_images = []
    augmented_images = []

    if 'original_hdf5' in files:
        original_images = extract_initial_images(files['original_hdf5'], num_samples=3)

    if 'augmented_hdf5' in files:
        augmented_images = extract_initial_images(files['augmented_hdf5'], num_samples=3)

    if original_images or augmented_images:
        save_initial_images(skill_name, original_images, augmented_images, images_output_dir)

    results['images'] = {
        'original_count': len(original_images),
        'augmented_count': len(augmented_images)
    }

    return results

def main():
    parser = argparse.ArgumentParser(description="Test post-processing results")
    parser.add_argument("--input_dir", type=str, required=True,
                      help="Directory containing post-processed files")
    parser.add_argument("--output_dir", type=str,
                      default="scripts/phase2/tests/validation_results",
                      help="Directory to save validation results")
    args = parser.parse_args()

    print("🧪 Post-Processing Validation Test Script")
    print("=" * 50)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    images_output_dir = os.path.join(args.output_dir, "initial_images")
    os.makedirs(images_output_dir, exist_ok=True)

    print(f"📂 Input directory: {args.input_dir}")
    print(f"📁 Output directory: {args.output_dir}")

    # Find all skills and their files
    skills = find_skill_files(args.input_dir)

    if not skills:
        print(f"❌ No skill files found in {args.input_dir}")
        return 1

    print(f"🔍 Found {len(skills)} skills to test")

    # Test each skill
    results = {
        'test_timestamp': datetime.now().isoformat(),
        'input_directory': args.input_dir,
        'total_skills': len(skills),
        'skills': {}
    }

    for skill_idx, (skill_name, files) in enumerate(skills.items()):
        print(f"\n[{skill_idx + 1}/{len(skills)}]", end="")

        try:
            skill_results = test_skill(skill_name, files, images_output_dir)
            results['skills'][skill_name] = skill_results

        except Exception as e:
            print(f"❌ Error testing {skill_name}: {e}")
            traceback.print_exc()
            results['skills'][skill_name] = {
                'error': str(e),
                'skill_name': skill_name
            }

    # Generate validation report
    generate_validation_report(results, args.output_dir)

    print(f"\n✅ Validation completed!")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"🖼️  Initial images saved to: {images_output_dir}")

    return 0

if __name__ == "__main__":
    exit(main())