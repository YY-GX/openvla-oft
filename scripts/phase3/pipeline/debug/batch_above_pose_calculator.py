#!/usr/bin/env python3
"""
Batch script to visualize above-pose for all BDDL files.

Simple script that iterates through all BDDL files and generates visualization images.
"""

import sys
import os
import json

# Add project paths
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')

from scripts.phase3.pipeline.utils.above_pose_calculator import visualize_above_pose

# Base directory for BDDL files
BDDL_BASE_DIR = "externals/boss/libero/libero/bddl_files/atomic_skills"

# Path to skill config
SKILL_CONFIG_PATH = "scripts/phase3/pipeline/config/skill_config.json"

# Project root
PROJECT_ROOT = "/mnt/arc/yygx/pkgs_baselines/openvla-oft"


def main():
    """Main function to iterate through all BDDL files and run above pose calculator."""

    # Read skill config
    config_path = os.path.join(PROJECT_ROOT, SKILL_CONFIG_PATH)
    print(f"📖 Reading skill config from: {config_path}")

    with open(config_path, 'r') as f:
        skill_config = json.load(f)

    # Collect all unique BDDL files
    all_bddl_files = set()
    for skill_info in skill_config.values():
        all_bddl_files.update(skill_info.get('bddl_files', []))

    all_bddl_files = sorted(all_bddl_files)

    print(f"📋 Found {len(all_bddl_files)} unique BDDL files\n")

    # Statistics
    successful = 0
    failed = 0
    failures = []

    # Process each BDDL file
    for i, bddl_filename in enumerate(all_bddl_files, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(all_bddl_files)}] Processing: {bddl_filename}")
        print(f"{'='*80}")

        # Construct full path
        bddl_path = os.path.join(PROJECT_ROOT, BDDL_BASE_DIR, bddl_filename)

        # Check if file exists
        if not os.path.exists(bddl_path):
            print(f"❌ BDDL file not found: {bddl_path}")
            failed += 1
            failures.append({
                "bddl_file": bddl_filename,
                "reason": "File not found"
            })
            continue

        # Run visualization (target_object will be auto-inferred)
        try:
            success = visualize_above_pose(
                bddl_file=bddl_path,
                target_object=None,  # Auto-infer from BDDL
                above_height=0.10,
                shift=False,
                output_dir=None  # Uses default
            )

            if success:
                print(f"✅ Success")
                successful += 1
            else:
                print(f"❌ Failed")
                failed += 1
                failures.append({
                    "bddl_file": bddl_filename,
                    "reason": "visualize_above_pose returned False"
                })

        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            failed += 1
            failures.append({
                "bddl_file": bddl_filename,
                "reason": str(e)
            })
            import traceback
            traceback.print_exc()

    # Save failures
    if failures:
        failures_path = os.path.join(PROJECT_ROOT, "scripts/phase3/pipeline/debug/failures.json")
        with open(failures_path, 'w') as f:
            json.dump(failures, f, indent=2)
        print(f"\n💾 Saved {len(failures)} failures to: {failures_path}")

    # Summary
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY")
    print(f"{'='*80}")
    print(f"Total: {len(all_bddl_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
