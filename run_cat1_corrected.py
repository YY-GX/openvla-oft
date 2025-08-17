#!/usr/bin/env python3

import os
import sys
import subprocess

# Change to the atomic_skills directory and run the corrected script
atomic_skills_dir = "externals/boss/libero/libero/bddl_files/atomic_skills"

print("Running corrected Category 1 BDDL splitting script...")
print(f"Working directory: {atomic_skills_dir}")

# Remove old files first
result = subprocess.run([
    "rm", "-f", "*_pick.bddl", "*_place.bddl", "cat1_split_map.json"
], cwd=atomic_skills_dir, shell=True)

# Change to the atomic_skills directory and run the script
result = subprocess.run([
    "python", "scripts/modify_cat1_bddl.py"
], cwd=atomic_skills_dir, capture_output=True, text=True)

print("STDOUT:")
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)  # Last 2000 chars

if result.stderr:
    print("STDERR:")
    print(result.stderr)

print(f"Return code: {result.returncode}")

if result.returncode == 0:
    print("✅ Script completed successfully!")
    
    # Check if mapping file was created
    mapping_file = os.path.join(atomic_skills_dir, "cat1_split_map.json")
    if os.path.exists(mapping_file):
        print(f"📋 Mapping file created: {mapping_file}")
        
        # Count generated files
        import json
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        
        total_generated = sum(len(files) for files in mapping.values())
        print(f"📊 Total files generated: {total_generated}")
        print(f"📁 Input files processed: {len(mapping)}")
    else:
        print("⚠️ Mapping file not found")
else:
    print("❌ Script failed!")