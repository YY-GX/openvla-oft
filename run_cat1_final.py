#!/usr/bin/env python3

import os
import sys
import subprocess

# Change to the atomic_skills directory and run the corrected script
atomic_skills_dir = "externals/boss/libero/libero/bddl_files/atomic_skills"

print("Running final corrected Category 1 BDDL splitting script...")
print(f"Working directory: {atomic_skills_dir}")

# Remove old files first
subprocess.run([
    "bash", "-c", "rm -f *_pick.bddl *_place.bddl cat1_split_map.json"
], cwd=atomic_skills_dir)

# Run the corrected script
result = subprocess.run([
    "python", "scripts/modify_cat1_bddl.py"
], cwd=atomic_skills_dir, capture_output=True, text=True)

print("Script completed!")
print(f"Return code: {result.returncode}")

if result.stderr:
    print("STDERR:")
    print(result.stderr[-1000:])  # Last 1000 chars

if result.returncode == 0:
    print("✅ Script completed successfully!")
    
    # Check results
    mapping_file = os.path.join(atomic_skills_dir, "cat1_split_map.json")
    if os.path.exists(mapping_file):
        import json
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        
        total_generated = sum(len(files) for files in mapping.values())
        print(f"📊 Total files generated: {total_generated}")
        print(f"📁 Input files processed: {len(mapping)}")
        
        # Test one file to verify quality
        if mapping:
            first_files = list(mapping.values())[0]
            if len(first_files) >= 2:
                pick_file = first_files[0]
                print(f"\n✅ Sample pick file: {os.path.basename(pick_file)}")
                if os.path.exists(pick_file):
                    with open(pick_file, 'r') as f:
                        content = f.read()
                        lines = content.split('\n')
                        # Show key sections
                        for i, line in enumerate(lines):
                            if '(:fixtures' in line or '(:objects' in line or '(:obj_of_interest' in line:
                                print(f"  {line}")
                                for j in range(i+1, min(i+4, len(lines))):
                                    if lines[j].strip() and not lines[j].strip().startswith('('):
                                        print(f"    {lines[j]}")
                                    elif lines[j].strip() == ')':
                                        break
    else:
        print("⚠️ Mapping file not found")
else:
    print("❌ Script failed!")
    print("STDOUT:")
    print(result.stdout[-1000:])  # Last 1000 chars