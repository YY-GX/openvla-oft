#!/usr/bin/env python3

import os
import sys
sys.path.append('externals/boss/libero/libero/bddl_files/atomic_skills/scripts')

from split_cat1_bddl import split_bddl_file, parse_language_for_objects

# Test with one file
input_file = "externals/boss/libero/libero/bddl_files/atomic_skills/cat1/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl"
output_dir = "/tmp"

print("Testing language parsing...")
test_languages = [
    "put the black bowl on the plate",
    "stack the middle black bowl on the back black bowl",
    "put the wine bottle in the cabinet"
]

for lang in test_languages:
    obj_a, obj_b, prep = parse_language_for_objects(lang)
    print(f"'{lang}' -> A:'{obj_a}', B:'{obj_b}', prep:'{prep}'")

print("\nTesting the split script...")
pick_file, place_file = split_bddl_file(input_file, output_dir)

if pick_file and place_file:
    print("✅ Script ran successfully!")
    
    print(f"\nPick file: {pick_file}")
    print("=" * 40)
    with open(pick_file, 'r') as f:
        content = f.read()
        print(content[:800] + "..." if len(content) > 800 else content)
        
    print(f"\nPlace file: {place_file}")
    print("=" * 40)
    with open(place_file, 'r') as f:
        content = f.read()
        print(content[:800] + "..." if len(content) > 800 else content)
        
    # Check bracket balance for both files
    for file_path in [pick_file, place_file]:
        with open(file_path, 'r') as f:
            content = f.read()
            open_count = content.count('(')
            close_count = content.count(')')
            file_type = "pick" if "pick" in file_path else "place"
            print(f"\n{file_type.upper()} file bracket check:")
            print(f"Opening: {open_count}, Closing: {close_count}, Balanced: {'✅' if open_count == close_count else '❌'}")
            
else:
    print("❌ Script failed!")