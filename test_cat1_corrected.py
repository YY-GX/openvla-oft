#!/usr/bin/env python3

import os
import sys
sys.path.append('externals/boss/libero/libero/bddl_files/atomic_skills/scripts')

from modify_cat1_bddl import split_bddl_file

# Test with one file
input_file = "externals/boss/libero/libero/bddl_files/atomic_skills/cat1/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl"
output_dir = "/tmp"

print("Testing the corrected cat1 split script...")
pick_file, place_file = split_bddl_file(input_file, output_dir)

if pick_file and place_file:
    print("✅ Script ran successfully!")
    
    print("\n" + "="*50)
    print("PICK FILE (should only have bowl + table):")
    print("="*50)
    with open(pick_file, 'r') as f:
        content = f.read()
        print(content)
        
    print("\n" + "="*50)
    print("PLACE FILE (should only have bowl + plate + table):")
    print("="*50)
    with open(place_file, 'r') as f:
        content = f.read()
        print(content)
        
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