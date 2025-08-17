#!/usr/bin/env python3

import os
import sys
sys.path.append('externals/boss/libero/libero/bddl_files/atomic_skills/scripts')

from modify_cat1_bddl import split_bddl_file

# Test with the problematic files
test_files = [
    "externals/boss/libero/libero/bddl_files/atomic_skills/cat1/KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet.bddl",
    "externals/boss/libero/libero/bddl_files/atomic_skills/cat1/KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle.bddl"
]

output_dir = "/tmp"

for input_file in test_files:
    filename = os.path.basename(input_file)
    print(f"\n{'='*60}")
    print(f"Testing: {filename}")
    print('='*60)
    
    pick_file, place_file = split_bddl_file(input_file, output_dir)
    
    if pick_file and place_file:
        print(f"\n✅ Generated pick file:")
        with open(pick_file, 'r') as f:
            content = f.read()
            print(content)
            
        print(f"\n✅ Generated place file:")
        with open(place_file, 'r') as f:
            content = f.read()
            print(content)
    else:
        print("❌ Failed to generate files")