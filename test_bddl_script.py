#!/usr/bin/env python3

import os
import sys
sys.path.append('externals/boss/libero/libero/bddl_files/atomic_skills/scripts')

from clean_bddl_final import clean_bddl_file

# Test with one file
input_file = "externals/boss/libero/libero/bddl_files/atomic_skills/cat2/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet.bddl"
output_file = "/tmp/test_cleaned.bddl"

print("Testing the fixed script...")
success = clean_bddl_file(input_file, output_file)

if success:
    print("✅ Script ran successfully!")
    print("\nGenerated file content:")
    with open(output_file, 'r') as f:
        content = f.read()
        print(content)
        
    # Check for proper bracket matching
    open_count = content.count('(')
    close_count = content.count(')')
    print(f"\nBracket count check:")
    print(f"Opening brackets: {open_count}")
    print(f"Closing brackets: {close_count}")
    print(f"Balanced: {'✅' if open_count == close_count else '❌'}")
else:
    print("❌ Script failed!")