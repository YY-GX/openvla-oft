#!/usr/bin/env python3
"""
Detailed analysis of specific BDDL error patterns identified.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import json

def analyze_missing_initial_states():
    """Analyze missing initial states like (Open drawer) in place files."""
    base_dir = Path("/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss/libero/libero/bddl_files/atomic_skills")
    original_dir = base_dir / "original_44_skills"
    
    missing_open_states = []
    
    # Focus on place files that should have Open states
    place_files = list(base_dir.glob("*place.bddl"))
    
    for place_file in place_files:
        # Find corresponding original
        base_name = place_file.name.replace('_place.bddl', '.bddl')
        orig_file = original_dir / base_name
        
        if not orig_file.exists():
            continue
            
        # Read original file for initial states
        with open(orig_file, 'r') as f:
            orig_content = f.read()
            
        # Read generated place file
        with open(place_file, 'r') as f:
            place_content = f.read()
            
        # Check for Open states in original
        orig_open_states = re.findall(r'\(Open\s+([^)]+)\)', orig_content)
        place_open_states = re.findall(r'\(Open\s+([^)]+)\)', place_content)
        
        missing_opens = set(orig_open_states) - set(place_open_states)
        
        if missing_opens:
            missing_open_states.append({
                'file': str(place_file),
                'missing_open_states': list(missing_opens),
                'original_opens': orig_open_states,
                'place_opens': place_open_states
            })
    
    return missing_open_states

def analyze_wrong_objects_in_place_files():
    """Find cases where place files have wrong objects (like red cup in chocolate pudding place)."""
    base_dir = Path("/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss/libero/libero/bddl_files/atomic_skills")
    original_dir = base_dir / "original_44_skills"
    
    wrong_objects = []
    
    place_files = list(base_dir.glob("*place.bddl"))
    
    for place_file in place_files:
        base_name = place_file.name.replace('_place.bddl', '.bddl')
        orig_file = original_dir / base_name
        
        if not orig_file.exists():
            continue
            
        # Parse both files
        with open(orig_file, 'r') as f:
            orig_content = f.read()
        with open(place_file, 'r') as f:
            place_content = f.read()
            
        # Extract objects from both
        orig_objects = set(re.findall(r'(\w+_\d+)\s*-\s*\w+', orig_content))
        place_objects = set(re.findall(r'(\w+_\d+)\s*-\s*\w+', place_content))
        
        # Find objects that don't belong
        extra_objects = place_objects - orig_objects
        missing_objects = orig_objects - place_objects
        
        if extra_objects or missing_objects:
            wrong_objects.append({
                'file': str(place_file),
                'extra_objects': list(extra_objects),
                'missing_objects': list(missing_objects),
                'task_type': 'place'
            })
    
    return wrong_objects

def analyze_missing_fixtures_in_pick_files():
    """Find pick files missing fixtures that are referenced in goals/regions."""
    base_dir = Path("/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss/libero/libero/bddl_files/atomic_skills")
    original_dir = base_dir / "original_44_skills"
    
    missing_fixture_refs = []
    
    pick_files = list(base_dir.glob("*pick.bddl"))
    
    for pick_file in pick_files:
        base_name = pick_file.name.replace('_pick.bddl', '.bddl')
        orig_file = original_dir / base_name
        
        if not orig_file.exists():
            continue
            
        with open(pick_file, 'r') as f:
            pick_content = f.read()
            
        # Extract defined fixtures
        fixtures_match = re.search(r'\(:fixtures\s*\n(.*?)\)', pick_content, re.DOTALL)
        defined_fixtures = set()
        if fixtures_match:
            fixture_lines = fixtures_match.group(1)
            defined_fixtures = set(re.findall(r'(\w+)\s*-\s*\w+', fixture_lines))
        
        # Extract referenced fixtures in regions and goals
        referenced_fixtures = set()
        
        # From regions that target fixtures
        region_targets = re.findall(r'\(:target\s+(\w+)\)', pick_content)
        referenced_fixtures.update(region_targets)
        
        # From goals that reference fixture regions  
        goal_fixtures = re.findall(r'(\w+_\d+)_\w+', pick_content)
        referenced_fixtures.update(goal_fixtures)
        
        # Check for missing fixtures
        missing_fixtures = referenced_fixtures - defined_fixtures - {'kitchen_table', 'living_room_table'}
        
        if missing_fixtures:
            missing_fixture_refs.append({
                'file': str(pick_file),
                'defined_fixtures': list(defined_fixtures),
                'referenced_fixtures': list(referenced_fixtures),
                'missing_fixtures': list(missing_fixtures)
            })
    
    return missing_fixture_refs

def find_specific_error_patterns():
    """Look for the specific issues mentioned in the user request."""
    print("Detailed BDDL Error Pattern Analysis")
    print("=" * 50)
    
    # 1. Missing initial states in place files
    print("\n1. MISSING INITIAL STATES (like missing Open drawer states)")
    print("-" * 60)
    missing_opens = analyze_missing_initial_states()
    
    if missing_opens:
        for issue in missing_opens[:10]:  # Show first 10
            print(f"File: {issue['file']}")
            print(f"  Missing Open states: {issue['missing_open_states']}")
            print(f"  Original had: {issue['original_opens']}")
            print()
    else:
        print("No missing Open states found.")
    
    print(f"Total files with missing Open states: {len(missing_opens)}")
    
    # 2. Wrong objects in place files  
    print("\n2. WRONG OBJECTS IN PLACE FILES")
    print("-" * 60)
    wrong_objs = analyze_wrong_objects_in_place_files()
    
    if wrong_objs:
        for issue in wrong_objs[:10]:  # Show first 10
            print(f"File: {issue['file']}")
            if issue['extra_objects']:
                print(f"  Extra objects: {issue['extra_objects']}")
            if issue['missing_objects']:
                print(f"  Missing objects: {issue['missing_objects']}")
            print()
    
    print(f"Total place files with wrong objects: {len(wrong_objs)}")
    
    # 3. Missing fixtures in pick files
    print("\n3. MISSING FIXTURES IN PICK FILES")
    print("-" * 60)
    missing_fixtures = analyze_missing_fixtures_in_pick_files()
    
    if missing_fixtures:
        for issue in missing_fixtures[:10]:  # Show first 10
            print(f"File: {issue['file']}")
            print(f"  Missing fixtures: {issue['missing_fixtures']}")
            print(f"  Defined: {issue['defined_fixtures']}")
            print(f"  Referenced: {issue['referenced_fixtures']}")
            print()
    
    print(f"Total pick files with missing fixtures: {len(missing_fixtures)}")
    
    # 4. Look for the specific red cup in chocolate pudding case
    print("\n4. SPECIFIC CASE: RED CUP IN CHOCOLATE PUDDING FILES")
    print("-" * 60)
    base_dir = Path("/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss/libero/libero/bddl_files/atomic_skills")
    
    chocolate_files = list(base_dir.glob("*chocolate_pudding*place.bddl"))
    for choc_file in chocolate_files:
        with open(choc_file, 'r') as f:
            content = f.read()
        
        # Check if it contains red_coffee_mug
        if 'red_coffee_mug' in content:
            print(f"FOUND: {choc_file} contains red_coffee_mug")
            
            # Check corresponding original
            base_name = choc_file.name.replace('_place.bddl', '.bddl')
            orig_file = base_dir / "original_44_skills" / base_name
            
            if orig_file.exists():
                with open(orig_file, 'r') as f:
                    orig_content = f.read()
                
                orig_objects = set(re.findall(r'(\w+_\d+)\s*-', orig_content))
                place_objects = set(re.findall(r'(\w+_\d+)\s*-', content))
                
                print(f"  Original objects: {orig_objects}")
                print(f"  Place file objects: {place_objects}")
                print(f"  Extra in place: {place_objects - orig_objects}")
                print()
    
    # Save detailed analysis
    detailed_report = {
        'missing_open_states': missing_opens,
        'wrong_objects_in_place_files': wrong_objs,
        'missing_fixtures_in_pick_files': missing_fixtures
    }
    
    with open('/mnt/arc/yygx/pkgs_baselines/openvla-oft/detailed_bddl_analysis.json', 'w') as f:
        json.dump(detailed_report, f, indent=2)
    
    print(f"\nDetailed analysis saved to: detailed_bddl_analysis.json")

if __name__ == "__main__":
    find_specific_error_patterns()