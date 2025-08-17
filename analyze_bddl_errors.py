#!/usr/bin/env python3
"""
Comprehensive BDDL file comparison script to identify systematic errors
in the generated atomic skills files compared to originals.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import json

def parse_bddl_file(filepath):
    """Parse a BDDL file and extract key components."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract sections using regex
    data = {
        'language': '',
        'regions': [],
        'fixtures': [],
        'objects': [],
        'obj_of_interest': [],
        'init': [],
        'goal': []
    }
    
    # Language line
    lang_match = re.search(r'\(:language\s+(.+?)\)', content)
    if lang_match:
        data['language'] = lang_match.group(1).strip()
    
    # Regions section
    regions_match = re.search(r'\(:regions\s*\n(.*?)\n\s*\)', content, re.DOTALL)
    if regions_match:
        regions_content = regions_match.group(1)
        # Extract region names (first word in parentheses)
        region_names = re.findall(r'\(\s*(\w+)', regions_content)
        data['regions'] = region_names
    
    # Fixtures section
    fixtures_match = re.search(r'\(:fixtures\s*\n(.*?)\)', content, re.DOTALL)
    if fixtures_match:
        fixtures_content = fixtures_match.group(1)
        # Extract fixture definitions like "microwave_1 - microwave"
        fixtures = re.findall(r'(\w+)\s*-\s*(\w+)', fixtures_content)
        data['fixtures'] = fixtures
    
    # Objects section
    objects_match = re.search(r'\(:objects\s*\n(.*?)\)', content, re.DOTALL)
    if objects_match:
        objects_content = objects_match.group(1)
        # Extract object definitions
        objects = re.findall(r'(\w+(?:\s+\w+)*)\s*-\s*(\w+)', objects_content)
        data['objects'] = objects
    
    # Obj_of_interest section
    obj_interest_match = re.search(r'\(:obj_of_interest\s*\n(.*?)\)', content, re.DOTALL)
    if obj_interest_match:
        obj_interest_content = obj_interest_match.group(1)
        objects = re.findall(r'(\w+)', obj_interest_content)
        data['obj_of_interest'] = objects
    
    # Init section
    init_match = re.search(r'\(:init\s*\n(.*?)\)', content, re.DOTALL)
    if init_match:
        init_content = init_match.group(1)
        # Extract all predicates like (On ...) (Open ...) (Close ...)
        predicates = re.findall(r'\((\w+)\s+([^)]+)\)', init_content)
        data['init'] = predicates
    
    # Goal section
    goal_match = re.search(r'\(:goal\s*\n(.*?)\)', content, re.DOTALL)
    if goal_match:
        goal_content = goal_match.group(1)
        predicates = re.findall(r'\((\w+)\s+([^)]+)\)', goal_content)
        data['goal'] = predicates
    
    return data

def find_issues(original_data, generated_data, filename):
    """Compare original and generated data to find issues."""
    issues = []
    
    # Check for wrong/missing objects
    orig_objects = set(obj[0] for obj in original_data['objects'])
    gen_objects = set(obj[0] for obj in generated_data['objects'])
    
    missing_objects = orig_objects - gen_objects
    extra_objects = gen_objects - orig_objects
    
    if missing_objects:
        issues.append({
            'type': 'missing_objects',
            'description': f"Missing objects: {missing_objects}",
            'objects': list(missing_objects)
        })
    
    if extra_objects:
        issues.append({
            'type': 'extra_objects', 
            'description': f"Extra objects: {extra_objects}",
            'objects': list(extra_objects)
        })
    
    # Check for missing fixtures
    orig_fixtures = set(fix[0] for fix in original_data['fixtures'])
    gen_fixtures = set(fix[0] for fix in generated_data['fixtures'])
    
    missing_fixtures = orig_fixtures - gen_fixtures
    if missing_fixtures:
        issues.append({
            'type': 'missing_fixtures',
            'description': f"Missing fixtures: {missing_fixtures}",
            'fixtures': list(missing_fixtures)
        })
    
    # Check for missing regions  
    orig_regions = set(original_data['regions'])
    gen_regions = set(generated_data['regions'])
    
    missing_regions = orig_regions - gen_regions
    if missing_regions:
        issues.append({
            'type': 'missing_regions',
            'description': f"Missing regions: {missing_regions}",
            'regions': list(missing_regions)
        })
    
    # Check for missing initial states
    orig_init = set((pred[0], pred[1]) for pred in original_data['init'])
    gen_init = set((pred[0], pred[1]) for pred in generated_data['init'])
    
    missing_init = orig_init - gen_init
    if missing_init:
        issues.append({
            'type': 'missing_initial_states',
            'description': f"Missing initial states: {missing_init}",
            'states': list(missing_init)
        })
    
    # Check for references to missing fixtures/objects in regions and init
    all_referenced_objects = set()
    all_referenced_fixtures = set()
    
    # Extract references from regions
    for region in generated_data['regions']:
        # Look for patterns like "microwave_1" in region names
        if '_' in region and region.split('_')[-1].isdigit():
            base_name = '_'.join(region.split('_')[:-1])
            all_referenced_objects.add(region.split('_')[0] + '_' + region.split('_')[-1])
    
    # Extract references from init states
    for pred_type, pred_args in generated_data['init']:
        words = pred_args.split()
        for word in words:
            if '_' in word and word.split('_')[-1].isdigit():
                all_referenced_objects.add(word)
            elif word in ['kitchen_table', 'living_room_table', 'microwave_1', 'wooden_cabinet_1']:
                all_referenced_fixtures.add(word)
    
    # Check if referenced objects/fixtures exist
    defined_objects = set(obj[0] for obj in generated_data['objects'])
    defined_fixtures = set(fix[0] for fix in generated_data['fixtures'])
    
    undefined_object_refs = all_referenced_objects - defined_objects
    undefined_fixture_refs = all_referenced_fixtures - defined_fixtures
    
    if undefined_object_refs:
        issues.append({
            'type': 'undefined_object_references',
            'description': f"References to undefined objects: {undefined_object_refs}",
            'objects': list(undefined_object_refs)
        })
    
    if undefined_fixture_refs:
        issues.append({
            'type': 'undefined_fixture_references', 
            'description': f"References to undefined fixtures: {undefined_fixture_refs}",
            'fixtures': list(undefined_fixture_refs)
        })
    
    return issues

def main():
    """Main analysis function."""
    base_dir = Path("/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss/libero/libero/bddl_files/atomic_skills")
    original_dir = base_dir / "original_44_skills"
    
    # Directories to check
    check_dirs = [
        base_dir,  # Root level generated files
        base_dir / "cat1",
        base_dir / "cat2", 
        base_dir / "modified_cat1",
        base_dir / "modified_cat2"
    ]
    
    all_issues = defaultdict(list)
    issue_counts = defaultdict(int)
    
    print("BDDL File Comparison Analysis")
    print("=" * 50)
    
    # Get all original files for comparison
    original_files = {}
    for orig_file in original_dir.glob("*.bddl"):
        original_files[orig_file.name] = parse_bddl_file(orig_file)
    
    print(f"Found {len(original_files)} original files")
    
    # Check each directory
    for check_dir in check_dirs:
        if not check_dir.exists():
            continue
            
        print(f"\nAnalyzing directory: {check_dir.name}")
        print("-" * 30)
        
        generated_files = list(check_dir.glob("*.bddl"))
        print(f"Found {len(generated_files)} generated files")
        
        for gen_file in generated_files:
            # For pick/place files, find corresponding original
            base_name = gen_file.name
            if base_name.endswith('_pick.bddl'):
                orig_name = base_name.replace('_pick.bddl', '.bddl')
            elif base_name.endswith('_place.bddl'): 
                orig_name = base_name.replace('_place.bddl', '.bddl')
            else:
                orig_name = base_name
            
            if orig_name not in original_files:
                print(f"  WARNING: No original found for {base_name}")
                continue
            
            # Parse and compare
            try:
                gen_data = parse_bddl_file(gen_file)
                orig_data = original_files[orig_name]
                
                issues = find_issues(orig_data, gen_data, gen_file.name)
                
                if issues:
                    all_issues[f"{check_dir.name}/{gen_file.name}"] = issues
                    print(f"  ISSUES in {gen_file.name}:")
                    for issue in issues:
                        print(f"    - {issue['type']}: {issue['description']}")
                        issue_counts[issue['type']] += 1
                        
            except Exception as e:
                print(f"  ERROR parsing {gen_file.name}: {e}")
    
    # Summary report
    print("\n" + "=" * 50)
    print("SUMMARY REPORT")
    print("=" * 50)
    
    print(f"\nTotal files with issues: {len(all_issues)}")
    print(f"Total issue instances: {sum(issue_counts.values())}")
    
    print("\nIssue types and counts:")
    for issue_type, count in sorted(issue_counts.items()):
        print(f"  {issue_type}: {count}")
    
    # Detailed breakdown by error type
    print("\n" + "=" * 50)
    print("DETAILED BREAKDOWN BY ERROR TYPE")
    print("=" * 50)
    
    issues_by_type = defaultdict(list)
    for filename, file_issues in all_issues.items():
        for issue in file_issues:
            issues_by_type[issue['type']].append((filename, issue))
    
    for issue_type, instances in issues_by_type.items():
        print(f"\n{issue_type.upper()} ({len(instances)} instances):")
        print("-" * 40)
        for filename, issue in instances:
            print(f"  {filename}: {issue['description']}")
    
    # Save detailed report to JSON
    report = {
        'summary': {
            'total_files_with_issues': len(all_issues),
            'total_issue_instances': sum(issue_counts.values()),
            'issue_type_counts': dict(issue_counts)
        },
        'detailed_issues': dict(all_issues),
        'issues_by_type': {k: [(f, i) for f, i in v] for k, v in issues_by_type.items()}
    }
    
    with open('/mnt/arc/yygx/pkgs_baselines/openvla-oft/bddl_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: bddl_analysis_report.json")

if __name__ == "__main__":
    main()