#!/usr/bin/env python3
"""
Task Sequence Planner for Long Horizon Manipulation Tasks.

This script provides high-level task planning by converting task names to skill sequences
and mapping language skills to their corresponding HDF5 dataset paths. Currently implements
fixed task planning based on predefined long_horizon_tasks.json but designed to be
extended with VLM-based planning in the future.

Key Functions:
- simple_fixed_task_planner(): Returns skill sequence for given task name
- get_skill_dataset_paths(): Maps language skills to HDF5 combined dataset paths
"""

import os
import sys
import json
import csv
import argparse
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add project root to path
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')


def load_long_horizon_tasks(tasks_file: str) -> Dict:
    """Load long horizon tasks from JSON file."""
    with open(tasks_file, 'r') as f:
        return json.load(f)


def load_skills_mapping_csv(csv_file: str) -> Dict[str, str]:
    """
    Load skills to combined_path mapping from CSV file.
    
    Args:
        csv_file: Path to skills_objects_mapping_updated.csv
        
    Returns:
        Dictionary mapping skill_name to combined_path
    """
    skills_mapping = {}
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                skill_name = row['skill_name'].strip().strip('"')  # Remove quotes
                combined_path = row['combined_path'].strip()
                
                if skill_name and combined_path:
                    skills_mapping[skill_name] = combined_path
        
        return skills_mapping
        
    except Exception as e:
        print(f"❌ Error loading CSV mapping: {e}")
        return {}


def simple_fixed_task_planner(task_name: str, tasks_file: str) -> Optional[List[str]]:
    """
    Simple fixed task planner that returns skill sequences for predefined tasks.
    
    This function implements fixed task planning based on the long_horizon_tasks.json
    file. In the future, this can be extended or replaced with VLM-based planning.
    
    Args:
        task_name: Name of the long horizon task
        tasks_file: Path to long_horizon_tasks.json
        
    Returns:
        List of language skill names in execution order, or None if task not found
    """
    # Load tasks
    try:
        tasks_data = load_long_horizon_tasks(tasks_file)
    except Exception as e:
        print(f"❌ Error loading tasks file: {e}")
        return None
    
    # Find matching task (case-insensitive)
    task_name_lower = task_name.lower().strip()
    
    for task in tasks_data['long_horizon_tasks']:
        if task['name'].lower().strip() == task_name_lower:
            print(f"✅ Found task plan for: {task['name']}")
            print(f"📊 Task ID: {task['task_id']}")
            print(f"📝 Description: {task['description']}")
            print(f"🔢 Estimated steps: {task['estimated_steps']}")
            print(f"🎯 Skills sequence: {len(task['skills'])} skills")
            
            return task['skills']
    
    # Show available tasks if not found
    available_tasks = [task['name'] for task in tasks_data['long_horizon_tasks']]
    print(f"❌ Task '{task_name}' not found")
    print(f"📋 Available tasks: {available_tasks}")
    
    return None


def extract_language_from_path(combined_path: str) -> str:
    """
    Extract language description from HDF5 combined path.
    
    Args:
        combined_path: Path like "datasets/.../pick_moka_pot_combined.hdf5"
        
    Returns:
        Language description like "pick moka pot"
    """
    # Extract filename and remove _combined.hdf5
    filename = os.path.basename(combined_path)
    if filename.endswith('_combined.hdf5'):
        filename = filename.replace('_combined.hdf5', '')  # Remove '_combined.hdf5'
    
    # Convert underscores to spaces
    language_desc = filename.replace('_', ' ')
    
    return language_desc


def get_skill_dataset_paths(skills: List[str], csv_file: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Map language skills to their corresponding HDF5 dataset combined paths and extract real language descriptions.
    
    Args:
        skills: List of language skill names from JSON
        csv_file: Path to skills_objects_mapping_updated.csv
        
    Returns:
        Tuple of (real_language_skills, skill_name -> combined_path mapping)
    """
    # Load skills mapping from CSV
    skills_mapping = load_skills_mapping_csv(csv_file)
    
    if not skills_mapping:
        print("❌ Failed to load skills mapping from CSV")
        return [], {}
    
    print(f"📋 Loaded {len(skills_mapping)} skills from CSV mapping")
    
    # Map skills to dataset paths and extract real language descriptions
    skill_paths = {}
    real_language_skills = []
    missing_skills = []
    
    for skill in skills:
        skill_clean = skill.strip()
        
        if skill_clean in skills_mapping:
            combined_path = skills_mapping[skill_clean]
            
            # Extract real language description from path
            real_language_desc = extract_language_from_path(combined_path)
            
            skill_paths[real_language_desc] = combined_path
            real_language_skills.append(real_language_desc)
            
            print(f"✅ '{skill_clean}' -> '{real_language_desc}' -> {os.path.basename(combined_path)}")
        else:
            missing_skills.append(skill_clean)
            print(f"❌ {skill_clean} -> NOT FOUND in CSV mapping")
    
    if missing_skills:
        print(f"\n⚠️  Warning: {len(missing_skills)} skills not found in mapping:")
        for skill in missing_skills:
            print(f"   - {skill}")
        
        print(f"\n💡 Available skills in CSV:")
        for available_skill in sorted(skills_mapping.keys())[:10]:
            print(f"   - {available_skill}")
        if len(skills_mapping) > 10:
            print(f"   ... and {len(skills_mapping) - 10} more")
    
    return real_language_skills, skill_paths


def plan_task_sequence(task_name: str, tasks_file: str = None, csv_file: str = None) -> Optional[List[str]]:
    """
    Main function for pipeline integration - returns actual language skill sequence.
    
    This is the function called by execute_long_horizon_pipeline.py.
    
    Args:
        task_name: Name of the long horizon task
        tasks_file: Path to long_horizon_tasks.json (optional, uses default)
        csv_file: Path to skills_objects_mapping_updated.csv (optional, uses default)
        
    Returns:
        List of real language skill descriptions extracted from HDF5 paths
    """
    # Use default paths if not provided
    if tasks_file is None:
        tasks_file = '/mnt/arc/yygx/pkgs_baselines/openvla-oft/prompts/for_ai/long_horizon_tasks.json'
    if csv_file is None:
        csv_file = '/mnt/arc/yygx/pkgs_baselines/openvla-oft/prompts/for_ai/skills_objects_mapping_updated.csv'
    
    print(f"🎯 Planning task sequence: {task_name}")
    
    # Step 1: Get raw skill sequence from JSON
    raw_skills_sequence = simple_fixed_task_planner(task_name, tasks_file)
    
    if not raw_skills_sequence:
        return None
    
    # Step 2: Extract real language descriptions from dataset paths
    real_language_skills, skill_paths = get_skill_dataset_paths(raw_skills_sequence, csv_file)
    
    print(f"\n✅ Final skill sequence ({len(real_language_skills)} skills):")
    for i, skill in enumerate(real_language_skills, 1):
        print(f"   {i}. {skill}")
    
    return real_language_skills


def plan_and_map_task(task_name: str, tasks_file: str, csv_file: str) -> Tuple[Optional[List[str]], Dict[str, str]]:
    """
    Complete task planning and dataset mapping pipeline.
    
    Args:
        task_name: Name of the long horizon task
        tasks_file: Path to long_horizon_tasks.json
        csv_file: Path to skills_objects_mapping_updated.csv
        
    Returns:
        Tuple of (real_language_skills, skill_to_dataset_mapping)
    """
    print(f"🎯 Planning task: {task_name}")
    print("=" * 60)
    
    # Step 1: Get raw skill sequence using fixed planner
    raw_skills_sequence = simple_fixed_task_planner(task_name, tasks_file)
    
    if not raw_skills_sequence:
        return None, {}
    
    print(f"\n📊 Raw skills from JSON:")
    for i, skill in enumerate(raw_skills_sequence, 1):
        print(f"   {i}. {skill}")
    
    # Step 2: Map skills to dataset paths and extract real language descriptions
    print(f"\n🔗 Extracting real language descriptions from HDF5 paths:")
    print("-" * 50)
    
    real_language_skills, skill_paths = get_skill_dataset_paths(raw_skills_sequence, csv_file)
    
    print(f"\n✅ Task planning summary:")
    print(f"   📝 Task: {task_name}")
    print(f"   🔢 Raw skills: {len(raw_skills_sequence)}")
    print(f"   🎯 Real language skills: {len(real_language_skills)}")
    print(f"   💾 Mapped datasets: {len(skill_paths)}")
    print(f"   ⚠️  Missing mappings: {len(raw_skills_sequence) - len(real_language_skills)}")
    
    return real_language_skills, skill_paths


def save_task_plan(task_name: str, skills_sequence: List[str], skill_paths: Dict[str, str], output_dir: str):
    """Save task plan results to JSON file."""
    output_data = {
        "task_name": task_name,
        "planning_method": "simple_fixed_task_planner",
        "timestamp": str(Path().resolve()),  # Simple timestamp
        "skills_sequence": skills_sequence,
        "skill_dataset_paths": skill_paths,
        "total_skills": len(skills_sequence),
        "mapped_skills": len(skill_paths),
        "missing_skills": [skill for skill in skills_sequence if skill not in skill_paths]
    }
    
    # Create output filename
    task_filename = task_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    output_file = os.path.join(output_dir, f"{task_filename}_plan.json")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Task plan saved to: {output_file}")
    return output_file


def main():
    """Main function for task sequence planning."""
    parser = argparse.ArgumentParser(description='Plan task sequences for long horizon manipulation')
    parser.add_argument('--task', type=str, required=True,
                        help='Name of the long horizon task to plan')
    parser.add_argument('--tasks-file', type=str, 
                        default='/mnt/arc/yygx/pkgs_baselines/openvla-oft/prompts/for_ai/long_horizon_tasks.json',
                        help='Path to long_horizon_tasks.json')
    parser.add_argument('--csv-file', type=str,
                        default='/mnt/arc/yygx/pkgs_baselines/openvla-oft/prompts/for_ai/skills_objects_mapping_updated.csv',
                        help='Path to skills_objects_mapping_updated.csv')
    parser.add_argument('--output-dir', type=str,
                        default='/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase1/task_plans',
                        help='Output directory for task plans')
    parser.add_argument('--save', action='store_true',
                        help='Save task plan to JSON file')
    
    args = parser.parse_args()
    
    print("🎯 Task Sequence Planner")
    print(f"📁 Tasks file: {args.tasks_file}")
    print(f"📋 CSV mapping: {args.csv_file}")
    print(f"💾 Output dir: {args.output_dir}")
    
    # Validate input files
    if not os.path.exists(args.tasks_file):
        print(f"❌ Tasks file not found: {args.tasks_file}")
        return
    
    if not os.path.exists(args.csv_file):
        print(f"❌ CSV file not found: {args.csv_file}")
        return
    
    # Plan task
    skills_sequence, skill_paths = plan_and_map_task(args.task, args.tasks_file, args.csv_file)
    
    if skills_sequence is None:
        print("❌ Task planning failed")
        return
    
    # Save if requested
    if args.save:
        output_file = save_task_plan(args.task, skills_sequence, skill_paths, args.output_dir)
    
    print("\n🎉 Task sequence planning completed!")
    
    # Show next steps
    if len(skill_paths) < len(skills_sequence):
        print("\n⚠️  Next steps:")
        print("   1. Update CSV mapping with missing skills")
        print("   2. Ensure all combined HDF5 files exist")
        print("   3. Re-run planning to verify complete mapping")


if __name__ == "__main__":
    main()