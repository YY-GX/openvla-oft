#!/usr/bin/env python3
"""
Add bddl_file and benchmark fields to tasks_and_skills.json.

This allows explicit BDDL file mapping instead of relying on name matching.
"""

import json
from pathlib import Path


def add_bddl_mapping():
    """Add bddl_file and benchmark fields to each task."""

    config_path = Path(__file__).parent / "tasks_and_skills.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # BDDL file mappings based on task names
    # Format: task_name -> (bddl_file, benchmark)
    bddl_mappings = {
        "Cooking Preparation Setup": ("LONG_HORIZON_cooking_preparation_setup", "long_horizon_tasks_v0"),
        "Complete Kitchen Organization": ("LONG_HORIZON_complete_kitchen_organization", "long_horizon_tasks_v0"),
        "Switch Table Objects": ("LONG_HORIZON_switch_table_objects", "long_horizon_tasks_v0"),
        "Pick White Bowl": ("LONG_HORIZON_pick_white_bowl", "long_horizon_tasks_v0"),
        "Pick Black Bowl": ("LONG_HORIZON_pick_black_bowl", "long_horizon_tasks_v0"),
        "Put The Black Bowl In The Bottom Drawer Of The Cabinet And Close It": (
            "LONG_HORIZON_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
            "long_horizon_tasks_v0"
        ),
        "Turn On The Stove And Put The Moka Pot On It": (
            "LONG_HORIZON_turn_on_the_stove_and_put_the_moka_pot_on_it",
            "long_horizon_tasks_v0"
        ),
    }

    # Update tasks
    for task in config['long_horizon_tasks']:
        task_name = task['name']

        # Handle task 8 (v1 task) - check by task_id
        if task['task_id'] == 8:
            task['bddl_file'] = 'LONG_HORIZON_complete_kitchen_organization'
            task['benchmark'] = 'long_horizon_tasks_v1'
            print(f"✓ Task {task['task_id']}: {task_name} (v1)")
        elif task_name in bddl_mappings:
            bddl_file, benchmark = bddl_mappings[task_name]
            task['bddl_file'] = bddl_file
            task['benchmark'] = benchmark
            print(f"✓ Task {task['task_id']}: {task_name} ({benchmark})")
        else:
            # Fallback: infer from task name
            bddl_file = "LONG_HORIZON_" + task_name.lower().replace(' ', '_')
            task['bddl_file'] = bddl_file
            task['benchmark'] = 'long_horizon_tasks_v0'
            print(f"⚠️  Task {task['task_id']}: {task_name} (inferred: {bddl_file})")

    # Update documentation
    if '_field_reference' in config:
        config['_field_reference']['long_horizon_tasks']['bddl_file'] = (
            "BDDL filename (e.g., 'LONG_HORIZON_complete_kitchen_organization')"
        )
        config['_field_reference']['long_horizon_tasks']['benchmark'] = (
            "LIBERO benchmark name (e.g., 'long_horizon_tasks_v0' or 'long_horizon_tasks_v1')"
        )

    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Updated {len(config['long_horizon_tasks'])} tasks with BDDL file mappings")
    print(f"💾 Saved to: {config_path}")


if __name__ == "__main__":
    add_bddl_mapping()
