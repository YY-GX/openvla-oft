#!/usr/bin/env python3
"""
Generate minimal task configuration from existing task_config.json.

Removes unused fields and creates a cleaner, more maintainable config.
"""

import json
from pathlib import Path


def generate_minimal_config():
    """Extract minimal necessary fields from task_config.json."""

    # Load existing config
    config_path = Path(__file__).parent / "task_config.json"
    with open(config_path, 'r') as f:
        full_config = json.load(f)

    # Extract minimal long horizon tasks (only task_id, name, skills)
    minimal_tasks = []
    for task in full_config.get('long_horizon_tasks', []):
        minimal_task = {
            'task_id': task['task_id'],
            'name': task['name'],
            'skills': task['skills']
        }
        minimal_tasks.append(minimal_task)

    # Extract minimal skill mappings (only target_object, language, bddl_predicates)
    minimal_skills = {}
    for skill_name, skill_data in full_config.get('skill_mappings', {}).items():
        minimal_skill = {
            'target_object': skill_data['target_object'],
            'language': skill_data['language'],
            'bddl_predicates': skill_data['bddl_predicates']
        }
        minimal_skills[skill_name] = minimal_skill

    # Create minimal config with documentation
    minimal_config = {
        "_comment": "Minimal task and skill configuration for Phase 3 evaluation pipeline",
        "_description": [
            "This config contains:",
            "  1. long_horizon_tasks: Task definitions with skill sequences",
            "  2. skill_mappings: Atomic skill definitions with target objects and success predicates",
            "",
            "Used by:",
            "  - evaluate_above.py: Gets target_object and language for each skill",
            "  - task_planner.py: Gets skill sequence for each long horizon task",
            "  - skill_checker.py: Gets bddl_predicates to check skill success"
        ],
        "_how_to_add_new_long_horizon_task": {
            "step_1": "Identify the skill sequence from your BDDL file (e.g., LONG_HORIZON_complete_kitchen_organization.bddl)",
            "step_2": "Add task entry to 'long_horizon_tasks' section below",
            "step_3": "For each skill in the sequence, check if it exists in 'skill_mappings'",
            "step_4": "If skill is NEW, add it to 'skill_mappings' with target_object, language, and bddl_predicates",
            "step_5": "Run evaluation: python scripts/phase3/pipeline/evaluation/eval_long_horizon.py --task_name 'Your Task Name'",
            "example": {
                "long_horizon_tasks_addition": {
                    "task_id": 8,
                    "name": "Complete Kitchen Organization",
                    "skills": [
                        "pick black bowl 1",
                        "place black bowl 1 on the plate 1",
                        "open the top drawer of the cabinet 1",
                        "pick ketchup",
                        "place ketchup in top drawer of the cabinet 1",
                        "close the top drawer of the cabinet 1"
                    ]
                },
                "skill_mappings_addition_if_new": {
                    "pick ketchup": {
                        "target_object": "ketchup_1_main",
                        "language": "pick ketchup",
                        "bddl_predicates": [["pickedup", "ketchup_1_main"]]
                    }
                }
            }
        },
        "_field_reference": {
            "long_horizon_tasks": {
                "task_id": "Unique integer ID for the task",
                "name": "Task name (must match when calling evaluation script)",
                "skills": "List of skill names in execution order (must exist in skill_mappings)"
            },
            "skill_mappings": {
                "target_object": "MuJoCo object name with _main suffix (e.g., akita_black_bowl_1_main)",
                "language": "Clean VLA instruction without numbers (e.g., 'pick black bowl')",
                "bddl_predicates": "List of [predicate, object1, object2] tuples for success checking"
            }
        },
        "long_horizon_tasks": minimal_tasks,
        "skill_mappings": minimal_skills
    }

    # Save minimal config
    output_path = Path(__file__).parent / "tasks_and_skills.json"
    with open(output_path, 'w') as f:
        json.dump(minimal_config, f, indent=2)

    # Print statistics
    original_lines = sum(1 for _ in open(config_path))
    new_lines = sum(1 for _ in open(output_path))

    print(f"✅ Generated minimal config: {output_path.name}")
    print(f"📊 Statistics:")
    print(f"   Original: {original_lines} lines")
    print(f"   New:      {new_lines} lines")
    print(f"   Reduction: {100 * (1 - new_lines/original_lines):.1f}%")
    print(f"\n📝 Structure:")
    print(f"   Long horizon tasks: {len(minimal_tasks)}")
    print(f"   Skill mappings:     {len(minimal_skills)}")
    print(f"\n💡 Removed unused fields:")
    print(f"   - object_mappings section (not used)")
    print(f"   - init_files (not used by evaluation)")
    print(f"   - bddl_files (not used by evaluation)")
    print(f"   - combined_path (not used by evaluation)")
    print(f"   - task descriptions, initial_states, goal_states (not used)")


if __name__ == "__main__":
    generate_minimal_config()
