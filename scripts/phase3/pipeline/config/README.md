# Phase 3 Configuration Files

## Overview

This directory contains configuration files for the Phase 3 evaluation pipeline.

## Configuration Files

### `tasks_and_skills.json` (Recommended - Clean & Minimal)

**New minimal configuration** with only essential fields needed for evaluation.

**Structure:**
- `long_horizon_tasks`: Task definitions with skill sequences (3 fields per task)
- `skill_mappings`: Atomic skill definitions (3 fields per skill)

**Usage:**
- Used by `evaluate_above.py`: Gets `target_object` and `language`
- Used by `task_planner.py`: Gets skill sequences for long horizon tasks
- Used by `skill_checker.py`: Gets `bddl_predicates` for success checking

**Advantages:**
- ✅ Clean and focused - only necessary fields
- ✅ Well-documented with inline examples
- ✅ Easy to extend with new tasks
- ✅ Clear field descriptions

---

### `task_config.json` (Legacy - Verbose)

**Original configuration** with many unused fields.

**Why it's verbose:**
- Contains `object_mappings` section (not used)
- Each skill has 6 fields, but only 3 are used
- Tasks have 7 fields, but only 3 are used
- No documentation on how to add new tasks

**Fields removed in new config:**
- ❌ `init_files` (not used by evaluation)
- ❌ `bddl_files` (not used by evaluation)
- ❌ `combined_path` (not used by evaluation)
- ❌ Task `description`, `initial_states`, `goal_states` (not used)
- ❌ `object_mappings` section (not used)

---

## How to Add a New Long Horizon Task

### Example: Adding a new task from `LONG_HORIZON_complete_kitchen_organization.bddl`

**Step 1:** Identify the skill sequence from your BDDL file

```
Skills:
  - pick black bowl 1
  - place black bowl 1 on the plate 1
  - open the top drawer of the cabinet 1
  - pick ketchup
  - place ketchup in top drawer of the cabinet 1
  - close the top drawer of the cabinet 1
```

**Step 2:** Add task entry to `long_horizon_tasks` in `tasks_and_skills.json`

```json
{
  "task_id": 8,
  "name": "Complete Kitchen Organization V2",
  "skills": [
    "pick black bowl 1",
    "place black bowl 1 on the plate 1",
    "open the top drawer of the cabinet 1",
    "pick ketchup",
    "place ketchup in top drawer of the cabinet 1",
    "close the top drawer of the cabinet 1"
  ]
}
```

**Step 3:** Check if all skills exist in `skill_mappings`

Most likely they already exist! If a skill is reused from another task, you don't need to add it again.

**Step 4:** If a skill is NEW, add it to `skill_mappings`

```json
"pick ketchup": {
  "target_object": "ketchup_1_main",
  "language": "pick ketchup",
  "bddl_predicates": [["pickedup", "ketchup_1_main"]]
}
```

**How to find these values:**
- `target_object`: MuJoCo object name with `_main` suffix (check `skill_config.json` or inspect environment)
- `language`: Clean VLA instruction without instance numbers
- `bddl_predicates`: Parse from BDDL goal section (e.g., `(pickedup ketchup_1)` → `["pickedup", "ketchup_1_main"]`)

**Step 5:** Run evaluation

```bash
python scripts/phase3/pipeline/evaluation/eval_long_horizon.py \
    --task_name 'Complete Kitchen Organization V2' \
    --num_trials 10
```

---

## Common BDDL Predicates

| Predicate | Format | Example | Meaning |
|-----------|--------|---------|---------|
| `pickedup` | `[predicate, object]` | `["pickedup", "akita_black_bowl_1_main"]` | Object lifted 3cm+ above initial height |
| `on` | `[predicate, obj1, obj2]` | `["on", "bowl", "plate_region"]` | obj1 is on top of obj2 |
| `in` | `[predicate, obj1, obj2]` | `["in", "ketchup", "drawer_region"]` | obj1 is inside obj2 (containment) |
| `open` | `[predicate, object]` | `["open", "wooden_cabinet_1_bottom"]` | Articulated object is open |
| `close` | `[predicate, object]` | `["close", "wooden_cabinet_1_bottom"]` | Articulated object is closed |
| `turnon` | `[predicate, object]` | `["turnon", "flat_stove_1_main"]` | Object is turned on |
| `turnoff` | `[predicate, object]` | `["turnoff", "flat_stove_1_main"]` | Object is turned off |

---

## Migrating to New Config

To migrate scripts to use `tasks_and_skills.json`:

**Option 1: Update config_loader.py** (Recommended - affects all scripts)

```python
# In scripts/phase3/pipeline/config/config_loader.py
config_path = os.path.join(script_dir, "tasks_and_skills.json")  # Changed from task_config.json
```

**Option 2: Update individual scripts**

```python
# In evaluate_above.py, task_planner.py, etc.
task_config_path = f"{script_dir}/config/tasks_and_skills.json"
```

---

## Regenerating the Config

If you need to regenerate `tasks_and_skills.json` from `task_config.json`:

```bash
python scripts/phase3/pipeline/config/generate_minimal_config.py
```

This will:
- Extract only essential fields
- Add comprehensive documentation
- Save as `tasks_and_skills.json`

---

## Field Reference

### `long_horizon_tasks` Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task_id` | int | Yes | Unique integer ID for the task |
| `name` | string | Yes | Task name (must match when calling evaluation script) |
| `skills` | list[string] | Yes | List of skill names in execution order (must exist in skill_mappings) |

### `skill_mappings` Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `target_object` | string | Yes | MuJoCo object name with `_main` suffix (e.g., `akita_black_bowl_1_main`) |
| `language` | string | Yes | Clean VLA instruction without numbers (e.g., `"pick black bowl"`) |
| `bddl_predicates` | list[list] | Yes | List of [predicate, object1, object2] tuples for success checking |

---

## Questions?

Check the inline documentation in `tasks_and_skills.json` - it has detailed examples and field descriptions.
