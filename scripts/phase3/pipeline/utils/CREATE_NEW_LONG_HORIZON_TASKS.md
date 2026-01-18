# Creating New Long-Horizon Tasks for Evaluation

This guide explains how to create new long-horizon tasks for evaluation with `evaluate_above.py`.

## Overview

The evaluation pipeline (`evaluate_above.py`) executes long-horizon tasks by:
1. Loading the BDDL file from `long_horizon_tasks_v0/` directory
2. Looking up the skill sequence from `task_config.json["long_horizon_tasks"]`
3. For each skill, getting target objects and language from `skill_mappings`
4. Executing VLA → checking BDDL predicates → moving to next skill

---

## TODO List

### 1. Create BDDL File

**Location:** `externals/boss/libero/libero/bddl_files/long_horizon_tasks_v0/`

**File naming:** `LONG_HORIZON_<your_task_name>.bddl`

**Structure:**
```lisp
(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
  (:domain robosuite)

  ;; Line 3: Language - comma-separated list of atomic skills (matches task_config skill names)
  (:language pick moka pot, place moka pot on the stove 2, turn on the stove 2, ...)

  ;; Lines 4-69: Regions - define spawn regions and target regions for objects
  (:regions
    (object_init_region
        (:target kitchen_table)
        (:ranges ((-0.1 0.1 -0.09 0.11)))
        (:yaw_rotation ((0.0 0.0)))
    )
    (target_region
        (:target fixture_name)
    )
  )

  ;; Lines 71-75: Fixtures - fixed furniture (stoves, cabinets, tables, microwaves)
  (:fixtures
    kitchen_table - kitchen_table
    flat_stove_1 flat_stove_2 - flat_stove
    microwave_1 - microwave
  )

  ;; Lines 77-80: Objects - movable objects (bowls, bottles, etc.)
  (:objects
    moka_pot_1 - moka_pot
    chefmate_8_frypan_1 - chefmate_8_frypan
  )

  ;; Lines 82-88: Objects of interest - relevant for success checking
  (:obj_of_interest
    moka_pot_1
    chefmate_8_frypan_1
    flat_stove_1
  )

  ;; Lines 90-96: Init - initial object placements
  (:init
    (On moka_pot_1 kitchen_table_moka_pot_init_region)
    (On chefmate_8_frypan_1 kitchen_table_frypan_init_region)
  )

  ;; Lines 98-106: Goal - goal predicates
  (:goal
    (And
      (On moka_pot_1 flat_stove_1_cook_region_1)
      (Turnon flat_stove_1)
      (Open microwave_1_heat_region)
    )
  )
)
```

**Common BDDL Predicates:**
- `(On object region)` - object must be on/in the region
- `(Turnon object)` - object must be turned on (stove, microwave)
- `(Open object)` - object must be open (drawer, microwave)
- `(Inside object container)` - object must be inside container
- `(NextTo object1 object2)` - spatial relationship

---

### 2. Register Task in Benchmark

**Location:** `externals/boss/libero/libero/benchmark/boss_task_map.py`

**Action:** Add your BDDL filename (without `.bddl` extension) to the `"long_horizon_tasks_v0"` list:

```python
boss_task_map = {
    # ... other suites ...
    "long_horizon_tasks_v0": [
        "LONG_HORIZON_cooking_preparation_setup",
        "LONG_HORIZON_complete_kitchen_organization",
        "LONG_HORIZON_switch_table_objects",
        "LONG_HORIZON_pick_white_bowl",
        "LONG_HORIZON_pick_black_bowl",
        "LONG_HORIZON_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "LONG_HORIZON_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "LONG_HORIZON_your_new_task",  # ← Add here
    ],
}
```

---

### 3. Update Task Count in Benchmark Init

**Location:** `externals/boss/libero/libero/benchmark/__init__.py`

**Action:** Update the range in the benchmark indices:

```python
benchmark_dict = {
    # ... other suites ...
    "long_horizon_tasks_v0": [i for i in range(0, 8)],  # ← Change 7 to 8 (if adding 1 task)
}
```

**Note:** The number should match the total count of tasks in the list.

---

### 4. Add Task Definition to task_config.json

**Location:** `scripts/phase3/pipeline/config/task_config.json`

#### Section 1: Add to `"long_horizon_tasks"` array

```json
{
  "long_horizon_tasks": [
    {
      "task_id": 1,
      "name": "Cooking Preparation Setup",
      "description": "...",
      "skills": [...],
      "estimated_steps": 7,
      "initial_states": [...],
      "goal_states": [...]
    },
    {
      "task_id": 4,  // ← Increment from last task
      "name": "Your Task Name",
      "description": "Clear description of what the task does",
      "skills": [
        "pick object 1",
        "place object 1 on target 1",
        "open the drawer",
        "pick object 2",
        "place object 2 in drawer"
      ],
      "estimated_steps": 5,
      "initial_states": [
        "object 1 is on table",
        "object 2 is on table",
        "drawer is closed"
      ],
      "goal_states": [
        "object 1 is on target 1",
        "object 2 is in drawer"
      ]
    }
  ]
}
```

#### Section 2: Add skill mappings for each atomic skill

For **each skill** in your task's `"skills"` list, add an entry to `"skill_mappings"` (if it doesn't already exist):

```json
{
  "skill_mappings": {
    "pick object 1": {
      "init_files": ["pick_object.init"],
      "bddl_files": [
        "KITCHEN_SCENE3_pick_object.bddl",
        "KITCHEN_SCENE5_pick_object.bddl"
      ],
      "combined_path": "datasets/hdf5_datasets/atomic_local_demos_augmented_pick_10_place_15_atomic_15/combined_by_language/pick_object_original.hdf5",
      "target_object": "object_1_main",
      "language": "pick object",
      "bddl_predicates": [["pickedup", "object_1_main"]]
    },
    "place object 1 on target 1": {
      "init_files": ["place_object_on_target.init"],
      "bddl_files": ["KITCHEN_SCENE3_place_object_on_target.bddl"],
      "combined_path": "datasets/.../place_object_on_target_original.hdf5",
      "target_object": "target_1_main",
      "language": "place object on target",
      "bddl_predicates": [["on", "object_1_main", "target_1_region"]]
    }
  }
}
```

**Key fields:**
- `init_files`: List of `.init` files for this skill (in `datasets/hdf5_datasets/atomic_above_fewer/all/`)
- `bddl_files`: List of atomic BDDL files for this skill
- `target_object`: MuJoCo object name that VLA should interact with
- `language`: Clean language instruction for VLA (without numbers)
- `bddl_predicates`: Success criteria predicates to check

---

### 5. Update object_mappings (if needed)

**Location:** `scripts/phase3/pipeline/config/task_config.json` (top of file)

If your task uses new objects, add mappings from simplified names to MuJoCo names:

```json
{
  "object_mappings": {
    "bowl_1": "akita_black_bowl_1",
    "moka_pot": "moka_pot_1",
    "plate_1": "plate_1",
    "stove_1": "flat_stove_1",
    "your_object": "mujoco_object_name",  // ← Add new mappings
    "cabinet_1": "wooden_cabinet_1"
  }
}
```

---

### 6. Ensure Atomic Skill Components Exist

For **each skill** in your task's `"skills"` list, verify these files exist:

#### Required Files Checklist:
- [ ] **Init file:** `datasets/hdf5_datasets/atomic_above_fewer/all/<skill_name>.init`
  - Contains initial states for starting this skill

- [ ] **Atomic BDDL file:** `externals/boss/libero/libero/bddl_files/atomic_skills/SCENE_X_<skill>_<pick|place>.bddl`
  - Defines the atomic skill's goal predicates

- [ ] **Skill mapping:** Entry in `task_config.json["skill_mappings"]`
  - Maps skill name to objects, language, and predicates

- [ ] **HDF5 dataset** (if using pre-trained VLA): `datasets/.../combined_by_language/<skill>_original.hdf5`
  - Training data for this skill (optional if VLA already trained)

---

### 7. Test Your New Task

Run evaluation:

```bash
python scripts/phase3/pipeline/evaluation/evaluate_above.py \
    --vla_checkpoint runs/your_checkpoint \
    --task_name "Your Task Name"
```

The system will:
1. Load BDDL from `long_horizon_tasks_v0/LONG_HORIZON_your_task_name.bddl`
2. Look up skill sequence from `task_config.json["long_horizon_tasks"]`
3. For each skill:
   - Get target object and language from `skill_mappings`
   - Move to above pose using MPlib
   - Execute VLA for skill
   - Check BDDL predicates for success
   - If success, move to next skill; if fail, retry or abort

---

## Summary of Files to Modify

| # | File | Action |
|---|------|--------|
| 1 | `externals/boss/libero/libero/bddl_files/long_horizon_tasks_v0/<TASK>.bddl` | **Create** new BDDL file |
| 2 | `externals/boss/libero/libero/benchmark/boss_task_map.py` | **Add** task name to list |
| 3 | `externals/boss/libero/libero/benchmark/__init__.py` | **Update** task count range |
| 4 | `scripts/phase3/pipeline/config/task_config.json` | **Add** task definition + skill mappings |
| 5 | Atomic skill components | **Verify** all `.init`, `.bddl`, `.hdf5` files exist |

---

## Example: Adding "Clean Kitchen" Task

### 1. Create BDDL
**File:** `externals/boss/libero/libero/bddl_files/long_horizon_tasks_v0/LONG_HORIZON_clean_kitchen.bddl`

```lisp
(define (problem LIBERO_Kitchen_Tabletop_Manipulation)
  (:domain robosuite)
  (:language pick black bowl, place black bowl on the plate, pick ketchup, place ketchup in the drawer)
  (:regions ...)
  (:fixtures ...)
  (:objects ...)
  (:init ...)
  (:goal ...)
)
```

### 2. Register in boss_task_map.py
```python
"long_horizon_tasks_v0": [
    ...,
    "LONG_HORIZON_clean_kitchen",  # ← Add
],
```

### 3. Update benchmark count
```python
"long_horizon_tasks_v0": [i for i in range(0, 8)],  # 7 → 8
```

### 4. Add to task_config.json
```json
{
  "task_id": 4,
  "name": "Clean Kitchen",
  "skills": [
    "pick black bowl",
    "place black bowl on the plate",
    "pick ketchup",
    "place ketchup in the drawer"
  ],
  ...
}
```

### 5. Test
```bash
python scripts/phase3/pipeline/evaluation/evaluate_above.py \
    --task_name "Clean Kitchen"
```

---

## Troubleshooting

### Task not found
- Check BDDL filename matches task name (spaces → underscores, lowercase)
- Verify task is in `boss_task_map.py` list
- Check benchmark range includes your task index

### Skill mapping not found
- Ensure all skills in `"skills"` array have entries in `"skill_mappings"`
- Check skill names match exactly (case-sensitive)

### BDDL predicate check fails
- Verify `bddl_predicates` in skill mapping match BDDL goal predicates
- Check object names match MuJoCo model (use `_main` suffix)
- Ensure region names are correct (e.g., `flat_stove_1_cook_region_1`)

---

## Additional Resources

- **BDDL Syntax:** See existing files in `externals/boss/libero/libero/bddl_files/long_horizon_tasks_v0/`
- **Atomic Skills:** Check `externals/boss/libero/libero/bddl_files/atomic_skills/` for examples
- **Object Names:** Inspect MuJoCo XML models or use `env.sim.model.body_names` to list available objects
