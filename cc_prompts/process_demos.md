# Prompt: Script for Generating Localized Robot Demonstrations

## 1. Project Overview

The primary goal of this task is to create a Python script that processes the large-scale `libero` robotics dataset. This script will extract short, meaningful sub-trajectories, or "local demos," from long, complex demonstrations. 

These local demos are crucial for training a Visual Language Action (VLA) model for the challenging "last-mile" portion of robotic manipulation tasks. By focusing the training data on these critical, contact-rich segments, we aim to significantly improve the model's robustness and ability to generalize.

***

## 2. Core Script Objectives

The final Python script must perform the following actions:

1.  **Parse and Categorize:** Iterate through a directory of `.bddl` files that define atomic skills. Categorize each skill as `pick`, `place`, or `atomic` based on its filename suffix (`_pick.bddl`, `_place.bddl`, or no suffix).
2.  **Map to Source Data:** For each atomic skill, identify the corresponding original, full-length demonstration HDF5 file using a provided JSON mapping file.
3.  **Simulate and Replay:** Programmatically load a `libero` simulation environment for each atomic skill and replay the full-length demonstration trajectory from the source HDF5 file.
4.  **Extract Local Segment:** During replay, extract a specific segment of the trajectory based on precise, category-specific slicing logic detailed below.
5.  **Save Output:** Aggregate all extracted local demo segments for a given atomic skill and save them into a single, new HDF5 file named after the skill.

***

## 3. Key Resources & Input Files

The script will need to access the following files and directories. 

* **Atomic Skill BDDL Files**
    * **Path:** `externals/boss/libero/libero/bddl_files/atomic_skills`
    * **Purpose:** This directory contains the definitions for all new atomic skills. The script will iterate through these files. The filename is critical for determining the skill category and the slicing logic to apply.

* **Category Split Map (`cat_split_map.json`)**
    * **Path:** `externals/boss/libero/libero/bddl_files/atomic_skills/cat_split_map.json`
    * **Purpose:** This JSON file maps the original `libero` tasks to their new, split atomic skills (`_pick` and `_place`) or keep unchanged if original ones are already atomic (no suffix). It is essential for finding the correct source demonstration data for split skills.

* **Original `libero` Demonstrations**
    * **Path:** `/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/libero_90_no_noops/`
    * **Purpose:** This directory contains the source HDF5 files (*.hdf5) with the full-length demonstration trajectories that need to be replayed and sliced.

* **Reference Implementation Script**
    * **Path:** `utils/generate_local_demos_h2.py`
    * **Purpose:** This script is a previous, simpler version of the required logic. It should be used as a **concrete reference** for how to perform several key operations:
        * Initializing a `libero` simulation environment.
        * Loading and replaying a demonstration trajectory step-by-step.
        * Detecting end-effector **contact** with objects in the scene.
        * Detecting a **gripper-closing** action from the action vectors.
        The new script should adopt and adapt these low-level mechanics.

***

## 4. Detailed Implementation Logic

The core of the script is the logic for slicing the demonstrations. This logic is different for each skill category.

### Slicing Logic for "Pick" Skills (`*_pick.bddl`)
* **Start Trigger:** The start of the local demo is determined by a key event. This event is either the **first contact** made by the end-effector or the **first time the gripper command transitions from negative (open) to positive (close)**. Let the timestep of this trigger be `trigger_timestep`.
* **Start Index:** `max(0, trigger_timestep - 5)`
* **End Index:** The final timestep of the simulation replay. The `libero` environment, when configured with a `_pick.bddl` file, will **terminate automatically** once the `PickedUp` success condition is met. The script should simply collect the trajectory up to this natural termination point.

### Slicing Logic for "Place" Skills (`*_place.bddl`)
* **Start Index:** `max(0, total_timesteps - 10)`, where `total_timesteps` is the total number of steps in the original demonstration. This captures the final approach and placement action.
* **End Index:** The final timestep of the original demonstration.

### Slicing Logic for Atomic Skills (Category 2, no suffix)
* **Start Trigger:** The same as the "pick" skill: the first detected **end-effector contact**. Let the timestep of this trigger be `trigger_timestep`.
* **Start Index:** `max(0, trigger_timestep - 6)`
* **End Index:** The final timestep of the original demonstration.

***

## 5. Code Structure and Environment

* **Script Location:** Please place the final script at `TBD/scripts/generate_local_demos.py`.
* **Output Location:** The generated HDF5 files should be saved to `TBD/datasets/libero_local_demos/`. The output filename should match the atomic skill BDDL filename, with the extension changed to `.hdf5` (e.g., `KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_pick.bddl` -> `KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_pick_demo.hdf5`).
* **Dependencies:** The script will rely on standard libraries like `h5py`, `numpy`, `tqdm`, and the `libero` robotics suite.

***

## 6. Required Features & Best Practices

* **Command-Line Interface:** The script **must** use Python's `argparse` library to accept paths for all input directories and the output directory.
* **Debug Mode:** Implement a mandatory `--debug` command-line flag. When this flag is used, the script must:
    * Process only **one** `.bddl` file for each category (`_pick`, `_place`, and atomic).
    * For each of those files, process only the **first** demonstration trajectory found in the source HDF5 file.
    * This feature is critical for enabling rapid testing and debugging of the full data pipeline.
* **Clear Logging:** The script should print clear and concise status updates to the console using `print` or a simple logger. This should include which file is being processed, the detected skill type, the number of demos found, and the start/end indices for each extracted slice.

***

## 7. Testing Requirements

Please provide a separate test script to verify the core functionality.

* **Test File Location:** `TBD/tests/test_data_generation.py`
* **Test Coverage:** The tests should, at a minimum, verify:
    * A function that correctly identifies skill types from a list of sample filenames.
    * A function that correctly uses the `cat_split_map.json` to find the source demo file for a given `_pick` or `_place` bddl.
    * A basic test that confirms an output HDF5 file is created after a mock processing run.

***

## 8. Final Deliverables

1.  The main data generation script: `generate_local_demos.py`
2.  The corresponding test script: `test_data_generation.py`