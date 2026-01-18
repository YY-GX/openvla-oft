#!/usr/bin/env python3
"""
Analyze grasp positions for pick skills to detect left vs right edge grasping.

For each demo:
1. Recreate LIBERO environment
2. Find stable grasp timestep (when gripper closes and holds object)
3. Set env state to that timestep
4. Extract gripper position from env.sim
5. Extract bowl position from env.sim using mujoco name
6. Compare X-coordinates to classify left/right grasp
7. Report statistics

Usage:
    python analyze_grasp_position.py --hdf5 datasets/hdf5_datasets/atomic_above_26_skills/all/pick_black_bowl.hdf5

pick:
[5, 6, 7, 8, 9, 10, 15, 20, 22, 27, 28, 29, 33, 34, 35, 36, 37, 49, 50, 51, 62, 75, 76, 77, 78, 83, 84, 88, 89, 90, 96, 97, 105, 106, 125, 126, 127, 128, 135, 142, 143, 146, 150, 151, 152, 157, 159, 165, 174, 175, 176, 177, 179, 180, 187, 188, 193, 197, 202, 203, 204, 205, 206, 207, 209, 210, 216, 219, 220, 226, 228, 233, 235, 236, 248, 249, 258, 259, 260, 262, 265, 266, 267, 269, 270, 278, 279, 287, 288, 295, 299, 300, 301, 305, 306, 307, 308, 313, 323, 324, 326, 332, 342, 345, 346, 347, 350, 352, 358, 359, 360, 361, 369, 370, 371, 372, 374, 392, 393, 396, 400, 401, 402, 407, 409, 410, 411, 420, 422, 423, 424, 425, 426, 427, 431, 432, 438, 440, 448, 449, 450, 451, 452, 456, 457, 462, 463, 467, 468, 469, 482, 483, 484, 486, 487, 496, 505, 508, 510, 524, 525, 531, 532, 533, 544, 550, 552, 553, 557, 558, 559, 561, 563, 564, 565, 572, 574, 584, 589, 590]    


"""

import argparse
import h5py
import json
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, '/mnt/arc/yygx/pkgs_baselines/openvla-oft')

from libero.libero.envs import OffScreenRenderEnv


# Paths
SKILL_CONFIG_PATH = "scripts/phase3/pipeline/config/skill_config.json"
BDDL_BASE_PATH = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss/libero/libero/bddl_files/atomic_skills"


def load_skill_config() -> Dict:
    """Load skill configuration with mujoco object names."""
    with open(SKILL_CONFIG_PATH, 'r') as f:
        return json.load(f)


def get_object_mujoco_names(skill_name: str, skill_config: Dict) -> Optional[List[str]]:
    """
    Get mujoco object names for the target object.

    For pick_black_bowl or place_black_bowl_on_the_plate:
    - Target object is black_bowl
    - Mujoco names: black_bowl_*_main (multiple variants exist)

    Returns list of mujoco names to search in sim
    """
    if skill_name not in skill_config:
        return None

    skill_info = skill_config[skill_name]

    # Get target object - returns list of mujoco names
    if 'target_object' in skill_info:
        target = skill_info['target_object']
        # Ensure it's a list
        if isinstance(target, str):
            return [target]
        return target

    # Fallback: infer from skill name
    if 'black_bowl' in skill_name:
        return ['akita_black_bowl_1_main', 'akita_black_bowl_2_main', 'akita_black_bowl_3_main']

    return None


def create_env_from_skill(skill_name: str, skill_config: Dict):
    """
    Create LIBERO environment from skill configuration.

    Args:
        skill_name: Name of the skill
        skill_config: Skill configuration dict

    Returns:
        Tuple of (env, bddl_file_name)
    """
    if skill_name not in skill_config:
        raise ValueError(f"Skill {skill_name} not found in config")

    skill_info = skill_config[skill_name]

    # Get BDDL file - use first one if multiple
    bddl_files = skill_info.get('bddl_files', [])
    if not bddl_files:
        raise ValueError(f"No BDDL files found for skill {skill_name}")

    bddl_file_name = bddl_files[0]  # Use first BDDL variant

    # Construct full BDDL file path
    bddl_file_path = os.path.join(BDDL_BASE_PATH, bddl_file_name)

    if not os.path.exists(bddl_file_path):
        raise ValueError(f"BDDL file not found: {bddl_file_path}")

    # Create environment directly from BDDL file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_path,
        camera_heights=256,
        camera_widths=256,
        horizon=10000
    )
    env.reset()

    return env, bddl_file_name




def get_object_position_from_sim(env, object_names: List[str]) -> Optional[np.ndarray]:
    """
    Get object position from simulation using mujoco names.

    Args:
        env: LIBERO environment
        object_names: List of possible mujoco object names

    Returns:
        (3,) array of xyz position, or None if not found
    """
    sim = env.sim

    # Search for object in sim model bodies using exact names
    for obj_name in object_names:
        for body_id in range(sim.model.nbody):
            body_name = sim.model.body_id2name(body_id)
            if body_name == obj_name:
                # Found the object body
                body_pos = sim.data.body_xpos[body_id].copy()
                return body_pos

    return None


def get_gripper_position_from_sim(env) -> np.ndarray:
    """
    Get gripper/end-effector position from simulation.

    Returns:
        (3,) array of xyz position
    """
    sim = env.sim

    # Try to find gripper site (most accurate)
    # site_names = ["gripper0_grip_site", "ee_site", "grip_site"]
    site_names = ["gripper0_grip_site"]
    site_names = ["robot0_joint7"]

    for site_name in site_names:
        try:
            site_id = sim.model.site_name2id(site_name)
            site_pos = sim.data.site_xpos[site_id].copy()
            return site_pos
        except:
            continue

    # Fallback: use body position of end effector body
    for body_id in range(sim.model.nbody):
        body_name = sim.model.body_id2name(body_id)
        if body_name and ("gripper" in body_name.lower() or "eef" in body_name.lower()):
            body_pos = sim.data.body_xpos[body_id].copy()
            return body_pos

    raise ValueError("Could not find gripper position in sim")




def classify_grasp_side(gripper_pos: np.ndarray,
                        bowl_pos: np.ndarray,
                        center_threshold: float = 0.01) -> str:
    """
    Classify grasp as left or right based on X-coordinate.

    Args:
        gripper_pos: (3,) gripper xyz position
        bowl_pos: (3,) bowl xyz position
        center_threshold: Distance threshold for center grasps (meters)

    Returns:
        "left" if gripper is left of bowl, "right" if right, "center" if very close
    """
    # Compare X coordinates (X axis is left-right in world frame)
    x_offset = gripper_pos[1] - bowl_pos[1]

    if abs(x_offset) < center_threshold:
        return "center"
    elif x_offset < 0:
        return "left"  # Gripper is to the left of bowl center
    else:
        return "right"  # Gripper is to the right of bowl center


def analyze_demo(hdf5_file: h5py.File,
                demo_idx: int,
                env,
                object_names: List[str],
                verbose: bool = False) -> Optional[Tuple[str, float]]:
    """
    Analyze a single demo to determine grasp position using the last timestep.

    Args:
        hdf5_file: Open HDF5 file
        demo_idx: Demo index
        env: LIBERO environment
        object_names: List of possible mujoco object names
        verbose: Print detailed info

    Returns:
        Tuple of (grasp_side, x_offset) or None if analysis failed
    """
    demo_key = f"demo_{demo_idx}"

    if demo_key not in hdf5_file['data']:
        if verbose:
            print(f"  Warning: {demo_key} not found in HDF5")
        return None

    demo = hdf5_file['data'][demo_key]

    # Load states
    try:
        states = demo['states'][:]  # (T, state_dim)
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not load states for {demo_key}: {e}")
        return None

    # Use last timestep (most stable state)
    last_t = len(states) - 1

    if last_t < 0:
        if verbose:
            print(f"  Warning: Empty trajectory for {demo_key}")
        return None

    # Set environment to last state
    try:
        env.sim.set_state_from_flattened(states[last_t])
        env.sim.forward()
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not set env state for {demo_key}: {e}")
        return None

    # Get positions from simulation
    try:
        gripper_pos = get_gripper_position_from_sim(env)
        bowl_pos = get_object_position_from_sim(env, object_names)
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not get positions from sim for {demo_key}: {e}")
        return None

    if bowl_pos is None:
        if verbose:
            print(f"  Warning: Could not find object in simulation (tried: {object_names})")
        return None

    # Classify grasp side
    grasp_side = classify_grasp_side(gripper_pos, bowl_pos)
    x_offset = gripper_pos[0] - bowl_pos[0]

    if verbose:
        print(f"  {demo_key}: timestep={last_t}, side={grasp_side}, offset={x_offset:.4f}m")

    return (grasp_side, x_offset)


def analyze_hdf5_file(hdf5_path: str, verbose: bool = False):
    """
    Analyze all demos in an HDF5 file for grasp positions.

    Args:
        hdf5_path: Path to HDF5 file
        verbose: Print detailed progress
    """
    print("=" * 80)
    print(f"Analyzing: {os.path.basename(hdf5_path)}")
    print("=" * 80)

    # Load skill config
    skill_config = load_skill_config()

    # Get skill name from filename
    skill_name = os.path.splitext(os.path.basename(hdf5_path))[0]

    # Get object names (list of possible mujoco names)
    object_names = get_object_mujoco_names(skill_name, skill_config)
    if object_names is None:
        print(f"Error: Could not find object names for skill {skill_name}")
        print(f"Available skills in config: {list(skill_config.keys())[:10]}...")
        return

    print(f"Skill: {skill_name}")
    print(f"Target object names: {object_names}")
    print()

    # Open HDF5 file
    with h5py.File(hdf5_path, 'r') as f:
        # Create environment
        try:
            print(f"Creating environment...")
            env, bddl_file = create_env_from_skill(skill_name, skill_config)
            print(f"  Using BDDL: {bddl_file}")
            print()
        except Exception as e:
            print(f"Error: Could not create environment: {e}")
            return

        # Get total number of demos
        n_demos = len(f['data'])
        print(f"Total demos: {n_demos}")
        print()

        # Analyze each demo
        results = {}
        left_indices = []
        right_indices = []
        center_indices = []
        failed_indices = []

        print("Analyzing demos...")
        for demo_idx in range(n_demos):
            result = analyze_demo(f, demo_idx, env, object_names, verbose=verbose)

            if result is None:
                failed_indices.append(demo_idx)
                continue

            grasp_side, x_offset = result
            results[demo_idx] = (grasp_side, x_offset)

            if grasp_side == "left":
                left_indices.append(demo_idx)
            elif grasp_side == "right":
                right_indices.append(demo_idx)
            else:
                center_indices.append(demo_idx)

            # Print progress every 10 demos
            if not verbose and (demo_idx + 1) % 10 == 0:
                print(f"  Processed {demo_idx + 1}/{n_demos} demos")

        env.close()

        # Print summary
        print()
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        total = len(results)
        n_left = len(left_indices)
        n_right = len(right_indices)
        n_center = len(center_indices)
        n_failed = len(failed_indices)

        print(f"Total demos:     {n_demos}")
        print(f"Analyzed:        {total}")
        print(f"Failed:          {n_failed}")
        print()
        print(f"Left grasps:     {n_left:3d} ({n_left/total*100:5.1f}%)")
        print(f"Right grasps:    {n_right:3d} ({n_right/total*100:5.1f}%)")
        print(f"Center grasps:   {n_center:3d} ({n_center/total*100:5.1f}%)")
        print()
        print(f"Left grasp indices:  {left_indices}")
        print(f"Right grasp indices: {right_indices}")
        if center_indices:
            print(f"Center grasp indices: {center_indices}")
        if failed_indices:
            print(f"Failed analysis indices: {failed_indices}")
        print()

        # Print detailed offsets for debugging
        if results:
            print("Detailed Analysis (using last timestep):")
            show_count = min(20, len(results))
            for demo_idx in sorted(results.keys())[:show_count]:
                side, offset = results[demo_idx]
                print(f"  demo_{demo_idx:3d}: {side:>6} | offset = {offset:+.4f}m")
            if total > show_count:
                print(f"  ... (showing first {show_count} of {total} demos)")

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze grasp positions in pick skill demos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_grasp_position.py --hdf5 datasets/hdf5_datasets/atomic_above_26_skills/all/pick_black_bowl.hdf5
  python analyze_grasp_position.py --hdf5 datasets/hdf5_datasets/atomic_above_26_skills/all/place_black_bowl_on_the_plate.hdf5 --verbose
        """
    )
    parser.add_argument(
        '--hdf5',
        type=str,
        required=True,
        help='Path to HDF5 file to analyze'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress for each demo'
    )

    args = parser.parse_args()

    if not os.path.exists(args.hdf5):
        print(f"Error: HDF5 file not found: {args.hdf5}")
        return

    analyze_hdf5_file(args.hdf5, verbose=args.verbose)


if __name__ == "__main__":
    main()
