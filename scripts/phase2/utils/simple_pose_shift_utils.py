#!/usr/bin/env python3
"""
Simple Pose Shifting Utilities for Phase 2 Implementation.

Simplified version of pose_shift_utils.py with basic random offset approach
based on motion planner inaccuracy analysis results:
- Position error: ~2cm → use ±3cm shifts
- Orientation error: ~28.6° → use ±45° shifts

Key Functions:
- generate_shifted_pose(): Creates a single shifted pose with simple random offsets
- find_family_pose(): Finds closest family pose from initial states
"""

import numpy as np
import os
import pickle
from typing import List, Tuple, Optional, Dict
from scipy.spatial.transform import Rotation as R

# Default configuration parameters based on motion planner analysis
DEFAULT_POSITION_SHIFT_RANGE = 0.03  # ±3cm for each axis
DEFAULT_ORIENTATION_SHIFT_RANGE = np.radians(45)  # ±45° for each axis


def collect_step_data(action: np.ndarray, obs: Dict, reward: float, done: bool, env) -> Dict:
    """
    Collect data from a single environment step according to hdf5_structure.md specifications.
    """
    step_data = {
        'actions': action,                                     # Action taken this step
        'dones': done,                                         # Episode termination flag
        'rewards': reward,                                     # Reward from step
        'states': env.sim.get_state().flatten(),              # Full simulation state (84-dim)
        'robot_states': obs['robot0_proprio-state'][:9],      # First 9 elements of proprio state
        'obs': {
            'joint_states': obs['robot0_joint_pos'],          # 7-DOF joint positions
            'gripper_states': obs['robot0_gripper_qpos'],     # 2-element gripper positions
            'ee_pos': obs['robot0_eef_pos'],                  # 3D end-effector position
            'ee_ori': R.from_quat(obs['robot0_eef_quat']).as_euler('xyz'),  # Convert quat to euler
            'ee_states': np.concatenate([                      # Combined pose (6-dim)
                obs['robot0_eef_pos'], 
                R.from_quat(obs['robot0_eef_quat']).as_euler('xyz')
            ]),
            'agentview_rgb': obs['agentview_image'],          # Agent viewpoint RGB image
            'eye_in_hand_rgb': obs['robot0_eye_in_hand_image'] # Wrist camera RGB image
        }
    }
    return step_data


def extract_ee_pose_from_initial_state(initial_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract end-effector pose from initial state array.

    Args:
        initial_state: Initial state array [joint(7) + gripper(2) + sim_state(...)]

    Returns:
        Tuple of (position, quaternion) extracted from sim_state
    """
    # Skip joint states (7) and gripper states (2), extract from sim_state
    sim_state = initial_state[9:]

    # For LIBERO, EE pose is typically at the beginning of sim_state
    # Position: first 3 elements, Quaternion: next 4 elements
    if len(sim_state) < 7:
        raise ValueError(f"Sim state too short: {len(sim_state)} < 7")

    position = sim_state[:3].copy()
    quaternion = sim_state[3:7].copy()

    return position, quaternion


def extract_ee_poses_from_initial_states(initial_states: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract all EE poses from list of initial states.

    Args:
        initial_states: List of initial state arrays

    Returns:
        Tuple of (positions, quaternions) arrays of shape (N, 3) and (N, 4)
    """
    positions = []
    quaternions = []

    for state in initial_states:
        pos, quat = extract_ee_pose_from_initial_state(state)
        positions.append(pos)
        quaternions.append(quat)

    return np.array(positions), np.array(quaternions)


def generate_shifted_pose(target_pose: Tuple[np.ndarray, np.ndarray],
                         position_shift_range: float = DEFAULT_POSITION_SHIFT_RANGE,
                         orientation_shift_range: float = DEFAULT_ORIENTATION_SHIFT_RANGE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single shifted pose with simple random offsets.

    Simplified approach based on motion planner inaccuracy analysis:
    - Position: Add random offset ±position_shift_range to each axis
    - Orientation: Add random offset ±orientation_shift_range to each axis in axis-angle space

    Args:
        target_pose: (position, quaternion) - original target pose
        position_shift_range: Random offset range for position (default: ±3cm)
        orientation_shift_range: Random offset range for orientation (default: ±45°)

    Returns:
        Tuple of (shifted_position, shifted_quaternion)
    """
    target_position, target_quaternion = target_pose

    # Simple position shift: add random offset to each axis
    position_offsets = np.random.uniform(-position_shift_range, position_shift_range, 3)
    shifted_position = target_position + position_offsets

    # Simple orientation shift: modify all 3 axes while ensuring total difference <= orientation_shift_range
    target_rotation = R.from_quat(target_quaternion)
    target_axis_angle = target_rotation.as_rotvec()

    # Generate random offsets for all 3 axes
    orientation_offsets = np.random.uniform(-orientation_shift_range, orientation_shift_range, 3)

    # Check if the magnitude exceeds the limit and scale down if needed
    offset_magnitude = np.linalg.norm(orientation_offsets)
    if offset_magnitude > orientation_shift_range:
        orientation_offsets = orientation_offsets * (orientation_shift_range / offset_magnitude)

    shifted_axis_angle = target_axis_angle + orientation_offsets
    shifted_rotation = R.from_rotvec(shifted_axis_angle)
    shifted_quaternion = shifted_rotation.as_quat()

    return shifted_position, shifted_quaternion


def compute_pose_distance(pos1: np.ndarray, quat1: np.ndarray,
                         pos2: np.ndarray, quat2: np.ndarray,
                         position_weight: float = 1.0,
                         orientation_weight: float = 0.1) -> float:
    """
    Compute combined 3D Euclidean + orientation distance between poses.

    Args:
        pos1, quat1: First pose
        pos2, quat2: Second pose
        position_weight: Weight for position distance
        orientation_weight: Weight for orientation distance

    Returns:
        Combined distance metric
    """
    # Euclidean distance for position
    pos_distance = np.linalg.norm(pos1 - pos2)

    # Quaternion distance (angle between rotations)
    rot1 = R.from_quat(quat1)
    rot2 = R.from_quat(quat2)
    relative_rot = rot1.inv() * rot2
    angle_distance = np.abs(relative_rot.as_rotvec()).sum()  # Sum of axis-angle components

    return position_weight * pos_distance + orientation_weight * angle_distance


def find_family_pose(shifted_pose: Tuple[np.ndarray, np.ndarray],
                    current_skill_initial_states: List[np.ndarray],
                    original_pose: Tuple[np.ndarray, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find family pose - super simple: just return the original pose.

    Args:
        shifted_pose: (position, quaternion) - the shifted target pose (not used)
        current_skill_initial_states: List of initial states for current skill (not used)
        original_pose: (position, quaternion) - the original pose to return

    Returns:
        (family_position, family_quaternion) - original pose
    """
    if original_pose is not None:
        original_position, original_quaternion = original_pose
        print(f"✅ Using original pose as family pose")
        print(f"   📍 Family pose: pos=[{original_position[0]:.3f}, {original_position[1]:.3f}, {original_position[2]:.3f}]")
        return original_position, original_quaternion

    # Fallback to old behavior if original_pose not provided
    if len(current_skill_initial_states) == 0:
        raise ValueError("No initial states provided")

    shifted_position, shifted_quaternion = shifted_pose

    # Extract EE poses from initial states
    positions, quaternions = extract_ee_poses_from_initial_states(current_skill_initial_states)

    # Find closest initial state pose
    min_distance = float('inf')
    closest_idx = 0

    for i, (pos, quat) in enumerate(zip(positions, quaternions)):
        distance = compute_pose_distance(shifted_position, shifted_quaternion, pos, quat)
        if distance < min_distance:
            min_distance = distance
            closest_idx = i

    family_position = positions[closest_idx]
    family_quaternion = quaternions[closest_idx]

    print(f"✅ Selected closest family pose from initial state {closest_idx}")
    print(f"   📏 Distance to shifted pose: {min_distance:.3f}")

    return family_position, family_quaternion


def load_initial_states_for_skill(skill_name: str,
                                 atomic_demos_path: str = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_local_demos/full_atomic_skills") -> List[np.ndarray]:
    """
    Load initial states for a specific skill.

    Args:
        skill_name: Name of the skill (e.g., "KITCHEN_SCENE1_open_the_top_drawer")
        atomic_demos_path: Path to atomic demos directory

    Returns:
        List of initial state arrays for the skill
    """
    init_file_path = f"{atomic_demos_path}/{skill_name}.init"

    try:
        with open(init_file_path, 'rb') as f:
            initial_states = pickle.load(f)

        print(f"📁 Loaded {len(initial_states)} initial states for skill: {skill_name}")
        return initial_states

    except FileNotFoundError:
        raise FileNotFoundError(f"Initial states file not found: {init_file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading initial states: {e}")


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing simple pose shifting utilities...")

    # Test with a sample skill
    skill_name = "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet"

    try:
        # Load initial states
        initial_states = load_initial_states_for_skill(skill_name)
        positions, quaternions = extract_ee_poses_from_initial_states(initial_states)

        # Test with 5 different initial states
        num_tests = min(5, len(initial_states))
        print(f"🔄 Testing with {num_tests} different initial states:")

        for i in range(num_tests):
            target_pose = (positions[i], quaternions[i])

            # Convert target orientation to axis-angle degrees
            target_rot = R.from_quat(target_pose[1])
            target_axis = target_rot.as_rotvec()
            target_axis_deg = np.degrees(target_axis)

            print(f"\n📋 Test {i+1}/5 - Initial State {i}")
            print(f"   Target: pos=[{target_pose[0][0]:.3f}, {target_pose[0][1]:.3f}, {target_pose[0][2]:.3f}], axis=[{target_axis_deg[0]:.1f}°, {target_axis_deg[1]:.1f}°, {target_axis_deg[2]:.1f}°]")

            # Generate shifted pose with default parameters
            shifted_pose = generate_shifted_pose(target_pose)

            # Convert shifted orientation to axis-angle degrees
            shifted_rot = R.from_quat(shifted_pose[1])
            shifted_axis = shifted_rot.as_rotvec()
            shifted_axis_deg = np.degrees(shifted_axis)

            # Calculate differences
            pos_diff = np.linalg.norm(shifted_pose[0] - target_pose[0])
            relative_rot = shifted_rot * target_rot.inv()
            ori_diff_degrees = np.degrees(relative_rot.magnitude())

            print(f"   Shifted: pos=[{shifted_pose[0][0]:.3f}, {shifted_pose[0][1]:.3f}, {shifted_pose[0][2]:.3f}], axis=[{shifted_axis_deg[0]:.1f}°, {shifted_axis_deg[1]:.1f}°, {shifted_axis_deg[2]:.1f}°] | Δ={pos_diff*100:.1f}cm, {ori_diff_degrees:.1f}°")

            # Find family pose
            family_pose = find_family_pose(shifted_pose, initial_states)
            family_rot = R.from_quat(family_pose[1])
            family_axis = family_rot.as_rotvec()
            family_axis_deg = np.degrees(family_axis)
            family_distance = compute_pose_distance(shifted_pose[0], shifted_pose[1], family_pose[0], family_pose[1])

            print(f"   Family:  pos=[{family_pose[0][0]:.3f}, {family_pose[0][1]:.3f}, {family_pose[0][2]:.3f}], axis=[{family_axis_deg[0]:.1f}°, {family_axis_deg[1]:.1f}°, {family_axis_deg[2]:.1f}°] | dist={family_distance:.3f}")

        # Test with custom parameters using first initial state
        print(f"\n🔧 Testing custom parameters (±5cm, ±45°):")
        target_pose = (positions[0], quaternions[0])

        for i in range(3):
            custom_shifted_pose = generate_shifted_pose(
                target_pose,
                position_shift_range=0.05,  # ±5cm
                orientation_shift_range=np.radians(45)  # ±45°
            )

            # Convert custom shifted orientation to axis-angle degrees
            custom_rot = R.from_quat(custom_shifted_pose[1])
            custom_axis = custom_rot.as_rotvec()
            custom_axis_deg = np.degrees(custom_axis)

            pos_diff = np.linalg.norm(custom_shifted_pose[0] - target_pose[0])
            target_rot = R.from_quat(target_pose[1])
            relative_rot = custom_rot * target_rot.inv()
            ori_diff_degrees = np.degrees(relative_rot.magnitude())

            print(f"   Custom {i+1}: pos=[{custom_shifted_pose[0][0]:.3f}, {custom_shifted_pose[0][1]:.3f}, {custom_shifted_pose[0][2]:.3f}], axis=[{custom_axis_deg[0]:.1f}°, {custom_axis_deg[1]:.1f}°, {custom_axis_deg[2]:.1f}°] | Δ={pos_diff*100:.1f}cm, {ori_diff_degrees:.1f}°")

        print(f"\n✅ Simple pose shifting utilities test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure the atomic demos path and skill files exist")