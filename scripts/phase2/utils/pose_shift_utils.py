#!/usr/bin/env python3
"""
Pose Shifting Utilities for Phase 2 Implementation.

This module provides core functions for generating shifted poses and finding family poses
to create augmented demonstrations for robust VLA training.

Key Functions:
- generate_shifted_poses(): Creates poses outside current skill's initial state distribution
- find_family_pose(): Selects collision-avoiding family poses using 3D relationships
"""

import numpy as np
import os
import pickle
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation as R
from scipy.stats import chi2

# Default configuration parameters
POSITION_STD_DEFAULT = 0.02  # meters
ORIENTATION_STD_DEFAULT = 0.1  # radians  
NUM_SAMPLES_DEFAULT = 20
DISTRIBUTION_THRESHOLD_PERCENTILE = 95.0
EE_BACKWARD_OFFSET = 0.05  # meters
MAX_RETRY_ATTEMPTS = 100  # Prevent infinite loops


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


def compute_pose_distribution_statistics(positions: np.ndarray, quaternions: np.ndarray, 
                                       outlier_percentile: float = 95.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute distribution statistics for poses, filtering outliers.
    
    Args:
        positions: Array of positions (N, 3)
        quaternions: Array of quaternions (N, 4)
        outlier_percentile: Percentile threshold for outlier removal (e.g., 95.0 removes top 5%)
        
    Returns:
        Tuple of (position_mean_cov, quaternion_mean_cov) where each is (mean, covariance)
    """
    # Filter outliers using Mahalanobis distance
    pos_mean = np.mean(positions, axis=0)
    pos_cov = np.cov(positions.T)
    
    # Compute Mahalanobis distances for position outliers
    pos_distances = []
    for pos in positions:
        try:
            cov_inv = np.linalg.inv(pos_cov)
        except np.linalg.LinAlgError:
            cov_reg = pos_cov + np.eye(pos_cov.shape[0]) * 1e-6
            cov_inv = np.linalg.inv(cov_reg)
        diff = pos - pos_mean
        dist = np.sqrt(diff.T @ cov_inv @ diff)
        pos_distances.append(dist)
    
    pos_distances = np.array(pos_distances)
    threshold = np.percentile(pos_distances, outlier_percentile)
    inlier_mask = pos_distances <= threshold
    
    # Filter positions and quaternions
    filtered_positions = positions[inlier_mask]
    filtered_quaternions = quaternions[inlier_mask]
    
    print(f"📊 Filtered {np.sum(inlier_mask)}/{len(positions)} poses (removed {np.sum(~inlier_mask)} outliers)")
    
    # Position statistics on filtered data
    pos_mean = np.mean(filtered_positions, axis=0)
    pos_cov = np.cov(filtered_positions.T)
    
    # Quaternion statistics (convert to axis-angle for proper statistics)
    rotations = R.from_quat(filtered_quaternions)
    axis_angles = rotations.as_rotvec()
    
    quat_mean = np.mean(axis_angles, axis=0)
    quat_cov = np.cov(axis_angles.T)
    
    return (pos_mean, pos_cov), (quat_mean, quat_cov)


def mahalanobis_distance(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """
    Compute Mahalanobis distance from point to distribution.
    
    Args:
        x: Point to test
        mean: Distribution mean
        cov: Distribution covariance matrix
        
    Returns:
        Mahalanobis distance
    """
    diff = x - mean
    
    # Handle singular covariance matrix
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # Add small regularization
        cov_reg = cov + np.eye(cov.shape[0]) * 1e-6
        cov_inv = np.linalg.inv(cov_reg)
    
    return np.sqrt(diff.T @ cov_inv @ diff)


def is_pose_outside_distribution(position: np.ndarray, 
                                quaternion: np.ndarray,
                                pos_stats: Tuple[np.ndarray, np.ndarray],
                                quat_stats: Tuple[np.ndarray, np.ndarray],
                                threshold_percentile: float = 95.0) -> bool:
    """
    Check if pose is outside the initial state distribution.
    
    Args:
        position: Position to test (3,)
        quaternion: Quaternion to test (4,) 
        pos_stats: (mean, covariance) for positions
        quat_stats: (mean, covariance) for quaternions (in axis-angle space)
        threshold_percentile: Percentile threshold for "outside distribution"
        
    Returns:
        True if pose is outside distribution
    """
    pos_mean, pos_cov = pos_stats
    quat_mean, quat_cov = quat_stats
    
    # Convert quaternion to axis-angle
    rotation = R.from_quat(quaternion)
    axis_angle = rotation.as_rotvec()
    
    # Compute Mahalanobis distances
    pos_distance = mahalanobis_distance(position, pos_mean, pos_cov)
    quat_distance = mahalanobis_distance(axis_angle, quat_mean, quat_cov)
    
    # Chi-squared threshold for given percentile
    # For 3D position and 3D orientation (6 DOF total)
    threshold = np.sqrt(chi2.ppf(threshold_percentile / 100.0, df=6))
    
    # Combined distance (simplified - could use more sophisticated fusion)
    combined_distance = np.sqrt(pos_distance**2 + quat_distance**2)
    
    return combined_distance > threshold


def sample_pose_in_se3(target_position: np.ndarray,
                      target_quaternion: np.ndarray,
                      position_std: float,
                      orientation_std: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a pose near target pose with Gaussian noise in SE(3).
    
    Args:
        target_position: Target position (3,)
        target_quaternion: Target quaternion (4,)
        position_std: Standard deviation for position noise (meters)
        orientation_std: Standard deviation for orientation noise (radians)
        
    Returns:
        Tuple of (sampled_position, sampled_quaternion)
    """
    # Sample position with Gaussian noise
    position_noise = np.random.normal(0, position_std, 3)
    sampled_position = target_position + position_noise
    
    # Sample orientation with Gaussian noise in axis-angle space
    target_rotation = R.from_quat(target_quaternion)
    target_axis_angle = target_rotation.as_rotvec()
    
    orientation_noise = np.random.normal(0, orientation_std, 3)
    sampled_axis_angle = target_axis_angle + orientation_noise
    
    sampled_rotation = R.from_rotvec(sampled_axis_angle)
    sampled_quaternion = sampled_rotation.as_quat()
    
    return sampled_position, sampled_quaternion


def generate_shifted_poses(target_pose: Tuple[np.ndarray, np.ndarray], 
                          current_skill_initial_states: List[np.ndarray],
                          num_samples: int = NUM_SAMPLES_DEFAULT,
                          position_std: float = POSITION_STD_DEFAULT,
                          orientation_std: float = ORIENTATION_STD_DEFAULT,
                          distribution_threshold_percentile: float = DISTRIBUTION_THRESHOLD_PERCENTILE) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate shifted poses outside the current skill's initial state distribution.
    
    Args:
        target_pose: (position, quaternion) - original target pose
        current_skill_initial_states: List of initial states for current skill only
        num_samples: Number of shifted poses to generate
        position_std: Gaussian std for position shifts (meters)
        orientation_std: Gaussian std for orientation shifts (radians)
        distribution_threshold_percentile: Percentile threshold for "outside distribution"
        
    Returns:
        List of (shifted_position, shifted_quaternion) tuples outside the distribution
    """
    if len(current_skill_initial_states) == 0:
        raise ValueError("No initial states provided")
    
    target_position, target_quaternion = target_pose
    
    # Extract EE poses from initial states
    positions, quaternions = extract_ee_poses_from_initial_states(current_skill_initial_states)
    
    # Compute distribution statistics
    pos_stats, quat_stats = compute_pose_distribution_statistics(positions, quaternions)
    
    shifted_poses = []
    attempts = 0
    
    print(f"🎯 Generating {num_samples} shifted poses outside distribution...")
    print(f"📊 Reference distribution: {len(current_skill_initial_states)} initial states")
    print(f"🔧 Position std: {position_std:.3f}m, Orientation std: {orientation_std:.3f}rad")
    
    while len(shifted_poses) < num_samples and attempts < MAX_RETRY_ATTEMPTS:
        # Sample a pose near the target
        sampled_pos, sampled_quat = sample_pose_in_se3(
            target_position, target_quaternion, position_std, orientation_std
        )
        
        # Check if it's outside the distribution
        if is_pose_outside_distribution(
            sampled_pos, sampled_quat, pos_stats, quat_stats, distribution_threshold_percentile
        ):
            shifted_poses.append((sampled_pos, sampled_quat))
            if len(shifted_poses) % 5 == 0:
                print(f"   ✅ Generated {len(shifted_poses)}/{num_samples} poses")
        
        attempts += 1
    
    if len(shifted_poses) < num_samples:
        print(f"⚠️  Warning: Only generated {len(shifted_poses)}/{num_samples} poses after {attempts} attempts")
        print(f"   Consider increasing position_std/orientation_std or decreasing threshold_percentile")
    else:
        print(f"✅ Successfully generated {len(shifted_poses)} shifted poses")
    
    return shifted_poses


def project_pose_onto_facing_direction(position: np.ndarray, quaternion: np.ndarray) -> float:
    """
    Project pose onto its facing direction (-Z axis).
    
    Args:
        position: Position (3,)
        quaternion: Quaternion (4,)
        
    Returns:
        Scalar projection onto facing direction
    """
    rotation = R.from_quat(quaternion)
    facing_direction = rotation.apply([0, 0, -1])  # -Z axis in world frame
    
    # Project position onto facing direction
    projection = np.dot(position, facing_direction)
    return projection


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
                    ee_backward_offset: float = EE_BACKWARD_OFFSET) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find family pose from demonstrations to avoid collision during motion planning.
    
    Args:
        shifted_pose: (position, quaternion) - the shifted target pose
        current_skill_initial_states: List of initial states for current skill
        ee_backward_offset: Distance threshold for forward/backward classification
        
    Returns:
        (family_position, family_quaternion) - selected family pose
    """
    if len(current_skill_initial_states) == 0:
        raise ValueError("No initial states provided")
    
    shifted_position, shifted_quaternion = shifted_pose
    
    # Extract EE poses from initial states
    positions, quaternions = extract_ee_poses_from_initial_states(current_skill_initial_states)
    
    # Project shifted pose and all initial poses onto facing direction
    shifted_projection = project_pose_onto_facing_direction(shifted_position, shifted_quaternion)
    
    initial_projections = []
    for i, (pos, quat) in enumerate(zip(positions, quaternions)):
        projection = project_pose_onto_facing_direction(pos, quat)
        initial_projections.append((projection, i, pos, quat))
    
    # Sort by projection value
    initial_projections.sort(key=lambda x: x[0])
    
    print(f"🎯 Finding family pose for shifted pose...")
    print(f"📍 Shifted pose projection: {shifted_projection:.3f}")
    print(f"📊 Initial pose projections range: {initial_projections[0][0]:.3f} to {initial_projections[-1][0]:.3f}")
    
    # Determine if shifted pose is forward of 90% of initial poses
    percentile_90_idx = int(0.9 * len(initial_projections))
    percentile_90_projection = initial_projections[percentile_90_idx][0]
    
    if shifted_projection > percentile_90_projection:
        # Shifted pose is forward of 90% - select closest initial pose behind it
        print(f"🔄 Shifted pose is forward of 90% (> {percentile_90_projection:.3f})")
        print(f"   Selecting closest initial pose behind shifted pose...")
        
        # Find poses behind shifted pose
        behind_poses = [(proj, idx, pos, quat) for proj, idx, pos, quat in initial_projections 
                       if proj <= shifted_projection]
        
        if not behind_poses:
            # Fallback: use the most backward pose
            print(f"   ⚠️  No poses behind shifted pose, using most backward pose")
            selected = initial_projections[0]
        else:
            # Select closest behind pose
            distances = [compute_pose_distance(shifted_position, shifted_quaternion, pos, quat) 
                        for _, _, pos, quat in behind_poses]
            closest_idx = np.argmin(distances)
            selected = behind_poses[closest_idx]
    else:
        # Shifted pose is not forward of 90% - select closest pose in front
        print(f"🔄 Shifted pose is not forward of 90% (<= {percentile_90_projection:.3f})")
        print(f"   Selecting closest initial pose in front of shifted pose...")
        
        # Find poses in front of shifted pose
        front_poses = [(proj, idx, pos, quat) for proj, idx, pos, quat in initial_projections 
                      if proj >= shifted_projection]
        
        if not front_poses:
            # Fallback: use the most forward pose
            print(f"   ⚠️  No poses in front of shifted pose, using most forward pose")
            selected = initial_projections[-1]
        else:
            # Select closest front pose
            distances = [compute_pose_distance(shifted_position, shifted_quaternion, pos, quat) 
                        for _, _, pos, quat in front_poses]
            closest_idx = np.argmin(distances)
            selected = front_poses[closest_idx]
    
    projection, state_idx, family_position, family_quaternion = selected
    
    print(f"✅ Selected family pose from initial state {state_idx}")
    print(f"   📍 Family pose projection: {projection:.3f}")
    print(f"   📏 Distance to shifted pose: {compute_pose_distance(shifted_position, shifted_quaternion, family_position, family_quaternion):.3f}")
    
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


# Testing and debugging utilities
def visualize_pose_distribution(initial_states: List[np.ndarray], 
                               shifted_poses: List[Tuple[np.ndarray, np.ndarray]],
                               save_path: Optional[str] = None):
    """
    Visualize pose distribution for debugging (optional matplotlib dependency).
    
    Args:
        initial_states: List of initial states
        shifted_poses: List of shifted poses
        save_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("⚠️  Matplotlib not available for visualization")
        return
    
    # Extract positions and orientations
    positions, quaternions = extract_ee_poses_from_initial_states(initial_states)
    shifted_positions = np.array([pos for pos, _ in shifted_poses])
    shifted_quaternions = np.array([quat for _, quat in shifted_poses])
    
    fig = plt.figure(figsize=(16, 4))
    
    # 3D scatter plot with orientation arrows
    ax1 = fig.add_subplot(141, projection='3d')
    
    # Plot initial states
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               c='blue', alpha=0.6, label='Initial States', s=20)
    
    # Add orientation arrows for initial states (sample every 3rd to avoid clutter)
    for i in range(0, len(positions), 3):
        pos = positions[i]
        quat = quaternions[i]
        rotation = R.from_quat(quat)
        # Draw arrow showing -Z direction (gripper pointing direction)
        direction = rotation.apply([0, 0, -0.05])  # 5cm arrow
        ax1.quiver(pos[0], pos[1], pos[2], direction[0], direction[1], direction[2], 
                  color='blue', alpha=0.7, arrow_length_ratio=0.3)
    
    # Plot shifted poses
    ax1.scatter(shifted_positions[:, 0], shifted_positions[:, 1], shifted_positions[:, 2], 
               c='red', alpha=0.8, label='Shifted Poses', s=30)
    
    # Add orientation arrows for shifted poses
    for i in range(len(shifted_positions)):
        pos = shifted_positions[i]
        quat = shifted_quaternions[i]
        rotation = R.from_quat(quat)
        # Draw arrow showing -Z direction (gripper pointing direction)
        direction = rotation.apply([0, 0, -0.05])  # 5cm arrow
        ax1.quiver(pos[0], pos[1], pos[2], direction[0], direction[1], direction[2], 
                  color='red', alpha=0.9, arrow_length_ratio=0.3)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()
    ax1.set_title('3D Pose Distribution')
    
    # X-Y projection with arrows
    ax2 = fig.add_subplot(142)
    
    # Plot initial states as arrows
    for i in range(0, len(positions), 2):  # Sample every 2nd to avoid clutter
        pos = positions[i]
        quat = quaternions[i]
        rotation = R.from_quat(quat)
        # Project -Z direction onto X-Y plane
        direction_3d = rotation.apply([0, 0, -0.03])  # 3cm arrow
        ax2.arrow(pos[0], pos[1], direction_3d[0], direction_3d[1], 
                 head_width=0.01, head_length=0.01, fc='blue', ec='blue', alpha=0.7)
    
    # Plot shifted poses as arrows (sample every 4th to reduce density with more samples)
    for i in range(0, len(shifted_positions), 4):
        pos = shifted_positions[i]
        quat = shifted_quaternions[i]
        rotation = R.from_quat(quat)
        # Project -Z direction onto X-Y plane
        direction_3d = rotation.apply([0, 0, -0.03])  # 3cm arrow
        ax2.arrow(pos[0], pos[1], direction_3d[0], direction_3d[1], 
                 head_width=0.01, head_length=0.01, fc='red', ec='red', alpha=0.9)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('X-Y Projection')
    
    # X-Z projection with arrows
    ax3 = fig.add_subplot(143)
    
    # Plot initial states as arrows
    for i in range(0, len(positions), 2):  # Sample every 2nd to avoid clutter
        pos = positions[i]
        quat = quaternions[i]
        rotation = R.from_quat(quat)
        # Project -Z direction onto X-Z plane
        direction_3d = rotation.apply([0, 0, -0.03])  # 3cm arrow
        ax3.arrow(pos[0], pos[2], direction_3d[0], direction_3d[2], 
                 head_width=0.01, head_length=0.01, fc='blue', ec='blue', alpha=0.7)
    
    # Plot shifted poses as arrows (sample every 4th to reduce density with more samples)
    for i in range(0, len(shifted_positions), 4):
        pos = shifted_positions[i]
        quat = shifted_quaternions[i]
        rotation = R.from_quat(quat)
        # Project -Z direction onto X-Z plane
        direction_3d = rotation.apply([0, 0, -0.03])  # 3cm arrow
        ax3.arrow(pos[0], pos[2], direction_3d[0], direction_3d[2], 
                 head_width=0.01, head_length=0.01, fc='red', ec='red', alpha=0.9)
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('X-Z Projection')
    
    # Y-Z projection with arrows
    ax4 = fig.add_subplot(144)
    
    # Plot initial states as arrows
    for i in range(0, len(positions), 2):  # Sample every 2nd to avoid clutter
        pos = positions[i]
        quat = quaternions[i]
        rotation = R.from_quat(quat)
        # Project -Z direction onto Y-Z plane
        direction_3d = rotation.apply([0, 0, -0.03])  # 3cm arrow
        ax4.arrow(pos[1], pos[2], direction_3d[1], direction_3d[2], 
                 head_width=0.01, head_length=0.01, fc='blue', ec='blue', alpha=0.7)
    
    # Plot shifted poses as arrows (sample every 4th to reduce density with more samples)
    for i in range(0, len(shifted_positions), 4):
        pos = shifted_positions[i]
        quat = shifted_quaternions[i]
        rotation = R.from_quat(quat)
        # Project -Z direction onto Y-Z plane
        direction_3d = rotation.apply([0, 0, -0.03])  # 3cm arrow
        ax4.arrow(pos[1], pos[2], direction_3d[1], direction_3d[2], 
                 head_width=0.01, head_length=0.01, fc='red', ec='red', alpha=0.9)
    
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('Y-Z Projection')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved pose distribution plot: {save_path}")
    else:
        plt.show()


def visualize_pose_axis_histograms(initial_states: List[np.ndarray],
                                  shifted_poses: List[Tuple[np.ndarray, np.ndarray]],
                                  save_path: Optional[str] = None):
    """
    Visualize per-axis position distributions for initial vs shifted poses.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  Matplotlib not available for visualization")
        return
    
    positions, _ = extract_ee_poses_from_initial_states(initial_states)
    shifted_positions = np.array([pos for pos, _ in shifted_poses])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axis_labels = ["X (m)", "Y (m)", "Z (m)"]
    colors = {"initial": "blue", "shifted": "red"}
    
    for axis_index, ax in enumerate(axes):
        ax.hist(positions[:, axis_index], bins=30, color=colors["initial"], alpha=0.5, density=True, label="Initial")
        ax.hist(shifted_positions[:, axis_index], bins=30, color=colors["shifted"], alpha=0.5, density=True, label="Shifted")
        ax.set_xlabel(axis_labels[axis_index])
        ax.set_ylabel("Density")
        ax.legend()
        ax.set_title(f"{axis_labels[axis_index]} Distribution")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved pose axis histograms: {save_path}")
    else:
        plt.show()


def visualize_single_shifted_pose_with_family(initial_states: List[np.ndarray],
                                            shifted_pose: Tuple[np.ndarray, np.ndarray],
                                            family_pose: Tuple[np.ndarray, np.ndarray],
                                            save_path: Optional[str] = None):
    """
    Visualize a single shifted pose (red) with its family pose (green) against initial states (blue).
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("⚠️  Matplotlib not available for visualization")
        return
    
    # Extract positions and orientations
    positions, quaternions = extract_ee_poses_from_initial_states(initial_states)
    shifted_pos, shifted_quat = shifted_pose
    family_pos, family_quat = family_pose
    
    fig = plt.figure(figsize=(16, 4))
    
    # 3D scatter plot with orientation arrows
    ax1 = fig.add_subplot(141, projection='3d')
    
    # Plot initial states
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               c='blue', alpha=0.6, label='Initial States', s=20)
    
    # Add orientation arrows for initial states (sample every 3rd to avoid clutter)
    for i in range(0, len(positions), 3):
        pos = positions[i]
        quat = quaternions[i]
        rotation = R.from_quat(quat)
        # Draw arrow showing -Z direction (gripper pointing direction)
        direction = rotation.apply([0, 0, -0.05])  # 5cm arrow
        ax1.quiver(pos[0], pos[1], pos[2], direction[0], direction[1], direction[2], 
                  color='blue', alpha=0.7, arrow_length_ratio=0.3)
    
    # Plot single shifted pose
    ax1.scatter(shifted_pos[0], shifted_pos[1], shifted_pos[2], 
               c='red', alpha=0.9, label='Shifted Pose', s=50)
    
    # Add orientation arrow for shifted pose
    rotation = R.from_quat(shifted_quat)
    direction = rotation.apply([0, 0, -0.05])  # 5cm arrow
    ax1.quiver(shifted_pos[0], shifted_pos[1], shifted_pos[2], direction[0], direction[1], direction[2], 
              color='red', alpha=0.9, arrow_length_ratio=0.3)
    
    # Plot family pose
    ax1.scatter(family_pos[0], family_pos[1], family_pos[2], 
               c='green', alpha=0.9, label='Family Pose', s=50)
    
    # Add orientation arrow for family pose
    rotation = R.from_quat(family_quat)
    direction = rotation.apply([0, 0, -0.05])  # 5cm arrow
    ax1.quiver(family_pos[0], family_pos[1], family_pos[2], direction[0], direction[1], direction[2], 
              color='green', alpha=0.9, arrow_length_ratio=0.3)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()
    ax1.set_title('3D Pose Distribution')
    
    # X-Y projection with arrows
    ax2 = fig.add_subplot(142)
    
    # Plot initial states as arrows
    for i in range(0, len(positions), 2):  # Sample every 2nd to avoid clutter
        pos = positions[i]
        quat = quaternions[i]
        rotation = R.from_quat(quat)
        # Project -Z direction onto X-Y plane
        direction_3d = rotation.apply([0, 0, -0.03])  # 3cm arrow
        ax2.arrow(pos[0], pos[1], direction_3d[0], direction_3d[1], 
                 head_width=0.01, head_length=0.01, fc='blue', ec='blue', alpha=0.7)
    
    # Plot shifted pose as arrow
    rotation = R.from_quat(shifted_quat)
    direction_3d = rotation.apply([0, 0, -0.03])  # 3cm arrow
    ax2.arrow(shifted_pos[0], shifted_pos[1], direction_3d[0], direction_3d[1], 
             head_width=0.01, head_length=0.01, fc='red', ec='red', alpha=0.9)
    
    # Plot family pose as arrow
    rotation = R.from_quat(family_quat)
    direction_3d = rotation.apply([0, 0, -0.03])  # 3cm arrow
    ax2.arrow(family_pos[0], family_pos[1], direction_3d[0], direction_3d[1], 
             head_width=0.01, head_length=0.01, fc='green', ec='green', alpha=0.9)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('X-Y Projection')
    
    # X-Z projection with arrows
    ax3 = fig.add_subplot(143)
    
    # Plot initial states as arrows
    for i in range(0, len(positions), 2):  # Sample every 2nd to avoid clutter
        pos = positions[i]
        quat = quaternions[i]
        rotation = R.from_quat(quat)
        # Project -Z direction onto X-Z plane
        direction_3d = rotation.apply([0, 0, -0.03])  # 3cm arrow
        ax3.arrow(pos[0], pos[2], direction_3d[0], direction_3d[2], 
                 head_width=0.01, head_length=0.01, fc='blue', ec='blue', alpha=0.7)
    
    # Plot shifted pose as arrow
    rotation = R.from_quat(shifted_quat)
    direction_3d = rotation.apply([0, 0, -0.03])  # 3cm arrow
    ax3.arrow(shifted_pos[0], shifted_pos[2], direction_3d[0], direction_3d[2], 
             head_width=0.01, head_length=0.01, fc='red', ec='red', alpha=0.9)
    
    # Plot family pose as arrow
    rotation = R.from_quat(family_quat)
    direction_3d = rotation.apply([0, 0, -0.03])  # 3cm arrow
    ax3.arrow(family_pos[0], family_pos[2], direction_3d[0], direction_3d[2], 
             head_width=0.01, head_length=0.01, fc='green', ec='green', alpha=0.9)
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('X-Z Projection')
    
    # Y-Z projection with arrows
    ax4 = fig.add_subplot(144)
    
    # Plot initial states as arrows
    for i in range(0, len(positions), 2):  # Sample every 2nd to avoid clutter
        pos = positions[i]
        quat = quaternions[i]
        rotation = R.from_quat(quat)
        # Project -Z direction onto Y-Z plane
        direction_3d = rotation.apply([0, 0, -0.03])  # 3cm arrow
        ax4.arrow(pos[1], pos[2], direction_3d[1], direction_3d[2], 
                 head_width=0.01, head_length=0.01, fc='blue', ec='blue', alpha=0.7)
    
    # Plot shifted pose as arrow
    rotation = R.from_quat(shifted_quat)
    direction_3d = rotation.apply([0, 0, -0.03])  # 3cm arrow
    ax4.arrow(shifted_pos[1], shifted_pos[2], direction_3d[1], direction_3d[2], 
             head_width=0.01, head_length=0.01, fc='red', ec='red', alpha=0.9)
    
    # Plot family pose as arrow
    rotation = R.from_quat(family_quat)
    direction_3d = rotation.apply([0, 0, -0.03])  # 3cm arrow
    ax4.arrow(family_pos[1], family_pos[2], direction_3d[1], direction_3d[2], 
             head_width=0.01, head_length=0.01, fc='green', ec='green', alpha=0.9)
    
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('Y-Z Projection')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved single shifted pose visualization: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing pose shifting utilities...")
    
    # Test with a sample skill
    skill_name = "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet"
    
    try:
        # Load initial states
        initial_states = load_initial_states_for_skill(skill_name)
        
        # Create multiple targets based on demo distribution (2 per axis)
        positions, quaternions = extract_ee_poses_from_initial_states(initial_states)
        
        # Use demo mean and std as base
        demo_pos_mean = np.mean(positions, axis=0)
        demo_pos_std = np.std(positions, axis=0)
        demo_quat_mean = np.mean(quaternions, axis=0)
        
        # Create multiple targets per axis for flatter distribution
        targets = []
        # Even more granular spacing to eliminate peaks in the middle
        offset_multipliers = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 1.9, 2.2, 2.6, 3.0]  # 11 distances for very flat distribution
        
        # X-axis targets (left/right with multiple distances)
        for mult in offset_multipliers:
            targets.append((demo_pos_mean + np.array([-mult * demo_pos_std[0], 0, 0]), demo_quat_mean))
            targets.append((demo_pos_mean + np.array([+mult * demo_pos_std[0], 0, 0]), demo_quat_mean))
        
        # Y-axis targets (front/back with multiple distances)
        for mult in offset_multipliers:
            targets.append((demo_pos_mean + np.array([0, -mult * demo_pos_std[1], 0]), demo_quat_mean))
            targets.append((demo_pos_mean + np.array([0, +mult * demo_pos_std[1], 0]), demo_quat_mean))
        
        # Z-axis targets (up/down with multiple distances)
        for mult in offset_multipliers:
            targets.append((demo_pos_mean + np.array([0, 0, -mult * demo_pos_std[2]]), demo_quat_mean))
            targets.append((demo_pos_mean + np.array([0, 0, +mult * demo_pos_std[2]]), demo_quat_mean))
        
        print(f"🎯 Created {len(targets)} targets around demo distribution")
        print(f"📊 Demo mean: pos={demo_pos_mean}, std: pos={demo_pos_std}")
        
        # Generate shifted poses from all targets
        all_shifted_poses = []
        samples_per_target = 4  # 4 samples per target = 264 total samples (66 targets * 4)
        
        for i, (target_pos, target_quat) in enumerate(targets):
            target_pose = (target_pos, target_quat)
            if i < 4:  # Only print first few targets to avoid spam
                print(f"   Target {i+1}: pos={target_pos}")
            
            shifted_poses = generate_shifted_poses(
                target_pose, 
                initial_states,
                num_samples=samples_per_target,
                position_std=0.04,  # Slightly larger std for better overlap between layers
                orientation_std=0.262  # 15 degrees in radians (15 * pi/180)
            )
            all_shifted_poses.extend(shifted_poses)
        
        shifted_poses = all_shifted_poses
        
        print(f"\n✅ Generated {len(shifted_poses)} shifted poses")
        
        # Test family pose finding
        if shifted_poses:
            family_pose = find_family_pose(shifted_poses[0], initial_states)
            print(f"✅ Found family pose: {family_pose[0]}")
        
        # Optional visualization
        try:
            # Ensure debug directory exists at repo root: images_videos/phase2/debug
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            debug_dir = os.path.join(repo_root, "images_videos", "phase2", "debug")
            os.makedirs(debug_dir, exist_ok=True)

            # File name components
            safe_skill = skill_name.replace("/", "_")
            pose_plot_path = os.path.join(debug_dir, f"{safe_skill}_pose_distribution.png")
            hist_plot_path = os.path.join(debug_dir, f"{safe_skill}_pose_axis_histograms.png")

            # Save distribution projections (includes 3D, XY, XZ, YZ) - show all samples
            visualize_pose_distribution(initial_states, shifted_poses, save_path=pose_plot_path)

            # Save per-axis histograms - show all samples
            visualize_pose_axis_histograms(initial_states, shifted_poses, save_path=hist_plot_path)
            
            # Generate 10 individual images showing single shifted poses with family poses
            print(f"🎨 Generating 10 individual shifted pose visualizations...")
            for i in range(min(10, len(shifted_poses))):
                shifted_pose = shifted_poses[i]
                family_pose = find_family_pose(shifted_pose, initial_states)
                
                # Create individual image filename
                individual_plot_path = os.path.join(debug_dir, f"{safe_skill}_shifted_pose_{i+1:02d}.png")
                
                # Generate visualization
                visualize_single_shifted_pose_with_family(
                    initial_states, 
                    shifted_pose, 
                    family_pose, 
                    save_path=individual_plot_path
                )
                
                print(f"   ✅ Generated individual plot {i+1}/10: {individual_plot_path}")
        except Exception as viz_err:
            print(f"📊 Skipping visualization (matplotlib not available or display issues): {viz_err}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure the atomic demos path and skill files exist")