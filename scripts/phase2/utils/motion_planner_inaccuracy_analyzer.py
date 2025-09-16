#!/usr/bin/env python3
"""
Motion Planner Inaccuracy Analyzer for Phase 2 Implementation.

This script analyzes the actual distribution of motion planner inaccuracies to guide
proper pose shifting for VLA demonstration augmentation. Instead of using arbitrary
pose shifts, this creates a data-driven approach based on real motion planner behavior.

The script:
1. Randomly initializes LIBERO environments
2. Samples poses within scene bounds and executes motion planner movements
3. Records ground truth vs actual pose pairs
4. Analyzes statistical distributions of pose deviations
5. Saves parameters for data-driven pose shifting

Key Functions:
- analyze_motion_planner_inaccuracy(): Main analysis function
- sample_valid_scene_poses(): Generate poses within table/scene bounds
- execute_motion_sequence(): Execute cartesian_linear movements and record deviations
- analyze_pose_deviations(): Calculate statistical parameters of pose errors
"""

import argparse
import json
import os
import sys
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Add project paths
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')

# LIBERO imports
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action

# Import motion planner from Phase 2
from scripts.phase2.utils.motion_planner import MotionPlanner


class MotionPlannerInaccuracyAnalyzer:
    """Analyze motion planner pose deviations to guide VLA demonstration augmentation."""

    def __init__(self, debug_mode: bool = False, save_plots: bool = True):
        """
        Initialize the analyzer.

        Args:
            debug_mode: Enable verbose debugging output
            save_plots: Whether to save visualization plots
        """
        self.debug_mode = debug_mode
        self.save_plots = save_plots
        self.pose_deviations = []  # Store (target_pose, actual_pose) pairs

        # Scene bounds for pose sampling (KITCHEN scene typical bounds)
        self.scene_bounds = {
            'x': [0.0, 0.8],    # Table width
            'y': [-0.4, 0.4],   # Table depth
            'z': [0.95, 1.3],   # Above table surface to reasonable height
        }

        # Orientation constraints (reasonable working orientations)
        self.orientation_constraints = {
            'pitch_range': [-np.pi/3, np.pi/6],  # -60° to 30° pitch
            'roll_range': [-np.pi/6, np.pi/6],   # ±30° roll
            'yaw_range': [-np.pi, np.pi]         # Full yaw range
        }

    def sample_valid_scene_poses(self, num_poses: int = 7, center_pos: np.ndarray = None,
                                max_distance_from_center: float = 0.10,
                                max_distance_between_poses: float = 0.10) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Sample valid poses within scene bounds, constraining distances to be reasonable for motion planner.

        Args:
            num_poses: Number of poses to sample
            center_pos: Center position to sample around. If None, uses scene center
            max_distance_from_center: Maximum L2 distance from center position (default: 10cm)
            max_distance_between_poses: Maximum L2 distance between consecutive poses (default: 10cm)

        Returns:
            List of (position, quaternion) tuples
        """
        poses = []

        # Use scene center if no center position provided
        if center_pos is None:
            center_pos = np.array([
                (self.scene_bounds['x'][0] + self.scene_bounds['x'][1]) / 2,
                (self.scene_bounds['y'][0] + self.scene_bounds['y'][1]) / 2,
                (self.scene_bounds['z'][0] + self.scene_bounds['z'][1]) / 2
            ])

        last_position = center_pos.copy()

        for i in range(num_poses):
            max_attempts = 100  # Prevent infinite loops

            for attempt in range(max_attempts):
                # Sample position within distance constraint from last position
                # Use spherical sampling to ensure uniform distribution
                direction = np.random.randn(3)
                direction = direction / np.linalg.norm(direction)  # Normalize to unit vector
                distance = np.random.uniform(0.02, max_distance_between_poses)  # Minimum 2cm, maximum max_distance_between_poses
                new_position = last_position + direction * distance

                # Check constraints:
                # 1. Position is within scene bounds
                # 2. Position is within max_distance_from_center from center_pos
                distance_from_center = np.linalg.norm(new_position - center_pos)

                if (self.scene_bounds['x'][0] <= new_position[0] <= self.scene_bounds['x'][1] and
                    self.scene_bounds['y'][0] <= new_position[1] <= self.scene_bounds['y'][1] and
                    self.scene_bounds['z'][0] <= new_position[2] <= self.scene_bounds['z'][1] and
                    distance_from_center <= max_distance_from_center):
                    break
            else:
                # If we can't find a valid position, use a position closer to center
                direction = (center_pos - last_position)
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                else:
                    direction = np.array([1, 0, 0])  # Default direction
                distance = min(0.05, max_distance_between_poses)  # Use smaller distance
                new_position = last_position + direction * distance

            # Sample orientation with constraints (keep orientations reasonable and reachable)
            pitch = np.random.uniform(-np.pi/18, np.pi/18)      # ±10° pitch
            roll = np.random.uniform(-np.pi/18, np.pi/18)       # ±10° roll
            yaw = np.random.uniform(-np.pi/18, np.pi/18)        # ±10° yaw

            # Create rotation from Euler angles (ZYX convention)
            rotation = R.from_euler('ZYX', [yaw, pitch, roll])
            quaternion = rotation.as_quat()  # [x, y, z, w]

            poses.append((new_position, quaternion))
            last_position = new_position.copy()

            if self.debug_mode:
                distance_from_prev = np.linalg.norm(new_position - (poses[-2][0] if len(poses) > 1 else center_pos))
                distance_from_center = np.linalg.norm(new_position - center_pos)
                print(f"   Sampled pose {len(poses)}: pos=[{new_position[0]:.3f}, {new_position[1]:.3f}, {new_position[2]:.3f}], "
                     f"dist_from_prev={distance_from_prev:.3f}m, dist_from_center={distance_from_center:.3f}m, "
                     f"euler=[{np.degrees(yaw):.1f}°, {np.degrees(pitch):.1f}°, {np.degrees(roll):.1f}°]")

        return poses

    def sample_independent_poses(self, num_poses: int = 7, center_pos: np.ndarray = None,
                                max_distance_from_center: float = 0.10, center_quat: np.ndarray = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Sample independent poses around a center position.

        Unlike sample_valid_scene_poses, this doesn't chain poses but samples each one
        independently around the center position. This is better for testing motion
        planner accuracy since each motion starts from the same reset state.

        Args:
            num_poses: Number of poses to sample
            center_pos: Center position to sample around
            max_distance_from_center: Maximum L2 distance from center (default: 10cm)
            center_quat: Center orientation to sample around

        Returns:
            List of (position, quaternion) tuples
        """
        poses = []

        # Use scene center if no center position provided
        if center_pos is None:
            center_pos = np.array([
                (self.scene_bounds['x'][0] + self.scene_bounds['x'][1]) / 2,
                (self.scene_bounds['y'][0] + self.scene_bounds['y'][1]) / 2,
                (self.scene_bounds['z'][0] + self.scene_bounds['z'][1]) / 2
            ])

        # Use identity quaternion if no center orientation provided
        if center_quat is None:
            center_quat = np.array([0, 0, 0, 1])  # Identity quaternion [x, y, z, w]

        for i in range(num_poses):
            max_attempts = 100  # Prevent infinite loops

            # Sample position within distance constraint from center position
            # Use spherical sampling to ensure uniform distribution
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)  # Normalize to unit vector
            distance = np.random.uniform(0.02, max_distance_from_center)  # Minimum 2cm, maximum max_distance_from_center
            new_position = center_pos + direction * distance

            # Note: Removed scene bounds check since we want to test motion planner accuracy
            # at the actual current EE position, which may be outside predefined scene bounds

            # Sample RELATIVE orientation changes (±10° from center orientation)
            delta_pitch = np.random.uniform(-np.pi/18, np.pi/18)  # ±10° pitch change
            delta_roll = np.random.uniform(-np.pi/18, np.pi/18)   # ±10° roll change
            delta_yaw = np.random.uniform(-np.pi/18, np.pi/18)    # ±10° yaw change

            # Create relative rotation from Euler angle deltas
            delta_rotation = R.from_euler('ZYX', [delta_yaw, delta_pitch, delta_roll])

            # Apply relative rotation to center orientation
            center_rotation = R.from_quat(center_quat)
            new_rotation = center_rotation * delta_rotation
            quaternion = new_rotation.as_quat()  # [x, y, z, w]

            poses.append((new_position, quaternion))

            if self.debug_mode:
                distance_from_center = np.linalg.norm(new_position - center_pos)
                print(f"   Sampled pose {len(poses)}: pos=[{new_position[0]:.3f}, {new_position[1]:.3f}, {new_position[2]:.3f}], "
                     f"dist_from_center={distance_from_center:.3f}m, euler=[{np.degrees(yaw):.1f}°, {np.degrees(pitch):.1f}°, {np.degrees(roll):.1f}°]")

        return poses

    def get_actual_ee_pose(self, env) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get actual end-effector pose from environment observation.

        Args:
            env: LIBERO environment

        Returns:
            Tuple of (position, quaternion) or (None, None) if failed
        """
        try:
            # Get observation via dummy action (6-DOF + gripper open)
            dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])  # 6-DOF + gripper open
            obs, _, _, _ = env.step(dummy_action)

            # Extract EE pose from observation space (standard LIBERO format)
            ee_pos = obs.get('robot0_eef_pos', None)
            ee_quat = obs.get('robot0_eef_quat', None)

            if ee_pos is None or ee_quat is None:
                if self.debug_mode:
                    print("   ⚠️ Warning: robot0_eef_pos or robot0_eef_quat not found in observation")
                return None, None

            # Copy to avoid reference issues
            ee_pos = ee_pos.copy()
            ee_quat = ee_quat.copy()

            return ee_pos, ee_quat

        except Exception as e:
            if self.debug_mode:
                print(f"   ❌ Error getting EE pose: {e}")
            return None, None

    def execute_motion_sequence(self, env, motion_planner: MotionPlanner,
                               target_poses: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict]:
        """
        Execute motion sequence and record deviations.

        Executes poses sequentially: reset → pose1 → pose2 → pose3 → ...
        This shows how errors accumulate through a sequence of motions.

        Args:
            env: LIBERO environment
            motion_planner: Motion planner instance
            target_poses: List of (position, quaternion) target poses

        Returns:
            List of deviation records
        """
        deviations = []

        # Reset environment once at the beginning of the sequence
        env.reset()
        if self.debug_mode:
            print(f"   🔄 Reset environment for sequence start")

        for i, (target_pos, target_quat) in enumerate(target_poses):
            if self.debug_mode:
                print(f"   🎯 Moving to pose {i+1}/{len(target_poses)}: "
                     f"pos=[{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")

            # Execute motion planning from current state (no reset between motions)
            success, obs = motion_planner.move_to_pose(target_pos, target_quat, num_steps=400)

            if not success:
                if self.debug_mode:
                    print(f"   ❌ Motion planning failed for pose {i+1}")
                continue

            # Get actual reached pose
            actual_pos, actual_quat = self.get_actual_ee_pose(env)

            if actual_pos is None or actual_quat is None:
                if self.debug_mode:
                    print(f"   ❌ Failed to get actual pose for pose {i+1}")
                continue

            # Calculate deviations
            pos_error = actual_pos - target_pos
            pos_error_norm = np.linalg.norm(pos_error)

            # Calculate orientation error (angular distance)
            target_rot = R.from_quat(target_quat)
            actual_rot = R.from_quat(actual_quat)
            relative_rot = actual_rot * target_rot.inv()
            orientation_error = relative_rot.magnitude()  # Angular distance in radians

            # Record deviation
            deviation_record = {
                'target_pos': target_pos.copy(),
                'target_quat': target_quat.copy(),
                'actual_pos': actual_pos.copy(),
                'actual_quat': actual_quat.copy(),
                'pos_error': pos_error.copy(),
                'pos_error_norm': pos_error_norm,
                'orientation_error': orientation_error,
                'success': success
            }

            deviations.append(deviation_record)

            if self.debug_mode:
                print(f"   📊 Deviation: pos_error={pos_error_norm:.4f}m, "
                     f"orient_error={np.degrees(orientation_error):.1f}°")

        return deviations

    def analyze_pose_deviations(self, all_deviations: List[Dict]) -> Dict:
        """
        Analyze statistical distributions of pose deviations.

        Args:
            all_deviations: List of all deviation records

        Returns:
            Dictionary containing statistical analysis results
        """
        if not all_deviations:
            print("❌ No deviations to analyze")
            return {}

        print(f"📊 Analyzing {len(all_deviations)} pose deviations...")

        # Extract error arrays
        pos_errors = np.array([d['pos_error'] for d in all_deviations])  # Shape: (N, 3)
        pos_error_norms = np.array([d['pos_error_norm'] for d in all_deviations])  # Shape: (N,)
        orientation_errors = np.array([d['orientation_error'] for d in all_deviations])  # Shape: (N,)

        # Position error statistics (per axis and norm)
        pos_stats = {
            'mean': np.mean(pos_errors, axis=0),
            'std': np.std(pos_errors, axis=0),
            'median': np.median(pos_errors, axis=0),
            'percentiles': {
                '95': np.percentile(pos_errors, 95, axis=0),
                '99': np.percentile(pos_errors, 99, axis=0)
            }
        }

        # Position error norm statistics
        pos_norm_stats = {
            'mean': np.mean(pos_error_norms),
            'std': np.std(pos_error_norms),
            'median': np.median(pos_error_norms),
            'percentiles': {
                '95': np.percentile(pos_error_norms, 95),
                '99': np.percentile(pos_error_norms, 99)
            }
        }

        # Orientation error statistics
        orient_stats = {
            'mean': np.mean(orientation_errors),
            'std': np.std(orientation_errors),
            'median': np.median(orientation_errors),
            'percentiles': {
                '95': np.percentile(orientation_errors, 95),
                '99': np.percentile(orientation_errors, 99)
            }
        }

        # Test for normal distribution (Shapiro-Wilk test)
        # For pose shifting guidance, we want to know if errors follow normal distribution
        pos_x_normality = stats.shapiro(pos_errors[:, 0]) if len(pos_errors) > 3 else None
        pos_y_normality = stats.shapiro(pos_errors[:, 1]) if len(pos_errors) > 3 else None
        pos_z_normality = stats.shapiro(pos_errors[:, 2]) if len(pos_errors) > 3 else None
        orient_normality = stats.shapiro(orientation_errors) if len(orientation_errors) > 3 else None

        # Correlation analysis (check if position and orientation errors are correlated)
        pos_orient_corr = np.corrcoef(pos_error_norms, orientation_errors)[0, 1] if len(pos_error_norms) > 1 else 0.0

        analysis_results = {
            'num_samples': len(all_deviations),
            'position_error_stats': pos_stats,
            'position_norm_stats': pos_norm_stats,
            'orientation_error_stats': orient_stats,
            'normality_tests': {
                'position_x': pos_x_normality,
                'position_y': pos_y_normality,
                'position_z': pos_z_normality,
                'orientation': orient_normality
            },
            'position_orientation_correlation': pos_orient_corr,
            'raw_deviations': all_deviations  # Store for further analysis
        }

        # Print summary
        print(f"📈 Position Error Statistics:")
        print(f"   Mean: [{pos_stats['mean'][0]:.4f}, {pos_stats['mean'][1]:.4f}, {pos_stats['mean'][2]:.4f}] m")
        print(f"   Std:  [{pos_stats['std'][0]:.4f}, {pos_stats['std'][1]:.4f}, {pos_stats['std'][2]:.4f}] m")
        print(f"   Norm: mean={pos_norm_stats['mean']:.4f}m, std={pos_norm_stats['std']:.4f}m")
        print(f"📈 Orientation Error Statistics:")
        print(f"   Mean: {np.degrees(orient_stats['mean']):.2f}°, Std: {np.degrees(orient_stats['std']):.2f}°")
        print(f"📈 Position-Orientation Correlation: {pos_orient_corr:.3f}")

        return analysis_results

    def generate_pose_shift_parameters(self, analysis_results: Dict) -> Dict:
        """
        Generate parameters for pose shifting based on motion planner inaccuracy analysis.

        Args:
            analysis_results: Results from analyze_pose_deviations()

        Returns:
            Dictionary containing pose shift parameters for VLA augmentation
        """
        if not analysis_results:
            return {}

        pos_stats = analysis_results['position_error_stats']
        orient_stats = analysis_results['orientation_error_stats']

        # Generate pose shift parameters based on observed inaccuracies
        # Use 2-3x the observed standard deviation to cover most of the inaccuracy distribution
        safety_factor = 2.5  # Safety factor to ensure good coverage

        pose_shift_params = {
            'position_std': (np.abs(pos_stats['std']) * safety_factor).tolist(),  # Per-axis standard deviation
            'orientation_std': float(orient_stats['std'] * safety_factor),        # Orientation standard deviation
            'position_mean_bias': pos_stats['mean'].tolist(),                     # Systematic bias if any
            'orientation_mean_bias': float(orient_stats['mean']),                 # Systematic orientation bias
            'correlation': analysis_results['position_orientation_correlation'],
            'distribution_type': 'normal',  # Assume normal for now, could be refined
            'safety_factor': safety_factor,
            'analysis_metadata': {
                'num_samples': analysis_results['num_samples'],
                'analysis_date': datetime.now().isoformat(),
                'position_percentile_95': analysis_results['position_error_stats']['percentiles']['95'].tolist(),
                'orientation_percentile_95': float(analysis_results['orientation_error_stats']['percentiles']['95'])
            }
        }

        print(f"🎯 Generated Pose Shift Parameters:")
        print(f"   Position Std: [{pose_shift_params['position_std'][0]:.4f}, "
              f"{pose_shift_params['position_std'][1]:.4f}, {pose_shift_params['position_std'][2]:.4f}] m")
        print(f"   Orientation Std: {np.degrees(pose_shift_params['orientation_std']):.2f}°")
        print(f"   Safety Factor: {safety_factor}")

        return pose_shift_params

    def save_analysis_results(self, analysis_results: Dict, pose_shift_params: Dict,
                             output_dir: str = "scripts/phase2/utils/motion_planner_analysis"):
        """
        Save analysis results and pose shift parameters.

        Args:
            analysis_results: Full analysis results
            pose_shift_params: Generated pose shift parameters
            output_dir: Output directory for saving results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save full analysis results
        analysis_file = os.path.join(output_dir, "motion_planner_inaccuracy_analysis.pkl")
        with open(analysis_file, 'wb') as f:
            pickle.dump(analysis_results, f)
        print(f"💾 Saved full analysis to: {analysis_file}")

        # Save pose shift parameters (for easy loading in other scripts)
        params_file = os.path.join(output_dir, "pose_shift_parameters.json")
        with open(params_file, 'w') as f:
            json.dump(pose_shift_params, f, indent=2)
        print(f"💾 Saved pose shift parameters to: {params_file}")

        # Save human-readable summary
        summary_file = os.path.join(output_dir, "analysis_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Motion Planner Inaccuracy Analysis Summary\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Samples: {analysis_results['num_samples']}\n\n")

            pos_stats = analysis_results['position_error_stats']
            f.write(f"Position Error Statistics (m):\n")
            f.write(f"  Mean: [{pos_stats['mean'][0]:.4f}, {pos_stats['mean'][1]:.4f}, {pos_stats['mean'][2]:.4f}]\n")
            f.write(f"  Std:  [{pos_stats['std'][0]:.4f}, {pos_stats['std'][1]:.4f}, {pos_stats['std'][2]:.4f}]\n")

            orient_stats = analysis_results['orientation_error_stats']
            f.write(f"\nOrientation Error Statistics:\n")
            f.write(f"  Mean: {np.degrees(orient_stats['mean']):.2f}°\n")
            f.write(f"  Std:  {np.degrees(orient_stats['std']):.2f}°\n")

            f.write(f"\nRecommended Pose Shift Parameters:\n")
            f.write(f"  Position Std: {pose_shift_params['position_std']}\n")
            f.write(f"  Orientation Std: {np.degrees(pose_shift_params['orientation_std']):.2f}°\n")

        print(f"💾 Saved summary to: {summary_file}")

        # Generate and save plots if requested
        if self.save_plots:
            self.save_visualization_plots(analysis_results, output_dir)

    def save_visualization_plots(self, analysis_results: Dict, output_dir: str):
        """Save visualization plots of the analysis results."""
        try:
            # Extract data for plotting
            deviations = analysis_results['raw_deviations']
            pos_errors = np.array([d['pos_error'] for d in deviations])
            pos_error_norms = np.array([d['pos_error_norm'] for d in deviations])
            orientation_errors = np.array([d['orientation_error'] for d in deviations])

            # Create plots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Motion Planner Inaccuracy Analysis', fontsize=16)

            # Position error distributions (per axis)
            for i, axis_name in enumerate(['X', 'Y', 'Z']):
                axes[0, i].hist(pos_errors[:, i], bins=20, alpha=0.7, density=True)
                axes[0, i].axvline(np.mean(pos_errors[:, i]), color='red', linestyle='--', label='Mean')
                axes[0, i].axvline(np.median(pos_errors[:, i]), color='green', linestyle='--', label='Median')
                axes[0, i].set_xlabel(f'{axis_name} Position Error (m)')
                axes[0, i].set_ylabel('Density')
                axes[0, i].set_title(f'{axis_name}-axis Position Error Distribution')
                axes[0, i].legend()
                axes[0, i].grid(True, alpha=0.3)

            # Position error norm distribution
            axes[1, 0].hist(pos_error_norms, bins=20, alpha=0.7, density=True)
            axes[1, 0].axvline(np.mean(pos_error_norms), color='red', linestyle='--', label='Mean')
            axes[1, 0].axvline(np.median(pos_error_norms), color='green', linestyle='--', label='Median')
            axes[1, 0].set_xlabel('Position Error Norm (m)')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Position Error Magnitude Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Orientation error distribution
            axes[1, 1].hist(np.degrees(orientation_errors), bins=20, alpha=0.7, density=True)
            axes[1, 1].axvline(np.degrees(np.mean(orientation_errors)), color='red', linestyle='--', label='Mean')
            axes[1, 1].axvline(np.degrees(np.median(orientation_errors)), color='green', linestyle='--', label='Median')
            axes[1, 1].set_xlabel('Orientation Error (degrees)')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Orientation Error Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            # Correlation plot
            axes[1, 2].scatter(pos_error_norms, np.degrees(orientation_errors), alpha=0.6)
            axes[1, 2].set_xlabel('Position Error Norm (m)')
            axes[1, 2].set_ylabel('Orientation Error (degrees)')
            axes[1, 2].set_title('Position vs Orientation Error Correlation')
            axes[1, 2].grid(True, alpha=0.3)

            # Add correlation coefficient
            corr = analysis_results['position_orientation_correlation']
            axes[1, 2].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                           transform=axes[1, 2].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))

            plt.tight_layout()

            # Save plot
            plot_file = os.path.join(output_dir, "motion_planner_inaccuracy_plots.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"📊 Saved visualization plots to: {plot_file}")

        except Exception as e:
            print(f"⚠️ Warning: Failed to save plots: {e}")

    def analyze_motion_planner_inaccuracy(self, task_name: str, num_iterations: int = 30,
                                        poses_per_iteration: int = 7, max_distance: float = 0.10) -> Dict:
        """
        Main function to analyze motion planner inaccuracy.

        Args:
            task_name: Name of the atomic skills task (e.g., 'KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_pick')
            num_iterations: Number of iterations to run
            poses_per_iteration: Number of poses to sample per iteration
            max_distance: Maximum distance in meters (controls both distance from center and between poses)

        Returns:
            Dictionary containing analysis results and pose shift parameters
        """
        print(f"🔍 Starting Motion Planner Inaccuracy Analysis")
        print(f"   Task name: {task_name}")
        print(f"   Iterations: {num_iterations}")
        print(f"   Poses per iteration: {poses_per_iteration}")
        print(f"   Max distance: {max_distance:.3f}m ({max_distance*100:.0f}cm)")
        print(f"   Total expected samples: {num_iterations * poses_per_iteration}")

        all_deviations = []

        for iteration in range(num_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{num_iterations}")

            # Set different random seed for each iteration to ensure different pose sampling
            iteration_seed = np.random.randint(0, 1000000) + iteration * 1000
            np.random.seed(iteration_seed)
            if self.debug_mode:
                print(f"   🎲 Random seed: {iteration_seed}")

            try:
                # Find task by name from benchmark
                benchmark_dict = benchmark.get_benchmark_dict()
                task_suite = benchmark_dict['atomic_skills']()
                task = None
                for task_id in range(task_suite.n_tasks):
                    if task_suite.get_task(task_id).name == task_name:
                        task = task_suite.get_task(task_id)
                        break

                if task is None:
                    print(f"❌ Task not found: {task_name}")
                    continue

                # Initialize environment with proper parameters (use horizon=2000 first)
                env, _ = get_libero_env(task, model_family='openvla', resolution=256, horizon=10000)

                # Set higher horizon for motion planning sequences (like in long horizon pipeline)
                if hasattr(env, 'env') and hasattr(env.env, 'horizon'):
                    env.env.horizon = 10000  # Increase episode length for motion planning sequences
                    if self.debug_mode:
                        print(f"   ✅ Set environment horizon to 10000 steps for motion planning")

                env.reset()

                # Initialize motion planner
                motion_planner = MotionPlanner(env, method="cartesian_linear", num_steps=400)

                # Get current EE pose from reset state as starting point for sampling
                current_ee_pos, current_ee_quat = self.get_actual_ee_pose(env)
                if current_ee_pos is None:
                    print(f"   ❌ Failed to get current EE pose for iteration {iteration + 1}")
                    continue

                # Use independent sampling around current EE position
                # Each pose is sampled independently within max_distance from current EE position
                target_poses = self.sample_independent_poses(
                    poses_per_iteration,
                    center_pos=current_ee_pos,
                    max_distance_from_center=max_distance,
                    center_quat=current_ee_quat
                )

                # Debug: Print all sampled target poses
                if self.debug_mode:
                    print(f"   📊 All sampled target poses for iteration {iteration + 1}:")
                    for idx, (pos, quat) in enumerate(target_poses):
                        print(f"     Pose {idx+1}: pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
                        rot = R.from_quat(quat)
                        euler = rot.as_euler('ZYX', degrees=True)
                        print(f"              quat=[{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}], euler=[{euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°]")

                if self.debug_mode:
                    print(f"   Sampled {len(target_poses)} poses for iteration {iteration + 1}")

                # Execute motion sequence and collect deviations
                deviations = self.execute_motion_sequence(env, motion_planner, target_poses)
                all_deviations.extend(deviations)

                print(f"   ✅ Collected {len(deviations)} successful pose deviations")

                # Clean up
                env.close()

            except Exception as e:
                print(f"   ❌ Error in iteration {iteration + 1}: {e}")
                if self.debug_mode:
                    import traceback
                    traceback.print_exc()
                continue

        print(f"\n📊 Analysis Complete: Collected {len(all_deviations)} total pose deviations")

        if not all_deviations:
            print("❌ No valid deviations collected. Analysis failed.")
            return {}

        # Analyze deviations
        analysis_results = self.analyze_pose_deviations(all_deviations)

        # Generate pose shift parameters
        pose_shift_params = self.generate_pose_shift_parameters(analysis_results)

        # Save results
        self.save_analysis_results(analysis_results, pose_shift_params)

        return {
            'analysis_results': analysis_results,
            'pose_shift_parameters': pose_shift_params
        }


def main():
    parser = argparse.ArgumentParser(description="Analyze motion planner inaccuracy for pose shifting guidance")
    parser.add_argument('--task_name', type=str,
                       default='KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_pick',
                       help='Atomic skills task name (default: KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_pick)')
    parser.add_argument('--task_bddl', type=str,
                       default='externals/boss/libero/libero/bddl_files/atomic_skills/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_pick.bddl',
                       help='Path to BDDL file (used to extract task name if --task_name not provided)')
    parser.add_argument('--num_iterations', type=int, default=10,
                       help='Number of iterations to run (default: 10)')
    parser.add_argument('--poses_per_iteration', type=int, default=7,
                       help='Number of poses to sample per iteration (default: 7)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose output')
    parser.add_argument('--no_plots', action='store_true',
                       help='Disable saving visualization plots')
    parser.add_argument('--output_dir', type=str,
                       default='scripts/phase2/utils/motion_planner_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--max_distance', type=float, default=0.10,
                       help='Maximum distance in meters (controls both distance from center and between poses, default: 0.10m = 10cm)')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = MotionPlannerInaccuracyAnalyzer(
        debug_mode=args.debug,
        save_plots=not args.no_plots
    )

    # Use the default task name
    task_name = args.task_name

    # Run analysis
    results = analyzer.analyze_motion_planner_inaccuracy(
        task_name=task_name,
        num_iterations=args.num_iterations,
        poses_per_iteration=args.poses_per_iteration,
        max_distance=args.max_distance
    )

    if results:
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"📝 Use the generated pose_shift_parameters.json for data-driven pose shifting")
    else:
        print(f"\n❌ Analysis failed!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())