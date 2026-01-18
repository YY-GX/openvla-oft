#!/usr/bin/env python3
"""
Test MPLib collision-aware motion planning: pick ketchup, place in top drawer.
"""

import sys
import os
import numpy as np
from datetime import datetime
import imageio
import importlib.util
import cv2

sys.path.insert(0, '/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.insert(0, '/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')

import libero.libero.envs as libero_envs
from libero.libero import benchmark, get_libero_path
from scripts.phase3.pipeline.motion_planning.mplib.planner import MPlibMotionPlanner

# Load GT pose calculator
spec = importlib.util.spec_from_file_location(
    "pose_calculator",
    "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase2/4_phase2_gt_pose_calculator_clean.py"
)
pose_calculator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pose_calculator)


def main():
    # Create environment
    print("Creating environment...")
    benchmark_dict = benchmark.get_benchmark_dict()
    bm = benchmark_dict["long_horizon_tasks_v0"]()
    task = bm.get_task(1)  # Complete Kitchen Organization
    bddl_path = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)

    env_args = {
        "bddl_file_name": bddl_path,
        "camera_heights": 256,
        "camera_widths": 256,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_camera_obs": True,
        "control_freq": 20,
    }

    env = libero_envs.OffScreenRenderEnv(**env_args)
    env.reset()
    print("✅ Environment created\n")

    # Initialize MPLib planner (collision_aware=True by default)
    motion_planner = MPlibMotionPlanner(env=env, collision_aware=True, velocity_factor=0.8, verbose=True)
    print("✅ MPLib planner initialized\n")

    # Pose pairs directory
    pose_pairs_dir = "datasets/hdf5_datasets/atomic_local_demos_augmented_pick_10_place_15_atomic_15/ee_target_obj_pairs/simple_extraction_obs_pose_v1"

    all_frames = []

    # TEST 1: Pick ketchup
    print("="*60)
    print("TEST 1: Reset → Pick Ketchup")
    print("="*60)

    result = pose_calculator.calculate_gt_local_pose("pick ketchup", env, pose_pairs_dir=pose_pairs_dir)
    if result is None:
        print("❌ Failed to calculate pose for 'pick ketchup'")
        return

    target_pos, target_quat, vla_language, target_object = result
    print(f"Target: {vla_language}")
    print(f"Object: {target_object}\n")

    success1, observations1 = motion_planner.move_to_pose(
        target_pos, target_quat,
        position_threshold=0.02,
        orientation_threshold=0.524,
        skill_type="pick",
        grasped_object_name=None
    )

    # Handle both list and single dict returns
    obs_list1 = observations1 if isinstance(observations1, list) else [observations1]
    all_frames.extend([obs['agentview_image'] for obs in obs_list1 if isinstance(obs, dict) and 'agentview_image' in obs])
    print(f"Result: {'✅ SUCCESS' if success1 else '❌ FAILED'}\n")

    # Move EE up 10cm after pick
    print("="*60)
    print("Moving EE up 10cm after pick")
    print("="*60)

    obs = env.env._get_observations()
    current_ee_pos = obs['robot0_eef_pos'].copy()
    current_ee_quat = obs['robot0_eef_quat'].copy()

    target_ee_pos_up = current_ee_pos.copy()
    target_ee_pos_up[2] += 0.10  # 10cm up

    print(f"Current Z: {current_ee_pos[2]:.3f}m → Target Z: {target_ee_pos_up[2]:.3f}m\n")

    success_up, observations_up = motion_planner.move_to_pose(
        target_ee_pos_up, current_ee_quat,
        position_threshold=0.02,
        orientation_threshold=0.524,
        skill_type="pick",  # Keep gripper closed
        grasped_object_name=None
    )

    obs_list_up = observations_up if isinstance(observations_up, list) else [observations_up]
    all_frames.extend([obs['agentview_image'] for obs in obs_list_up if isinstance(obs, dict) and 'agentview_image' in obs])
    print(f"Result: {'✅ SUCCESS' if success_up else '❌ FAILED'}\n")

    # TEST 2: Place ketchup in top drawer
    print("="*60)
    print("TEST 2: Pick Ketchup → Place in Top Drawer")
    print("="*60)

    result = pose_calculator.calculate_gt_local_pose(
        "place ketchup in top drawer of the cabinet 1", env, pose_pairs_dir=pose_pairs_dir
    )
    if result is None:
        print("❌ Failed to calculate pose for 'place ketchup in top drawer'")
        return

    target_pos, target_quat, vla_language, target_object = result
    print(f"Target: {vla_language}")
    print(f"Object: {target_object}\n")

    success2, observations2 = motion_planner.move_to_pose(
        target_pos, target_quat,
        position_threshold=0.02,
        orientation_threshold=0.524,
        skill_type="place",
        grasped_object_name=None  # Not actually grasping in simulation
    )

    # Handle both list and single dict returns
    obs_list2 = observations2 if isinstance(observations2, list) else [observations2]
    all_frames.extend([obs['agentview_image'] for obs in obs_list2 if isinstance(obs, dict) and 'agentview_image' in obs])
    print(f"Result: {'✅ SUCCESS' if success2 else '❌ FAILED'}\n")

    # Save video (rotate 180 degrees clockwise)
    if all_frames:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f"/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase3/pipeline/motion_planning/mplib/examples/videos/collision_test_{timestamp}.mp4"
        # Rotate all frames 180 degrees
        rotated_frames = [cv2.rotate(frame, cv2.ROTATE_180) for frame in all_frames]
        imageio.mimsave(video_path, rotated_frames, fps=20)
        print(f"💾 Video saved: {video_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"{'='*60}")
    print(f"Test 1 (pick ketchup): {'✅ SUCCESS' if success1 else '❌ FAILED'}")
    print(f"Test 2 (place in drawer): {'✅ SUCCESS' if success2 else '❌ FAILED'}")
    print(f"{'='*60}\n")

    env.close()


if __name__ == "__main__":
    main()
