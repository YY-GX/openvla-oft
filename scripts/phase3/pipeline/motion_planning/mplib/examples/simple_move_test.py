#!/usr/bin/env python3
"""
Simple MPLib motion planner test: Move to target poses in Complete Kitchen Organization env.
Usage: python simple_move_test.py
"""

import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from datetime import datetime
import imageio

sys.path.insert(0, '/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.insert(0, '/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')

import libero.libero.envs as libero_envs
from libero.libero import benchmark, get_libero_path
from scripts.phase3.pipeline.motion_planning.mplib.planner import MPlibMotionPlanner
from scripts.phase3.pipeline.motion_planning.mplib.action_utils import get_controller_robot_pose
from scripts.phase3.pipeline.motion_planning.mplib.utils.ee_pose_utils import libero_to_controller_pose
import robosuite.utils.transform_utils as T

def main():
    # Create Complete Kitchen Organization environment
    benchmark_dict = benchmark.get_benchmark_dict()
    bm = benchmark_dict["long_horizon_tasks_v0"]()
    task = bm.get_task(1)  # Task ID 1 = Complete Kitchen Organization

    bddl_path = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)

    env_args = {
        "bddl_file_name": bddl_path,
        "camera_heights": 256,
        "camera_widths": 256,
        "has_renderer": False,
        "has_offscreen_renderer": True,  # Enable for video recording
        "ignore_done": True,
        "use_camera_obs": True,  # Enable camera observations
        "control_freq": 20,
    }

    env = libero_envs.OffScreenRenderEnv(**env_args)
    env.reset()
    print("✅ Environment created")

    motion_planner = MPlibMotionPlanner(env=env, collision_aware=True, velocity_factor=0.9, verbose=False, use_agentview_only=False)
    print("✅ MPLib motion planner initialized\n")

    # Get reset pose using controller
    reset_pos, reset_rot_mat = get_controller_robot_pose(env, "right")
    reset_quat = T.mat2quat(reset_rot_mat)  # [x, y, z, w] - scipy format

    print(f"Reset pose from get_controller_robot_pose:")
    print(f"  Position: {reset_pos}")
    print(f"  Quaternion [x,y,z,w]: {reset_quat}")

    reset_rot = R.from_quat(reset_quat)  # Already in scipy [x,y,z,w] format

    # TEST 1: Move +10cm along X axis (controller frame)
    print(f"\n{'='*60}")
    print("TEST 1: +10cm along X axis")
    print(f"{'='*60}")
    shift_pos = reset_pos + np.array([0.10, 0.0, 0.0])  # +10cm X only
    # Convert from [x,y,z,w] to [w,x,y,z] format for move_to_pose
    shift_quat_wxyz = np.array([reset_quat[3], reset_quat[0], reset_quat[1], reset_quat[2]])  # [w,x,y,z]

    print(f"🎯 Target (controller): pos={shift_pos}, quat={shift_quat_wxyz}")
    success1, observations1 = motion_planner.move_to_pose(
        shift_pos, shift_quat_wxyz,
        position_threshold=0.02,
        orientation_threshold=0.524,
        skill_type='pick',
        grasped_object_name=None,
        is_libero_pose=False,
        save_pointcloud=True
    )

    final_pos1, final_rot_mat1 = get_controller_robot_pose(env, "right")
    final_quat1 = T.mat2quat(final_rot_mat1)  # [x,y,z,w]
    pos_err1 = np.linalg.norm(final_pos1 - shift_pos)
    final_rot1 = R.from_quat(final_quat1)
    # Convert shift_quat_wxyz back to [x,y,z,w] for scipy
    shift_quat_xyzw = np.array([shift_quat_wxyz[1], shift_quat_wxyz[2], shift_quat_wxyz[3], shift_quat_wxyz[0]])
    target_rot1 = R.from_quat(shift_quat_xyzw)
    ori_err_rad1 = (target_rot1 * final_rot1.inv()).magnitude()
    ori_err1 = np.degrees(ori_err_rad1)
    print(f"✅ Success: {success1}")
    print(f"📏 Position error: {pos_err1:.4f}m ({pos_err1*100:.2f}cm)")
    print(f"📐 Orientation error: {ori_err1:.1f}° ({ori_err_rad1:.4f}rad)")
    frames1 = [obs['agentview_image'] for obs in observations1 if 'agentview_image' in obs]

    # TEST 2: Original target pose (specified in LIBERO frame)
    env.reset()

    # Target pose in LIBERO frame
    libero_target_pos = np.array([-0.0976, -0.3535, 1.1772])
    # libero_target_rot = R.from_rotvec(np.deg2rad([-176.9, 3.6, -28.8])) 
    libero_target_rot = R.from_rotvec(np.deg2rad([-170.498, 33.267, -10.437]))
    print(f"\n🎯 Target (LIBERO): pos={libero_target_pos}, axis={np.degrees(libero_target_rot.as_rotvec())}°")

    # Prepare quaternion in [w,x,y,z] for planner (it converts internally)
    lt_xyzw = libero_target_rot.as_quat()
    libero_target_quat_wxyz = np.array([lt_xyzw[3], lt_xyzw[0], lt_xyzw[1], lt_xyzw[2]])

    success2, observations2 = motion_planner.move_to_pose(
        libero_target_pos, libero_target_quat_wxyz,
        position_threshold=0.02,
        orientation_threshold=0.524,
        skill_type='pick',
        grasped_object_name=None,
        is_libero_pose=True,
        save_pointcloud=True
    )

    # For error metrics, compare against controller-frame target
    target_pos2, target_rot_mat2 = libero_to_controller_pose(libero_target_pos, libero_target_rot.as_matrix())
    final_pos2, final_rot_mat2 = get_controller_robot_pose(env, "right")
    final_quat2 = T.mat2quat(final_rot_mat2)
    pos_err2 = np.linalg.norm(final_pos2 - target_pos2)
    final_rot2 = R.from_quat(final_quat2)
    target_quat2 = T.mat2quat(target_rot_mat2)
    target_rot2 = R.from_quat(target_quat2)
    ori_err_rad2 = (target_rot2 * final_rot2.inv()).magnitude()
    ori_err2 = np.degrees(ori_err_rad2)
    print(f"✅ Success: {success2}")
    print(f"📏 Position error: {pos_err2:.4f}m ({pos_err2*100:.2f}cm)")
    print(f"📐 Orientation error: {ori_err2:.1f}° ({ori_err_rad2:.4f}rad)")
    frames2 = [obs['agentview_image'] for obs in observations2 if 'agentview_image' in obs]

    # Save videos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase3/pipeline/motion_planning/mplib/examples/videos"

    if frames1:
        video_path1 = f"{video_dir}/test1_shift_{timestamp}.mp4"
        imageio.mimsave(video_path1, frames1, fps=20)
        print(f"\n💾 Video 1 saved: {video_path1}")

    if frames2:
        video_path2 = f"{video_dir}/test2_target_{timestamp}.mp4"
        imageio.mimsave(video_path2, frames2, fps=20)
        print(f"💾 Video 2 saved: {video_path2}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"{'='*60}")
    print(f"Test 1 (shift): Success={success1}, pos_err={pos_err1*100:.1f}cm, ori_err={ori_err1:.1f}°")
    print(f"Test 2 (target): Success={success2}, pos_err={pos_err2*100:.1f}cm, ori_err={ori_err2:.1f}°")
    print(f"{'='*60}\n")

    env.close()


if __name__ == "__main__":
    main()
