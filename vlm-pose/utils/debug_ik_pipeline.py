import sys
sys.path.append('..')

import argparse
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import cv2
import h5py

from utils.tracik_tools import solve_ik, pose6d_to_matrix
from scipy.spatial.transform import Rotation as R

# Import LIBERO env creation utility
from experiments.robot.libero.libero_utils import get_libero_env_agentview_zoomout
from libero.libero import benchmark


def save_env_image(env, save_path):
    if hasattr(env, 'get_observation'):
        obs = env.get_observation()
    else:
        obs = env.env._get_observations()
    img = obs["agentview_image"][::-1, ::-1]
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description="Debug IK pipeline by round-tripping reset pose.")
    parser.add_argument('--skill_name', type=str, default='KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet', help='Skill/task name to use')
    parser.add_argument('--save_dir', type=str, default='runs/eval_results/vlm_pose_generator/debug_ik', help='Directory to save images')
    parser.add_argument('--resolution', type=int, default=256, help='Image resolution')
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create env for the skill
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_90"]()  # Instantiate the suite
    task_id = None
    for i in range(task_suite.n_tasks):
        if task_suite.get_task(i).name == args.skill_name:
            task_id = i
            break
    if task_id is None:
        raise ValueError(f"Task {args.skill_name} not found in libero_90 suite.")
    task = task_suite.get_task(task_id)
    env, _ = get_libero_env_agentview_zoomout(task, model_family="openvla", resolution=args.resolution)

    # 2. Reset robot (should be in neutral pose)
    obs = env.reset()
    robot = env.env.robots[0]
    reset_joints = robot._joint_positions.copy()
    print(f"[DEBUG] Reset joint angles: {reset_joints}")

    # 3. FK: get EE pose from reset joints using TracIK
    from tracikpy import TracIKSolver
    urdf_path = 'externals/tracikpy/data/franka_panda.urdf'
    base_link = 'panda_link0'
    tip_link = 'panda_hand'
    ik_solver = TracIKSolver(urdf_path, base_link, tip_link)
    fk_pose_mat = ik_solver.fk(reset_joints)
    print(f"[DEBUG] FK pose matrix from reset joints:\n{fk_pose_mat}")
    # Convert to 6D pose (xyz + axis-angle)
    pos = fk_pose_mat[:3, 3]
    rot = R.from_matrix(fk_pose_mat[:3, :3])
    axis_angle = rot.as_rotvec()
    pose6d = np.concatenate([pos, axis_angle])
    print(f"[DEBUG] FK pose as 6D (xyz + axis-angle): {pose6d}")

    # 4. Save image of reset pose
    save_env_image(env, save_dir / 'reset_pose.png')
    print(f"[INFO] Saved reset pose image to {save_dir / 'reset_pose.png'}")

    # 5. IK: run IK on the FK pose
    ik_joints = solve_ik(fk_pose_mat)
    print(f"[DEBUG] IK joints from FK pose: {ik_joints}")

    # 6. Set robot to IK-computed joints
    if ik_joints is not None:
        robot.set_robot_joint_positions(ik_joints)
        env.env.sim.forward()
        if hasattr(env.env, '_update_observables'):
            env.env._update_observables(force=True)
        # Print sim joint positions after set
        try:
            print(f"[DEBUG] Sim joint positions after IK set: {robot._joint_positions}")
        except Exception as e:
            print(f"[DEBUG] Sim joint debug failed: {e}")
        save_env_image(env, save_dir / 'ik_pose.png')
        print(f"[INFO] Saved IK pose image to {save_dir / 'ik_pose.png'}")
    else:
        print("[WARNING] IK failed for FK pose!")

    # 7. Load the first demonstration for this skill and extract the first timestep pose
    # Use extract_local_pairs.py as reference
    # Try to find the demo file for this skill
    demo_dir = os.path.join("datasets/hdf5_datasets/libero_90_no_noops/")
    demo_files = [f for f in os.listdir(demo_dir) if f.startswith(args.skill_name) and f.endswith("_demo.hdf5")]
    if not demo_files:
        print(f"[WARNING] No demo file found for skill {args.skill_name} in {demo_dir}")
    else:
        demo_path = os.path.join(demo_dir, demo_files[0])
        print(f"[DEBUG] Loading demo file: {demo_path}")
        with h5py.File(demo_path, 'r') as f:
            # Get the first demo key
            demo_keys = list(f['data'].keys())
            if not demo_keys:
                print(f"[WARNING] No demos found in file {demo_path}")
            else:
                demo = f['data'][demo_keys[0]]
                ee_pos = demo['obs']['ee_pos'][()]
                ee_ori = demo['obs']['ee_ori'][()]
                if len(ee_pos) > 0 and len(ee_ori) > 0:
                    first_pose = np.concatenate([ee_pos[0], ee_ori[0]])
                    print(f"[DEBUG] First timestep pose from demo: {first_pose}")
                    # Use IK pipeline to set joints for this pose and save image
                    print(f"[DEBUG] Running IK for first demo pose...")
                    first_pose_mat = pose6d_to_matrix(first_pose)
                    demo_ik_joints = solve_ik(first_pose_mat)
                    print(f"[DEBUG] IK joints from first demo pose: {demo_ik_joints}")
                    if demo_ik_joints is not None:
                        robot.set_robot_joint_positions(demo_ik_joints)
                        env.env.sim.forward()
                        if hasattr(env.env, '_update_observables'):
                            env.env._update_observables(force=True)
                        try:
                            print(f"[DEBUG] Sim joint positions after demo IK set: {robot._joint_positions}")
                        except Exception as e:
                            print(f"[DEBUG] Sim joint debug failed: {e}")
                        save_env_image(env, save_dir / 'demo_first_pose_ik.png')
                        print(f"[INFO] Saved demo first pose IK image to {save_dir / 'demo_first_pose_ik.png'}")
                    else:
                        print("[WARNING] IK failed for first demo pose!")

                    # Manually apply an offset to the first demo pose and try IK
                    offset = np.array([0.667, 0.0, -0.81, 0.0, 0.0, 0.0])
                    offset_pose = first_pose + offset
                    print(f"[DEBUG] Applying manual offset: {offset}")
                    print(f"[DEBUG] Offset demo pose: {offset_pose}")
                    offset_pose_mat = pose6d_to_matrix(offset_pose)
                    offset_ik_joints = solve_ik(offset_pose_mat)
                    print(f"[DEBUG] IK joints from offset demo pose: {offset_ik_joints}")
                    if offset_ik_joints is not None:
                        robot.set_robot_joint_positions(offset_ik_joints)
                        env.env.sim.forward()
                        if hasattr(env.env, '_update_observables'):
                            env.env._update_observables(force=True)
                        try:
                            print(f"[DEBUG] Sim joint positions after offset demo IK set: {robot._joint_positions}")
                        except Exception as e:
                            print(f"[DEBUG] Sim joint debug failed: {e}")
                        save_env_image(env, save_dir / 'demo_first_pose_offset_ik.png')
                        print(f"[INFO] Saved offset demo first pose IK image to {save_dir / 'demo_first_pose_offset_ik.png'}")
                    else:
                        print("[WARNING] IK failed for offset demo pose!")

                else:
                    print(f"[WARNING] Demo has no ee_pos or ee_ori data.")

    print("[DONE] Compare the two images. If the pipeline is correct, the robot should look identical in both.")

if __name__ == '__main__':
    main() 