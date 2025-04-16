import argparse
import json
import os
import pickle
import h5py
import numpy as np
import robosuite.utils.transform_utils as T
from tqdm import tqdm
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
import cv2

IMAGE_RESOLUTION = 256

def is_noop(action, prev_action=None, threshold=1e-4):
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    return np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_action[-1]

def extract_local_init(actions, joint_states, gripper_states, states, pre_grasp_steps=3):
    gripper_commands = actions[:, -1]
    transition_idx = None
    for i in range(1, len(gripper_commands)):
        if gripper_commands[i - 1] < 0 and gripper_commands[i] > 0:
            transition_idx = i
            break
    if transition_idx is None:
        return None
    start_idx = max(0, transition_idx - pre_grasp_steps)
    return np.concatenate([joint_states[start_idx], gripper_states[start_idx], states[start_idx]])

def main(args):
    print(f"Extracting oss local inits from task suite: {args.libero_task_suite}")
    os.makedirs(os.path.join(args.dest_dir, "oss_local_init"), exist_ok=True)
    success_count_dict = {}

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    if args.is_debug:
        task_id_ls = [17, 19, 22, 25]
    else:
        task_id_ls = list(range(num_tasks_in_suite))

    for task_id in tqdm(task_id_ls, desc="Tasks"):
        task = task_suite.get_task(task_id)
        task_name = task.name
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)

        task_name = task_name.split("_with_")[0]
        orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task_name}_demo.hdf5")
        if not os.path.exists(orig_data_path):
            print(f"Missing: {orig_data_path}")
            continue

        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        local_init_states = []
        task_successes = 0

        if args.is_debug:
            orig_data_keys = orig_data.keys()[0]
            imgs = []
        else:
            orig_data_keys = orig_data.keys()

        for i in range(orig_data_keys):
            demo_data = orig_data[f"demo_{i}"]
            orig_actions = demo_data["actions"][()]
            orig_states = demo_data["states"][()]

            env.reset()

            # yy: TODO here - double check whether this is correct
            try:
                env.set_init_state(orig_states[0])
            except ValueError as e:
                print("Init file does not match new BDDL task (e.g., objects added/removed). Falling back to random reset.")

            for _ in range(10):
                obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

            states = []
            actions = []
            gripper_states = []
            joint_states = []

            for _, action in enumerate(orig_actions):
                if args.is_debug:
                    imgs.append(obs['agentview_image'])
                prev_action = actions[-1] if len(actions) > 0 else None
                if is_noop(action, prev_action):
                    continue

                if states == []:
                    states.append(orig_states[0])
                else:
                    states.append(env.sim.get_state().flatten())

                actions.append(action)
                gripper_states.append(obs["robot0_gripper_qpos"])
                joint_states.append(obs["robot0_joint_pos"])

                obs, reward, done, info = env.step(action.tolist())

            if done:
                task_successes += 1
                try:
                    action_array = np.array(actions)
                    local_init = extract_local_init(action_array, joint_states, gripper_states, states, args.pre_grasp_steps)
                    if local_init is not None:
                        local_init_states.append(local_init)
                except Exception as e:
                    print(f"Error processing demo {i} of {task_name}: {e}")

        if args.is_debug:
            debug_video_path = os.path.join(args.dest_dir, "oss_local_init", f"debug_{task_id}_{task_name}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 20  # Change if needed
            h, w = imgs[0].shape[:2]
            writer = cv2.VideoWriter(debug_video_path, fourcc, fps, (w, h))
            for img in imgs:
                writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV
            writer.release()

        # Save the .init file if there's any success
        if len(local_init_states) > 0:
            init_save_path = os.path.join(args.dest_dir, "oss_local_init", f"{task_name}_local_init.init")
            with open(init_save_path, 'wb') as f:
                pickle.dump(np.array(local_init_states), f)
            print(f"Saved {len(local_init_states)} init states to {init_save_path}")
        else:
            print("[WARNING] NO local init states saved!!!")

        success_count_dict[task_name] = task_successes
        print(success_count_dict)
        orig_data_file.close()

    # Save success counts
    if not args.is_debug:
        success_path = os.path.join(args.dest_dir, "oss_local_init", "success_counts.pkl")
        with open(success_path, 'wb') as f:
            pickle.dump(success_count_dict, f)
        print(f"\nSaved success counts to: {success_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, required=True,
                        choices=["libero_90", "ch1"], default="ch1")
    parser.add_argument("--libero_raw_data_dir", type=str, required=True,
                        help="Path to directory with raw regenerated HDF5 demos",
                        default="/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/datasets/libero_90/")
    parser.add_argument("--dest_dir", type=str, required=True,
                        help="Destination dir to save .init files and success count",
                        default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/local_demos_libero_90_openvla_no_noops_pre_3")
    parser.add_argument("--pre_grasp_steps", type=int, default=3)
    parser.add_argument("--is_debug", type=int, default=0)
    args = parser.parse_args()

    main(args)
