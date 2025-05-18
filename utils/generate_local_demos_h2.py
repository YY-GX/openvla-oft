import h5py
import numpy as np
import os
import glob
import pickle
import argparse
from tqdm import tqdm
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env

"""
This script use hueristic (detect contact point timesteps), h2, to extract local demos from the original demo files.
"""

# Recursive HDF5 loader
def load_hdf5_to_dict(file_path):
    def recursively_extract(group):
        result = {}
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Dataset):
                result[key] = item[()]
            elif isinstance(item, h5py.Group):
                result[key] = recursively_extract(item)
        return result

    with h5py.File(file_path, 'r') as file:
        return recursively_extract(file)

# Save dict to HDF5
def save_dict_to_hdf5(data_dict, save_path):
    def recursively_save(h5file, path, dic):
        for key, item in dic.items():
            full_path = f"{path}/{key}" if path else key
            if isinstance(item, dict):
                grp = h5file.create_group(full_path)
                recursively_save(h5file, full_path, item)
            else:
                h5file.create_dataset(full_path, data=item)

    with h5py.File(save_path, 'w') as h5file:
        recursively_save(h5file, '', data_dict)

def extract_and_save_initial_states(local_data_dict, save_path, init_offset_in_local):
    os.makedirs("local_inits", exist_ok=True)
    all_states = []

    for demo_key in local_data_dict['data']:
        obs = local_data_dict['data'][demo_key]['obs']
        joint = obs['joint_states'][init_offset_in_local]
        gripper = obs['gripper_states'][init_offset_in_local]
        extra_state = local_data_dict['data'][demo_key]['states'][init_offset_in_local]
        state = np.concatenate([joint, gripper, extra_state])
        all_states.append(state)

    all_states = np.array(all_states)
    with open(save_path, 'wb') as f:
        pickle.dump(all_states, f)
    print(f"Saved initial states ({all_states.shape[0]} demos) to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract local demos based on contact or gripper closing.")
    parser.add_argument("--bddl_folder", type=str, default="/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/libero/bddl_files/libero_90/*.bddl")
    parser.add_argument("--raw_demo_dir", type=str, default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/libero_90_no_noops/")
    parser.add_argument("--dest_dir", type=str, default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/")
    parser.add_argument("--pre_contact_init_steps", type=int, default=3, help="Steps before contact for init extraction.")
    parser.add_argument("--pre_contact_demo_steps", type=int, default=8, help="Steps before contact for demo extraction.")
    args = parser.parse_args()

    assert args.pre_contact_demo_steps >= args.pre_contact_init_steps, "Demo steps must be >= init steps"

    folder_suffix = f"local_demos_libero_90_openvla_no_noops_pre_init_{args.pre_contact_init_steps}_pre_demo_{args.pre_contact_demo_steps}"
    args.dest_dir = os.path.join(args.dest_dir, folder_suffix)
    os.makedirs(args.dest_dir, exist_ok=True)

    demo_files = sorted(glob.glob(args.bddl_folder))
    demo_files = [os.path.join(args.raw_demo_dir, os.path.basename(bddl).replace(".bddl", "_demo.hdf5")) for bddl in demo_files]

    for file_path in tqdm(demo_files, desc="Processing demo files"):
        data_dict = load_hdf5_to_dict(file_path)
        local_data_dict = {'data': {}}
        task_name = os.path.basename(file_path).split("_demo.hdf5")[0]

        for demo_key in tqdm(data_dict['data'], desc=f"Processing {os.path.basename(file_path)}", leave=False):
            demo = data_dict['data'][demo_key]
            actions = demo['actions']
            gripper_commands = actions[:, -1]

            # 1. Try to detect gripper closing
            transition_idx = None
            for i in range(1, len(gripper_commands)):
                if gripper_commands[i - 1] < 0 and gripper_commands[i] > 0:
                    transition_idx = i
                    print(f"Found gripper closing at {transition_idx} in {file_path} / {demo_key}")
                    break

            # 2. Fallback to contact detection if no gripper closing
            if transition_idx is None:
                print(f"No gripper closing found in {file_path} / {demo_key}, trying contact-based detection.")
                bm_name = "libero_90"
                task_suite = benchmark.get_benchmark_dict()[bm_name]()
                task = [t for t in task_suite.tasks if t.name == task_name][0]
                env, _ = get_libero_env(task, model_family="openvla", resolution=256)

                EE_GEOM_NAMES = [
                    "robot0_eef", "robot0_gripper0_finger0", "robot0_gripper0_finger1",
                    "gripper0_hand_collision", "gripper0_finger0_collision", "gripper0_finger1_collision"
                ]
                states = demo['states']

                for t, sim_state in enumerate(states):
                    env.set_init_state(sim_state)
                    for j in range(env.sim.data.ncon):
                        contact = env.sim.data.contact[j]
                        g1 = env.sim.model.geom_id2name(contact.geom1)
                        g2 = env.sim.model.geom_id2name(contact.geom2)
                        if g1 in EE_GEOM_NAMES or g2 in EE_GEOM_NAMES:
                            transition_idx = t
                            print(f"Found EE contact at {transition_idx} in {file_path} / {demo_key}")
                            break
                    if transition_idx is not None:
                        break

            if transition_idx is None:
                print(f"Skipping {file_path} / {demo_key} due to no detected event.")
                continue

            # Slicing bounds
            start_idx_demo = max(0, transition_idx - args.pre_contact_demo_steps)
            end_idx = len(gripper_commands)

            # Extract local demo
            local_demo = {}
            for key in demo:
                if isinstance(demo[key], dict):
                    local_demo[key] = {subkey: demo[key][subkey][start_idx_demo:end_idx] for subkey in demo[key]}
                else:
                    local_demo[key] = demo[key][start_idx_demo:end_idx]
            local_data_dict['data'][demo_key] = local_demo

        # Save local demo
        save_name = os.path.basename(file_path)
        save_path = os.path.join(args.dest_dir, save_name)
        save_dict_to_hdf5(local_data_dict, save_path)
        print(f"Saved local demo to {save_path}")

        # Calculate init offset relative to local demo
        init_offset_in_local = args.pre_contact_demo_steps - args.pre_contact_init_steps

        # Save initial states
        init_save_path = os.path.join(args.dest_dir, f"{task_name}_local_init.init")
        if not os.path.exists(init_save_path):
            extract_and_save_initial_states(local_data_dict, init_save_path, init_offset_in_local)
        else:
            print(f"Init already exists for {task_name}, skipping.")
