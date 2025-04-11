import h5py
import numpy as np
import os
import glob
import pickle
from tqdm import tqdm

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

def extract_and_save_initial_states(local_data_dict, save_path):
    os.makedirs("local_inits", exist_ok=True)
    all_states = []

    for demo_key in local_data_dict['data']:
        obs = local_data_dict['data'][demo_key]['obs']
        joint = obs['joint_states'][0]     # [7]
        gripper = obs['gripper_states'][0] # [2]
        extra_state = obs['states'][0]     # whatever its dim is
        state = np.concatenate([joint, gripper, extra_state])
        all_states.append(state)

    all_states = np.array(all_states)  # [num_demos, ?]
    with open(save_path, 'wb') as f:
        pickle.dump(all_states, f)
    print(f"Saved initial states ({all_states.shape[0]} demos) to {save_path}")

# Main function
if __name__ == "__main__":
    pre_grasp_steps = 3

    bddl_folder = "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/libero/bddl_files/libero_90/*.bddl"
    raw_demo_dir = "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/datasets/libero_90_openvla_no_noops_full/"
    dest_pth = f"/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/datasets/local_demos_libero_90_openvla_no_noops_pre_{pre_grasp_steps}/"
    os.makedirs(dest_pth, exist_ok=True)

    demo_files = np.array([
        os.path.join(raw_demo_dir, os.path.basename(bddl).replace(".bddl", "_demo.hdf5"))
        for bddl in sorted(glob.glob(bddl_folder))
    ])

    for file_path in tqdm(demo_files, desc="Processing demo files"):
        data_dict = load_hdf5_to_dict(file_path)
        local_data_dict = {'data': {}}
        task_name = os.path.basename(file_path).split("_demo.hdf5")[0]

        for demo_key in tqdm(data_dict['data'], desc=f"Processing {os.path.basename(file_path)}", leave=False):
            demo = data_dict['data'][demo_key]
            actions = demo['actions']
            gripper_commands = actions[:, -1]

            transition_idx = None
            for i in range(1, len(gripper_commands)):
                if gripper_commands[i - 1] < 0 and gripper_commands[i] > 0:
                    transition_idx = i
                    break

            if transition_idx is None:
                print(f"No gripper closing transition found in {file_path} / {demo_key}.")
                continue

            start_idx = max(0, transition_idx - pre_grasp_steps)
            end_idx = len(gripper_commands)

            local_demo = {}
            for key in demo:
                if isinstance(demo[key], dict):
                    local_demo[key] = {
                        subkey: demo[key][subkey][start_idx:end_idx]
                        for subkey in demo[key]
                    }
                else:
                    local_demo[key] = demo[key][start_idx:end_idx]

            local_data_dict['data'][demo_key] = local_demo

        # Save local demo file
        save_name = os.path.basename(file_path)
        save_path = os.path.join(dest_pth, save_name)
        save_dict_to_hdf5(local_data_dict, save_path)
        print(f"Saved local demo to {save_path}")

        # Save initial state
        init_save_path = os.path.join("local_inits", f"{task_name}_local_init.init")
        if not os.path.exists(init_save_path):
            extract_and_save_initial_states(local_data_dict, init_save_path)
        else:
            print(f"Init already exists for {task_name}, skipping.")
