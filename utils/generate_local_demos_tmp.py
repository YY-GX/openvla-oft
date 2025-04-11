import h5py
import numpy as np
import os
import glob
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


# Main function
if __name__ == "__main__":
    """
    Important information:
    1. gripper open -> close: -1 -> 1
    2. to access action: data_dict['data']['demo_0']['actions'] [116, 7]
    3. to access obs: data_dict['data']['demo_0']['obs']['agentview_rgb'] [116, 256, 256, 3]
    4. to access obs: data_dict['data']['demo_0']['obs']['eye_in_hand_rgb'] [116, 256, 256, 3]
    """

    pre_grasp_steps = 3

    bddl_folder = "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/libero/bddl_files/libero_90/*.bddl"
    demos_file = np.array([
        os.path.join(
            "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/datasets/libero_90_openvla_no_noops_full/",
            bddl_file.split('/')[-1].split('.')[0] + "_demo.hdf5"
        ) for bddl_file in sorted(glob.glob(bddl_folder))
    ])

    dest_pth = f"/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/datasets/local_demos_libero_90_openvla_no_noops_pre_{pre_grasp_steps}/"  # Make sure this directory exists
    os.makedirs(dest_pth, exist_ok=True)

    for file_path in tqdm(demos_file, desc="Processing demo files"):
        data_dict = load_hdf5_to_dict(file_path)
        local_data_dict = {'data': {}}

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
                    local_demo[key] = {}
                    for subkey in demo[key]:
                        local_demo[key][subkey] = demo[key][subkey][start_idx:end_idx]
                else:
                    local_demo[key] = demo[key][start_idx:end_idx]

            local_data_dict['data'][demo_key] = local_demo

        # Save new HDF5 file
        save_name = os.path.basename(file_path)
        save_path = os.path.join(dest_pth, save_name)
        save_dict_to_hdf5(local_data_dict, save_path)
        print(f"Saved local demo to {save_path}")
