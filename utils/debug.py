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
    5. to access ee: data_dict['data']['demo_0']['obs']['ee_pos'] [116, 3]; data_dict['data']['demo_0']['obs']['ee_ori'] [116, 3]
    6. to access joint: data_dict['data']['demo_0']['obs']['joint_states'] [116, 7]
    6. to access gripper: data_dict['data']['demo_0']['obs']['gripper_states'] [116, 2]
    """

    file_path = "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/datasets/libero_90_openvla_no_noops_full/KITCHEN_SCENE7_open_the_microwave_demo.hdf5"
    file_path = "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/datasets/libero_90_openvla_no_noops_full/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_demo.hdf5"
    file_path = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/libero_90_no_noops/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_demo.hdf5"

    # folder_path1 = "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/datasets/libero_90/"
    # folder_path2 = "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/datasets/libero_90_openvla_no_noops_full/"

    # file_path = "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/datasets/local_demos_libero_90_openvla_no_noops_pre_3/KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it_demo.hdf5"

    data_dict = load_hdf5_to_dict(file_path)

    # import os
    # for file in os.listdir(folder_path1):
    #     print(file)
    #     print(len(load_hdf5_to_dict(os.path.join(folder_path1, file))['data']))
    # for file in os.listdir(folder_path2):
    #     print(file)
    #     print(len(load_hdf5_to_dict(os.path.join(folder_path2, file))['data']))

    a = 1

