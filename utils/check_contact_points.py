import os
import h5py
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env

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

if __name__ == "__main__":
    data_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/libero_90_no_noops/"
    bm_name = "boss_44"

    task_suite = benchmark.get_benchmark_dict()[bm_name]()
    EE_GEOM_NAMES = [
        "robot0_eef",
        "robot0_gripper0_finger0",
        "robot0_gripper0_finger1",
        "gripper0_hand_collision",
        "gripper0_finger0_collision",
        "gripper0_finger1_collision",
    ]

    for task in task_suite.tasks:
        task_name = task.name
        demo_file = os.path.join(data_dir, f"{task_name}_demo.hdf5")
        if not os.path.exists(demo_file):
            print(f"[WARN] Missing demo file for {task_name}")
            continue

        data_dict = load_hdf5_to_dict(demo_file)
        demo_keys = list(data_dict['data'].keys())
        demo = data_dict['data'][demo_keys[0]]
        states = demo['states']
        actions = demo['actions']
        total_steps = len(states)

        env, _ = get_libero_env(task, model_family="openvla", resolution=256)

        print(f"\n[TASK] {task_name} ({total_steps} timesteps)")

        # --- Gripper Closing Detection ---
        gripper_closed = False
        for t, action in enumerate(actions):
            gripper_cmd = action[-1]
            if gripper_cmd > 0:  # assuming positive means closing
                print(f"  [GRIPPER CLOSE] at timestep {t+1} / {total_steps}")
                gripper_closed = True
                break
        if not gripper_closed:
            print(f"  [INFO] No gripper closing found.")

        # --- Contact Detection ---
        contact_found = False
        for t, sim_state in enumerate(states):
            env.set_init_state(sim_state)
            for i in range(env.sim.data.ncon):
                contact = env.sim.data.contact[i]
                g1 = env.sim.model.geom_id2name(contact.geom1)
                g2 = env.sim.model.geom_id2name(contact.geom2)
                if g1 in EE_GEOM_NAMES or g2 in EE_GEOM_NAMES:
                    print(f"  [CONTACT] {g1} <-> {g2} at timestep {t+1} / {total_steps}")
                    contact_found = True
                    break
            if contact_found:
                break
        if not contact_found:
            print(f"  [INFO] No EE contact found.")
