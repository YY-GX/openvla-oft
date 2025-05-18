import os
import h5py
import numpy as np
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env

def load_hdf5_to_dict(file_path):
    with h5py.File(file_path, 'r') as file:
        return {k: v[()] if isinstance(v, h5py.Dataset) else None for k, v in file['data'].items()}

if __name__ == "__main__":
    data_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/libero_90_no_noops/"
    bm_name = "boss_44"

    task = benchmark.get_benchmark_dict()[bm_name]().tasks[0]
    task_name = task.name
    print(f"[INFO] Task: {task_name}")

    env, _ = get_libero_env(task, model_family="openvla", resolution=256)
    demo_file = os.path.join(data_dir, f"{task_name}_demo.hdf5")

    with h5py.File(demo_file, 'r') as f:
        demo_keys = list(f['data'].keys())
        demo = f['data'][demo_keys[0]]
        states = demo['states']

    EE_GEOM_NAMES = ["robot0_eef", "robot0_gripper0_finger0", "robot0_gripper0_finger1"]

    for t, sim_state in enumerate(states):
        env.set_init_state(sim_state[()])
        n_contacts = env.sim.data.ncon
        if n_contacts == 0:
            continue

        print(f"[t={t}] {n_contacts} contacts:")
        for i in range(n_contacts):
            contact = env.sim.data.contact[i]
            g1 = env.sim.model.geom_id2name(contact.geom1)
            g2 = env.sim.model.geom_id2name(contact.geom2)
            ee_hit = g1 in EE_GEOM_NAMES or g2 in EE_GEOM_NAMES
            print(f"  {g1} <-> {g2}" + ("  [EE CONTACT]" if ee_hit else ""))
