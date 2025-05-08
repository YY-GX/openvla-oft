import os
import pickle
import numpy as np
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image, get_libero_wrist_image

def save_initial_images(task_name: str, saved_num: int, init_dir: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    init_path = os.path.join(init_dir, f"{task_name}_local_init.init")
    if not os.path.exists(init_path):
        raise FileNotFoundError(f"Init file not found at {init_path}")

    with open(init_path, 'rb') as f:
        all_states = pickle.load(f)

    print(f"[INFO] Found {len(all_states)} initial states in {init_path}")
    saved_num = min(saved_num, len(all_states))

    # Get task and environment
    task_suite = benchmark.get_benchmark_dict()["libero_local1"]()
    task = [t for t in task_suite.tasks if t.name == task_name.replace("_local", "")][0]
    env, _ = get_libero_env(task, model_family="openvla", resolution=256)

    for i in range(saved_num):
        sim_state = all_states[i][9:]
        obs = env.set_init_state(sim_state)

        agent_img = get_libero_image(obs)
        wrist_img = get_libero_wrist_image(obs)

        # Save images
        agent_path = os.path.join(save_dir, f"{task_name}_agent_{i}.png")
        wrist_path = os.path.join(save_dir, f"{task_name}_wrist_{i}.png")
        agent_img.save(agent_path)
        wrist_img.save(wrist_path)

        print(f"[✔] Saved: {agent_path} and {wrist_path}")


if __name__ == "__main__":
    task_name = "LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray"
    saved_num = 10
    init_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/local_demos_libero_90_openvla_no_noops_pre_3"
    save_dir = f"./debug_inits_imgs/{task_name}/"
    save_initial_images(task_name, saved_num, init_dir, save_dir)

