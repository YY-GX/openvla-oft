import os
import pickle
import numpy as np
from PIL import Image
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image


def load_random_init(init_dir, task_name):
    init_path = os.path.join(init_dir, f"{task_name}_local_init.init")
    if not os.path.exists(init_path):
        raise FileNotFoundError(f"Init file not found at {init_path}")

    with open(init_path, 'rb') as f:
        all_states = pickle.load(f)

    if len(all_states) == 0:
        raise ValueError(f"No states found in {init_path}")

    idx = np.random.randint(len(all_states))
    sim_state = all_states[idx][9:]  # skip first 9 metadata entries
    return sim_state


def render_from_init(benchmark_name, init_dir, task_name, save_dir, num_samples=5):
    os.makedirs(save_dir, exist_ok=True)

    # Load benchmark and find task
    task_suite = benchmark.get_benchmark_dict()[benchmark_name]()
    task_matches = [t for t in task_suite.tasks if t.name == task_name]
    if not task_matches:
        print(f"[WARN] Task '{task_name}' not found in benchmark '{benchmark_name}'. Skipping.")
        return
    task = task_matches[0]

    # Create environment
    env, _ = get_libero_env(task, model_family="openvla", resolution=256)

    # Render specified number of inits
    for i in range(num_samples):
        try:
            sim_state = load_random_init(init_dir, task_name)
        except (FileNotFoundError, ValueError) as e:
            print(f"[WARN] {e}")
            continue

        obs = env.set_init_state(sim_state)
        img = get_libero_image(obs)

        img_path = os.path.join(save_dir, f"{task_name}_from_init_{i}.png")
        Image.fromarray(img).save(img_path)
        print(f"[✔] Saved image: {img_path}")


if __name__ == "__main__":
    gl_benchmark_name = "gl_size"
    init_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/local_demos_libero_90_openvla_no_noops_pre_3/"
    save_base_dir = "./gl_size_rendered_inits"

    # Get task names from gl_size
    gl_benchmark = benchmark.get_benchmark_dict()[gl_benchmark_name]()
    gl_task_names = [t.name for t in gl_benchmark.tasks]

    for task_name in gl_task_names:
        print(f"[INFO] Processing task: {task_name}")

        render_from_init(
            gl_benchmark_name,
            init_dir,
            task_name,
            os.path.join(save_base_dir, task_name),
            num_samples=1
        )
