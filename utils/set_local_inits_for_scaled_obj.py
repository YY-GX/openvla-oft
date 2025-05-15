import os
import pickle
import numpy as np
from PIL import Image
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image, get_libero_wrist_image


def load_random_init(init_dir, task_name):
    init_path = os.path.join(init_dir, f"{task_name}_local_init.init")
    if not os.path.exists(init_path):
        raise FileNotFoundError(f"Init file not found at {init_path}")

    with open(init_path, 'rb') as f:
        all_states = pickle.load(f)

    if len(all_states) == 0:
        raise ValueError(f"No states found in {init_path}")

    idx = np.random.randint(len(all_states))
    sim_state = all_states[idx][9:]  # skip metadata
    return sim_state


def render_and_save_combined_view(benchmark_name, init_dir, task_name, save_dir, num_samples=5):
    os.makedirs(save_dir, exist_ok=True)

    # Get benchmark and task
    task_suite = benchmark.get_benchmark_dict()[benchmark_name]()
    task_matches = [t for t in task_suite.tasks if t.name == task_name]
    if not task_matches:
        print(f"[WARN] Task '{task_name}' not found in benchmark '{benchmark_name}'. Skipping.")
        return
    task = task_matches[0]

    # Create environment
    env, _ = get_libero_env(task, model_family="openvla", resolution=256)

    for i in range(num_samples):
        try:
            sim_state = load_random_init(init_dir, task_name)
        except (FileNotFoundError, ValueError) as e:
            print(f"[WARN] {e}")
            continue

        obs = env.set_init_state(sim_state)

        # Get both views
        agent_img = get_libero_image(obs)
        wrist_img = get_libero_wrist_image(obs)

        # Combine them side by side
        combined_width = agent_img.shape[1] + wrist_img.shape[1]
        combined_height = max(agent_img.shape[0], wrist_img.shape[0])
        combined_img = Image.new('RGB', (combined_width, combined_height))
        combined_img.paste(Image.fromarray(agent_img), (0, 0))
        combined_img.paste(Image.fromarray(wrist_img), (agent_img.shape[1], 0))

        # Save
        img_path = os.path.join(save_dir, f"{task_name}_combined_{i}.png")
        combined_img.save(img_path)
        print(f"[✔] Saved combined image: {img_path}")


if __name__ == "__main__":
    gl_benchmark_name = "gl_size"
    gl_benchmark_name = "libero_local3"
    init_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/local_demos_libero_90_openvla_no_noops_pre_3/"
    save_base_dir = f"./{gl_benchmark_name}_combined_inits"

    # Collect task names from gl_size
    gl_benchmark = benchmark.get_benchmark_dict()[gl_benchmark_name]()
    gl_task_names = [t.name for t in gl_benchmark.tasks]

    for task_name in gl_task_names:
        print(f"[INFO] Processing task: {task_name}")
        render_and_save_combined_view(
            gl_benchmark_name,
            init_dir,
            task_name,
            os.path.join(save_base_dir, task_name),
            num_samples=1
        )
