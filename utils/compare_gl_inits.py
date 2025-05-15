import os
import torch
from PIL import Image
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image


def load_init_states(init_dir, task_name):
    init_path = os.path.join(init_dir, f"{task_name}.init")
    if not os.path.exists(init_path):
        raise FileNotFoundError(f"Init file not found at {init_path}")
    return torch.load(init_path)


def compare_suites(boss_benchmark_name, gl_benchmark_name, boss_init_dir, gl_init_dir, save_dir, num_samples=5):
    os.makedirs(save_dir, exist_ok=True)

    # Load both benchmarks
    boss_benchmark = benchmark.get_benchmark_dict()[boss_benchmark_name]()
    gl_benchmark = benchmark.get_benchmark_dict()[gl_benchmark_name]()

    # Collect task names from gl_benchmark
    gl_task_names = [t.name for t in gl_benchmark.tasks]

    print(f"[INFO] Tasks in {gl_benchmark_name}: {gl_task_names}")

    for task_name in gl_task_names:
        print(f"[INFO] Processing task: {task_name}")

        # Lookup matching task in boss_benchmark
        boss_task_matches = [t for t in boss_benchmark.tasks if t.name == task_name]
        if not boss_task_matches:
            print(f"[WARN] Task '{task_name}' not found in {boss_benchmark_name}. Skipping.")
            continue
        boss_task = boss_task_matches[0]
        gl_task = [t for t in gl_benchmark.tasks if t.name == task_name][0]

        # Load init states
        try:
            boss_states = load_init_states(boss_init_dir, task_name)
            gl_states = load_init_states(gl_init_dir, task_name)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue

        min_samples = min(num_samples, len(boss_states), len(gl_states))
        print(f"[INFO] Comparing {min_samples} samples for task {task_name}")

        # Create environments
        boss_env, _ = get_libero_env(boss_task, model_family="openvla", resolution=256)
        gl_env, _ = get_libero_env(gl_task, model_family="openvla", resolution=256)

        for i in range(min_samples):
            # Set and render states
            boss_obs = boss_env.set_init_state(boss_states[i][9:])
            gl_obs = gl_env.set_init_state(gl_states[i][9:])

            boss_img = get_libero_image(boss_obs)
            gl_img = get_libero_image(gl_obs)

            # Create combined image
            combined_img = Image.new('RGB', (boss_img.shape[1] * 2, boss_img.shape[0]))
            combined_img.paste(Image.fromarray(boss_img), (0, 0))
            combined_img.paste(Image.fromarray(gl_img), (boss_img.shape[1], 0))

            task_save_dir = os.path.join(save_dir, task_name)
            os.makedirs(task_save_dir, exist_ok=True)
            combined_img_path = os.path.join(task_save_dir, f"comparison_{i}.png")
            combined_img.save(combined_img_path)
            print(f"[✔] Saved comparison: {combined_img_path}")


# Example Usage
if __name__ == "__main__":
    compare_suites(
        boss_benchmark_name="boss_44",
        gl_benchmark_name="gl_size",
        boss_init_dir="/mnt/arc/yygx/paper_codebases/RA-L_25/BOSS/libero/libero/init_files/boss_44",
        gl_init_dir="/mnt/arc/yygx/paper_codebases/RA-L_25/BOSS/libero/libero/init_files/gl_size",
        save_dir="./comparison_images",
        num_samples=5
    )
