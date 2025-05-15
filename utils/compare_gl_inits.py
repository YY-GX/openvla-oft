import os
from PIL import Image
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image

def render_fresh_env_images(benchmark_name, task_name, save_dir, num_samples=5):
    os.makedirs(save_dir, exist_ok=True)

    # Load the benchmark and find the task
    task_suite = benchmark.get_benchmark_dict()[benchmark_name]()
    task_matches = [t for t in task_suite.tasks if t.name == task_name]
    if not task_matches:
        print(f"[WARN] Task '{task_name}' not found in benchmark '{benchmark_name}'. Skipping.")
        return
    task = task_matches[0]

    # Create environment
    env, _ = get_libero_env(task, model_family="openvla", resolution=256)

    # Render fresh resets
    for i in range(num_samples):
        obs = env.reset()
        img = get_libero_image(obs)

        img_path = os.path.join(save_dir, f"{task_name}_{i}.png")
        Image.fromarray(img).save(img_path)
        print(f"[✔] Saved image: {img_path}")


if __name__ == "__main__":
    save_base_dir = "./comparison_images"

    # Get task names from gl_size only
    gl_benchmark = benchmark.get_benchmark_dict()["gl_size"]()
    gl_task_names = [t.name for t in gl_benchmark.tasks]

    for task_name in gl_task_names:
        print(f"[INFO] Processing task: {task_name}")

        # Render from gl_size
        render_fresh_env_images(
            "gl_size",
            task_name,
            os.path.join(save_base_dir, "gl_size", task_name),
            num_samples=5
        )

        # Render from boss_44
        render_fresh_env_images(
            "boss_44",
            task_name,
            os.path.join(save_base_dir, "boss_44", task_name),
            num_samples=5
        )
