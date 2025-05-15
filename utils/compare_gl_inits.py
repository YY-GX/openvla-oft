import os
import pickle
import numpy as np
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image
from PIL import Image, ImageDraw

def load_init_states(init_dir, task_name):
    init_path = os.path.join(init_dir, f"{task_name}.init")
    if not os.path.exists(init_path):
        raise FileNotFoundError(f"Init file not found at {init_path}")
    with open(init_path, 'rb') as f:
        return pickle.load(f)

def visualize_comparison(task_name, num_samples, suite1_dir, suite2_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Load init states from both suites
    states_suite1 = load_init_states(suite1_dir, task_name)
    states_suite2 = load_init_states(suite2_dir, task_name)

    print(f"[INFO] {len(states_suite1)} states in suite1, {len(states_suite2)} in suite2")

    # Get environment setup
    task_suite = benchmark.get_benchmark_dict()["libero_local2"]()
    task = [t for t in task_suite.tasks if t.name == task_name.replace("_local", "")][0]
    env, _ = get_libero_env(task, model_family="openvla", resolution=256)

    for i in range(num_samples):
        sim_state_1 = states_suite1[i][9:]
        sim_state_2 = states_suite2[i][9:]

        obs1 = env.set_init_state(sim_state_1)
        obs2 = env.set_init_state(sim_state_2)

        img1 = get_libero_image(obs1)
        img2 = get_libero_image(obs2)

        # Combine side by side
        combined_img = Image.new('RGB', (img1.shape[1]*2, img1.shape[0]))
        combined_img.paste(Image.fromarray(img1), (0, 0))
        combined_img.paste(Image.fromarray(img2), (img1.shape[1], 0))

        output_path = os.path.join(save_dir, f"{task_name}_comparison_{i}.png")
        combined_img.save(output_path)

        print(f"[✔] Saved comparison image: {output_path}")

# Example usage
if __name__ == "__main__":
    task_name = "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet"
    num_samples = 5

    # Directories with .init files
    suite1_dir = "/mnt/arc/yygx/paper_codebases/RA-L_25/BOSS/libero/libero/init_files/boss_44"
    suite2_dir = "/mnt/arc/yygx/paper_codebases/RA-L_25/BOSS/libero/libero/init_files/gl_size"

    # Where to save the comparison images
    save_dir = f"./comparison_images/{task_name}"

    visualize_comparison(task_name, num_samples, suite1_dir, suite2_dir, save_dir)
