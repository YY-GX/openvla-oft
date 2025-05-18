import os
import pickle
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env

def log_obs_keys(benchmark_name):
    task_suite = benchmark.get_benchmark_dict()[benchmark_name]()
    for task in task_suite.tasks:
        task_name = task.name
        print(f"\n[INFO] Inspecting task: {task_name}")

        # Create environment
        env, _ = get_libero_env(task, model_family="openvla", resolution=256)

        obs = env.set_init_state(sim_state)
        obs_keys = list(obs.keys())
        print(f"[OBS KEYS] {obs_keys}")
        print("==========================")

if __name__ == "__main__":
    bm_name = "boss_44"
    log_obs_keys(bm_name)
