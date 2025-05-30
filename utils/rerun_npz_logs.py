import argparse
import random
from pathlib import Path
import numpy as np
import rerun as rr
import time
import random

def visualize_npz_rollout(
    file_path: Path,
    name: str = "rollout",
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
):
    data = np.load(file_path)

    rr.init(name, spawn=(mode == "local"))

    if mode == "distant":
        rr.serve(open_browser=False, web_port=web_port, ws_port=ws_port)

    num_frames = len(data["agentview_image"])
    for i in range(num_frames):
        rr.set_time_sequence("frame_index", i)

        rr.log("camera/agentview", rr.Image(data["agentview_image"][i][::-1, :, :]))
        rr.log("camera/wristview", rr.Image(data["wrist_image"][i]))

        # print(data["actions"].shape)
        # print(data["joint_states"].shape)

        if "actions" in data:
            if data["actions"].shape[0] != 0 and (i < data["actions"].shape[0]):
                for j, val in enumerate(data["actions"][i]):
                    rr.log(f"action/{j}", rr.Scalar(val))

        if "joint_states" in data:
            for j, val in enumerate(data["joint_states"][i]):
                rr.log(f"joint_state/{j}", rr.Scalar(val))

        if "gripper_states" in data:
            for j, val in enumerate(data["gripper_states"][i]):
                rr.log(f"gripper_state/{j}", rr.Scalar(val))

    if mode == "distant":
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("[INFO] Server stopped.")


def find_npz(results_folder: Path, task_name: str, preferred: str = "succ", fail_idx=None) -> Path:
    task_dir = results_folder / task_name
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    files = sorted(task_dir.glob(f"{preferred}_*.npz"))
    if not files:
        raise FileNotFoundError(f"No '{preferred}' npz files found in {task_dir}")
    if fail_idx:
        return files[fail_idx]
    else:
        return files[random.choice(list(range(len(files))))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-folder", type=Path, default=Path("./runs/eval_results/wrist_only_True_v2/"), help="Base eval results folder")
    parser.add_argument("--task-name", type=str, required=True, help="Task name (e.g. put_bowl)")
    parser.add_argument("--preferred", type=str, choices=["succ", "fail"], default="fail", help="Which type of episode to load")
    parser.add_argument("--fail-idx", type=int, default=None,  help="Which episode to load")
    parser.add_argument("--mode", type=str, choices=["local", "distant"], default="distant", help="Viewer mode")
    parser.add_argument("--web-port", type=int, default=9090)
    parser.add_argument("--ws-port", type=int, default=9087)
    parser.add_argument("--name", type=str, default="rollout", help="Name of the Rerun stream")
    args = parser.parse_args()

    file_path = find_npz(args.results_folder, args.task_name, args.preferred, args.fail_idx)
    print(f"[INFO] Visualizing: {file_path}")

    visualize_npz_rollout(
        file_path=file_path,
        name=args.name,
        mode=args.mode,
        web_port=args.web_port,
        ws_port=args.ws_port,
    )


if __name__ == "__main__":
    main()
