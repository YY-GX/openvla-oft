#!/usr/bin/env python3

import argparse
import logging
import time
from pathlib import Path

import rerun as rr
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

def visualize_rlds(
    data_dir: str,
    dataset_name: str,
    episode_index: int,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    save: bool = False,
    output_dir: Path | None = None,
) -> Path | None:
    if save:
        assert output_dir is not None, "Provide --output-dir when using --save"

    logging.info("Loading dataset...")
    ds = tfds.load(dataset_name, data_dir=data_dir)
    train_set = list(ds["train"])

    print(f"Loaded {len(train_set)} episodes from {dataset_name}")
    if episode_index >= len(train_set):
        raise IndexError(f"Episode index {episode_index} is out of range. Dataset only has {len(train_set)} episodes.")

    episode = train_set[episode_index]
    repo_id = f"{dataset_name}/episode_{episode_index}"

    logging.info("Starting Rerun...")
    spawn_local_viewer = mode == "local" and not save
    rr.init(repo_id, spawn=spawn_local_viewer)

    if mode == "distant":
        rr.serve(open_browser=False, web_port=web_port, ws_port=ws_port)

    steps = episode["steps"]
    for i, step in enumerate(steps):
        rr.set_time_sequence("frame_index", i)

        # Images
        if "image" in step["observation"]:
            rr.log("image", rr.Image(step["observation"]["image"].numpy()))
        if "wrist_image" in step["observation"]:
            rr.log("wrist_image", rr.Image(step["observation"]["wrist_image"].numpy()))

        # Scalars
        for name in ["reward", "discount", "is_first", "is_last", "is_terminal"]:
            if name in step:
                rr.log(name, rr.Scalar(float(step[name].numpy())))

        # Vectors
        if "state" in step["observation"]:
            state_np = step["observation"]["state"].numpy()
            for idx, val in enumerate(state_np):
                rr.log(f"state/{idx}", rr.Scalar(val))

            gripper_np = state_np[-2:]
            for idx, val in enumerate(gripper_np):
                rr.log(f"gripper_state/{idx}", rr.Scalar(val))

        if "joint_state" in step["observation"]:
            for idx, val in enumerate(step["observation"]["joint_state"].numpy()):
                rr.log(f"joint_state/{idx}", rr.Scalar(val))

        if "action" in step:
            for idx, val in enumerate(step["action"].numpy()):
                rr.log(f"action/{idx}", rr.Scalar(val))

        if i == 0 and "language_instruction" in step:
            rr.log("instruction", rr.TextDocument(step["language_instruction"].numpy().decode("utf-8")))

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        rrd_path = output_dir / f"{dataset_name.replace('/', '_')}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        logging.info(f"Saved .rrd file to {rrd_path}")
        return rrd_path

    if mode == "distant" and not save:
        try:
            logging.info(f"Rerun server running at ws://localhost:{ws_port} — connect from local machine.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Interrupted by user. Shutting down.")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, required=True, help="Path to local TFDS data_dir")
    parser.add_argument("--dataset-name", type=str, required=True, help="Name of dataset folder (e.g. libero_local1)")
    parser.add_argument("--episode-index", type=int, default=None, help="Which episode to visualize (ignored if --language-description is set)")
    parser.add_argument("--language-description", type=str, default=None, help="Language instruction to match episode (overrides episode index if set)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save .rrd file (if --save is set)")
    parser.add_argument("--mode", type=str, default="local", choices=["local", "distant"], help="Viewer mode")
    parser.add_argument("--web-port", type=int, default=9090, help="Web UI port for distant rerun")
    parser.add_argument("--ws-port", type=int, default=9087, help="WebSocket port for distant rerun")
    parser.add_argument("--save", action="store_true", help="Save .rrd file instead of spawning viewer")

    args = parser.parse_args()

    # Handle language description filtering
    if args.language_description:
        logging.info(f"Searching for episode with instruction: \"{args.language_description}\"")
        ds = tfds.load(args.dataset_name, data_dir=args.data_dir)
        train_set = list(ds["train"])
        match_found = False

        for idx, ep in enumerate(train_set):
            try:
                first_step = next(iter(ep["steps"]))
                if "language_instruction" in first_step:
                    lang = first_step["language_instruction"].numpy().decode("utf-8").strip()
                    print(f"|{lang}|")
                    print(f"|{args.language_description.strip()}|")
                    if lang == args.language_description.strip():
                        logging.info(f"Found matching episode at index {idx}")
                        args.episode_index = idx
                        match_found = True
                        break
            except Exception as e:
                logging.warning(f"Skipping episode {idx} due to error: {e}")

        if not match_found:
            logging.error(f"No episode found with instruction: \"{args.language_description}\"")
            return

    if args.episode_index is None:
        raise ValueError("You must specify either --episode-index or --language-description")

    visualize_rlds(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        episode_index=args.episode_index,
        mode=args.mode,
        web_port=args.web_port,
        ws_port=args.ws_port,
        save=args.save,
        output_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()
