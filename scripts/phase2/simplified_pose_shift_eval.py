#!/usr/bin/env python3
"""
simplified_pose_shift_eval.py

Simplified atomic skills evaluation to visualize pose shifting effects.
Compares VLA performance between:
1. Shifted initial pose -> VLA execution
2. Family pose (closest demo) -> VLA execution

Saves videos for both scenarios to analyze VLA generalization.

python scripts/phase2/simplified_pose_shift_eval.py \
      --atomic_demos_path datasets/hdf5_datasets/atomic_local_demos/augmented_atomic_skills \
      --num_trials_per_skill 5 \
      --save_videos true \
      --save_failure_videos true \
      --exp_name "baseline_vla_pose_shift_test" \
      --shift_position_std 0.05 \
      --shift_orientation_std 1.047 

python scripts/phase2/simplified_pose_shift_eval.py \
    --pretrained_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/atomic_skills_augmented_debug/1.0.0/openvla-7b+libero_atomic_skills_augmented_debug+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_debug--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--70000_chkpt \
    --atomic_demos_path datasets/hdf5_datasets/atomic_local_demos_augmented/combined/debug \
    --focused_eval_mode True
"""
import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
import h5py
import pickle
import traceback
import imageio
from datetime import datetime

# Add project paths
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.robot_utils import (
    normalize_gripper_action,
    invert_gripper_action,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    get_model,
    get_image_resize_size,
    get_action,
)

# Phase 2 utilities
from scripts.phase2.utils.motion_planner import MotionPlanner
from scripts.phase2.utils.simple_pose_shift_utils import (
    generate_shifted_pose,
    find_family_pose,
    load_initial_states_for_skill,
)

# Initialize logging
logging.basicConfig(level=logging.INFO)


@dataclass
class SimplifiedConfig:
    """Simplified configuration for pose shift evaluation."""

    # Model configuration
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/atomic_skills/1.0.1/openvla-7b+libero_atomic_skills+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--200000_chkpt"

    # Model-specific parameters
    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_proprio: bool = True                         # Whether to include proprio state in input
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy
    load_in_8bit: bool = False                       # Load with 8-bit quantization
    load_in_4bit: bool = False                       # Load with 4-bit quantization
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (wrist cam only)

    # Environment configuration
    task_suite_name: str = "atomic_skills"           # Task suite
    env_img_res: int = 256                           # Environment image resolution

    # Simplified pose shifting parameters
    shift_position_std: float = 0.05                 # Position shift range (±5cm for each axis)
    shift_orientation_std: float = 1.047             # Orientation shift range (±60° total)

    # Evaluation parameters
    num_trials_per_skill: int = 1                    # Number of trials per skill
    max_steps: int = 220                             # Maximum steps per episode

    # Data paths
    atomic_demos_path: str = "datasets/hdf5_datasets/atomic_local_demos_augmented/combined/debug"

    # Output configuration
    save_videos: bool = True                         # Save rollout videos
    save_failure_videos: bool = True                 # Save videos even for failed episodes
    exp_name: str = "simplified_pose_shift_eval"     # Experiment name for output directory

    # Test mode
    test_mode: bool = False                          # Test mode: only test setup without loading full model

    # Specific skill for testing (empty = all skills)
    target_skill: str = ""                           # Test specific skill (e.g., "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet")

    # Focused evaluation mode for 3 specific skills
    focused_eval_mode: bool = False                  # If True, evaluates only 3 specific skills with 10 trials each

    is_depth: bool = False                           # Whether to use depth images (permanently set to False)

    # Action un-normalization key
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key


def load_vla_model_and_processor(cfg: SimplifiedConfig):
    """Load VLA model and processor."""
    if cfg.test_mode:
        print("🧪 Test mode: Skipping model loading")
        return None, None, None, None

    print(f"🤖 Loading VLA model: {cfg.model_family}")
    print(f"   Checkpoint: {cfg.pretrained_checkpoint}")

    # Load VLA model using get_model like the original script
    vla = get_model(cfg)

    # Get VLA components - following original script pattern
    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, vla.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, vla)

    print(f"✅ VLA model loaded successfully")
    return vla, processor, action_head, proprio_projector


def check_unnorm_key(cfg: SimplifiedConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = "atomic_skills"  # Default for atomic skills

    # For atomic_skills, the model was trained with "libero_atomic_skills" key
    if unnorm_key == "atomic_skills":
        unnorm_key = "libero_atomic_skills"

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    # If still not found, try to find any key containing "atomic" or fallback to bridge_orig
    if unnorm_key not in model.norm_stats:
        available_keys = list(model.norm_stats.keys())
        print(f"Warning: Expected key '{unnorm_key}' not found. Available keys: {available_keys}")

        # Try to find atomic skills related key
        atomic_keys = [k for k in available_keys if "atomic" in k.lower() or "libero" in k.lower()]
        if atomic_keys:
            unnorm_key = atomic_keys[0]
            print(f"Using atomic-related key: {unnorm_key}")
        else:
            # Fallback to bridge_orig as it's a common normalization key
            unnorm_key = "bridge_orig"
            print(f"Using fallback key: {unnorm_key}")

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def prepare_observation_for_policy(obs, resize_size):
    """Prepare observation for policy input - wrist cam only (from original script)."""
    # Get preprocessed images - wrist cam only
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict - use end-effector pose like working script
    observation = {
        "full_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation


def process_action(action, model_family):
    """Process action before sending to environment (from working script)."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def generate_vla_action(cfg, vla, observation, task_description, processor, action_head, proprio_projector):
    """Generate VLA action using original script pattern."""
    if cfg.test_mode:
        # Return dummy action in test mode
        return get_libero_dummy_action("libero_spatial")

    # Get actions using the original get_action function
    actions = get_action(
        cfg,
        vla,
        observation,
        task_description,
        processor=processor,
        action_head=action_head,
        proprio_projector=proprio_projector,
        noisy_action_projector=None,  # Not using diffusion
        use_film=cfg.use_film,
    )

    # Return the first action (original script uses action queue)
    return actions[0] if len(actions) > 0 else get_libero_dummy_action("libero_spatial")


def load_atomic_skill_metadata(cfg: SimplifiedConfig):
    """Load atomic skills metadata by dynamically discovering tasks with .init files."""
    print("📊 Loading atomic skills metadata...")

    # Initialize task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    
    # Get tasks with initial state files
    atomic_skills = []
    for task_id in range(task_suite.n_tasks):
        task_name = task_suite.get_task(task_id).name
        # Check for .init files - try both _original.init, _augmented.init, and plain .init
        original_init_path = os.path.join(cfg.atomic_demos_path, f"{task_name}_original.init")
        augmented_init_path = os.path.join(cfg.atomic_demos_path, f"{task_name}_augmented.init")
        plain_init_path = os.path.join(cfg.atomic_demos_path, f"{task_name}.init")

        if os.path.exists(original_init_path) or os.path.exists(augmented_init_path) or os.path.exists(plain_init_path):
            atomic_skills.append(task_name)
        else:
            print(f"[WARNING] {task_name} does not have corresponding atomic init file.")
    
    print(f"✅ Loaded {len(atomic_skills)} atomic skills with .init files")
    return atomic_skills


def evaluate_skill_with_pose_shift(skill_name: str, cfg: SimplifiedConfig, vla, processor, action_head, proprio_projector, video_dir: Path, resize_size):
    """
    Evaluate a single skill with pose shifting.

    Workflow:
    1. Load initial state and generate shifted pose
    2. Evaluate: Initial state -> Move to shifted pose -> VLA execution (save video)
    3. Find family pose for the shifted pose
    4. Evaluate: Initial state -> Move to family pose -> VLA execution (save video)
    """
    print(f"\n{'='*80}")
    print(f"🎯 Evaluating skill with pose shifts: {skill_name}")
    print(f"{'='*80}")

    try:
        # Load initial states for this skill
        initial_states = load_initial_states_for_skill(skill_name, cfg.atomic_demos_path)
        if len(initial_states) == 0:
            print(f"❌ No initial states found for skill: {skill_name}")
            return None

        # Use the first initial state as reference
        reference_initial_state = initial_states[0]

        # Extract skill instruction from skill name
        instruction = skill_name.replace("_", " ").replace("KITCHEN SCENE1", "").replace("KITCHEN SCENE2", "").replace("KITCHEN SCENE9", "").replace("KITCHEN SCENE10", "").strip()
        print(f"💬 Instruction: {instruction}")

        results = {}

        # Evaluation 1: Initial state -> Generate shifted pose from actual current -> VLA execution
        print(f"\n🔄 Evaluation 1: Initial state -> Generate shifted pose -> VLA execution")
        shifted_result = evaluate_single_trial_with_shift(
            skill_name, reference_initial_state, instruction, cfg,
            vla, processor, action_head, proprio_projector,
            video_dir, video_suffix="shifted", initial_states=initial_states, resize_size=resize_size
        )
        results["shifted"] = shifted_result

        # Evaluation 2: Initial state -> Direct VLA execution (no motion planning)
        print(f"\n🔄 Evaluation 2: Initial state -> Direct VLA execution")
        family_result = evaluate_single_trial_direct(
            skill_name, reference_initial_state, instruction,
            cfg, vla, processor, action_head, proprio_projector,
            video_dir, video_suffix="baseline", resize_size=resize_size
        )

        results["family"] = family_result

        # Summary
        print(f"\n📊 Results Summary for {skill_name}:")
        print(f"   Shifted pose success: {shifted_result['success'] if shifted_result else 'Failed'}")
        print(f"   Baseline (direct) success: {family_result['success'] if family_result else 'Failed'}")

        return results

    except Exception as e:
        print(f"❌ Error evaluating skill {skill_name}: {e}")
        traceback.print_exc()
        return None


def evaluate_single_trial_with_shift(skill_name: str, initial_state: np.ndarray, instruction: str, cfg: SimplifiedConfig,
                                    vla, processor, action_head, proprio_projector, video_dir: Path, video_suffix: str, initial_states: list, resize_size):
    """
    Evaluate a single trial with shift generation: Initial state -> Generate shifted pose from current -> VLA execution.
    """
    try:
        # Find task by name (following original script pattern)
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict["atomic_skills"]()
        task = None
        for task_id in range(task_suite.n_tasks):
            if task_suite.get_task(task_id).name == skill_name:
                task = task_suite.get_task(task_id)
                break

        if task is None:
            print(f"   ❌ Task not found: {skill_name}")
            return {"success": False, "reason": "task_not_found"}

        # Create environment using the task object
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res, horizon=10000)

        # Reset environment first (critical for proper initialization)
        env.reset()

        # Set initial state using proper Libero method
        print(f"   🔄 Setting initial simulation state...")
        # Use the full initial state format: [joint(7) + gripper(2) + sim_state]
        # The sim_state part (from index 9:) is used for complete environment reset
        sim_state = initial_state[9:]  # Extract simulation state

        # Use env.set_init_state() for proper environment reset (same as generation script)
        env.set_init_state(sim_state)

        # Get observations after setting state
        obs = env.env._get_observations()

        # Get actual current EE pose from observation
        current_ee_pos = obs['robot0_eef_pos'].copy()
        current_ee_quat = obs['robot0_eef_quat'].copy()

        current_ee_axisangle = quat2axisangle(current_ee_quat)
        print(f"   📍 Current EE: pos=[{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}] ori=[{np.degrees(current_ee_axisangle[0]):.1f}°, {np.degrees(current_ee_axisangle[1]):.1f}°, {np.degrees(current_ee_axisangle[2]):.1f}°]")

        # Generate shifted pose from actual current pose
        current_pose = (current_ee_pos, current_ee_quat)
        shifted_pose = generate_shifted_pose(
            current_pose,
            position_shift_range=cfg.shift_position_std,
            orientation_shift_range=cfg.shift_orientation_std
        )
        shifted_pos, shifted_quat = shifted_pose

        shifted_axisangle = quat2axisangle(shifted_quat)
        print(f"   🎯 Shifted EE: pos=[{shifted_pos[0]:.3f}, {shifted_pos[1]:.3f}, {shifted_pos[2]:.3f}] ori=[{np.degrees(shifted_axisangle[0]):.1f}°, {np.degrees(shifted_axisangle[1]):.1f}°, {np.degrees(shifted_axisangle[2]):.1f}°]")
        pos_diff = np.linalg.norm(shifted_pos - current_ee_pos)
        print(f"   📏 Position difference: {pos_diff*100:.1f}cm")

        # Create motion planner
        motion_planner = MotionPlanner(env, method="cartesian_linear", num_steps=50)

        # Move to shifted pose with tight thresholds
        print(f"   🚀 Moving to shifted pose using motion planner...")
        position_threshold = 0.01  # 1cm
        orientation_threshold = np.radians(5)  # 5 degrees
        move_success, final_obs = motion_planner.move_to_pose(
            shifted_pos, shifted_quat,
            position_threshold=position_threshold,
            orientation_threshold=orientation_threshold
        )

        if not move_success:
            print(f"   ❌ Motion planner failed to reach shifted pose")
            return {"success": False, "reason": "motion_planner_failed"}

        print(f"   ✅ Motion planner successfully reached shifted pose")

        # Start VLA evaluation from shifted pose
        print(f"   🤖 Starting VLA evaluation from shifted pose...")

        # Debug: Print EE pose and gripper state before VLA execution
        current_ee_pos = final_obs['robot0_eef_pos'].copy()
        current_ee_quat = final_obs['robot0_eef_quat'].copy()
        current_ee_axisangle = quat2axisangle(current_ee_quat)
        gripper_state = final_obs['robot0_gripper_qpos'].copy()
        print(f"   📍 EVAL 1 - EE pose: pos=[{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}] ori=[{np.degrees(current_ee_axisangle[0]):.1f}°, {np.degrees(current_ee_axisangle[1]):.1f}°, {np.degrees(current_ee_axisangle[2]):.1f}°] gripper=[{gripper_state[0]:.3f}, {gripper_state[1]:.3f}]")

        # Initialize video recording
        frames = []
        if cfg.save_videos:
            # Record initial frame
            img = get_libero_image(final_obs)
            frames.append(img)

        # Initialize action queue (like working script)
        action_queue = deque(maxlen=cfg.num_open_loop_steps)

        # VLA execution loop
        success = False
        obs = final_obs

        for step in range(cfg.max_steps):
            # If action queue is empty, requery model (like working script)
            if len(action_queue) == 0:
                # Prepare observation for policy
                observation = prepare_observation_for_policy(obs, resize_size)
                # Get actions using VLA
                actions = get_action(
                    cfg,
                    vla,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=None,  # Not using diffusion
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action (like working script)
            action = process_action(action, cfg.model_family)


            # Execute action
            obs, reward, done, info = env.step(action.tolist())

            # Record frame
            if cfg.save_videos:
                img = get_libero_image(obs)
                frames.append(img)

            # Check for task completion
            if done:
                success = True  # done=True means task completed successfully
                print(f"   ✅ Episode finished at step {step+1}: Success")
                break

        if not done:
            print(f"   ⏰ Episode timed out after {cfg.max_steps} steps")

        # Save video (only if video_suffix is provided)
        video_path = None
        if cfg.save_videos and video_suffix and (success or cfg.save_failure_videos):
            video_name = f"{skill_name}_{video_suffix}_{'success' if success else 'failure'}.mp4"
            video_path = video_dir / video_name
            video_writer = imageio.get_writer(str(video_path), fps=30)
            for img in frames:
                video_writer.append_data(img)
            video_writer.close()
            print(f"   🎥 Video saved: {video_path}")

        return {
            "success": success,
            "steps": step + 1 if done else cfg.max_steps,
            "video_path": video_path,
            "shifted_pose": shifted_pose,  # Return shifted pose for family pose generation
            "original_pose": current_pose   # Return original pose for family pose generation
        }

    except Exception as e:
        print(f"   ❌ Trial evaluation failed: {e}")
        traceback.print_exc()
        return {"success": False, "reason": f"evaluation_error: {e}"}


def evaluate_single_trial(skill_name: str, initial_state: np.ndarray, target_pose: tuple, instruction: str,
                         cfg: SimplifiedConfig, vla, processor, action_head, proprio_projector,
                         video_dir: Path, video_suffix: str, resize_size):
    """
    Evaluate a single trial: Initial state -> Move to target pose -> VLA execution.
    """
    target_pos, target_quat = target_pose
    print(f"   🎯 Target pose: pos=[{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")

    try:
        # Find task by name (following original script pattern)
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict["atomic_skills"]()
        task = None
        for task_id in range(task_suite.n_tasks):
            if task_suite.get_task(task_id).name == skill_name:
                task = task_suite.get_task(task_id)
                break

        if task is None:
            print(f"   ❌ Task not found: {skill_name}")
            return {"success": False, "reason": "task_not_found"}

        # Create environment using the task object
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res, horizon=10000)

        # Reset environment first (critical for proper initialization)
        env.reset()

        # Set initial state using proper Libero method
        print(f"   🔄 Setting initial simulation state...")
        # Use the full initial state format: [joint(7) + gripper(2) + sim_state]
        # The sim_state part (from index 9:) is used for complete environment reset
        sim_state = initial_state[9:]  # Extract simulation state

        # Use env.set_init_state() for proper environment reset (same as generation script)
        env.set_init_state(sim_state)

        # Get observations after setting state
        obs = env.env._get_observations()

        # Get actual current EE pose from observation
        current_ee_pos = obs['robot0_eef_pos'].copy()
        current_ee_quat = obs['robot0_eef_quat'].copy()

        current_ee_axisangle = quat2axisangle(current_ee_quat)
        target_axisangle = quat2axisangle(target_quat)
        
        print(f"   📍 Actual current EE: pos=[{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}] ori=[{np.degrees(current_ee_axisangle[0]):.1f}°, {np.degrees(current_ee_axisangle[1]):.1f}°, {np.degrees(current_ee_axisangle[2]):.1f}°]")
        print(f"   🎯 Target EE (input):  pos=[{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] ori=[{np.degrees(target_axisangle[0]):.1f}°, {np.degrees(target_axisangle[1]):.1f}°, {np.degrees(target_axisangle[2]):.1f}°]")

        # Create motion planner
        motion_planner = MotionPlanner(env, method="cartesian_linear", num_steps=50)

        # Move to target pose with tight thresholds
        print(f"   🚀 Moving to target pose using motion planner...")
        position_threshold = 0.01  # 1cm
        orientation_threshold = np.radians(5)  # 5 degrees
        move_success, final_obs = motion_planner.move_to_pose(
            target_pos, target_quat,
            position_threshold=position_threshold,
            orientation_threshold=orientation_threshold
        )

        if not move_success:
            print(f"   ❌ Motion planner failed to reach target pose")
            return {"success": False, "reason": "motion_planner_failed"}

        print(f"   ✅ Motion planner successfully reached target pose")

        # Start VLA evaluation from this pose
        print(f"   🤖 Starting VLA evaluation from target pose...")

        # Initialize video recording
        frames = []
        if cfg.save_videos:
            # Record initial frame
            img = get_libero_image(final_obs)
            frames.append(img)

        # Initialize action queue (like working script)
        action_queue = deque(maxlen=cfg.num_open_loop_steps)

        # VLA execution loop
        success = False
        obs = final_obs

        for step in range(cfg.max_steps):
            # If action queue is empty, requery model (like working script)
            if len(action_queue) == 0:
                # Prepare observation for policy
                observation = prepare_observation_for_policy(obs, resize_size)
                # Get actions using VLA
                actions = get_action(
                    cfg,
                    vla,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=None,  # Not using diffusion
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action (like working script)
            action = process_action(action, cfg.model_family)


            # Execute action
            obs, reward, done, info = env.step(action.tolist())

            # Record frame
            if cfg.save_videos:
                img = get_libero_image(obs)
                frames.append(img)

            # Check for task completion
            if done:
                success = True  # done=True means task completed successfully
                print(f"   ✅ Episode finished at step {step+1}: Success")
                break

        if not done:
            print(f"   ⏰ Episode timed out after {cfg.max_steps} steps")

        # Save video (only if video_suffix is provided)
        video_path = None
        if cfg.save_videos and video_suffix and (success or cfg.save_failure_videos):
            video_name = f"{skill_name}_{video_suffix}_{'success' if success else 'failure'}.mp4"
            video_path = video_dir / video_name
            video_writer = imageio.get_writer(str(video_path), fps=30)
            for img in frames:
                video_writer.append_data(img)
            video_writer.close()
            print(f"   🎥 Video saved: {video_path}")

        return {
            "success": success,
            "steps": step + 1 if done else cfg.max_steps,
            "video_path": video_path
        }

    except Exception as e:
        print(f"   ❌ Trial evaluation failed: {e}")
        traceback.print_exc()
        return {"success": False, "reason": f"evaluation_error: {e}"}


def evaluate_single_trial_direct(skill_name: str, initial_state: np.ndarray, instruction: str,
                                cfg: SimplifiedConfig, vla, processor, action_head, proprio_projector,
                                video_dir: Path, video_suffix: str, resize_size):
    """
    Evaluate a single trial: Initial state -> Direct VLA execution (no motion planning).
    """
    print(f"   🎯 Direct VLA execution from initial state")

    try:
        # Find task by name (following original script pattern)
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict["atomic_skills"]()
        task = None
        for task_id in range(task_suite.n_tasks):
            if task_suite.get_task(task_id).name == skill_name:
                task = task_suite.get_task(task_id)
                break

        if task is None:
            print(f"   ❌ Task not found: {skill_name}")
            return {"success": False, "reason": "task_not_found"}

        # Create environment using the task object
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res, horizon=10000)

        # Reset environment first (critical for proper initialization)
        env.reset()

        # Set initial state using proper Libero method
        print(f"   🔄 Setting initial simulation state...")
        # Use the full initial state format: [joint(7) + gripper(2) + sim_state]
        # The sim_state part (from index 9:) is used for complete environment reset
        sim_state = initial_state[9:]  # Extract simulation state

        # Use env.set_init_state() for proper environment reset (same as generation script)
        env.set_init_state(sim_state)

        # Get observations after setting state
        obs = env.env._get_observations()

        # Get current EE pose for logging
        current_ee_pos = obs['robot0_eef_pos'].copy()
        current_ee_quat = obs['robot0_eef_quat'].copy()
        current_ee_axisangle = quat2axisangle(current_ee_quat)
        print(f"   📍 Current EE: pos=[{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}] ori=[{np.degrees(current_ee_axisangle[0]):.1f}°, {np.degrees(current_ee_axisangle[1]):.1f}°, {np.degrees(current_ee_axisangle[2]):.1f}°]")

        # Start VLA evaluation directly from initial state
        print(f"   🤖 Starting VLA evaluation from initial state...")

        # Debug: Print EE pose and gripper state before VLA execution
        current_ee_pos = obs['robot0_eef_pos'].copy()
        current_ee_quat = obs['robot0_eef_quat'].copy()
        current_ee_axisangle = quat2axisangle(current_ee_quat)
        gripper_state = obs['robot0_gripper_qpos'].copy()
        print(f"   📍 EVAL 2 - EE pose: pos=[{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}] ori=[{np.degrees(current_ee_axisangle[0]):.1f}°, {np.degrees(current_ee_axisangle[1]):.1f}°, {np.degrees(current_ee_axisangle[2]):.1f}°] gripper=[{gripper_state[0]:.3f}, {gripper_state[1]:.3f}]")

        # Initialize video recording
        frames = []
        if cfg.save_videos:
            # Record initial frame
            img = get_libero_image(obs)
            frames.append(img)

        # Initialize action queue (like working script)
        action_queue = deque(maxlen=cfg.num_open_loop_steps)

        # VLA execution loop
        success = False

        for step in range(cfg.max_steps):
            # If action queue is empty, requery model (like working script)
            if len(action_queue) == 0:
                # Prepare observation for policy
                observation = prepare_observation_for_policy(obs, resize_size)
                # Get actions using VLA
                actions = get_action(
                    cfg,
                    vla,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=None,  # Not using diffusion
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action (like working script)
            action = process_action(action, cfg.model_family)


            # Execute action
            obs, reward, done, info = env.step(action.tolist())

            # Record frame
            if cfg.save_videos:
                img = get_libero_image(obs)
                frames.append(img)

            # Check for task completion
            if done:
                success = True  # done=True means task completed successfully
                print(f"   ✅ Episode finished at step {step+1}: Success")
                break

        if not done:
            print(f"   ⏰ Episode timed out after {cfg.max_steps} steps")

        # Save video (only if video_suffix is provided)
        video_path = None
        if cfg.save_videos and video_suffix and (success or cfg.save_failure_videos):
            video_name = f"{skill_name}_{video_suffix}_{'success' if success else 'failure'}.mp4"
            video_path = video_dir / video_name
            video_writer = imageio.get_writer(str(video_path), fps=30)
            for img in frames:
                video_writer.append_data(img)
            video_writer.close()
            print(f"   🎥 Video saved: {video_path}")

        return {
            "success": success,
            "steps": step + 1 if done else cfg.max_steps,
            "video_path": video_path
        }

    except Exception as e:
        print(f"   ❌ Trial evaluation failed: {e}")
        traceback.print_exc()
        return {"success": False, "reason": f"evaluation_error: {e}"}


def evaluate_skill_focused_mode(skill_name: str, cfg: SimplifiedConfig, vla, processor, action_head, proprio_projector, video_dir: Path, resize_size):
    """
    Evaluate a single skill in focused mode with multiple trials.

    Runs 10 trials for both shifted and original evaluations, saves 1 success + 1 failure
    video per evaluation type (4 videos total per skill), and returns success rates for both.

    Expected videos per skill:
    - {skill_name}_shifted_success.mp4 (if any success found)
    - {skill_name}_shifted_failure.mp4 (if any failure found)
    - {skill_name}_original_success.mp4 (if any success found)
    - {skill_name}_original_failure.mp4 (if any failure found)
    """
    print(f"\n{'='*80}")
    print(f"🎯 FOCUSED EVALUATION: {skill_name}")
    print(f"{'='*80}")

    try:
        # Load initial states for this skill
        initial_states = load_initial_states_for_skill(skill_name, cfg.atomic_demos_path)
        if len(initial_states) == 0:
            print(f"❌ No initial states found for skill: {skill_name}")
            return None

        # Extract skill instruction from skill name
        instruction = skill_name.replace("_", " ").replace("KITCHEN SCENE1", "").replace("KITCHEN SCENE2", "").replace("KITCHEN SCENE9", "").replace("KITCHEN SCENE10", "").strip()
        print(f"💬 Instruction: {instruction}")

        results = {
            "shifted": {"success_rate": 0.0, "successes": 0, "total_trials": cfg.num_trials_per_skill},
            "original": {"success_rate": 0.0, "successes": 0, "total_trials": cfg.num_trials_per_skill}
        }

        # Run shifted evaluation trials
        print(f"\n🔄 SHIFTED EVALUATION: {cfg.num_trials_per_skill} trials")
        shifted_successes = 0
        shifted_success_video_saved = False
        shifted_failure_video_saved = False

        for trial_idx in range(cfg.num_trials_per_skill):
            print(f"  Trial {trial_idx + 1}/{cfg.num_trials_per_skill}...")

            # Use different initial states for variety
            reference_initial_state = initial_states[trial_idx % len(initial_states)]

            # Determine if we should save video for this trial
            save_video_for_trial = False
            video_suffix = None

            trial_result = evaluate_single_trial_with_shift(
                skill_name, reference_initial_state, instruction, cfg,
                vla, processor, action_head, proprio_projector,
                video_dir, video_suffix=None,  # Don't save yet, decide after
                initial_states=initial_states, resize_size=resize_size
            )

            trial_success = trial_result and trial_result.get("success", False)
            if trial_success:
                shifted_successes += 1
                # Save first success video
                if not shifted_success_video_saved:
                    video_suffix = f"shifted_success"
                    save_video_for_trial = True
                    shifted_success_video_saved = True
            else:
                # Save first failure video
                if not shifted_failure_video_saved:
                    video_suffix = f"shifted_failure"
                    save_video_for_trial = True
                    shifted_failure_video_saved = True

            # Re-run trial with video saving if needed
            if save_video_for_trial:
                print(f"   🎥 Re-running trial to save {video_suffix} video...")
                trial_result = evaluate_single_trial_with_shift(
                    skill_name, reference_initial_state, instruction, cfg,
                    vla, processor, action_head, proprio_projector,
                    video_dir, video_suffix=video_suffix,
                    initial_states=initial_states, resize_size=resize_size
                )

        results["shifted"]["successes"] = shifted_successes
        results["shifted"]["success_rate"] = shifted_successes / cfg.num_trials_per_skill

        # Run original evaluation trials
        print(f"\n🔄 ORIGINAL EVALUATION: {cfg.num_trials_per_skill} trials")
        original_successes = 0
        original_success_video_saved = False
        original_failure_video_saved = False

        for trial_idx in range(cfg.num_trials_per_skill):
            print(f"  Trial {trial_idx + 1}/{cfg.num_trials_per_skill}...")

            # Use different initial states for variety
            reference_initial_state = initial_states[trial_idx % len(initial_states)]

            # Determine if we should save video for this trial
            save_video_for_trial = False
            video_suffix = None

            trial_result = evaluate_single_trial_direct(
                skill_name, reference_initial_state, instruction,
                cfg, vla, processor, action_head, proprio_projector,
                video_dir, video_suffix=None,  # Don't save yet, decide after
                resize_size=resize_size
            )

            trial_success = trial_result and trial_result.get("success", False)
            if trial_success:
                original_successes += 1
                # Save first success video
                if not original_success_video_saved:
                    video_suffix = f"original_success"
                    save_video_for_trial = True
                    original_success_video_saved = True
            else:
                # Save first failure video
                if not original_failure_video_saved:
                    video_suffix = f"original_failure"
                    save_video_for_trial = True
                    original_failure_video_saved = True

            # Re-run trial with video saving if needed
            if save_video_for_trial:
                print(f"   🎥 Re-running trial to save {video_suffix} video...")
                trial_result = evaluate_single_trial_direct(
                    skill_name, reference_initial_state, instruction,
                    cfg, vla, processor, action_head, proprio_projector,
                    video_dir, video_suffix=video_suffix,
                    resize_size=resize_size
                )

        results["original"]["successes"] = original_successes
        results["original"]["success_rate"] = original_successes / cfg.num_trials_per_skill

        # Print trial summary
        print(f"\n📊 TRIAL SUMMARY for {skill_name}:")
        print(f"   Shifted: {shifted_successes}/{cfg.num_trials_per_skill} ({results['shifted']['success_rate']*100:.1f}%)")
        print(f"   Original: {original_successes}/{cfg.num_trials_per_skill} ({results['original']['success_rate']*100:.1f}%)")

        return results

    except Exception as e:
        print(f"❌ Error evaluating skill {skill_name}: {e}")
        traceback.print_exc()
        return None


def run_simplified_evaluation(cfg: SimplifiedConfig):
    """Run simplified pose shift evaluation."""
    print(f"🚀 Starting Simplified Pose Shift Evaluation")
    print(f"   Model: {cfg.model_family}")
    print(f"   Position shift: ±{cfg.shift_position_std*100:.1f}cm")
    print(f"   Orientation shift: ±{np.degrees(cfg.shift_orientation_std):.1f}°")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/{cfg.exp_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Create video directories with shift parameters in name
    video_base_dir = Path("images_videos/atomic_local_vla_eval_videos_with_shifts")
    pos_std_cm = cfg.shift_position_std * 100
    ori_std_deg = np.degrees(cfg.shift_orientation_std)
    video_dir = video_base_dir / f"simplified_eval_{timestamp}_pos{pos_std_cm:.1f}cm_ori{ori_std_deg:.1f}deg"
    video_dir.mkdir(parents=True, exist_ok=True)
    print(f"🎥 Video directory: {video_dir}")

    # Load VLA model and processor
    vla, processor, action_head, proprio_projector = load_vla_model_and_processor(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Load atomic skills
    atomic_skills = load_atomic_skill_metadata(cfg)

    # Filter to target skill if specified
    if cfg.target_skill:
        if cfg.target_skill in atomic_skills:
            atomic_skills = [cfg.target_skill]
            print(f"🎯 Targeting specific skill: {cfg.target_skill}")
        else:
            print(f"❌ Target skill not found: {cfg.target_skill}")
            return

    # Handle focused evaluation mode for 3 specific skills
    if cfg.focused_eval_mode:
        focused_skills = [
            "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
            "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_pick",
            "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet"
        ]

        # Check which focused skills are available
        available_focused_skills = [skill for skill in focused_skills if skill in atomic_skills]
        if not available_focused_skills:
            print(f"❌ None of the focused skills found in available atomic skills")
            return

        atomic_skills = available_focused_skills
        print(f"🎯 Focused evaluation mode: {len(atomic_skills)} skills with 10 trials each")
        print(f"   Skills: {atomic_skills}")

        # Override num_trials_per_skill for focused mode
        cfg.num_trials_per_skill = 10

    # Evaluate each skill
    all_results = {}

    if cfg.focused_eval_mode:
        # Use focused evaluation with multiple trials and success rate tracking
        for skill_name in tqdm.tqdm(atomic_skills, desc="Evaluating skills"):
            skill_results = evaluate_skill_focused_mode(
                skill_name, cfg, vla, processor, action_head, proprio_projector, video_dir, resize_size
            )
            all_results[skill_name] = skill_results
    else:
        # Use standard single-trial evaluation
        for skill_name in tqdm.tqdm(atomic_skills, desc="Evaluating skills"):
            skill_results = evaluate_skill_with_pose_shift(
                skill_name, cfg, vla, processor, action_head, proprio_projector, video_dir, resize_size
            )
            all_results[skill_name] = skill_results

    # Save results summary
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print final summary
    print(f"\n{'='*80}")
    print(f"📊 FINAL EVALUATION SUMMARY")
    print(f"{'='*80}")

    if cfg.focused_eval_mode:
        # Focused mode: show detailed success rates for each skill
        print(f"🎯 FOCUSED EVALUATION RESULTS (10 trials per skill)")
        print(f"{'='*60}")

        for skill_name, results in all_results.items():
            if results:
                shifted_rate = results.get("shifted", {}).get("success_rate", 0.0) * 100
                original_rate = results.get("original", {}).get("success_rate", 0.0) * 100

                print(f"\n🔸 {skill_name}:")
                print(f"   Shifted:  {results['shifted']['successes']}/{results['shifted']['total_trials']} ({shifted_rate:.1f}%)")
                print(f"   Original: {results['original']['successes']}/{results['original']['total_trials']} ({original_rate:.1f}%)")
                print(f"   Gap:      {original_rate - shifted_rate:.1f}%")

        # Calculate overall averages
        total_skills = len([r for r in all_results.values() if r])
        if total_skills > 0:
            avg_shifted_rate = sum(r.get("shifted", {}).get("success_rate", 0.0) for r in all_results.values() if r) / total_skills * 100
            avg_original_rate = sum(r.get("original", {}).get("success_rate", 0.0) for r in all_results.values() if r) / total_skills * 100

            print(f"\n📊 OVERALL AVERAGES:")
            print(f"   Shifted:  {avg_shifted_rate:.1f}%")
            print(f"   Original: {avg_original_rate:.1f}%")
            print(f"   Gap:      {avg_original_rate - avg_shifted_rate:.1f}%")

    else:
        # Standard mode: single trial results
        shifted_successes = 0
        baseline_successes = 0
        total_skills = 0

        for skill_name, results in all_results.items():
            if results:
                total_skills += 1
                if results.get("shifted", {}).get("success", False):
                    shifted_successes += 1
                if results.get("family", {}).get("success", False):
                    baseline_successes += 1

        if total_skills > 0:
            shifted_rate = shifted_successes / total_skills * 100
            baseline_rate = baseline_successes / total_skills * 100

            print(f"Total skills evaluated: {total_skills}")
            print(f"Shifted pose success rate: {shifted_successes}/{total_skills} ({shifted_rate:.1f}%)")
            print(f"Baseline (direct) success rate: {baseline_successes}/{total_skills} ({baseline_rate:.1f}%)")
            print(f"Generalization gap: {baseline_rate - shifted_rate:.1f}%")

    print(f"📁 Results saved to: {results_path}")
    print(f"🎥 Videos saved to: {video_dir}")


if __name__ == "__main__":
    cfg = draccus.parse(SimplifiedConfig)
    run_simplified_evaluation(cfg)