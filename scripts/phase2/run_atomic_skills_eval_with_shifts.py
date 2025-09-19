#!/usr/bin/env python3
"""
run_atomic_skills_eval_with_shifts.py

Enhanced atomic skills evaluation with pose shifting capabilities.
Based on run_libero_eval_local.py but adapted for atomic skills evaluation
with robustness testing through initial state pose shifting.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
import h5py
import pickle
import traceback
import re
import imageio
from datetime import datetime

import wandb

# Add project paths (following Phase 1 pattern)
import os
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
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

# Import Phase 2 utilities
from scripts.phase2.utils.simple_pose_shift_utils import (
    generate_shifted_pose,
    find_family_pose,
    load_initial_states_for_skill,
    extract_ee_pose_from_initial_state
)

# Import motion planner from Phase 1
from scripts.phase2.utils.motion_planner import MotionPlanner


# Define task suite constants
class TaskSuite(str, Enum):
    ATOMIC_SKILLS = "atomic_skills"


# Define max steps for atomic skills
TASK_MAX_STEPS = {
    TaskSuite.ATOMIC_SKILLS: 400,  # Set reasonable max steps for atomic skills
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/atomic_skills/1.0.1/openvla-7b+libero_atomic_skills+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--200000_chkpt"     # Default atomic skills checkpoint

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (wrist cam only)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # Atomic skills environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "atomic_skills"           # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 20                    # Number of rollouts per task
    is_depth: bool = False                           # Whether to use depth images (permanently set to False)
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Phase 2: Simple pose shifting parameters
    #################################################################################################################
    enable_pose_shifts: bool = True                 # Enable pose shifting for robustness testing
    shift_position_std: float = 0.03                 # Position shift range (±3cm for each axis) - based on motion planner analysis
    shift_orientation_std: float = 0.785             # Orientation shift range (±45° total) - based on motion planner analysis
    shift_max_attempts: int = 100                    # Maximum attempts to find valid shifted pose
    
    # Debug mode for pose shifting
    debug_pose_shifts: bool = False                  # Debug mode: test 3 skills and show pose comparisons
    debug_specific_skill: str = ""                   # Debug specific skill (e.g., "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet")
    
    # Simple comparison mode
    simple_comparison: bool = False                   # Simple comparison mode: 1 trial per skill, save 2 videos (with/without shifts)
    test_mode: bool = False                           # Test mode: only test setup without loading full model

    # Motion planner parameters for pose shifting
    motion_planner_method: str = "cartesian_linear"  # Motion planner method for pose shifting
    motion_planner_steps: int = 400                  # Number of steps for cartesian linear interpolation
    motion_planner_pos_gain: float = 5.0             # Position gain for motion planner
    motion_planner_ori_gain: float = 5.0             # Orientation gain for motion planner

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "yygx"                       # Name of WandB entity
    wandb_project: str = "openvla-oft-atomic-skills-eval-with-shifts"   # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    #################################################################################################################
    # Atomic skills specific paths
    #################################################################################################################
    atomic_demos_path: str = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_local_demos/full_atomic_skills"
                                                      # Path to atomic skills demos and initial states
    atomic_combined_path: str = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_local_demos/full_atomic_skills_combined"
                                                      # Path to combined atomic skills demos
    bddl_files_path: str = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss/libero/libero/bddl_files/atomic_skills"
                                                      # Path to BDDL files for language extraction

    # fmt: on


def extract_language_from_bddl(bddl_file_path: str) -> str:
    """Extract the language description from a BDDL file."""
    try:
        with open(bddl_file_path, 'r') as f:
            content = f.read()
        
        # Look for (:language ...) pattern
        match = re.search(r'\(:language\s+(.*?)\)', content)
        if match:
            language = match.group(1).strip()
            return language
        else:
            print(f"Warning: No language found in {bddl_file_path}")
            return None
    except FileNotFoundError:
        print(f"Warning: BDDL file not found: {bddl_file_path}")
        return None
    except Exception as e:
        print(f"Error reading BDDL file {bddl_file_path}: {e}")
        return None


def get_combined_language_for_task(task_name: str, cfg: GenerateConfig) -> str:
    """
    Get the combined language for a task by extracting from BDDL file.
    This replaces the grab_language_from_filename approach with proper BDDL parsing.
    """
    # Construct BDDL file path
    bddl_file_path = os.path.join(cfg.bddl_files_path, f"{task_name}.bddl")
    
    # Extract language from BDDL file
    language = extract_language_from_bddl(bddl_file_path)
    
    if language:
        return language
    else:
        # Fallback to filename-based extraction if BDDL parsing fails
        print(f"Warning: Could not extract language from BDDL for {task_name}, using filename fallback")
        # Remove .bddl extension if present
        task_name_clean = task_name.replace('.bddl', '')
        # Extract language part after SCENE
        if "SCENE" in task_name_clean:
            parts = task_name_clean.split("SCENE")
            if len(parts) > 1:
                language_part = parts[1]
                # Remove scene number and underscore
                if language_part[0].isdigit():
                    language_part = language_part[1:]
                if language_part.startswith('_'):
                    language_part = language_part[1:]
                return language_part.replace('_', ' ')
        return task_name_clean.replace('_', ' ')


def apply_pose_shift_to_initial_state(initial_state: np.ndarray, 
                                    shifted_pose: tuple,
                                    motion_planner,
                                    env) -> Optional[np.ndarray]:
    """
    Apply pose shift to initial state using motion planner and IK.
    
    Args:
        initial_state: Original initial state array
        shifted_pose: (position, quaternion) - target shifted pose
        motion_planner: Motion planner instance
        env: Environment instance
        
    Returns:
        Modified initial state with shifted pose, or None if failed
    """
    try:
        # Set the original initial state
        env.set_init_state(initial_state[9:])  # Skip joint+gripper states
        
        # Get current EE pose
        dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])  # 6-DOF + gripper open
        obs, _, _, _ = env.step(dummy_action)
        current_ee_pos = obs['robot0_eef_pos']
        current_ee_quat = obs['robot0_eef_quat']
        
        print(f"   🎯 Shifting from current pose to target shifted pose...")
        print(f"      Current: pos=[{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}]")
        
        shifted_position, shifted_quaternion = shifted_pose
        print(f"      Target:  pos=[{shifted_position[0]:.3f}, {shifted_position[1]:.3f}, {shifted_position[2]:.3f}]")
        
        # Use motion planner to move to shifted pose with tight thresholds (1cm position, 10° orientation)
        position_threshold = 0.01  # 1cm
        orientation_threshold = np.radians(5)  # 5 degrees in radians
        move_success, final_obs = motion_planner.move_to_pose(shifted_position, shifted_quaternion,
                                                            position_threshold=position_threshold,
                                                            orientation_threshold=orientation_threshold)
        
        if not move_success:
            print(f"   ❌ Motion planner failed to reach shifted pose")
            return None
        
        # Get the final state after shifting
        final_state = env.sim.get_state().flatten()
        
        # Construct new initial state: [joints + gripper + final_sim_state]
        final_joint_states = final_obs['robot0_joint_pos']  # 7 elements
        final_gripper_states = final_obs['robot0_gripper_qpos']  # 2 elements
        
        shifted_initial_state = np.concatenate([
            final_joint_states, 
            final_gripper_states, 
            final_state
        ])
        
        print(f"   ✅ Successfully applied pose shift")
        return shifted_initial_state
        
    except Exception as e:
        print(f"   ❌ Error applying pose shift: {e}")
        return None


def debug_pose_shifts(cfg: GenerateConfig):
    """
    Debug mode: Test pose shifting on specific skill or 3 random skills and print pose comparisons.
    """
    import random  # Import at the top of the function
    
    if cfg.debug_specific_skill:
        print(f"🔍 DEBUG MODE: Testing pose shifting on specific skill: {cfg.debug_specific_skill}")
        selected_skills = [cfg.debug_specific_skill]
    else:
        print("🔍 DEBUG MODE: Testing pose shifting on 3 random skills")
        # Get available atomic skills (based on actual .init files found)
        atomic_skills = [
            "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
            "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
            "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet",
            "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
            "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_pick",
            "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_place",
            "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet_pick",
            "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate_pick",
            "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate_pick",
            "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate_pick",
            "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove_pick",
            "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove_pick",
            "KITCHEN_SCENE3_turn_on_the_stove"
        ]
        
        # Sample 3 random skills
        selected_skills = random.sample(atomic_skills, min(3, len(atomic_skills)))
    
    print("=" * 80)
    
    for i, skill_name in enumerate(selected_skills, 1):
        total_skills = len(selected_skills)
        print(f"\n🎯 SKILL {i}/{total_skills}: {skill_name}")
        print("-" * 60)
        
        try:
            # Load initial states for this skill
            skill_initial_states = load_initial_states_for_skill(skill_name, cfg.atomic_demos_path)
            print(f"📊 Loaded {len(skill_initial_states)} initial states for skill")

            # Select a random initial state
            selected_state = random.choice(skill_initial_states)
            
            # Get EE pose from observation space (not simulation state)
            print("🔧 Creating environment to extract observation space EE pose...")
            
            # Find task by name
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict[cfg.task_suite_name]()
            task = None
            for task_id in range(task_suite.n_tasks):
                if task_suite.get_task(task_id).name == skill_name:
                    task = task_suite.get_task(task_id)
                    break
            
            if task is None:
                print(f"❌ Task not found: {skill_name}")
                continue
                
            env, _ = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res, horizon=10000)
            
            # Set initial state
            env.sim.set_state_from_flattened(selected_state[9:])  # Skip joint(7) + gripper(2)
            env.sim.forward()
            
            # Run dummy action to get observations
            dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])  # 6-DOF + gripper open
            obs, _, _, _ = env.step(dummy_action)
            
            # Extract EE pose from observation space
            original_pos = obs['robot0_eef_pos'].copy()
            original_quat = obs['robot0_eef_quat'].copy()
            
            env.close()
            
            print(f"📍 ORIGINAL EE POSE:")
            print(f"   Position: [{original_pos[0]:.4f}, {original_pos[1]:.4f}, {original_pos[2]:.4f}]")
            print(f"   Quaternion: [{original_quat[0]:.4f}, {original_quat[1]:.4f}, {original_quat[2]:.4f}, {original_quat[3]:.4f}]")
            
            # Generate shifted pose using simple approach
            original_pose = (original_pos, original_quat)
            shifted_pose = generate_shifted_pose(
                original_pose,
                position_shift_range=cfg.shift_position_std,
                orientation_shift_range=cfg.shift_orientation_std
            )

            if shifted_pose:
                # Show shifted pose
                shifted_pos, shifted_quat = shifted_pose
                print(f"🎯 SHIFTED EE POSE:")
                print(f"   Position: [{shifted_pos[0]:.4f}, {shifted_pos[1]:.4f}, {shifted_pos[2]:.4f}]")
                print(f"   Quaternion: [{shifted_quat[0]:.4f}, {shifted_quat[1]:.4f}, {shifted_quat[2]:.4f}, {shifted_quat[3]:.4f}]")
                
                # Calculate differences
                pos_diff = np.linalg.norm(shifted_pos - original_pos)
                quat_diff = np.linalg.norm(shifted_quat - original_quat)
                
                # Calculate per-axis differences
                pos_diff_x = abs(shifted_pos[0] - original_pos[0])
                pos_diff_y = abs(shifted_pos[1] - original_pos[1])
                pos_diff_z = abs(shifted_pos[2] - original_pos[2])
                
                print(f"📏 DISTANCE ANALYSIS:")
                print(f"   Total position distance: {pos_diff:.4f}m")
                print(f"   X-axis difference: {pos_diff_x:.4f}m")
                print(f"   Y-axis difference: {pos_diff_y:.4f}m")
                print(f"   Z-axis difference: {pos_diff_z:.4f}m")
                print(f"   Quaternion distance: {quat_diff:.4f}")
                
                # Check if distance is reasonable
                if pos_diff > 0.5:
                    print(f"   ⚠️  WARNING: Large position shift ({pos_diff:.4f}m > 0.5m)")
                elif pos_diff > 0.2:
                    print(f"   ⚠️  MODERATE: Medium position shift ({pos_diff:.4f}m)")
                else:
                    print(f"   ✅ REASONABLE: Small position shift ({pos_diff:.4f}m)")
                
                # Show family pose
                family_pose = find_family_pose(shifted_pose, skill_initial_states)
                family_pos, family_quat = family_pose
                print(f"👨‍👩‍👧‍👦 FAMILY POSE:")
                print(f"   Position: [{family_pos[0]:.4f}, {family_pos[1]:.4f}, {family_pos[2]:.4f}]")
                print(f"   Quaternion: [{family_quat[0]:.4f}, {family_quat[1]:.4f}, {family_quat[2]:.4f}, {family_quat[3]:.4f}]")
                
            else:
                print("❌ No shifted pose generated")
                
        except Exception as e:
            print(f"❌ Error testing skill {skill_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("🔍 DEBUG MODE COMPLETED")
    return


def test_simple_comparison_setup(cfg: GenerateConfig):
    """Test simple comparison setup without loading full model."""
    print("🧪 Testing environment setup and pose generation...")
    
    # Initialize task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    
    # Get first task with initial state file
    task_id = None
    for test_task_id in range(min(5, task_suite.n_tasks)):  # Test first 5 tasks only
        task_name = task_suite.get_task(test_task_id).name
        init_path = os.path.join(cfg.atomic_demos_path, f"{task_name}.init")
        if os.path.exists(init_path):
            task_id = test_task_id
            break
    
    if task_id is None:
        print("❌ No tasks with initial state files found")
        return
    
    task = task_suite.get_task(task_id)
    task_name = task.name
    print(f"🎯 Testing with task: {task_name}")
    
    try:
        # Test environment creation
        print("🔧 Testing environment creation...")
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res, horizon=10000)
        print(f"✅ Environment created successfully")
        
        # Test initial state loading
        print("🔧 Testing initial state loading...")
        initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, None)
        if all_initial_states:
            print(f"✅ Loaded {len(all_initial_states)} initial states")
            
            # Test basic state setting
            print("🔧 Testing state setting...")
            selected_state = all_initial_states[0]
            obs = run_without_pose_shifts(env, selected_state)
            if obs is not None:
                print("✅ Basic state setting successful")
            else:
                print("❌ Basic state setting failed")
                
        else:
            print("❌ No initial states loaded")
            
        env.close()
        print("🧪 Setup test completed successfully!")
        
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        import traceback
        traceback.print_exc()


def simple_comparison_mode(cfg: GenerateConfig):
    """
    Simple comparison mode: Run num_trials_per_task trials per skill, save videos (with/without pose shifts).
    """
    import random  # Import random for selecting initial states

    print(f"🎬 SIMPLE COMPARISON MODE: Running {cfg.num_trials_per_task} trials per skill with/without pose shifts")
    print("=" * 80)
    
    # Early test mode - just verify environment loading and pose shifting
    if hasattr(cfg, 'test_mode') and cfg.test_mode:
        print("🧪 TEST MODE: Only testing environment setup and pose generation")
        test_simple_comparison_setup(cfg)
        return
    
    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = Path("images_videos/atomic_local_vla_eval_videos_with_shifts") / f"comparison_{timestamp}"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Saving comparison videos to: {comparison_dir}")
    
    # Initialize task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    
    # Get tasks with initial state files
    task_id_ls = []
    for task_id in range(task_suite.n_tasks):
        task_name = task_suite.get_task(task_id).name
        init_path = os.path.join(cfg.atomic_demos_path, f"{task_name}.init")
        if os.path.exists(init_path):
            task_id_ls.append(task_id)
    
    print(f"📊 Found {len(task_id_ls)} tasks with initial state files")
    
    # Load model and components
    print("🔧 Loading model and components...")
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    
    # Track overall results
    total_success_baseline = 0
    total_success_shifted = 0
    total_trials = 0

    for task_id in task_id_ls:
        task = task_suite.get_task(task_id)
        task_name = task.name
        print(f"\n🎯 Processing task: {task_name}")
        print("-" * 60)

        # Track results for this task
        task_success_baseline = 0
        task_success_shifted = 0

        try:
            # Load initial states once per task
            initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, None)
            if not all_initial_states:
                print(f"❌ No initial states found for {task_name}")
                continue

            # Run multiple trials for this task
            for trial_idx in range(cfg.num_trials_per_task):
                print(f"\n📋 Trial {trial_idx + 1}/{cfg.num_trials_per_task} for {task_name}")
                print("-" * 40)

                # Create environment for this trial
                env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res, horizon=10000)

                # Select random initial state for this trial
                selected_initial_state = random.choice(all_initial_states)

                # Run WITHOUT pose shifting
                print("📹 Running WITHOUT pose shifting...")
                obs = run_without_pose_shifts(env, selected_initial_state)
                success_baseline = False
                if obs is not None:
                    # Create temporary cfg with pose shifts disabled for baseline
                    baseline_cfg = cfg
                    baseline_cfg.enable_pose_shifts = False

                    success_baseline, replay_images_baseline, obs_list_baseline, actions_baseline = run_episode(
                        baseline_cfg, env, task_name, task_description, model, resize_size,
                        processor, action_head, proprio_projector, noisy_action_projector
                    )

                    # Save baseline video for first trial only
                    if trial_idx == 0:
                        baseline_video_path = comparison_dir / f"{task_name}_baseline.mp4"
                        import imageio
                        video_writer = imageio.get_writer(str(baseline_video_path), fps=30)
                        for obs in obs_list_baseline:
                            # Use agent view image for video
                            video_writer.append_data(obs['agentview_image'])
                        video_writer.close()
                        print(f"   💾 Saved baseline video: {baseline_video_path}")

                # Reset environment for second run
                env.close()
                env, _ = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res, horizon=10000)

                # Run WITH pose shifting
                print("📹 Running WITH pose shifting...")
                obs = run_with_pose_shifts(env, selected_initial_state, task_name, cfg)
                success_shifted = False
                if obs is not None:
                    # Create temporary cfg with pose shifts enabled for shifted run
                    shifted_cfg = cfg
                    shifted_cfg.enable_pose_shifts = True

                    success_shifted, replay_images_shifted, obs_list_shifted, actions_shifted = run_episode(
                        shifted_cfg, env, task_name, task_description, model, resize_size,
                        processor, action_head, proprio_projector, noisy_action_projector
                    )

                    # Save shifted video for first trial only
                    if trial_idx == 0:
                        shifted_video_path = comparison_dir / f"{task_name}_shifted.mp4"
                        video_writer = imageio.get_writer(str(shifted_video_path), fps=30)
                        for obs in obs_list_shifted:
                            # Use agent view image for video
                            video_writer.append_data(obs['agentview_image'])
                        video_writer.close()
                        print(f"   💾 Saved shifted video: {shifted_video_path}")

                # Track results for this trial
                if success_baseline:
                    task_success_baseline += 1
                    total_success_baseline += 1
                if success_shifted:
                    task_success_shifted += 1
                    total_success_shifted += 1
                total_trials += 1

                print(f"   📊 Trial {trial_idx + 1} results: Baseline={'✅' if success_baseline else '❌'}, Shifted={'✅' if success_shifted else '❌'}")

                # Close environment after trial
                env.close()

            # Print task summary
            baseline_rate = task_success_baseline / cfg.num_trials_per_task
            shifted_rate = task_success_shifted / cfg.num_trials_per_task
            print(f"\n📊 Task Summary for {task_name}:")
            print(f"   Baseline: {task_success_baseline}/{cfg.num_trials_per_task} ({baseline_rate:.1%})")
            print(f"   Shifted:  {task_success_shifted}/{cfg.num_trials_per_task} ({shifted_rate:.1%})")
            if baseline_rate > shifted_rate:
                print(f"   🎯 Good candidate for augmentation! (baseline works, shifts fail)")
            elif baseline_rate == shifted_rate:
                print(f"   ✅ Already robust to pose shifts")
            else:
                print(f"   ⚠️  Unexpected: shifted performs better than baseline")

        except Exception as e:
            print(f"❌ Error processing {task_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print overall summary
    overall_baseline_rate = total_success_baseline / total_trials if total_trials > 0 else 0
    overall_shifted_rate = total_success_shifted / total_trials if total_trials > 0 else 0

    print(f"\n🎬 SIMPLE COMPARISON COMPLETED")
    print("=" * 80)
    print(f"📊 Overall Results:")
    print(f"   Total tasks evaluated: {len(task_id_ls)}")
    print(f"   Total trials: {total_trials}")
    print(f"   Baseline success rate: {total_success_baseline}/{total_trials} ({overall_baseline_rate:.1%})")
    print(f"   Shifted success rate:  {total_success_shifted}/{total_trials} ({overall_shifted_rate:.1%})")
    print(f"   Robustness gap: {overall_baseline_rate - overall_shifted_rate:.1%}")
    print(f"📁 Videos saved in: {comparison_dir}")
    return


def run_without_pose_shifts(env, selected_initial_state):
    """Run without pose shifts - just set initial state."""
    sim_state = selected_initial_state[9:]  # Skip first 9 dims (joint + gripper states)
    obs = env.set_init_state(sim_state)
    return obs


def run_with_pose_shifts(env, selected_initial_state, task_name, cfg):
    """Run with pose shifts - apply pose shifting logic."""
    # Create a temporary cfg with pose shifts enabled for this specific run
    temp_cfg = cfg
    temp_cfg.enable_pose_shifts = True
    
    # Use the existing set_atomic_inits_with_shifts function
    obs = set_atomic_inits_with_shifts(temp_cfg, env, task_name)
    return obs


def set_atomic_inits_with_shifts(cfg, env, task_name):
    """
    Enhanced set_atomic_inits with optional pose shifting capability.
    
    Args:
        cfg: Configuration object
        env: Environment instance
        task_name: Task name
        
    Returns:
        Initial observation, or None if failed
    """
    init_path = os.path.join(cfg.atomic_demos_path, f"{task_name}.init")
    
    if not os.path.exists(init_path):
        print(f"Warning: Init file not found: {init_path}")
        return None
        
    with open(init_path, 'rb') as f:
        all_states = pickle.load(f)  # shape: [num_demos, 9+*]
    
    if len(all_states) == 0:
        return None
    if len(all_states) >= 20:
        all_states = all_states[:20]  # Limit to 20 inits for fair evaluation

    # Select random initial state
    idx = np.random.randint(len(all_states))
    selected_initial_state = all_states[idx]
    
    if not cfg.enable_pose_shifts:
        # Original behavior: just use the selected state
        sim_state = selected_initial_state[9:]  # Skip first 9 dims (joint + gripper states)
        obs = env.set_init_state(sim_state)
        return obs
    
    # Phase 2: Apply pose shifting
    print(f"🎯 Phase 2: Applying pose shifting for robustness testing...")
    print(f"📊 Shift parameters: pos_std={cfg.shift_position_std:.3f}m, ori_std={cfg.shift_orientation_std:.3f}rad")
    
    try:
        # Load all initial states for current skill for distribution analysis
        skill_initial_states = load_initial_states_for_skill(task_name, cfg.atomic_demos_path)
        
        # Get current EE pose from observation space (not simulation state)
        print("🔧 Getting current EE pose from observation space...")
        dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])  # 6-DOF + gripper open
        current_obs, _, _, _ = env.step(dummy_action)
        original_ee_pos = current_obs['robot0_eef_pos'].copy()
        original_ee_quat = current_obs['robot0_eef_quat'].copy()
        original_pose = (original_ee_pos, original_ee_quat)
        
        # Initialize motion planner
        motion_planner = MotionPlanner(
            env,
            method=cfg.motion_planner_method,
            num_steps=cfg.motion_planner_steps,
            pos_gain=cfg.motion_planner_pos_gain,
            ori_gain=cfg.motion_planner_ori_gain
        )
        
        # Try to find a valid shifted pose
        for attempt in range(cfg.shift_max_attempts):
            print(f"\n🔄 Shift attempt {attempt + 1}/{cfg.shift_max_attempts}")
            
            try:
                # Generate shifted pose using simple approach
                shifted_pose = generate_shifted_pose(
                    original_pose,
                    position_shift_range=cfg.shift_position_std,
                    orientation_shift_range=cfg.shift_orientation_std
                )

                if not shifted_pose:
                    print(f"   ❌ No shifted pose generated")
                    continue
                
                # Apply pose shift to initial state
                shifted_initial_state = apply_pose_shift_to_initial_state(
                    selected_initial_state, shifted_pose, motion_planner, env
                )
                
                if shifted_initial_state is not None:
                    # Success! Use the shifted initial state
                    print(f"   ✅ Successfully applied pose shift on attempt {attempt + 1}")
                    
                    # Set the final shifted state
                    sim_state = shifted_initial_state[9:]
                    obs = env.set_init_state(sim_state)
                    return obs
                    
            except Exception as e:
                print(f"   ❌ Shift attempt {attempt + 1} failed: {e}")
                continue
        
        # All shift attempts failed, fall back to original state
        print(f"⚠️  All {cfg.shift_max_attempts} shift attempts failed, using original initial state")
        sim_state = selected_initial_state[9:]
        obs = env.set_init_state(sim_state)
        return obs
        
    except Exception as e:
        print(f"❌ Error in pose shifting pipeline: {e}")
        print(f"   Falling back to original initial state")
        
        # Fall back to original behavior
        sim_state = selected_initial_state[9:]
        obs = env.set_init_state(sim_state)
        return obs


def get_eval_results_folder(cfg):
    """Create new folders for evaluation results."""
    # Use the specified path structure
    base = Path("/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/atomic_skills/1.0.1/eval_results_with_shifts")
    
    # Create timestamped subfolder for this evaluation run
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if cfg.enable_pose_shifts:
        folder = base / f"eval_run_shifts_{timestamp}"
    else:
        folder = base / f"eval_run_baseline_{timestamp}"
    
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_all_success_rates_to(results_folder, task_success_rates):
    """Save success info as JSON with BDDL filename as key and success rate as value."""
    # Save as pickle (original format)
    pickle_path = results_folder / "success_rates.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(task_success_rates, f)
    print(f"[✔] Saved success rates (pickle) to: {pickle_path}")
    
    # Save as JSON (BDDL filename -> success rate)
    json_path = results_folder / "success_rates.json"
    with open(json_path, "w") as f:
        json.dump(task_success_rates, f, indent=2)
    print(f"[✔] Saved success rates (JSON) to: {json_path}")
    
    # Also save a simple text summary
    summary_path = results_folder / "success_rates_summary.txt"
    with open(summary_path, "w") as f:
        f.write("BDDL Filename -> Success Rate\n")
        f.write("=" * 50 + "\n")
        for bddl_filename, success_rate in sorted(task_success_rates.items()):
            f.write(f"{bddl_filename}: {success_rate:.2%}\n")
    print(f"[✔] Saved success rates summary to: {summary_path}")


def maybe_save_rollout_obs(
        cfg,
        obs_list,
        actions,
        task_name: str,
        episode_idx: int,
        success: bool,
        results_folder: Path,
        success_count: int,
        failure_count: int,
        max_failures=3,
        max_successes=1,
):
    """Save rollouts for visualization."""
    if success and success_count >= max_successes:
        return success_count, failure_count
    if not success and failure_count >= max_failures:
        return success_count, failure_count

    # Extract fields - wrist cam only
    joint_states = np.array([o["robot0_joint_pos"] for o in obs_list])
    gripper_states = np.array([o["robot0_gripper_qpos"] for o in obs_list])
    wrist_imgs = np.array([o["robot0_eye_in_hand_image"] for o in obs_list])
    actions_np = np.array(actions)

    task_dir = results_folder / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    tag = "succ" if success else "fail"
    shift_tag = "shift" if cfg.enable_pose_shifts else "baseline"
    file_path = task_dir / f"{tag}_{shift_tag}_{episode_idx}.npz"

    np.savez_compressed(
        file_path,
        joint_states=joint_states,
        gripper_states=gripper_states,
        wrist_image=wrist_imgs,
        actions=actions_np,
        success=success,
        pose_shifts_enabled=cfg.enable_pose_shifts,
    )
    print(f"[💾] Saved rollout to: {file_path}")

    if success:
        success_count += 1
    else:
        failure_count += 1
    return success_count, failure_count


def save_atomic_skills_video(
    obs_list,
    task_name: str,
    episode_idx: int,
    success: bool,
    success_count: int,
    failure_count: int,
    cfg,
    max_successes: int = 1,
    max_failures: int = 1,
):
    """
    Save video for atomic skills evaluation with pose shifting.
    Records at most 1 success and 1 failure video per task.
    Creates both agent view and wrist cam videos.
    """
    # Check if we should record this video
    if success and success_count >= max_successes:
        return success_count, failure_count
    if not success and failure_count >= max_failures:
        return success_count, failure_count
    
    # Create video directory
    if cfg.enable_pose_shifts:
        video_dir = Path("images_videos/atomic_local_vla_eval_videos_with_shifts")
    else:
        video_dir = Path("images_videos/atomic_local_vla_eval_videos_baseline")
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract images
    agent_imgs = np.array([o["agentview_image"] for o in obs_list])
    wrist_imgs = np.array([o["robot0_eye_in_hand_image"] for o in obs_list])
    
    # Create filename
    tag = "success" if success else "failure"
    shift_tag = "shift" if cfg.enable_pose_shifts else "baseline"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save agent view video
    agent_video_path = video_dir / f"{task_name}_{tag}_{shift_tag}_ep{episode_idx}_{timestamp}_agentview.mp4"
    agent_writer = imageio.get_writer(str(agent_video_path), fps=30)
    for img in agent_imgs:
        agent_writer.append_data(img)
    agent_writer.close()
    
    # Save wrist cam video
    wrist_video_path = video_dir / f"{task_name}_{tag}_{shift_tag}_ep{episode_idx}_{timestamp}_wristcam.mp4"
    wrist_writer = imageio.get_writer(str(wrist_video_path), fps=30)
    for img in wrist_imgs:
        wrist_writer.append_data(img)
    wrist_writer.close()
    
    print(f"[🎥] Saved {tag} videos for {task_name} ({shift_tag}):")
    print(f"    Agent view: {agent_video_path}")
    print(f"    Wrist cam:  {wrist_video_path}")
    
    # Update counters
    if success:
        success_count += 1
    else:
        failure_count += 1
    
    return success_count, failure_count


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"
    
    # Validate Phase 2 parameters
    if cfg.enable_pose_shifts:
        assert cfg.shift_position_std > 0, "shift_position_std must be positive"
        assert cfg.shift_orientation_std > 0, "shift_orientation_std must be positive"
        assert cfg.shift_max_attempts > 0, "shift_max_attempts must be positive"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)
    print(f"[INFO] model.norm_stats: {model.norm_stats}; cfg.task_suite_name: {cfg.task_suite_name};")

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # For atomic_skills, the model was trained with "libero_atomic_skills" key
    if unnorm_key == "atomic_skills":
        unnorm_key = "libero_atomic_skills"

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.enable_pose_shifts:
        run_id += "--with_pose_shifts"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # For atomic skills, we use our own atomic init files
    task = task_suite.get_task(task_id)
    task_name = task.name
    
    try:
        # Load initial states from atomic init file
        all_initial_states = load_initial_states_for_skill(task_name, cfg.atomic_demos_path)
        if all_initial_states:
            log_message(f"Loaded {len(all_initial_states)} initial states for {task_name}", log_file)
            return all_initial_states, all_initial_states
        else:
            log_message(f"No initial states found for {task_name}", log_file)
            return None, None
    except Exception as e:
        log_message(f"Error loading initial states for {task_name}: {e}", log_file)
        return None, None


def prepare_observation(obs, resize_size, cfg):
    """Prepare observation for policy input - wrist cam only."""
    # Get preprocessed images - wrist cam only
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    img = wrist_img
    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_name: str,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set atomic skills initial states with optional pose shifting
    obs = set_atomic_inits_with_shifts(cfg, env, task_name)
    if obs is None:
        print(f"[WARNING] Task {task_name} doesn't have corresponding atomic init file!")
        return None, None, None, None  # this means this task doesn't contain atomic init

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
               "{NUM_ACTIONS_CHUNK} constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    all_obs, all_actions = [], []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    try:
        while t < max_steps:
            all_obs.append(obs)

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size, cfg)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())

            all_actions.append(action.tolist())

            if done:
                success = True
                break
            t += 1

    except Exception as e:
        error_details = traceback.format_exc()
        log_message(f"Episode error:\n{error_details}", log_file)
        exit(1)

    return success, replay_images, all_obs, all_actions


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    results_folder,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    saved_successes, saved_failures = 0, 0
    video_successes, video_failures = 0, 0

    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    # Use large horizon to accommodate motion planning for pose shifts
    horizon = 2000 if cfg.enable_pose_shifts else None
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res, horizon=horizon)
    task_name = task.name

    # Get combined language for the task
    combined_language = get_combined_language_for_task(task_name, cfg)
    log_message(f"Using combined language: '{combined_language}'", log_file)
    
    # Log pose shifting status
    if cfg.enable_pose_shifts:
        log_message(f"🎯 Phase 2: Pose shifting ENABLED", log_file)
        log_message(f"   Position std: {cfg.shift_position_std:.3f}m", log_file)
        log_message(f"   Orientation std: {cfg.shift_orientation_std:.3f}rad", log_file)
        log_message(f"   Max attempts: {cfg.shift_max_attempts}", log_file)
    else:
        log_message(f"📊 Baseline: Pose shifting DISABLED", log_file)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {combined_language}", log_file)

        # For atomic skills, we don't use initial states from Libero
        # The set_atomic_inits_with_shifts function handles the initialization
        initial_state = None

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images, obs_list, actions = run_episode(
            cfg,
            env,
            task_name,
            combined_language,  # Use combined language instead of task_description
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
        )

        # Jump the task if it doesn't contain atomic init
        if success is None and replay_images is None:
            task_episodes = None
            break

        # Save rollout for rerun
        saved_successes, saved_failures = maybe_save_rollout_obs(
            cfg,
            obs_list,
            actions,
            task_name,
            episode_idx,
            success,
            results_folder,
            saved_successes,
            saved_failures,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save atomic skills videos (max 1 success, 1 failure per task)
        video_successes, video_failures = save_atomic_skills_video(
            obs_list,
            task_name,
            episode_idx,
            success,
            video_successes,
            video_failures,
            cfg,
            max_successes=1,
            max_failures=1,
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    if task_episodes is None:
        task_success_rate = -1
    else:
        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb_data = {
            f"success_rate/{combined_language}": task_success_rate,
            f"num_episodes/{combined_language}": task_episodes,
        }
        if cfg.enable_pose_shifts:
            wandb_data[f"pose_shifts_enabled/{combined_language}"] = True
        wandb.log(wandb_data)

    return total_episodes, total_successes, task_success_rate


@draccus.wrap()
def eval_atomic_skills(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on atomic skills benchmark tasks."""

    # Check for debug mode
    if cfg.debug_pose_shifts:
        debug_pose_shifts(cfg)
        return 0.0  # Exit early in debug mode
    
    # Check for simple comparison mode
    if cfg.simple_comparison:
        simple_comparison_mode(cfg)
        return 0.0  # Exit early in comparison mode

    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize atomic skills task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)
    
    # Log Phase 2 configuration
    if cfg.enable_pose_shifts:
        log_message(f"🎯 Phase 2: Pose shifting enabled", log_file)
        log_message(f"   Position std: {cfg.shift_position_std:.3f}m", log_file)
        log_message(f"   Orientation std: {cfg.shift_orientation_std:.3f}rad", log_file)
        log_message(f"   Max attempts: {cfg.shift_max_attempts}", log_file)
        log_message(f"   Motion planner: {cfg.motion_planner_method}", log_file)
    else:
        log_message(f"📊 Baseline evaluation: Pose shifting disabled", log_file)

    # Pre-check which tasks have atomic init files
    task_id_ls = []
    for task_id in range(num_tasks):
        task_name = task_suite.get_task(task_id).name
        init_path = os.path.join(cfg.atomic_demos_path, f"{task_name}.init")
        if not os.path.exists(init_path):
            print(f"[ERROR] {task_name} does not have corresponding atomic init file.")
        else:
            task_id_ls.append(task_id)

    results_folder = get_eval_results_folder(cfg)
    task_success_rates = {}

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_id_ls):
        total_episodes, total_successes, task_success_rate = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            results_folder,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
        )

        task_name = task_suite.get_task(task_id).name
        task_success_rates[task_name] = task_success_rate
        
        # Save progress incrementally
        save_all_success_rates_to(results_folder, task_success_rates)

    # Save success info
    save_all_success_rates_to(results_folder, task_success_rates)

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    
    if cfg.enable_pose_shifts:
        log_message(f"🎯 Phase 2: Evaluation completed with pose shifting", log_file)
    else:
        log_message(f"📊 Baseline evaluation completed", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb_data = {
            "success_rate/total": final_success_rate,
            "num_episodes/total": total_episodes,
            "pose_shifts_enabled": cfg.enable_pose_shifts,
        }
        wandb.log(wandb_data)
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_atomic_skills()