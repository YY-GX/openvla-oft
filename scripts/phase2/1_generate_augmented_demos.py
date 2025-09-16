#!/usr/bin/env python3
"""
1_generate_augmented_demos.py

Generate augmented demonstrations with pose shifting for robust VLA training.
Based on generate_local_demos_fixed_single_replay.py but enhanced with Phase 2 pose shifting.

This script creates additional training data by:
1. Loading original demonstrations
2. Generating shifted poses outside initial state distribution  
3. Using motion planner to navigate from shifted poses to family poses
4. Saving successful augmented demonstrations with metadata
"""

import argparse
import json
import os
import glob
import numpy as np
# Handle h5py import issue
try:
    import h5py
except ImportError as e:
    print(f"h5py import error: {e}")
    print("Try: conda install -c conda-forge h5py --force-reinstall")
    exit(1)
import pickle
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from datetime import datetime

# Add project paths (following Phase 1 pattern)
import sys
import os
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env

# Import Phase 2 utilities
from scripts.phase2.utils.simple_pose_shift_utils import (
    generate_shifted_pose,
    find_family_pose,
    load_initial_states_for_skill,
    extract_ee_pose_from_initial_state,
    collect_step_data
)

# Import motion planner from Phase 1
from scripts.phase2.utils.motion_planner import MotionPlanner


# Configuration parameters
DEFAULT_POSITION_SHIFT_RANGE = 0.05  # ±5cm
DEFAULT_ORIENTATION_SHIFT_RANGE_DEG = 60  # ±60 degrees
MOTION_PLANNER_METHOD = "cartesian_linear"
DEFAULT_MOTION_PLANNER_STEPS = 50 # old: 400
DEFAULT_MOTION_PLANNER_POS_GAIN = 5.0
DEFAULT_MOTION_PLANNER_ORI_GAIN = 5.0
DEFAULT_POSITION_THRESHOLD = 0.005  # 0.5cm
DEFAULT_ORIENTATION_THRESHOLD_DEG = 5  # 5 degrees

# Global logging configuration
VERBOSE = False
DEBUG_MODE = False
DEBUG_SKILL_MODE = False

def log_always(message: str):
    """Log critical information that should always be shown."""
    print(message)

def log_info(message: str):
    """Log important information (shown in normal and verbose modes)."""
    print(message)

def log_verbose(message: str):
    """Log detailed information only when verbose mode is enabled."""
    if VERBOSE:
        print(message)

def log_debug(message: str):
    """Log debug information only in debug modes."""
    if DEBUG_MODE or DEBUG_SKILL_MODE or VERBOSE:
        print(f"🐛 {message}")

def log_success(message: str, verbose_only: bool = False):
    """Log success messages with ✅ prefix."""
    if verbose_only:
        log_verbose(f"✅ {message}")
    else:
        print(f"✅ {message}")

def log_error(message: str):
    """Log error messages with ❌ prefix."""
    print(f"❌ {message}")

def log_warning(message: str, verbose_only: bool = False):
    """Log warning messages with ⚠️ prefix."""
    if verbose_only:
        log_verbose(f"⚠️ {message}")
    else:
        print(f"⚠️ {message}")

def log_mp_result(success: bool, pos_err: float, ori_err: float, target_pos: np.ndarray, current_pos: np.ndarray):
    """Log motion planning results with key metrics (always shown)."""
    status = "SUCCESS" if success else "FAILED"
    log_info(f"   Motion Planning {status}: pos_err={pos_err*1000:.1f}mm, ori_err={ori_err:.1f}°")
    log_verbose(f"   Target EE: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    log_verbose(f"   Current EE: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")


def categorize_skill_file(bddl_filename: str) -> str:
    """
    Categorize a BDDL file based on its filename suffix.
    
    Args:
        bddl_filename: Name of the BDDL file
        
    Returns:
        Category: 'pick', 'place', or 'atomic'
    """
    if bddl_filename.endswith('_pick.bddl'):
        return 'pick'
    elif bddl_filename.endswith('_place.bddl'):
        return 'place'
    else:
        return 'atomic'


def find_original_bddl_from_mapping(atomic_bddl_path: str, cat_split_map: Dict) -> Optional[str]:
    """
    Find the original BDDL file path for a given atomic skill using the mapping.
    
    Args:
        atomic_bddl_path: Path to the atomic skill BDDL file
        cat_split_map: Loaded cat_split_map.json data
        
    Returns:
        Original BDDL file path or None if not found
    """
    # Check if this atomic skill path is in the values of the mapping
    for original_path, atomic_paths in cat_split_map.items():
        if atomic_bddl_path in atomic_paths:
            return original_path
    return None


def get_demo_filename_from_bddl(original_bddl_path: str) -> str:
    """
    Convert a BDDL file path to its corresponding demo HDF5 filename.
    
    Args:
        original_bddl_path: Path to the original BDDL file
        
    Returns:
        Demo HDF5 filename
    """
    # Extract the base name without path and .bddl extension
    base_name = os.path.basename(original_bddl_path).replace('.bddl', '')
    return f"{base_name}_demo.hdf5"


def discover_atomic_skills(atomic_skills_dir: str, debug: bool = False) -> List[Tuple[str, str]]:
    """
    Discover and categorize all atomic skill BDDL files.

    Args:
        atomic_skills_dir: Directory containing atomic skill BDDL files
        debug: If True, return specific debug tasks

    Returns:
        List of (file_path, category) tuples
    """
    atomic_skills = []

    # Get all BDDL files in the main atomic_skills directory (not subdirectories)
    bddl_pattern = os.path.join(atomic_skills_dir, "*.bddl")
    bddl_files = sorted(glob.glob(bddl_pattern))

    for bddl_file in bddl_files:
        category = categorize_skill_file(os.path.basename(bddl_file))
        atomic_skills.append((bddl_file, category))

    if debug:
        # Return specific debug tasks
        debug_task_names = [
            "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet.bddl",  # atomic
            "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_pick.bddl",   # pick
            "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_place.bddl"   # place
        ]

        debug_skills = []
        for file_path, category in atomic_skills:
            bddl_filename = os.path.basename(file_path)
            if bddl_filename in debug_task_names:
                debug_skills.append((file_path, category))

        log_debug(f"Debug mode: Selected specific tasks:")
        for file_path, category in debug_skills:
            log_debug(f"   - {os.path.basename(file_path)} ({category})")

        return debug_skills

    return atomic_skills


def map_to_source_demos(atomic_skills: List[Tuple[str, str]], 
                       cat_split_map: Dict, 
                       raw_demo_dir: str) -> List[Dict]:
    """
    Map atomic skills to their source demonstration files.
    
    Args:
        atomic_skills: List of (file_path, category) tuples
        cat_split_map: Loaded cat_split_map.json data
        raw_demo_dir: Directory containing original demo HDF5 files
        
    Returns:
        List of dictionaries with mapping information
    """
    mappings = []
    
    for atomic_bddl_path, category in atomic_skills:
        # Find the original BDDL file for this atomic skill
        original_bddl_path = find_original_bddl_from_mapping(atomic_bddl_path, cat_split_map)
        
        if original_bddl_path is None:
            log_warning(f"No mapping found for {atomic_bddl_path}", verbose_only=True)
            continue

        # Get the demo filename
        demo_filename = get_demo_filename_from_bddl(original_bddl_path)
        demo_full_path = os.path.join(raw_demo_dir, demo_filename)

        # Check if the demo file exists
        if not os.path.exists(demo_full_path):
            log_warning(f"Demo file not found: {demo_full_path}", verbose_only=True)
            continue
        
        mapping_info = {
            'atomic_bddl_path': atomic_bddl_path,
            'category': category,
            'original_bddl_path': original_bddl_path,
            'demo_filename': demo_filename,
            'demo_full_path': demo_full_path,
            'skill_name': os.path.basename(atomic_bddl_path).replace('.bddl', '')
        }
        mappings.append(mapping_info)
    
    return mappings


def load_hdf5_demo_data(demo_file_path: str) -> Dict:
    """Load demonstration data from HDF5 file."""
    def recursively_extract(group):
        result = {}
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Dataset):
                result[key] = item[()]
            elif isinstance(item, h5py.Group):
                result[key] = recursively_extract(item)
        return result

    with h5py.File(demo_file_path, 'r') as file:
        return recursively_extract(file)


def detect_trigger_timestep(actions: np.ndarray, env, states: np.ndarray) -> Optional[int]:
    """
    Detect trigger timestep for pick skills (contact or gripper closing).

    This function implements the trigger detection logic for pick skills by:
    1. First attempting gripper closing detection (gripper command: negative -> positive)
    2. Falling back to end-effector contact detection if no gripper closing found

    Args:
        actions: Action array from demonstration (timesteps, 7)
        env: Libero environment for contact detection
        states: Simulation states for replay during contact detection

    Returns:
        Timestep of trigger event, or None if not found
    """
    # First try gripper closing detection
    gripper_commands = actions[:, -1]  # Last dimension is gripper
    for i in range(1, len(gripper_commands)):
        if gripper_commands[i - 1] < 0 and gripper_commands[i] > 0:
            print(f"Found gripper closing at timestep {i}")
            return i

    # Fallback to contact detection
    print("No gripper closing found, trying contact detection...")
    EE_GEOM_NAMES = [
        "robot0_eef", "robot0_gripper0_finger0", "robot0_gripper0_finger1",
        "gripper0_hand_collision", "gripper0_finger0_collision", "gripper0_finger1_collision"
    ]

    for t, sim_state in enumerate(states):
        env.set_init_state(sim_state)
        for j in range(env.sim.data.ncon):
            contact = env.sim.data.contact[j]
            g1 = env.sim.model.geom_id2name(contact.geom1)
            g2 = env.sim.model.geom_id2name(contact.geom2)
            if g1 in EE_GEOM_NAMES or g2 in EE_GEOM_NAMES:
                print(f"Found EE contact at timestep {t}")
                return t

    print("No trigger found")
    return None


def detect_trigger_timestep_contact_only(actions: np.ndarray, env, states: np.ndarray) -> Optional[int]:
    """
    Detect trigger timestep for atomic skills using proper replay method.

    FIXED: Previously used env.set_init_state() which fails to properly update
    contact detection. Now uses env.step() replay method for accurate contact detection.

    Args:
        actions: Action array from demonstration
        env: Libero environment for contact detection
        states: Simulation states (unused in fixed version)

    Returns:
        Timestep of the trigger event, or None if not found.
    """
    print("Detecting contact trigger for atomic skill...")
    EE_GEOM_NAMES = [
        "robot0_eef", "robot0_gripper0_finger0", "robot0_gripper0_finger1",
        "gripper0_hand_collision", "gripper0_finger0_collision", "gripper0_finger1_collision"
    ]

    # Use proper replay method instead of set_init_state
    env.reset()
    for t, action in enumerate(actions):
        # Check contacts BEFORE taking action
        for j in range(env.sim.data.ncon):
            contact = env.sim.data.contact[j]
            g1 = env.sim.model.geom_id2name(contact.geom1)
            g2 = env.sim.model.geom_id2name(contact.geom2)
            if g1 in EE_GEOM_NAMES or g2 in EE_GEOM_NAMES:
                print(f"Found EE contact at timestep {t}")
                return t

        # Take the action
        obs, reward, done, _ = env.step(action)
        if done:
            break

    print("No contact trigger found")
    return None




def convert_steps_to_demo(step_data_list: List[Dict]) -> Dict:
    """Convert collected step data to demo format for HDF5 saving."""
    # Safety check: ensure step_data_list is a list of dictionaries
    if not isinstance(step_data_list, list):
        raise TypeError(f"step_data_list must be a list, got {type(step_data_list)}")
    
    if len(step_data_list) == 0:
        raise ValueError("step_data_list cannot be empty")
    
    # Check if all elements are dictionaries
    for i, step in enumerate(step_data_list):
        if not isinstance(step, dict):
            raise TypeError(f"step_data_list[{i}] must be a dict, got {type(step)}: {step}")
    
    demo_data = {}
    demo_data['actions'] = np.array([step['actions'] for step in step_data_list])
    demo_data['dones'] = np.array([step['dones'] for step in step_data_list], dtype=np.uint8)
    demo_data['rewards'] = np.array([step['rewards'] for step in step_data_list], dtype=np.uint8)
    demo_data['robot_states'] = np.array([step['robot_states'] for step in step_data_list])
    demo_data['states'] = np.array([step['states'] for step in step_data_list])
    
    # Convert observations
    demo_data['obs'] = {}
    obs_keys = step_data_list[0]['obs'].keys()
    for key in obs_keys:
        obs_data = np.array([step['obs'][key] for step in step_data_list])
        demo_data['obs'][key] = obs_data
    
    return demo_data


def save_multiple_demos_to_hdf5(demos_data: List[Dict], output_file_path: str):
    """Save multiple demo data to HDF5 file with correct structure."""
    with h5py.File(output_file_path, 'w') as h5file:
        data_group = h5file.create_group('data')
        
        for demo_idx, demo_data in enumerate(demos_data):
            demo_group = data_group.create_group(f'demo_{demo_idx}')
            
            # Core datasets
            demo_group.create_dataset('actions', data=demo_data['actions'])
            demo_group.create_dataset('dones', data=demo_data['dones'])
            demo_group.create_dataset('rewards', data=demo_data['rewards'])
            demo_group.create_dataset('robot_states', data=demo_data['robot_states'])
            demo_group.create_dataset('states', data=demo_data['states'])
            
            # Observation group
            obs_group = demo_group.create_group('obs')
            for obs_key, obs_data in demo_data['obs'].items():
                obs_group.create_dataset(obs_key, data=obs_data)


def extract_and_save_initial_states_original(original_demos: List[List[Dict]], init_file_path: str, init_offset: int):
    """Extract and save initial states from original demos (same as old script)."""
    all_states = []

    for demo_steps in original_demos:
        if len(demo_steps) <= init_offset:
            print(f"Warning: Not enough steps ({len(demo_steps)}) for init_offset {init_offset}")
            init_idx = 0
        else:
            init_idx = init_offset

        init_step = demo_steps[init_idx]

        # Combine joint, gripper, and extra states (following reference script pattern)
        joint_states = init_step['obs']['joint_states']
        gripper_states = init_step['obs']['gripper_states']
        extra_state = init_step['states']

        initial_state = np.concatenate([joint_states, gripper_states, extra_state])
        all_states.append(initial_state)

    with open(init_file_path, 'wb') as f:
        pickle.dump(all_states, f)


def create_empty_failure_video(skill_name: str, demo_type: str, demo_key: str, output_dir: str, fps: int = 30):
    """
    Create an empty failure video with a text overlay indicating the failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple black frame with failure text
    height, width = 480, 640  # Standard video dimensions
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add text overlay
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"FAILURE: {demo_key}"
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    cv2.putText(black_frame, text, (text_x, text_y), font, 1, (0, 0, 255), 2)
    
    # Create video filenames with _failure suffix
    agentview_path = os.path.join(output_dir, f"{skill_name}_{demo_type}_{demo_key}_failure_agentview.mp4")
    wrist_path = os.path.join(output_dir, f"{skill_name}_{demo_type}_{demo_key}_failure_wrist.mp4")
    
    # Save the same frame for both cameras
    save_video_frames([black_frame], agentview_path, fps)
    save_video_frames([black_frame], wrist_path, fps)
    
    print(f"🎬 Saved failure videos for {demo_key} ({demo_type}):")
    print(f"   AgentView: {agentview_path}")
    print(f"   Wrist: {wrist_path}")


def save_debug_videos(step_data_list: List[Dict], skill_name: str, demo_type: str, demo_key: str,
                     output_dir: str = "./debug_videos", fps: int = 30, is_failure: bool = False):
    """
    Save debug videos for both camera views from step data.

    Args:
        step_data_list: List of step data dictionaries
        skill_name: Name of the skill being processed
        demo_type: Type of demo ("original" or "augmented")
        demo_key: Demo identifier
        output_dir: Directory to save videos
        fps: Frames per second for video
    """
    if not step_data_list:
        print(f"⚠️  No step data to create videos for {demo_key} - creating empty failure video")
        # Create empty failure video
        create_empty_failure_video(skill_name, demo_type, demo_key, output_dir, fps)
        return

    os.makedirs(output_dir, exist_ok=True)

    # Extract frames from step data
    agentview_frames = []
    wrist_frames = []

    for step_data in step_data_list:
        if 'obs' in step_data:
            agentview_frames.append(step_data['obs']['agentview_rgb'])
            wrist_frames.append(step_data['obs']['eye_in_hand_rgb'])

    # Save agentview video
    failure_suffix = "_failure" if is_failure else ""
    agentview_filename = f"{skill_name}_{demo_type}_{demo_key}{failure_suffix}_agentview.mp4"
    agentview_path = os.path.join(output_dir, agentview_filename)
    save_video_frames(agentview_frames, agentview_path, fps)

    # Save wrist camera video
    wrist_filename = f"{skill_name}_{demo_type}_{demo_key}{failure_suffix}_wrist.mp4"
    wrist_path = os.path.join(output_dir, wrist_filename)
    save_video_frames(wrist_frames, wrist_path, fps)

    print(f"🎬 Saved debug videos for {demo_key} ({demo_type}):")
    print(f"   Agentview: {agentview_path}")
    print(f"   Wrist cam: {wrist_path}")


def save_video_frames(frames: List[np.ndarray], output_path: str, fps: int = 30):
    """
    Save a list of RGB frames as MP4 video.

    Args:
        frames: List of RGB frames (H, W, 3)
        output_path: Path for output video file
        fps: Frames per second
    """
    if not frames:
        print(f"⚠️  No frames to save for {output_path}")
        return

    try:
        # Get frame dimensions
        height, width = frames[0].shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not video_writer.isOpened():
            print(f"❌ Failed to open video writer for {output_path}")
            return

        # Write frames
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()
        print(f"✅ Video saved: {output_path} ({len(frames)} frames, {width}x{height})")

    except Exception as e:
        print(f"❌ Error saving video {output_path}: {e}")


def save_initial_images_debug(collected_steps: List[Dict], skill_name: str, demo_type: str, demo_key: str,
                             init_offset: int, output_dir: str):
    """
    Save initial images (at init_offset) for debugging purposes in debug mode.

    Args:
        collected_steps: List of step data from a successful demo
        skill_name: Name of the skill being processed
        demo_type: Type of demo ("original" or "augmented")
        demo_key: Demo identifier
        init_offset: The init_offset used (for filename)
        output_dir: Base output directory
    """
    if not collected_steps:
        log_verbose(f"No steps to extract initial images from for {demo_key}")
        return

    # Create initial images subdirectory
    init_images_dir = os.path.join(output_dir, "initial_images")
    os.makedirs(init_images_dir, exist_ok=True)

    # Determine which step to use as "initial" based on init_offset
    # For collected demos, the init_offset-th step represents our initial state
    if len(collected_steps) <= init_offset:
        # If demo is shorter than init_offset, use the first step
        init_step_idx = 0
        log_verbose(f"Demo shorter than init_offset ({len(collected_steps)} <= {init_offset}), using first step")
    else:
        init_step_idx = init_offset

    init_step = collected_steps[init_step_idx]

    if 'obs' not in init_step:
        log_verbose(f"No observation data in initial step for {demo_key}")
        return

    obs = init_step['obs']

    # Save agentview image
    if 'agentview_rgb' in obs:
        agentview_img = obs['agentview_rgb']
        agentview_filename = f"{skill_name}_{demo_type}_{demo_key}_init{init_offset}_agentview.png"
        agentview_path = os.path.join(init_images_dir, agentview_filename)

        try:
            # Convert RGB to BGR for OpenCV
            agentview_bgr = cv2.cvtColor(agentview_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(agentview_path, agentview_bgr)
            log_verbose(f"💾 Saved agentview initial image: {agentview_path}")
        except Exception as e:
            log_verbose(f"❌ Failed to save agentview image: {e}")

    # Save wrist camera image
    if 'eye_in_hand_rgb' in obs:
        wrist_img = obs['eye_in_hand_rgb']
        wrist_filename = f"{skill_name}_{demo_type}_{demo_key}_init{init_offset}_wrist.png"
        wrist_path = os.path.join(init_images_dir, wrist_filename)

        try:
            # Convert RGB to BGR for OpenCV
            wrist_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(wrist_path, wrist_bgr)
            log_verbose(f"💾 Saved wrist initial image: {wrist_path}")
        except Exception as e:
            log_verbose(f"❌ Failed to save wrist image: {e}")

    log_debug(f"Initial images saved for {demo_key} ({demo_type}) with init_offset={init_offset}")


def extract_and_save_initial_states_augmented(augmented_demos: List[List[Dict]], init_file_path: str, init_offset: int):
    """Extract and save initial states from augmented demos (shifted pose as initial state)."""
    all_states = []

    # Process augmented demos: initial state is the actual shifted pose (first step, index 0)
    for demo_steps in augmented_demos:
        if len(demo_steps) == 0:
            print(f"Warning: Empty augmented demo")
            continue

        init_step = demo_steps[0]  # First step contains the shifted pose state

        # Combine joint, gripper, and extra states (following reference script pattern)
        joint_states = init_step['obs']['joint_states']
        gripper_states = init_step['obs']['gripper_states']
        extra_state = init_step['states']

        initial_state = np.concatenate([joint_states, gripper_states, extra_state])
        all_states.append(initial_state)

    with open(init_file_path, 'wb') as f:
        pickle.dump(all_states, f)


def apply_pose_shifting_augmentation(demo_data: Dict,
                                   demo_key: str,
                                   skill_name: str,
                                   current_skill_initial_states: List[np.ndarray],
                                   env,
                                   args,
                                   trigger_timestep: int,
                                   skill_type: str = "atomic") -> Tuple[List[List[Dict]], List[Dict]]:
    """
    Apply pose shifting augmentation to a single demonstration.

    Args:
        demo_data: Single demo data from HDF5
        demo_key: Demo identifier
        skill_name: Skill name
        current_skill_initial_states: Initial states for current skill
        env: Environment instance
        args: Command line arguments
        trigger_timestep: Detected trigger timestep
        skill_type: Type of skill ("atomic", "pick", "place")

    Returns:
        Tuple of (successful_augmented_demos, augmentation_metadata)
    """
    actions = demo_data['actions']
    states = demo_data['states']

    # Calculate collection bounds based on skill type
    if skill_type == "place":
        # For place skills: start_idx = trigger_timestep - place_offset
        start_idx = max(0, trigger_timestep - args.place_offset)
    elif skill_type == "pick":
        # For pick skills: start_idx = trigger_timestep - pick_offset
        start_idx = max(0, trigger_timestep - args.pick_offset)
    else:  # atomic
        # For atomic skills: start_idx = trigger_timestep - atomic_offset
        start_idx = max(0, trigger_timestep - args.atomic_offset)

    print(f"\n🎯 Phase 2: Applying pose shifting augmentation to {demo_key}")
    print(f"   Original demo length: {len(actions)} steps")
    print(f"   Trigger at: {trigger_timestep}, Start collection at: {start_idx}")

    successful_augmented_demos = []
    augmentation_metadata = []

    try:
        # Extract original pose at start_idx for shifting base
        env.reset()
        for t in range(start_idx):
            if t < len(actions):
                env.step(actions[t])

        obs, _, _, _ = env.step(np.zeros(7))  # Dummy step to get observation
        original_ee_pos = obs['robot0_eef_pos']
        original_ee_quat = obs['robot0_eef_quat']
        original_pose = (original_ee_pos, original_ee_quat)

        print(f"   📍 Original pose at start_idx: pos=[{original_ee_pos[0]:.3f}, {original_ee_pos[1]:.3f}, {original_ee_pos[2]:.3f}]")

        # Step 2: Generate shifted pose using new simple approach
        print(f"   🎯 Step 2: Generating shifted pose...")
        shifted_position, shifted_quaternion = generate_shifted_pose(
            original_pose,
            position_shift_range=args.position_shift_range,
            orientation_shift_range=np.radians(args.orientation_shift_range_deg)
        )
        shifted_pose = (shifted_position, shifted_quaternion)

        print(f"   🎯 Generated shifted pose: pos=[{shifted_position[0]:.3f}, {shifted_position[1]:.3f}, {shifted_position[2]:.3f}]")

        # Initialize motion planner with verbose flag
        motion_planner = MotionPlanner(
            env,
            method=MOTION_PLANNER_METHOD,
            num_steps=args.motion_planner_steps,
            pos_gain=args.motion_planner_pos_gain,
            ori_gain=args.motion_planner_ori_gain,
            verbose=VERBOSE or DEBUG_MODE or DEBUG_SKILL_MODE
        )

        # Reset environment and replay to start_idx
        env.reset()
        for t in range(start_idx):
            if t < len(actions):
                env.step(actions[t])

        # Step 3: Move to shifted pose using motion planner (NO DATA COLLECTION)
        print(f"   🚀 Step 3: Moving to shifted pose using {MOTION_PLANNER_METHOD}...")
        
        # Move to shifted pose WITHOUT collecting data
        move1_success, obs_after_shift = motion_planner.move_to_pose(
            shifted_position,
            shifted_quaternion,
            position_threshold=args.position_threshold,
            orientation_threshold=np.radians(args.orientation_threshold_deg),
            skill_type=skill_type
        )

        if not move1_success:
            print(f"   ❌ Failed to move to shifted pose")
            return [], []

        print(f"   ✅ Successfully moved to shifted pose")

        # Step 4: Find family pose (use original pose)
        print(f"   🔍 Step 4: Finding family pose (using original pose)...")
        family_position, family_quaternion = find_family_pose(shifted_pose, current_skill_initial_states, original_pose)
        family_pose = (family_position, family_quaternion)

        # Step 5: Move from shifted pose to family pose (WITH DATA COLLECTION)
        print(f"   🚀 Step 5: Moving from shifted pose to family pose...")
        
        # Start collecting data from shifted pose to family pose
        collected_steps = []
        
        move2_success, motion_planning_steps = motion_planner.move_to_pose(
            family_position,
            family_quaternion,
            position_threshold=args.position_threshold,
            orientation_threshold=np.radians(args.orientation_threshold_deg),
            collect_data=True,
            skill_type=skill_type
        )

        if not move2_success:
            print(f"   ❌ Failed to move to family pose - motion planning threshold not met")
            return [], []

        print(f"   ✅ Successfully moved to family pose")
        
        # Collect the motion planning steps from shifted pose to family pose
        if isinstance(motion_planning_steps, list):
            collected_steps.extend(motion_planning_steps)
        else:
            print(f"   ⚠️ Warning: motion_planning_steps is not a list: {type(motion_planning_steps)}")
            print(f"   ⚠️ Skipping motion planning step collection")

        # Step 6: Continue replay from family pose to completion
        print(f"   🎬 Step 6: Continuing replay from family pose to completion...")

        replay_success = False

        # Continue from start_idx in the original trajectory
        for t in range(start_idx, len(actions)):
            action = actions[t]
            obs, reward, done, _ = env.step(action)

            # Collect step data
            step_data = collect_step_data(action, obs, reward, done, env)
            collected_steps.append(step_data)

            if done:
                # BDDL goal achieved - task is successful
                replay_success = True
                print(f"   ✅ Replay completed successfully at step {t}")
                break

        if replay_success and len(collected_steps) > 0:
            print(f"   🎉 Augmentation SUCCESS: Collected {len(collected_steps)} steps")
            successful_augmented_demos.append(collected_steps)

            # Save initial images in debug mode (augmented demo)
            if DEBUG_MODE or DEBUG_SKILL_MODE:
                save_initial_images_debug(collected_steps, skill_name, "augmented", demo_key, args.init_offset, args.output_dir)

            # Store metadata
            shift_applied = [
                shifted_position[0] - original_ee_pos[0],
                shifted_position[1] - original_ee_pos[1],
                shifted_position[2] - original_ee_pos[2],
                shifted_quaternion[0] - original_ee_quat[0],
                shifted_quaternion[1] - original_ee_quat[1],
                shifted_quaternion[2] - original_ee_quat[2]
            ]

            metadata = {
                'demo_key': demo_key,
                'shift_applied': shift_applied,
                'original_pose': [original_ee_pos.tolist(), original_ee_quat.tolist()],
                'shifted_pose': [shifted_position.tolist(), shifted_quaternion.tolist()],
                'family_pose': [family_position.tolist(), family_quaternion.tolist()],
                'steps_collected': len(collected_steps)
            }
            augmentation_metadata.append(metadata)
        else:
            print(f"   ❌ Replay failed or no steps collected")
            # Save failure video in debug modes with whatever steps were collected
            if args.debug or args.debug_skill:
                debug_video_dir = os.path.join(args.output_dir, "debug_videos")
                save_debug_videos(collected_steps, skill_name, "augmented", demo_key, debug_video_dir, is_failure=True)

    except Exception as e:
        print(f"   ❌ Augmentation failed with error: {e}")
        return [], []

    print(f"\n📊 Augmentation summary for {demo_key}: {len(successful_augmented_demos)} successful augmentations")

    return successful_augmented_demos, augmentation_metadata


def process_pick_skill_with_augmentation(mapping: Dict, args) -> Tuple[bool, str]:
    """
    Process a single pick skill with Phase 2 pose shifting augmentation.
    Returns (success, message) tuple.
    """
    skill_name = mapping['skill_name']
    atomic_bddl_path = mapping['atomic_bddl_path']
    demo_file_path = mapping['demo_full_path']

    print(f"\n🎯 Phase 2: Processing Pick Skill with Augmentation: {skill_name}")
    print(f"=" * 80)

    try:
        # Load demonstration data
        print("📁 Loading demonstration data...")
        demo_data = load_hdf5_demo_data(demo_file_path)

        if 'data' not in demo_data:
            return False, "No 'data' group found in HDF5 file"

        demo_keys = list(demo_data['data'].keys())
        if args.debug_skill:
            demo_keys = demo_keys[:args.debug_num_demos]
        elif args.debug:
            demo_keys = demo_keys[:args.debug_num_demos]

        print(f"📊 Found {len(demo_keys)} demonstrations")

        # Load current skill initial states for distribution analysis
        print(f"📁 Loading initial states for skill: {skill_name}")
        try:
            current_skill_initial_states = load_initial_states_for_skill(skill_name, args.atomic_demos_path)
        except Exception as e:
            print(f"❌ Failed to load initial states: {e}")
            return False, f"Failed to load initial states: {e}"

        # Initialize environment
        print("🌍 Initializing libero environment...")
        bm_name = "atomic_skills"
        task_suite = benchmark.get_benchmark_dict()[bm_name]()

        # Find task by extracting task name from atomic BDDL path
        atomic_task_name = os.path.basename(mapping['atomic_bddl_path']).replace('.bddl', '')
        task = None
        for t in task_suite.tasks:
            if t.name == atomic_task_name:
                task = t
                break

        if task is None:
            return False, f"Task not found: {atomic_task_name}"

        # Use large horizon to accommodate motion planning for pose shifts
        env, _ = get_libero_env(task, model_family="openvla", resolution=256, horizon=2000)

        # Process each demonstration
        original_successful_demos = []
        all_augmented_demos = []
        all_augmentation_metadata = []

        for demo_idx, demo_key in enumerate(tqdm(demo_keys, desc=f"Processing {skill_name} demos")):
            demo = demo_data['data'][demo_key]
            actions = demo['actions']
            states = demo['states']

            print(f"\n📋 Processing {demo_key} ({demo_idx + 1}/{len(demo_keys)})")
            print(f"   Demo length: {len(actions)} steps")

            # Detect trigger timestep for pick skills (gripper closing + contact detection)
            trigger_timestep = detect_trigger_timestep(actions, env, states)
            if trigger_timestep is None:
                print(f"❌ FAILURE: No trigger found for {demo_key}")
                continue

            print(f"✅ Trigger found at timestep {trigger_timestep}")

            # Process original demo (exactly like the old script - simple replay and record)
            start_idx = max(0, trigger_timestep - args.pick_offset)

            # Single replay for original demo (same as old script)
            env.reset()
            full_trajectory = []
            actual_completion_step = None
            replay_success = False

            # Replay entire trajectory and collect all data
            for t in range(len(actions)):
                action = actions[t]
                obs, reward, done, _ = env.step(action)

                # Collect step data for entire trajectory
                step_data = collect_step_data(action, obs, reward, done, env)
                full_trajectory.append(step_data)

                if done:
                    replay_success = True
                    actual_completion_step = t
                    break

            # Slice trajectory based on trigger and completion (same as old script)
            if replay_success and actual_completion_step is not None:
                desired_start_idx = max(0, trigger_timestep - args.pick_offset)
                if desired_start_idx <= actual_completion_step:
                    start_idx = desired_start_idx
                else:
                    # Fallback: collect at least min_steps_fallback steps before completion
                    min_steps_needed = args.min_steps_fallback
                    start_idx = max(0, actual_completion_step - min_steps_needed + 1)

                # Slice the trajectory
                end_idx = actual_completion_step + 1  # Include completion step
                collected_steps = full_trajectory[start_idx:end_idx]

                if len(collected_steps) > 0:
                    print(f"✅ Original demo SUCCESS: {demo_key} - collected {len(collected_steps)} steps")
                    original_successful_demos.append(collected_steps)

                    # Save initial images in debug mode (pick skill)
                    if args.debug or args.debug_skill:
                        save_initial_images_debug(collected_steps, skill_name, "original", demo_key, args.init_offset, args.output_dir)

                    # Phase 2: Apply pose shifting augmentation (multiple iterations)
                    if not args.disable_augmentation:
                        num_iterations = args.debug_num_iterations if (args.debug_skill or args.debug) else args.num_augmentation_iterations
                        for iteration in range(num_iterations):
                            print(f"\n🔄 Augmentation iteration {iteration + 1}/{num_iterations} for {demo_key}")
                            augmented_demos, augmentation_metadata = apply_pose_shifting_augmentation(
                                demo, f"{demo_key}_iter{iteration}", skill_name, current_skill_initial_states,
                                env, args, trigger_timestep, skill_type="pick"
                            )

                            all_augmented_demos.extend(augmented_demos)
                            all_augmentation_metadata.extend(augmentation_metadata)
                    else:
                        print(f"⏭️  Skipping augmentation (disabled)")
                else:
                    print(f"❌ Original demo FAILURE: {demo_key} - no steps collected")
                    # Save failure video in debug modes with whatever steps were collected
                    if args.debug or args.debug_skill:
                        debug_video_dir = os.path.join(args.output_dir, "debug_videos")
                        save_debug_videos(full_trajectory, skill_name, "original", demo_key, debug_video_dir, is_failure=True)
            else:
                print(f"❌ Original demo FAILURE: {demo_key} - replay unsuccessful")
                # Save failure video in debug modes with whatever steps were collected
                if args.debug or args.debug_skill:
                    debug_video_dir = os.path.join(args.output_dir, "debug_videos")
                    save_debug_videos(full_trajectory, skill_name, "original", demo_key, debug_video_dir)

        env.close()

        # Save results if we have successful demos (same logic as atomic skills)
        success_messages = []

        # Save original demos if we have them
        if original_successful_demos:
            # Save original demos to separate HDF5 file
            original_demos_data = []
            for demo_steps in original_successful_demos:
                demo_data_formatted = convert_steps_to_demo(demo_steps)
                original_demos_data.append(demo_data_formatted)

            original_output_filename = f"{skill_name}_original_demo.hdf5"
            original_output_path = os.path.join(args.output_dir, original_output_filename)
            save_multiple_demos_to_hdf5(original_demos_data, original_output_path)
            print(f"💾 Saved original HDF5 demo: {original_output_path}")

            # Save original initial states
            original_init_filename = f"{skill_name}_original.init"
            original_init_path = os.path.join(args.output_dir, original_init_filename)
            extract_and_save_initial_states_original(original_successful_demos, original_init_path, args.init_offset)
            print(f"💾 Saved original initial states: {original_init_path}")

            # Generate debug videos for original demos if in debug mode
            if args.debug:
                debug_video_dir = os.path.join(args.output_dir, "debug_videos")
                for i, demo_steps in enumerate(original_successful_demos):
                    save_debug_videos(demo_steps, skill_name, "original", f"demo_{demo_idx}", debug_video_dir)

            success_messages.append(f"original demos: {len(original_successful_demos)}")

        # Save augmented demos if we have them
        if all_augmented_demos:
            # Save augmented demos to separate HDF5 file
            augmented_demos_data = []
            for demo_steps in all_augmented_demos:
                demo_data_formatted = convert_steps_to_demo(demo_steps)
                augmented_demos_data.append(demo_data_formatted)

            augmented_output_filename = f"{skill_name}_augmented_demo.hdf5"
            augmented_output_path = os.path.join(args.output_dir, augmented_output_filename)
            save_multiple_demos_to_hdf5(augmented_demos_data, augmented_output_path)
            print(f"💾 Saved augmented HDF5 demo: {augmented_output_path}")

            # Save augmented initial states
            augmented_init_filename = f"{skill_name}_augmented.init"
            augmented_init_path = os.path.join(args.output_dir, augmented_init_filename)
            extract_and_save_initial_states_augmented(all_augmented_demos, augmented_init_path, args.init_offset)
            print(f"💾 Saved augmented initial states: {augmented_init_path}")

            # Generate debug videos for augmented demos if in debug mode
            if args.debug:
                debug_video_dir = os.path.join(args.output_dir, "debug_videos")
                for i, demo_steps in enumerate(all_augmented_demos):
                    save_debug_videos(demo_steps, skill_name, "augmented", f"demo_{demo_idx}", debug_video_dir)

            success_messages.append(f"augmented demos: {len(all_augmented_demos)}")

        if original_successful_demos or all_augmented_demos:
            # Calculate success rates
            original_success_rate = len(original_successful_demos) / len(demo_keys) if demo_keys else 0
            total_possible_augmented = len(demo_keys) * (args.debug_num_iterations if (args.debug_skill or args.debug) else args.num_augmentation_iterations)
            augmented_success_rate = len(all_augmented_demos) / total_possible_augmented if total_possible_augmented > 0 else 0

            print(f"📊 Success rates for {skill_name}:")
            print(f"   Original demos: {original_success_rate:.2f} ({len(original_successful_demos)}/{len(demo_keys)})")
            print(f"   Augmented demos: {augmented_success_rate:.2f} ({len(all_augmented_demos)}/{total_possible_augmented})")

            # Store success rate in global dict for later JSON export
            if not hasattr(process_atomic_skill_with_augmentation, 'skill_statistics'):
                process_atomic_skill_with_augmentation.skill_statistics = {}

            process_atomic_skill_with_augmentation.skill_statistics[skill_name] = {
                'original_demos': len(original_successful_demos),
                'augmented_demos': len(all_augmented_demos),
                'total_demos': len(original_successful_demos) + len(all_augmented_demos),
                'original_input_demos': len(demo_keys),
                'total_possible_augmented': total_possible_augmented,
                'original_success_rate': original_success_rate,
                'augmented_success_rate': augmented_success_rate,
                'augmentation_metadata': all_augmentation_metadata
            }

            return True, f"Successfully processed {', '.join(success_messages)}"
        else:
            return False, "No successful demonstrations found"

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Error processing {skill_name}: {error_details}")
        return False, f"Error processing {skill_name}: {str(e)}"


def process_place_skill_with_augmentation(mapping: Dict, args) -> Tuple[bool, str]:
    """
    Process a single place skill with Phase 2 pose shifting augmentation.
    Returns (success, message) tuple.
    """
    skill_name = mapping['skill_name']
    atomic_bddl_path = mapping['atomic_bddl_path']
    demo_file_path = mapping['demo_full_path']

    print(f"\n🎯 Phase 2: Processing Place Skill with Augmentation: {skill_name}")
    print(f"=" * 80)

    try:
        # Load demonstration data
        print("📁 Loading demonstration data...")
        demo_data = load_hdf5_demo_data(demo_file_path)

        if 'data' not in demo_data:
            return False, "No 'data' group found in HDF5 file"

        demo_keys = list(demo_data['data'].keys())
        if args.debug_skill:
            demo_keys = demo_keys[:args.debug_num_demos]
        elif args.debug:
            demo_keys = demo_keys[:args.debug_num_demos]

        print(f"📊 Found {len(demo_keys)} demonstrations")

        # Load current skill initial states for distribution analysis
        print(f"📁 Loading initial states for skill: {skill_name}")
        try:
            current_skill_initial_states = load_initial_states_for_skill(skill_name, args.atomic_demos_path)
        except Exception as e:
            print(f"❌ Failed to load initial states: {e}")
            return False, f"Failed to load initial states: {e}"

        # Initialize environment
        print("🌍 Initializing libero environment...")
        bm_name = "atomic_skills"
        task_suite = benchmark.get_benchmark_dict()[bm_name]()

        # Find task by extracting task name from atomic BDDL path
        atomic_task_name = os.path.basename(mapping['atomic_bddl_path']).replace('.bddl', '')
        task = None
        for t in task_suite.tasks:
            if t.name == atomic_task_name:
                task = t
                break

        if task is None:
            return False, f"Task not found: {atomic_task_name}"

        # Use large horizon to accommodate motion planning for pose shifts
        env, _ = get_libero_env(task, model_family="openvla", resolution=256, horizon=2000)

        # Process each demonstration
        original_successful_demos = []
        all_augmented_demos = []
        all_augmentation_metadata = []

        for demo_idx, demo_key in enumerate(tqdm(demo_keys, desc=f"Processing {skill_name} demos")):
            demo = demo_data['data'][demo_key]
            actions = demo['actions']

            print(f"\n📋 Processing {demo_key} ({demo_idx + 1}/{len(demo_keys)})")
            print(f"   Demo length: {len(actions)} steps")

            # Place skills: start from total_timesteps - place_offset to end
            total_timesteps = len(actions)
            start_idx = max(0, total_timesteps - args.place_offset)

            # Quick check: simulate to find actual completion point (same as original script)
            temp_env, _ = get_libero_env(task, model_family="openvla", resolution=256)
            temp_obs = temp_env.reset()
            actual_completion_step = None
            for temp_t in range(len(actions)):
                temp_action = actions[temp_t]
                temp_obs, temp_reward, temp_done, _ = temp_env.step(temp_action)
                if temp_done:
                    actual_completion_step = temp_t
                    break
            temp_env.close()

            if actual_completion_step is not None:
                # Try to collect the desired place_offset steps, but ensure we get at least some data
                desired_start_idx = max(0, actual_completion_step - args.place_offset + 1)
                if desired_start_idx <= actual_completion_step:
                    # We can collect the desired number of steps
                    start_idx = desired_start_idx
                    expected_steps = actual_completion_step - desired_start_idx + 1
                    print(f"Task completes at step {actual_completion_step}, collecting {expected_steps} steps from {start_idx}")
                else:
                    # Fallback: collect at least min_steps_fallback steps before completion
                    min_steps_needed = args.min_steps_fallback
                    start_idx = max(0, actual_completion_step - min_steps_needed + 1)
                    expected_steps = actual_completion_step - start_idx + 1
                    print(f"Task completes at step {actual_completion_step}, fallback to {expected_steps} steps from {start_idx}")

            print(f"Collection range: steps {start_idx} to {total_timesteps-1} (place skill)")

            # Reset environment and replay (same as original script)
            obs = env.reset()
            collected_steps = []
            replay_success = False

            # Replay trajectory and collect from start_idx
            for t in range(len(actions)):
                action = actions[t]
                obs, reward, done, _ = env.step(action)

                # Start collecting from start_idx
                if t >= start_idx:
                    step_data = collect_step_data(action, obs, reward, done, env)
                    collected_steps.append(step_data)

                if done:
                    # BDDL goal achieved - task is successful
                    replay_success = True
                    break

            if replay_success and len(collected_steps) > 0:
                print(f"✅ Original demo SUCCESS: {demo_key} - collected {len(collected_steps)} steps")
                original_successful_demos.append(collected_steps)

                # Save initial images in debug mode (place skill)
                if args.debug or args.debug_skill:
                    save_initial_images_debug(collected_steps, skill_name, "original", demo_key, args.init_offset, args.output_dir)

                # Phase 2: Apply pose shifting augmentation (multiple iterations)
                # For place skills, use the actual completion step as trigger and calculate proper start_idx
                if not args.disable_augmentation:
                    num_iterations = args.debug_num_iterations if args.debug else args.num_augmentation_iterations
                    for iteration in range(num_iterations):
                        print(f"\n🔄 Augmentation iteration {iteration + 1}/{num_iterations} for {demo_key}")
                        # For place skills: trigger = completion step, start_idx = completion - place_offset
                        place_trigger_timestep = actual_completion_step if actual_completion_step is not None else len(actions) - 1
                        augmented_demos, augmentation_metadata = apply_pose_shifting_augmentation(
                            demo, f"{demo_key}_iter{iteration}", skill_name, current_skill_initial_states,
                            env, args, place_trigger_timestep, skill_type="place"
                        )

                        all_augmented_demos.extend(augmented_demos)
                        all_augmentation_metadata.extend(augmentation_metadata)
                else:
                    print(f"⏭️  Skipping augmentation (disabled)")
            else:
                print(f"❌ Original demo FAILURE: {demo_key} - replay unsuccessful or no steps collected")
                # Save failure video in debug modes with whatever steps were collected
                if args.debug or args.debug_skill:
                    debug_video_dir = os.path.join(args.output_dir, "debug_videos")
                    save_debug_videos(collected_steps, skill_name, "original", demo_key, debug_video_dir, is_failure=True)

        env.close()

        # Save results if we have successful demos (same logic as atomic skills)
        success_messages = []

        # Save original demos if we have them
        if original_successful_demos:
            # Save original demos to separate HDF5 file
            original_demos_data = []
            for demo_steps in original_successful_demos:
                demo_data_formatted = convert_steps_to_demo(demo_steps)
                original_demos_data.append(demo_data_formatted)

            original_output_filename = f"{skill_name}_original_demo.hdf5"
            original_output_path = os.path.join(args.output_dir, original_output_filename)
            save_multiple_demos_to_hdf5(original_demos_data, original_output_path)
            print(f"💾 Saved original HDF5 demo: {original_output_path}")

            # Save original initial states
            original_init_filename = f"{skill_name}_original.init"
            original_init_path = os.path.join(args.output_dir, original_init_filename)
            extract_and_save_initial_states_original(original_successful_demos, original_init_path, args.init_offset)
            print(f"💾 Saved original initial states: {original_init_path}")

            # Generate debug videos for original demos if in debug mode
            if args.debug:
                debug_video_dir = os.path.join(args.output_dir, "debug_videos")
                for i, demo_steps in enumerate(original_successful_demos):
                    save_debug_videos(demo_steps, skill_name, "original", f"demo_{demo_idx}", debug_video_dir)

            success_messages.append(f"original demos: {len(original_successful_demos)}")

        # Save augmented demos if we have them
        if all_augmented_demos:
            # Save augmented demos to separate HDF5 file
            augmented_demos_data = []
            for demo_steps in all_augmented_demos:
                demo_data_formatted = convert_steps_to_demo(demo_steps)
                augmented_demos_data.append(demo_data_formatted)

            augmented_output_filename = f"{skill_name}_augmented_demo.hdf5"
            augmented_output_path = os.path.join(args.output_dir, augmented_output_filename)
            save_multiple_demos_to_hdf5(augmented_demos_data, augmented_output_path)
            print(f"💾 Saved augmented HDF5 demo: {augmented_output_path}")

            # Save augmented initial states
            augmented_init_filename = f"{skill_name}_augmented.init"
            augmented_init_path = os.path.join(args.output_dir, augmented_init_filename)
            extract_and_save_initial_states_augmented(all_augmented_demos, augmented_init_path, args.init_offset)
            print(f"💾 Saved augmented initial states: {augmented_init_path}")

            # Generate debug videos for augmented demos if in debug mode
            if args.debug:
                debug_video_dir = os.path.join(args.output_dir, "debug_videos")
                for i, demo_steps in enumerate(all_augmented_demos):
                    save_debug_videos(demo_steps, skill_name, "augmented", f"demo_{demo_idx}", debug_video_dir)

            success_messages.append(f"augmented demos: {len(all_augmented_demos)}")

        if original_successful_demos or all_augmented_demos:
            # Calculate success rates
            original_success_rate = len(original_successful_demos) / len(demo_keys) if demo_keys else 0
            total_possible_augmented = len(demo_keys) * (args.debug_num_iterations if (args.debug_skill or args.debug) else args.num_augmentation_iterations)
            augmented_success_rate = len(all_augmented_demos) / total_possible_augmented if total_possible_augmented > 0 else 0

            print(f"📊 Success rates for {skill_name}:")
            print(f"   Original demos: {original_success_rate:.2f} ({len(original_successful_demos)}/{len(demo_keys)})")
            print(f"   Augmented demos: {augmented_success_rate:.2f} ({len(all_augmented_demos)}/{total_possible_augmented})")

            # Store success rate in global dict for later JSON export
            if not hasattr(process_atomic_skill_with_augmentation, 'skill_statistics'):
                process_atomic_skill_with_augmentation.skill_statistics = {}

            process_atomic_skill_with_augmentation.skill_statistics[skill_name] = {
                'original_demos': len(original_successful_demos),
                'augmented_demos': len(all_augmented_demos),
                'total_demos': len(original_successful_demos) + len(all_augmented_demos),
                'original_input_demos': len(demo_keys),
                'total_possible_augmented': total_possible_augmented,
                'original_success_rate': original_success_rate,
                'augmented_success_rate': augmented_success_rate,
                'augmentation_metadata': all_augmentation_metadata
            }

            return True, f"Successfully processed {', '.join(success_messages)}"
        else:
            return False, "No successful demonstrations found"

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Error processing {skill_name}: {error_details}")
        return False, f"Error processing {skill_name}: {str(e)}"


def process_atomic_skill_with_augmentation(mapping: Dict, args) -> Tuple[bool, str]:
    """
    Process a single atomic skill with Phase 2 pose shifting augmentation.
    Returns (success, message) tuple.
    """
    skill_name = mapping['skill_name']
    atomic_bddl_path = mapping['atomic_bddl_path']
    demo_file_path = mapping['demo_full_path']
    
    print(f"\n🎯 Phase 2: Processing Atomic Skill with Augmentation: {skill_name}")
    print(f"=" * 80)
    
    try:
        # Load demonstration data
        print("📁 Loading demonstration data...")
        demo_data = load_hdf5_demo_data(demo_file_path)
        
        if 'data' not in demo_data:
            return False, "No 'data' group found in HDF5 file"
        
        demo_keys = list(demo_data['data'].keys())
        if args.debug_skill:
            demo_keys = demo_keys[:args.debug_num_demos]
        elif args.debug:
            demo_keys = demo_keys[:args.debug_num_demos]
        
        print(f"📊 Found {len(demo_keys)} demonstrations")
        
        # Load current skill initial states for distribution analysis
        print(f"📁 Loading initial states for skill: {skill_name}")
        try:
            current_skill_initial_states = load_initial_states_for_skill(skill_name, args.atomic_demos_path)
        except Exception as e:
            print(f"❌ Failed to load initial states: {e}")
            return False, f"Failed to load initial states: {e}"
        
        # Initialize environment
        print("🌍 Initializing libero environment...")
        bm_name = "atomic_skills"
        task_suite = benchmark.get_benchmark_dict()[bm_name]()
        
        # Find task by extracting task name from atomic BDDL path
        atomic_task_name = os.path.basename(mapping['atomic_bddl_path']).replace('.bddl', '')
        task = None
        for t in task_suite.tasks:
            if t.name == atomic_task_name:
                task = t
                break
        
        if task is None:
            return False, f"Task not found: {atomic_task_name}"
        
        # Use large horizon to accommodate motion planning for pose shifts
        env, _ = get_libero_env(task, model_family="openvla", resolution=256, horizon=2000)
        
        # Process each demonstration
        original_successful_demos = []
        all_augmented_demos = []
        all_augmentation_metadata = []

        for demo_idx, demo_key in enumerate(tqdm(demo_keys, desc=f"Processing {skill_name} demos")):
            demo = demo_data['data'][demo_key]
            actions = demo['actions']
            states = demo['states']

            print(f"\n📋 Processing {demo_key} ({demo_idx + 1}/{len(demo_keys)})")
            print(f"   Demo length: {len(actions)} steps")

            # Detect trigger timestep for atomic skills (contact detection only)
            trigger_timestep = detect_trigger_timestep_contact_only(actions, env, states)
            if trigger_timestep is None:
                print(f"❌ FAILURE: No contact trigger found for {demo_key}")
                continue

            print(f"✅ Contact trigger found at timestep {trigger_timestep}")

            # Process original demo (exactly like the old script - simple replay and record)
            start_idx = max(0, trigger_timestep - args.atomic_offset)

            # Single replay for original demo (same as old script)
            env.reset()
            full_trajectory = []
            actual_completion_step = None
            replay_success = False

            # Replay entire trajectory and collect all data
            for t in range(len(actions)):
                action = actions[t]
                obs, reward, done, _ = env.step(action)

                # Collect step data for entire trajectory
                step_data = collect_step_data(action, obs, reward, done, env)
                full_trajectory.append(step_data)

                if done:
                    replay_success = True
                    actual_completion_step = t
                    break

            # Slice trajectory based on trigger and completion (same as old script)
            if replay_success and actual_completion_step is not None:
                desired_start_idx = max(0, trigger_timestep - args.atomic_offset)
                if desired_start_idx <= actual_completion_step:
                    start_idx = desired_start_idx
                else:
                    # Fallback: collect at least min_steps_fallback steps before completion
                    min_steps_needed = args.min_steps_fallback
                    start_idx = max(0, actual_completion_step - min_steps_needed + 1)

                # Slice the trajectory
                end_idx = actual_completion_step + 1  # Include completion step
                collected_steps = full_trajectory[start_idx:end_idx]

                if len(collected_steps) > 0:
                    print(f"✅ Original demo SUCCESS: {demo_key} - collected {len(collected_steps)} steps")
                    original_successful_demos.append(collected_steps)

                    # Save initial images in debug mode (atomic skill)
                    if args.debug or args.debug_skill:
                        save_initial_images_debug(collected_steps, skill_name, "original", demo_key, args.init_offset, args.output_dir)

                    # Phase 2: Apply pose shifting augmentation (multiple iterations)
                    if not args.disable_augmentation:
                        num_iterations = args.debug_num_iterations if (args.debug_skill or args.debug) else args.num_augmentation_iterations
                        for iteration in range(num_iterations):
                            print(f"\n🔄 Augmentation iteration {iteration + 1}/{num_iterations} for {demo_key}")
                            augmented_demos, augmentation_metadata = apply_pose_shifting_augmentation(
                                demo, f"{demo_key}_iter{iteration}", skill_name, current_skill_initial_states,
                                env, args, trigger_timestep, skill_type="atomic"
                            )

                            all_augmented_demos.extend(augmented_demos)
                            all_augmentation_metadata.extend(augmentation_metadata)
                    else:
                        print(f"⏭️  Skipping augmentation (disabled)")
                else:
                    print(f"❌ Original demo FAILURE: {demo_key} - no steps collected")
                    # Save failure video in debug modes with whatever steps were collected
                    if args.debug or args.debug_skill:
                        debug_video_dir = os.path.join(args.output_dir, "debug_videos")
                        save_debug_videos(full_trajectory, skill_name, "original", demo_key, debug_video_dir, is_failure=True)
            else:
                print(f"❌ Original demo FAILURE: {demo_key} - replay unsuccessful")
                # Save failure video in debug modes with whatever steps were collected
                if args.debug or args.debug_skill:
                    debug_video_dir = os.path.join(args.output_dir, "debug_videos")
                    save_debug_videos(full_trajectory, skill_name, "original", demo_key, debug_video_dir)
        
        env.close()

        # Save results if we have successful demos
        success_messages = []

        # Save original demos if we have them
        if original_successful_demos:
            # Save original demos to separate HDF5 file
            original_demos_data = []
            for demo_steps in original_successful_demos:
                demo_data_formatted = convert_steps_to_demo(demo_steps)
                original_demos_data.append(demo_data_formatted)

            original_output_filename = f"{skill_name}_original_demo.hdf5"
            original_output_path = os.path.join(args.output_dir, original_output_filename)
            save_multiple_demos_to_hdf5(original_demos_data, original_output_path)
            print(f"💾 Saved original HDF5 demo: {original_output_path}")

            # Save original initial states
            original_init_filename = f"{skill_name}_original.init"
            original_init_path = os.path.join(args.output_dir, original_init_filename)
            extract_and_save_initial_states_original(original_successful_demos, original_init_path, args.init_offset)
            print(f"💾 Saved original initial states: {original_init_path}")

            # Generate debug videos for original demos if in debug mode
            if args.debug:
                debug_video_dir = os.path.join(args.output_dir, "debug_videos")
                for i, demo_steps in enumerate(original_successful_demos):
                    save_debug_videos(demo_steps, skill_name, "original", f"demo_{demo_idx}", debug_video_dir)

            success_messages.append(f"original demos: {len(original_successful_demos)}")

        # Save augmented demos if we have them
        if all_augmented_demos:
            # Save augmented demos to separate HDF5 file
            augmented_demos_data = []
            for demo_steps in all_augmented_demos:
                demo_data_formatted = convert_steps_to_demo(demo_steps)
                augmented_demos_data.append(demo_data_formatted)

            augmented_output_filename = f"{skill_name}_augmented_demo.hdf5"
            augmented_output_path = os.path.join(args.output_dir, augmented_output_filename)
            save_multiple_demos_to_hdf5(augmented_demos_data, augmented_output_path)
            print(f"💾 Saved augmented HDF5 demo: {augmented_output_path}")

            # Save augmented initial states
            augmented_init_filename = f"{skill_name}_augmented.init"
            augmented_init_path = os.path.join(args.output_dir, augmented_init_filename)
            extract_and_save_initial_states_augmented(all_augmented_demos, augmented_init_path, args.init_offset)
            print(f"💾 Saved augmented initial states: {augmented_init_path}")

            # Generate debug videos for augmented demos if in debug mode
            if args.debug:
                debug_video_dir = os.path.join(args.output_dir, "debug_videos")
                for i, demo_steps in enumerate(all_augmented_demos):
                    save_debug_videos(demo_steps, skill_name, "augmented", f"demo_{demo_idx}", debug_video_dir)

            success_messages.append(f"augmented demos: {len(all_augmented_demos)}")

        if original_successful_demos or all_augmented_demos:
            # Calculate success rates
            original_success_rate = len(original_successful_demos) / len(demo_keys) if demo_keys else 0
            total_possible_augmented = len(demo_keys) * (args.debug_num_iterations if (args.debug_skill or args.debug) else args.num_augmentation_iterations)
            augmented_success_rate = len(all_augmented_demos) / total_possible_augmented if total_possible_augmented > 0 else 0

            print(f"📊 Success rates for {skill_name}:")
            print(f"   Original demos: {original_success_rate:.2f} ({len(original_successful_demos)}/{len(demo_keys)})")
            print(f"   Augmented demos: {augmented_success_rate:.2f} ({len(all_augmented_demos)}/{total_possible_augmented})")

            # Store success rate in global dict for later JSON export
            if not hasattr(process_atomic_skill_with_augmentation, 'skill_statistics'):
                process_atomic_skill_with_augmentation.skill_statistics = {}

            process_atomic_skill_with_augmentation.skill_statistics[skill_name] = {
                'original_demos': len(original_successful_demos),
                'augmented_demos': len(all_augmented_demos),
                'total_demos': len(original_successful_demos) + len(all_augmented_demos),
                'original_input_demos': len(demo_keys),
                'total_possible_augmented': total_possible_augmented,
                'original_success_rate': original_success_rate,
                'augmented_success_rate': augmented_success_rate,
                'augmentation_metadata': all_augmentation_metadata
            }

            return True, f"Successfully processed {', '.join(success_messages)}"
        else:
            return False, "No successful demonstrations found"
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return False, f"Error processing {skill_name}: {str(e)}\n{error_details}"


def save_augmentation_metadata(output_dir: str):
    """Save comprehensive augmentation metadata to JSON file."""
    if not hasattr(process_atomic_skill_with_augmentation, 'skill_statistics'):
        print("⚠️  No skill statistics to save")
        return
    
    skill_stats = process_atomic_skill_with_augmentation.skill_statistics
    
    # Create comprehensive metadata
    metadata = {
        'generation_timestamp': datetime.now().isoformat(),
        'total_skills_processed': len(skill_stats),
        'overall_statistics': {
            'total_original_demos': sum(stats['original_demos'] for stats in skill_stats.values()),
            'total_augmented_demos': sum(stats['augmented_demos'] for stats in skill_stats.values()),
            'total_demos': sum(stats['total_demos'] for stats in skill_stats.values()),
            'total_input_demos': sum(stats['original_input_demos'] for stats in skill_stats.values()),
        },
        'per_skill_statistics': skill_stats
    }
    
    # Calculate overall rates
    if metadata['overall_statistics']['total_input_demos'] > 0:
        metadata['overall_statistics']['overall_original_success_rate'] = (
            metadata['overall_statistics']['total_original_demos'] / 
            metadata['overall_statistics']['total_input_demos']
        )
        metadata['overall_statistics']['overall_augmentation_ratio'] = (
            metadata['overall_statistics']['total_augmented_demos'] / 
            metadata['overall_statistics']['total_input_demos']
        )
        metadata['overall_statistics']['overall_demo_multiplication_factor'] = (
            metadata['overall_statistics']['total_demos'] / 
            metadata['overall_statistics']['total_input_demos']
        )
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "augmentation_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Saved augmentation metadata: {metadata_path}")
    
    # Print summary
    print(f"\n📊 Phase 2 Augmentation Summary:")
    print(f"   Skills processed: {metadata['total_skills_processed']}")
    print(f"   Input demos: {metadata['overall_statistics']['total_input_demos']}")
    print(f"   Original demos generated: {metadata['overall_statistics']['total_original_demos']}")
    print(f"   Augmented demos generated: {metadata['overall_statistics']['total_augmented_demos']}")
    print(f"   Total demos: {metadata['overall_statistics']['total_demos']}")
    if 'overall_demo_multiplication_factor' in metadata['overall_statistics']:
        print(f"   Demo multiplication factor: {metadata['overall_statistics']['overall_demo_multiplication_factor']:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Generate augmented demos with Phase 2 pose shifting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in debug mode (processes one file per category + 2 demos each)
  python 1_generate_augmented_demos.py --debug

  # Full run with augmentation
  python 1_generate_augmented_demos.py --output_dir /path/to/output

  # Disable augmentation (Phase 1 behavior)
  python 1_generate_augmented_demos.py --disable_augmentation
        """
    )

    # ============================================================================
    # CORE CONFIGURATION
    # ============================================================================

    # Logging control
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (shows detailed progress)"
    )

    # ============================================================================
    # INPUT/OUTPUT PATHS
    # ============================================================================
    parser.add_argument(
        "--atomic_skills_dir", 
        type=str,
        default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss/libero/libero/bddl_files/atomic_skills",
        help="Directory containing atomic skill BDDL files"
    )
    parser.add_argument(
        "--cat_split_map_file",
        type=str,
        default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss/libero/libero/bddl_files/atomic_skills/cat_split_map.json",
        help="Path to cat_split_map.json mapping file"
    )
    parser.add_argument(
        "--raw_demo_dir",
        type=str,
        default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/libero_90_no_noops/",
        help="Directory containing original demonstration HDF5 files"
    )
    parser.add_argument(
        "--atomic_demos_path",
        type=str,
        default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_local_demos/full_atomic_skills",
        help="Path to atomic demos for initial states"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_local_demos/augmented_atomic_skills",
        help="Output directory for generated augmented HDF5 files"
    )

    # ============================================================================
    # PROCESSING PARAMETERS
    # ============================================================================
    parser.add_argument(
        "--init_offset",
        type=int,
        default=15,
        help="Steps before trigger for initial state extraction"
    )
    parser.add_argument(
        "--pick_offset",
        type=int,
        default=25,
        help="Steps before trigger for pick skill data collection"
    )
    parser.add_argument(
        "--place_offset",
        type=int,
        default=25,
        help="Steps from end for place skill data collection"
    )
    parser.add_argument(
        "--atomic_offset",
        type=int,
        default=25,
        help="Steps before trigger for atomic skill data collection"
    )
    parser.add_argument(
        "--min_steps_fallback",
        type=int,
        default=3,
        help="Minimum steps to collect as fallback when collection window is too late"
    )

    # ============================================================================
    # PHASE 2 POSE SHIFTING PARAMETERS
    # ============================================================================
    parser.add_argument(
        "--position_shift_range",
        type=float,
        default=DEFAULT_POSITION_SHIFT_RANGE,
        help="Position shift range (±meters)"
    )
    parser.add_argument(
        "--orientation_shift_range_deg",
        type=float,
        default=DEFAULT_ORIENTATION_SHIFT_RANGE_DEG,
        help="Orientation shift range (±degrees)"
    )

    # ============================================================================
    # MOTION PLANNER PARAMETERS
    # ============================================================================
    parser.add_argument(
        "--motion_planner_steps",
        type=int,
        default=DEFAULT_MOTION_PLANNER_STEPS,
        help="Number of steps for motion planner"
    )
    parser.add_argument(
        "--motion_planner_pos_gain",
        type=float,
        default=DEFAULT_MOTION_PLANNER_POS_GAIN,
        help="Position gain for motion planner"
    )
    parser.add_argument(
        "--motion_planner_ori_gain",
        type=float,
        default=DEFAULT_MOTION_PLANNER_ORI_GAIN,
        help="Orientation gain for motion planner"
    )
    parser.add_argument(
        "--position_threshold",
        type=float,
        default=DEFAULT_POSITION_THRESHOLD,
        help="Position threshold for motion planner success (meters)"
    )
    parser.add_argument(
        "--orientation_threshold_deg",
        type=float,
        default=DEFAULT_ORIENTATION_THRESHOLD_DEG,
        help="Orientation threshold for motion planner success (degrees)"
    )
    parser.add_argument(
        "--num_augmentation_iterations",
        type=int,
        default=5,
        help="Number of augmentation iterations per demo (default: 5)"
    )
    parser.add_argument(
        "--disable_augmentation",
        action="store_true",
        help="Disable pose shifting augmentation (Phase 1 behavior)"
    )

    # ============================================================================
    # DEBUG AND TESTING MODES
    # ============================================================================
    # Debug mode - processes 3 representative skills (pick/place/atomic)
    parser.add_argument(
        "--debug", "--debug_mode",
        action="store_true",
        help="Debug mode: process 3 skills (1 pick, 1 place, 1 atomic) for testing"
    )

    # Debug skill mode - processes single specified skill
    parser.add_argument(
        "--debug_skill", "--debug_skill_mode",
        action="store_true",
        help="Debug single skill: process one specific skill for detailed analysis"
    )
    parser.add_argument(
        "--debug_skill_name",
        type=str,
        default="KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
        help="Skill name to debug in debug_skill mode"
    )

    # Debug mode parameters (apply to both --debug and --debug_skill modes)
    parser.add_argument(
        "--debug_num_demos",
        type=int,
        default=5,
        help="Number of demos per skill to process in debug modes (default: 5)"
    )
    parser.add_argument(
        "--debug_num_iterations",
        type=int,
        default=2,
        help="Number of augmentation iterations per demo in debug modes (default: 2)"
    )
    
    args = parser.parse_args()

    # Set global logging flags
    global VERBOSE, DEBUG_MODE, DEBUG_SKILL_MODE
    VERBOSE = args.verbose
    DEBUG_MODE = args.debug
    DEBUG_SKILL_MODE = args.debug_skill

    # Set debug output directory if debug mode is enabled
    if args.debug:
        args.output_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_local_demos_augmented/debug_mode"
        log_debug(f"Debug mode enabled: Output will be saved to {args.output_dir}")

    # Set debug skill mode output directory and parameters
    if args.debug_skill:
        args.output_dir = f"/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_local_demos_augmented/debug_skill_mode"
        log_debug(f"Debug skill mode enabled: Processing skill '{args.debug_skill_name}'")
        log_verbose(f"   Output will be saved to {args.output_dir}")
        log_verbose(f"   Will process {args.debug_num_demos} demos with {args.debug_num_iterations} iterations each")
    
    # Validate input paths
    if not os.path.exists(args.atomic_skills_dir):
        raise FileNotFoundError(f"Atomic skills directory not found: {args.atomic_skills_dir}")
    
    if not os.path.exists(args.cat_split_map_file):
        raise FileNotFoundError(f"cat_split_map.json not found: {args.cat_split_map_file}")
    
    if not os.path.exists(args.raw_demo_dir):
        raise FileNotFoundError(f"Raw demo directory not found: {args.raw_demo_dir}")
    
    if not os.path.exists(args.atomic_demos_path):
        raise FileNotFoundError(f"Atomic demos path not found: {args.atomic_demos_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    log_info("🎯 Phase 2: Atomic Skills Augmented Demo Generation")
    log_info("=" * 60)
    log_verbose(f"Atomic skills dir: {args.atomic_skills_dir}")
    log_verbose(f"Cat split map: {args.cat_split_map_file}")
    log_verbose(f"Raw demo dir: {args.raw_demo_dir}")
    log_verbose(f"Atomic demos path: {args.atomic_demos_path}")
    log_info(f"Output dir: {args.output_dir}")
    log_info(f"Augmentation enabled: {not args.disable_augmentation}")
    if not args.disable_augmentation:
        log_info(f"Position shift: ±{args.position_shift_range:.3f}m, Orientation shift: ±{args.orientation_shift_range_deg:.1f}°")
        log_verbose(f"Motion planner: {args.motion_planner_steps} steps, gains pos={args.motion_planner_pos_gain}/ori={args.motion_planner_ori_gain}")
        log_verbose(f"Motion planner thresholds: pos={args.position_threshold:.3f}m, ori={args.orientation_threshold_deg:.1f}°")
        iterations_per_demo = args.debug_num_iterations if (args.debug_skill or args.debug) else args.num_augmentation_iterations
        log_info(f"Augmentation iterations per demo: {iterations_per_demo}")
        if args.debug_skill:
            log_debug(f"Debug skill mode: {args.debug_skill_name}, {args.debug_num_demos} demos, {args.debug_num_iterations} iterations each")
        elif args.debug:
            log_debug(f"Debug mode: 3 skills (pick/place/atomic), {args.debug_num_demos} demos, {args.debug_num_iterations} iterations each")
    log_info("")
    
    # Load the category split mapping
    log_verbose("📁 Loading cat_split_map.json...")
    with open(args.cat_split_map_file, 'r') as f:
        cat_split_map = json.load(f)
    log_verbose(f"Loaded {len(cat_split_map)} mappings")

    # Discover atomic skill files
    log_verbose("\n🔍 Discovering atomic skill BDDL files...")
    atomic_skills = discover_atomic_skills(args.atomic_skills_dir, debug=args.debug)

    # Print summary of discovered files
    pick_count = sum(1 for _, category in atomic_skills if category == 'pick')
    place_count = sum(1 for _, category in atomic_skills if category == 'place')
    atomic_count = sum(1 for _, category in atomic_skills if category == 'atomic')

    log_info(f"Found {len(atomic_skills)} atomic skill files: {pick_count} pick, {place_count} place, {atomic_count} atomic")
    log_verbose(f"  - Pick skills: {pick_count}")
    log_verbose(f"  - Place skills: {place_count}")
    log_verbose(f"  - Atomic skills: {atomic_count}")

    # Map to source demonstration files
    log_verbose("\n🗺️  Mapping atomic skills to source demo files...")
    mappings = map_to_source_demos(atomic_skills, cat_split_map, args.raw_demo_dir)

    log_info(f"Successfully mapped {len(mappings)} atomic skills to demo files")
    
    # In debug mode, process all categories; in debug skill mode, process one specific skill; in normal mode, focus on atomic skills for Phase 2
    if args.debug_skill:
        # Find the specific skill to debug
        debug_skill_mapping = None
        for mapping in mappings:
            if mapping['skill_name'] == args.debug_skill_name:
                debug_skill_mapping = mapping
                break
        
        if debug_skill_mapping is None:
            available_skills = [m['skill_name'] for m in mappings]
            raise ValueError(f"Debug skill '{args.debug_skill_name}' not found. Available skills: {available_skills}")
        
        skills_to_process = [debug_skill_mapping]
        print(f"\n🎯 Debug Skill Mode: Processing skill '{args.debug_skill_name}' ({debug_skill_mapping['category']}) with augmentation")
        print(f"   Will process 5 demos with 2 iterations each")
    elif args.debug:
        skills_to_process = mappings  # Process all categories (pick, place, atomic) in debug mode
        print(f"\n🎯 Debug Mode: Processing {len(skills_to_process)} skills (all categories) with augmentation")
        print(f"   Skills selected for debug:")
        for mapping in skills_to_process:
            print(f"   - {mapping['skill_name']} ({mapping['category']})")
    else:
        skills_to_process = [m for m in mappings if m['category'] == 'atomic']
        print(f"\n🎯 Phase 2: Processing {len(skills_to_process)} atomic skills with augmentation")

    total_successful = 0
    total_failed = 0

    for mapping in tqdm(skills_to_process, desc="Processing skills"):
        category = mapping['category']
        if category == 'pick':
            success, message = process_pick_skill_with_augmentation(mapping, args)
        elif category == 'place':
            success, message = process_place_skill_with_augmentation(mapping, args)
        elif category == 'atomic':
            success, message = process_atomic_skill_with_augmentation(mapping, args)
        else:
            success, message = False, f"Unknown skill category: {category}"

        if success:
            total_successful += 1
            print(f"✅ {mapping['skill_name']}: {message}")
        else:
            total_failed += 1
            print(f"❌ {mapping['skill_name']}: {message}")
    
    # Save augmentation metadata
    save_augmentation_metadata(args.output_dir)
    
    print(f"\n🎯 Phase 2 Complete")
    print(f"=" * 60)
    print(f"Successfully processed: {total_successful}")
    print(f"Failed: {total_failed}")
    print(f"Total skills: {len(skills_to_process)}")

    # Enhanced augmentation success rate logging
    if hasattr(process_atomic_skill_with_augmentation, 'skill_statistics'):
        skill_stats = process_atomic_skill_with_augmentation.skill_statistics

        print(f"\n📊 Detailed Augmentation Success Analysis:")
        print(f"=" * 60)

        total_original_demos = 0
        total_augmented_demos = 0
        total_augmentation_attempts = 0
        total_trigger_found_demos = 0

        for skill_name, stats in skill_stats.items():
            total_original_demos += stats['original_demos']
            total_augmented_demos += stats['augmented_demos']
            total_augmentation_attempts += stats['total_possible_augmented']

            # Calculate trigger detection success (original demos indicate trigger was found)
            total_trigger_found_demos += stats['original_demos']

            print(f"\n{skill_name}:")
            print(f"  Original demos: {stats['original_demos']}/{stats['original_input_demos']} ({stats['original_success_rate']:.1%})")
            print(f"  Augmented demos: {stats['augmented_demos']}/{stats['total_possible_augmented']} ({stats['augmented_success_rate']:.1%})")

            # Show augmentation success rate for demos with triggers found
            if stats['original_demos'] > 0:
                augmentation_attempts_with_trigger = stats['original_demos'] * (args.debug_num_iterations if (args.debug_skill or args.debug) else args.num_augmentation_iterations)
                augmentation_success_rate_with_trigger = stats['augmented_demos'] / augmentation_attempts_with_trigger if augmentation_attempts_with_trigger > 0 else 0
                print(f"  Augmentation success (trigger found): {stats['augmented_demos']}/{augmentation_attempts_with_trigger} ({augmentation_success_rate_with_trigger:.1%})")

        print(f"\n🔢 Overall Statistics:")
        print(f"Original demo success rate: {total_original_demos}/{sum(stats['original_input_demos'] for stats in skill_stats.values())} ({total_original_demos/sum(stats['original_input_demos'] for stats in skill_stats.values()) if sum(stats['original_input_demos'] for stats in skill_stats.values()) > 0 else 0:.1%})")
        print(f"Total augmentation attempts: {total_augmentation_attempts}")
        print(f"Total augmented demos saved: {total_augmented_demos}")
        print(f"Overall augmentation success rate: {total_augmented_demos}/{total_augmentation_attempts} ({total_augmented_demos/total_augmentation_attempts if total_augmentation_attempts > 0 else 0:.1%})")

        # Calculate success rate for demos where trigger was found
        if total_trigger_found_demos > 0:
            augmentation_attempts_with_triggers = total_trigger_found_demos * (args.debug_num_iterations if (args.debug_skill or args.debug) else args.num_augmentation_iterations)
            augmentation_success_rate_with_triggers = total_augmented_demos / augmentation_attempts_with_triggers
            print(f"Augmentation success rate (trigger found): {total_augmented_demos}/{augmentation_attempts_with_triggers} ({augmentation_success_rate_with_triggers:.1%})")

    if total_successful > 0:
        print(f"\nOutput files saved to: {args.output_dir}")
        print("- *_original_demo.hdf5: Original demonstration data")
        print("- *_augmented_demo.hdf5: Augmented demonstration data")
        print("- *_original.init: Initial states for original demos")
        print("- *_augmented.init: Initial states for augmented demos")
        print("- augmentation_metadata.json: Comprehensive statistics and metadata")
        if args.debug or args.debug_skill:
            print("- debug_videos/: Debug videos (agentview and wrist cam) for all demos")
            print("- initial_images/: Initial state images (agentview and wrist cam) for investigating init_offset")




if __name__ == '__main__':
    main()
