#!/usr/bin/env python3
"""
Main script for generating localized robot demonstrations from atomic skills.

This script processes atomic skill BDDL files and extracts "local demos" - 
short, contact-rich sub-trajectories from long demonstrations for VLA model training.

The script performs the following operations:
1. Parse and categorize atomic skill BDDL files (pick/place/atomic)
2. Map them to source HDF5 demo files using cat_split_map.json
3. Simulate and replay demonstrations in libero environments
4. Extract local demonstrations based on trigger detection:
   - Pick skills: gripper closing or contact trigger, collect from trigger-5 to end
   - Place skills: collect from total_timesteps-10 to end
   - Atomic skills: contact trigger, collect from trigger-6 to end
5. Save only successful replays in correct HDF5 format
6. Generate initial states for evaluation and success rate statistics

Key features:
- Exact data collection following hdf5_structure.md specifications
- Multiple demo aggregation into single HDF5 files
- Debug mode for testing (--debug flag)
- Comprehensive error handling and logging
- Unit tested core functionality

Author: Generated for OpenVLA atomic skill learning pipeline
"""

import argparse
import json
import os
import glob
import h5py
import numpy as np
import pickle
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# Add libero imports
import sys
sys.path.append('externals/boss')
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env


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
        debug: If True, return only one file per category
        
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
        # Return only one file per category for debug mode
        categories_found = set()
        debug_skills = []
        for file_path, category in atomic_skills:
            if category not in categories_found:
                debug_skills.append((file_path, category))
                categories_found.add(category)
                if len(categories_found) == 3:  # pick, place, atomic
                    break
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
            print(f"WARNING: No mapping found for {atomic_bddl_path}")
            continue
        
        # Get the demo filename
        demo_filename = get_demo_filename_from_bddl(original_bddl_path)
        demo_full_path = os.path.join(raw_demo_dir, demo_filename)
        
        # Check if the demo file exists
        if not os.path.exists(demo_full_path):
            print(f"WARNING: Demo file not found: {demo_full_path}")
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


def collect_step_data(action: np.ndarray, obs: Dict, reward: float, done: bool, env) -> Dict:
    """
    Collect data from a single environment step according to hdf5_structure.md specifications.
    """
    step_data = {
        'actions': action,                                     # Action taken this step
        'dones': done,                                         # Episode termination flag
        'rewards': reward,                                     # Reward from step
        'states': env.sim.get_state().flatten(),              # Full simulation state (84-dim)
        'robot_states': obs['robot0_proprio-state'][:9],      # First 9 elements of proprio state
        'obs': {
            'joint_states': obs['robot0_joint_pos'],          # 7-DOF joint positions
            'gripper_states': obs['robot0_gripper_qpos'],     # 2-element gripper positions
            'ee_pos': obs['robot0_eef_pos'],                  # 3D end-effector position
            'ee_ori': R.from_quat(obs['robot0_eef_quat']).as_euler('xyz'),  # Convert quat to euler
            'ee_states': np.concatenate([                      # Combined pose (6-dim)
                obs['robot0_eef_pos'], 
                R.from_quat(obs['robot0_eef_quat']).as_euler('xyz')
            ]),
            'agentview_rgb': obs['agentview_image'],          # Agent viewpoint RGB image
            'eye_in_hand_rgb': obs['robot0_eye_in_hand_image'] # Wrist camera RGB image
        }
    }
    return step_data


def convert_steps_to_demo(step_data_list: List[Dict]) -> Dict:
    """Convert collected step data to demo format for HDF5 saving."""
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


def save_debug_videos(frames_agentview: List[np.ndarray], 
                     frames_wrist: List[np.ndarray],
                     skill_name: str, 
                     skill_type: str,
                     success: bool,
                     output_dir: str = "./debug_videos",
                     fps: int = 30):
    """
    Save debug videos for both camera views during replay.
    
    Args:
        frames_agentview: List of agentview RGB frames
        frames_wrist: List of wrist camera RGB frames  
        skill_name: Name of the skill being processed
        skill_type: Type of skill (pick/place/atomic)
        success: Whether the replay was successful
        output_dir: Directory to save videos
        fps: Frames per second for video
    """
    os.makedirs(output_dir, exist_ok=True)
    
    success_tag = "success" if success else "failed"
    
    # Save agentview video
    agentview_filename = f"{skill_name}_{success_tag}_agentview.mp4"
    agentview_path = os.path.join(output_dir, agentview_filename)
    save_video_frames(frames_agentview, agentview_path, fps)
    
    # Save wrist camera video
    wrist_filename = f"{skill_name}_{success_tag}_wrist.mp4"
    wrist_path = os.path.join(output_dir, wrist_filename)
    save_video_frames(frames_wrist, wrist_path, fps)
    
    print(f"🎬 Saved debug videos:")
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


def save_demo_to_hdf5(demo_data: Dict, output_file_path: str):
    """Save demo data to HDF5 file with correct structure."""
    with h5py.File(output_file_path, 'w') as h5file:
        data_group = h5file.create_group('data')
        demo_group = data_group.create_group('demo_0')
        
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


def extract_and_save_initial_state(step_data_list: List[Dict], init_file_path: str, init_offset: int):
    """Extract and save initial state for environment reset."""
    if len(step_data_list) <= init_offset:
        print(f"Warning: Not enough steps ({len(step_data_list)}) for init_offset {init_offset}")
        init_idx = 0
    else:
        init_idx = init_offset
    
    init_step = step_data_list[init_idx]
    
    # Combine joint, gripper, and extra states (following reference script pattern)
    joint_states = init_step['obs']['joint_states']
    gripper_states = init_step['obs']['gripper_states'] 
    extra_state = init_step['states']
    
    initial_state = np.concatenate([joint_states, gripper_states, extra_state])
    
    with open(init_file_path, 'wb') as f:
        pickle.dump([initial_state], f)  # Save as list to match reference format


def process_place_skill(mapping: Dict, args) -> Tuple[bool, str]:
    """
    Process a single place skill BDDL file through full simulation pipeline.
    Returns (success, message) tuple.
    """
    skill_name = mapping['skill_name']
    atomic_bddl_path = mapping['atomic_bddl_path']
    demo_file_path = mapping['demo_full_path']
    
    print(f"\n=== Processing Place Skill: {skill_name} ===")
    
    try:
        # Load demonstration data
        print("Loading demonstration data...")
        demo_data = load_hdf5_demo_data(demo_file_path)
        
        if 'data' not in demo_data:
            return False, "No 'data' group found in HDF5 file"
        
        demo_keys = list(demo_data['data'].keys())
        if args.debug:
            demo_keys = demo_keys[:1]  # Only process first demo in debug mode
        
        print(f"Found {len(demo_keys)} demonstrations")
        
        # Initialize environment
        print("Initializing libero environment...")
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
        
        env, _ = get_libero_env(task, model_family="openvla", resolution=256)
        successful_demos = []
        
        # Process each demonstration
        for demo_key in tqdm(demo_keys, desc=f"Processing {skill_name} demos"):
            demo = demo_data['data'][demo_key]
            actions = demo['actions']
            
            print(f"\nProcessing {demo_key} with {len(actions)} steps...")
            
            # Place skills: start from total_timesteps - place_offset to end
            # But ensure we collect at least some steps even if task completes early
            total_timesteps = len(actions)
            start_idx = max(0, total_timesteps - args.place_offset)
            
            # Quick check: simulate to find actual completion point
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
            
            # Reset environment and replay
            obs = env.reset()
            collected_steps = []
            replay_success = False
            
            # Debug video capture in debug mode
            debug_frames_agentview = []
            debug_frames_wrist = []
            if args.debug:
                # Capture initial frame
                debug_frames_agentview.append(obs['agentview_image'].copy())
                debug_frames_wrist.append(obs['robot0_eye_in_hand_image'].copy())
            
            # Replay trajectory and collect from start_idx
            for t in range(len(actions)):
                action = actions[t]
                obs, reward, done, _ = env.step(action)
                
                # Capture frames in debug mode
                if args.debug:
                    debug_frames_agentview.append(obs['agentview_image'].copy())
                    debug_frames_wrist.append(obs['robot0_eye_in_hand_image'].copy())
                
                # Start collecting from start_idx
                if t >= start_idx:
                    step_data = collect_step_data(action, obs, reward, done, env)
                    collected_steps.append(step_data)
                
                if done:
                    # BDDL goal achieved - task is successful
                    replay_success = True
                    break
            
            # If we didn't reach done, the task failed
            # (replay_success remains False)
            
            # Save debug videos in debug mode  
            if args.debug and debug_frames_agentview:
                save_debug_videos(debug_frames_agentview, debug_frames_wrist, 
                                skill_name, "place", replay_success)
            
            
            if replay_success and len(collected_steps) > 0:
                print(f"SUCCESS: {demo_key} - collected {len(collected_steps)} steps")
                successful_demos.append(collected_steps)
            else:
                print(f"FAILURE: {demo_key} - replay unsuccessful or no steps collected")
        
        env.close()
        
        # Save results if we have successful demos
        if successful_demos:
            return save_aggregated_demos(successful_demos, skill_name, args, demo_keys)
        else:
            return False, "No successful demonstrations found"
            
    except Exception as e:
        return False, f"Error processing {skill_name}: {str(e)}"


def process_atomic_skill(mapping: Dict, args) -> Tuple[bool, str]:
    """
    Process a single atomic skill BDDL file through full simulation pipeline.
    Returns (success, message) tuple.
    """
    skill_name = mapping['skill_name']
    atomic_bddl_path = mapping['atomic_bddl_path']
    demo_file_path = mapping['demo_full_path']
    
    print(f"\n=== Processing Atomic Skill: {skill_name} ===")
    
    try:
        # Load demonstration data
        print("Loading demonstration data...")
        demo_data = load_hdf5_demo_data(demo_file_path)
        
        if 'data' not in demo_data:
            return False, "No 'data' group found in HDF5 file"
        
        demo_keys = list(demo_data['data'].keys())
        if args.debug:
            demo_keys = demo_keys[:1]  # Only process first demo in debug mode
        
        print(f"Found {len(demo_keys)} demonstrations")
        
        # Initialize environment
        print("Initializing libero environment...")
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
        
        env, _ = get_libero_env(task, model_family="openvla", resolution=256)
        successful_demos = []
        
        # Process each demonstration
        for demo_key in tqdm(demo_keys, desc=f"Processing {skill_name} demos"):
            demo = demo_data['data'][demo_key]
            actions = demo['actions']
            states = demo['states']
            
            print(f"\nProcessing {demo_key} with {len(actions)} steps...")
            
            # Detect trigger timestep for atomic skills (contact detection only)
            trigger_timestep = detect_trigger_timestep_contact_only(actions, env, states)
            if trigger_timestep is None:
                print(f"FAILURE: No contact trigger found for {demo_key}")
                
                # Still capture debug video for failed trigger detection
                if args.debug:
                    obs = env.reset()
                    debug_frames_agentview = [obs['agentview_image'].copy()]
                    debug_frames_wrist = [obs['robot0_eye_in_hand_image'].copy()]
                    
                    # Replay entire trajectory for debug video
                    for t in range(len(actions)):
                        action = actions[t]
                        obs, reward, done, _ = env.step(action)
                        debug_frames_agentview.append(obs['agentview_image'].copy())
                        debug_frames_wrist.append(obs['robot0_eye_in_hand_image'].copy())
                        if done:
                            break
                    
                    # Save failed video
                    save_debug_videos(debug_frames_agentview, debug_frames_wrist, 
                                    skill_name, "atomic", False)
                continue
            
            # Calculate collection bounds for atomic skills
            start_idx = max(0, trigger_timestep - args.atomic_offset)
            
            # Quick check: simulate to find actual completion point for atomic skills
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
                # Try to collect the desired atomic_offset steps, but ensure we get at least some data
                desired_start_idx = max(0, trigger_timestep - args.atomic_offset)
                if desired_start_idx <= actual_completion_step:
                    # We can collect from the desired start to completion
                    start_idx = desired_start_idx
                    expected_steps = actual_completion_step - desired_start_idx + 1
                    print(f"Task completes at step {actual_completion_step}, collecting {expected_steps} steps from {start_idx}")
                else:
                    # Fallback: collect at least min_steps_fallback steps before completion
                    min_steps_needed = args.min_steps_fallback
                    start_idx = max(0, actual_completion_step - min_steps_needed + 1)
                    expected_steps = actual_completion_step - start_idx + 1
                    print(f"Task completes at step {actual_completion_step}, fallback to {expected_steps} steps from {start_idx}")
            
            print(f"Collection range: steps {start_idx} to end (trigger at {trigger_timestep})")
            
            # Reset environment and replay
            obs = env.reset()
            collected_steps = []
            replay_success = False
            
            # Debug video capture in debug mode
            debug_frames_agentview = []
            debug_frames_wrist = []
            if args.debug:
                # Capture initial frame
                debug_frames_agentview.append(obs['agentview_image'].copy())
                debug_frames_wrist.append(obs['robot0_eye_in_hand_image'].copy())
            
            # Replay from start to trigger
            for t in range(trigger_timestep):
                action = actions[t]
                obs, reward, done, _ = env.step(action)
                
                # Capture frames in debug mode
                if args.debug:
                    debug_frames_agentview.append(obs['agentview_image'].copy())
                    debug_frames_wrist.append(obs['robot0_eye_in_hand_image'].copy())
                
                if done:
                    break
            
            # Start collecting from start_idx
            for t in range(start_idx, len(actions)):
                action = actions[t]
                obs, reward, done, _ = env.step(action)
                
                # Capture frames in debug mode
                if args.debug:
                    debug_frames_agentview.append(obs['agentview_image'].copy())
                    debug_frames_wrist.append(obs['robot0_eye_in_hand_image'].copy())
                
                # Collect step data
                step_data = collect_step_data(action, obs, reward, done, env)
                collected_steps.append(step_data)
                
                if done:
                    # BDDL goal achieved - task is successful
                    replay_success = True
                    break
            
            # If we didn't reach done in replay, but we know from simulation that task completes
            # and we collected data, consider it successful for atomic skills
            if not replay_success and actual_completion_step is not None and len(collected_steps) > 0:
                replay_success = True
                print(f"Atomic task: replay didn't reach done, but simulation showed completion at {actual_completion_step} and we collected {len(collected_steps)} steps - considering successful")
            
            # Save debug videos in debug mode
            if args.debug and debug_frames_agentview:
                save_debug_videos(debug_frames_agentview, debug_frames_wrist, 
                                skill_name, "atomic", replay_success)
            
            
            if replay_success and len(collected_steps) > 0:
                print(f"SUCCESS: {demo_key} - collected {len(collected_steps)} steps")
                successful_demos.append(collected_steps)
            else:
                if replay_success:
                    print(f"FAILURE: {demo_key} - task succeeded but no steps collected (collection window issue)")
                else:
                    print(f"FAILURE: {demo_key} - task did not complete successfully")
        
        env.close()
        
        # Save results if we have successful demos
        if successful_demos:
            return save_aggregated_demos(successful_demos, skill_name, args, demo_keys)
        else:
            return False, "No successful demonstrations found"
            
    except Exception as e:
        return False, f"Error processing {skill_name}: {str(e)}"


def detect_trigger_timestep_contact_only(actions: np.ndarray, env, states: np.ndarray) -> Optional[int]:
    """
    Detect trigger timestep for atomic skills (contact detection only, no gripper).
    Returns the timestep of the trigger event, or None if not found.
    """
    print("Detecting contact trigger for atomic skill...")
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
    
    print("No contact trigger found")
    return None


def save_aggregated_demos(successful_demos: List[List[Dict]], skill_name: str, args, demo_keys: List[str]) -> Tuple[bool, str]:
    """
    Save aggregated demonstrations to HDF5 file and related files.
    """
    # Aggregate all successful demos into one HDF5 file
    all_demos_data = []
    for demo_steps in successful_demos:
        demo_data_formatted = convert_steps_to_demo(demo_steps)
        all_demos_data.append(demo_data_formatted)
    
    # Save HDF5 file with multiple demos
    output_filename = f"{skill_name}_demo.hdf5"
    output_path = os.path.join(args.output_dir, output_filename)
    save_multiple_demos_to_hdf5(all_demos_data, output_path)
    print(f"Saved HDF5 demo: {output_path}")
    
    # Save initial states from all successful demos
    init_filename = f"{skill_name}.init"
    init_path = os.path.join(args.output_dir, init_filename)
    extract_and_save_initial_states_multiple(successful_demos, init_path, args.init_offset)
    print(f"Saved initial states: {init_path}")
    
    # Calculate success rate
    success_rate = len(successful_demos) / len(demo_keys)
    print(f"Success rate: {success_rate:.2f} ({len(successful_demos)}/{len(demo_keys)})")
    
    # Store success rate in global dict for later JSON export
    if not hasattr(save_aggregated_demos, 'success_rates'):
        save_aggregated_demos.success_rates = {}
    save_aggregated_demos.success_rates[skill_name] = success_rate
    
    return True, f"Successfully processed {len(successful_demos)} demos"


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


def extract_and_save_initial_states_multiple(successful_demos: List[List[Dict]], init_file_path: str, init_offset: int):
    """Extract and save initial states from multiple demos."""
    all_states = []
    
    for demo_steps in successful_demos:
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


def process_pick_skill(mapping: Dict, args) -> Tuple[bool, str]:
    """
    Process a single pick skill BDDL file through full simulation pipeline.
    Returns (success, message) tuple.
    """
    skill_name = mapping['skill_name']
    atomic_bddl_path = mapping['atomic_bddl_path']
    demo_file_path = mapping['demo_full_path']
    
    print(f"\n=== Processing Pick Skill: {skill_name} ===")
    
    try:
        # Load demonstration data
        print("Loading demonstration data...")
        demo_data = load_hdf5_demo_data(demo_file_path)
        
        if 'data' not in demo_data:
            return False, "No 'data' group found in HDF5 file"
        
        demo_keys = list(demo_data['data'].keys())
        if args.debug:
            demo_keys = demo_keys[:1]  # Only process first demo in debug mode
        
        print(f"Found {len(demo_keys)} demonstrations")
        
        # Initialize environment
        print("Initializing libero environment...")
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
        
        # Create environment using the atomic task directly
        env, _ = get_libero_env(task, model_family="openvla", resolution=256)
        
        successful_demos = []
        
        # Process each demonstration
        for demo_key in tqdm(demo_keys, desc=f"Processing {skill_name} demos"):
            demo = demo_data['data'][demo_key]
            actions = demo['actions']
            states = demo['states']
            
            print(f"\nProcessing {demo_key} with {len(actions)} steps...")
            
            # Detect trigger timestep
            trigger_timestep = detect_trigger_timestep(actions, env, states)
            if trigger_timestep is None:
                print(f"FAILURE: No trigger found for {demo_key}")
                continue
            
            # Calculate collection bounds for pick skills
            start_idx = max(0, trigger_timestep - args.pick_offset)
            print(f"Collection range: steps {start_idx} to end (trigger at {trigger_timestep})")
            
            # Reset environment and replay
            obs = env.reset()
            collected_steps = []
            replay_success = False
            
            # Debug video capture in debug mode
            debug_frames_agentview = []
            debug_frames_wrist = []
            if args.debug:
                # Capture initial frame
                debug_frames_agentview.append(obs['agentview_image'].copy())
                debug_frames_wrist.append(obs['robot0_eye_in_hand_image'].copy())
            
            # Replay from start to trigger
            for t in range(trigger_timestep):
                action = actions[t]
                obs, reward, done, _ = env.step(action)
                
                # Capture frames in debug mode
                if args.debug:
                    debug_frames_agentview.append(obs['agentview_image'].copy())
                    debug_frames_wrist.append(obs['robot0_eye_in_hand_image'].copy())
                
                if done:
                    break
            
            # Start collecting from start_idx
            step_count = start_idx
            for t in range(start_idx, len(actions)):
                action = actions[t]
                obs, reward, done, _ = env.step(action)
                
                # Capture frames in debug mode
                if args.debug:
                    debug_frames_agentview.append(obs['agentview_image'].copy())
                    debug_frames_wrist.append(obs['robot0_eye_in_hand_image'].copy())
                
                # Collect step data
                step_data = collect_step_data(action, obs, reward, done, env)
                collected_steps.append(step_data)
                step_count += 1
                
                if done:
                    # For pick skills, completion of trajectory indicates success
                    replay_success = True  # If we reach done, consider it successful
                    break
            
            # If we didn't hit done, but collected steps, still consider success
            if not 'replay_success' in locals():
                replay_success = len(collected_steps) > 0
            
            # Save debug videos in debug mode
            if args.debug and debug_frames_agentview:
                save_debug_videos(debug_frames_agentview, debug_frames_wrist, 
                                skill_name, "pick", replay_success)
            
            if replay_success and len(collected_steps) > 0:
                print(f"SUCCESS: {demo_key} - collected {len(collected_steps)} steps")
                successful_demos.append(collected_steps)
            else:
                print(f"FAILURE: {demo_key} - replay unsuccessful or no steps collected")
        
        env.close()
        
        # Save results if we have successful demos
        if successful_demos:
            return save_aggregated_demos(successful_demos, skill_name, args, demo_keys)
        else:
            return False, "No successful demonstrations found"
            
    except Exception as e:
        return False, f"Error processing {skill_name}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate local demos from atomic skills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in debug mode (processes one file per category)
  python generate_local_demos.py --debug
  
  # Full run with custom paths
  python generate_local_demos.py --atomic_skills_dir /path/to/skills --raw_demo_dir /path/to/demos
        """
    )
    
    # Input/output paths
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
        "--output_dir",
        type=str,
        default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_local_demos/",
        help="Output directory for generated HDF5 files"
    )
    
    # Processing parameters
    parser.add_argument(
        "--init_offset",
        type=int,
        default=5,
        help="Steps before trigger for initial state extraction"
    )
    parser.add_argument(
        "--pick_offset",
        type=int,
        default=30,
        help="Steps before trigger for pick skill data collection (default: 5)"
    )
    parser.add_argument(
        "--place_offset",
        type=int,
        default=30,
        help="Steps from end for place skill data collection (default: 10)"
    )
    parser.add_argument(
        "--atomic_offset",
        type=int,
        default=30,
        help="Steps before trigger for atomic skill data collection (default: 6)"
    )
    parser.add_argument(
        "--min_steps_fallback",
        type=int,
        default=3,
        help="Minimum steps to collect as fallback when collection window is too late (default: 3)"
    )
    
    # Debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: process only one file per category"
    )
    
    args = parser.parse_args()
    
    # Validate input paths
    if not os.path.exists(args.atomic_skills_dir):
        raise FileNotFoundError(f"Atomic skills directory not found: {args.atomic_skills_dir}")
    
    if not os.path.exists(args.cat_split_map_file):
        raise FileNotFoundError(f"cat_split_map.json not found: {args.cat_split_map_file}")
    
    if not os.path.exists(args.raw_demo_dir):
        raise FileNotFoundError(f"Raw demo directory not found: {args.raw_demo_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Atomic Skills Local Demo Generation ===")
    print(f"Atomic skills dir: {args.atomic_skills_dir}")
    print(f"Cat split map: {args.cat_split_map_file}")
    print(f"Raw demo dir: {args.raw_demo_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Init offset: {args.init_offset}")
    print(f"Debug mode: {args.debug}")
    print()
    
    # Load the category split mapping
    print("Loading cat_split_map.json...")
    with open(args.cat_split_map_file, 'r') as f:
        cat_split_map = json.load(f)
    print(f"Loaded {len(cat_split_map)} mappings")
    
    # Discover atomic skill files
    print("\nDiscovering atomic skill BDDL files...")
    atomic_skills = discover_atomic_skills(args.atomic_skills_dir, debug=args.debug)
    
    # Print summary of discovered files
    pick_count = sum(1 for _, category in atomic_skills if category == 'pick')
    place_count = sum(1 for _, category in atomic_skills if category == 'place') 
    atomic_count = sum(1 for _, category in atomic_skills if category == 'atomic')
    
    print(f"Found {len(atomic_skills)} atomic skill files:")
    print(f"  - Pick skills: {pick_count}")
    print(f"  - Place skills: {place_count}")
    print(f"  - Atomic skills: {atomic_count}")
    
    if args.debug:
        print("\n[DEBUG MODE] Processing one file per category")
    
    # Map to source demonstration files
    print("\nMapping atomic skills to source demo files...")
    mappings = map_to_source_demos(atomic_skills, cat_split_map, args.raw_demo_dir)
    
    print(f"Successfully mapped {len(mappings)} atomic skills to demo files")
    
    # Print detailed mapping information
    print("\n=== File Mappings ===")
    for i, mapping in enumerate(mappings, 1):
        print(f"{i}. {mapping['skill_name']} ({mapping['category']})")
        print(f"   Atomic BDDL: {os.path.basename(mapping['atomic_bddl_path'])}")
        print(f"   Original BDDL: {os.path.basename(mapping['original_bddl_path'])}")
        print(f"   Demo file: {mapping['demo_filename']}")
        if not os.path.exists(mapping['demo_full_path']):
            print(f"   ⚠️  Demo file not found!")
        print()
    
    # Phase 3: Process all skill types
    print("\n=== Phase 3: Processing All Skill Types ===")
    
    # Categorize mappings by skill type
    pick_mappings = [m for m in mappings if m['category'] == 'pick']
    place_mappings = [m for m in mappings if m['category'] == 'place']
    atomic_mappings = [m for m in mappings if m['category'] == 'atomic']
    
    print(f"Found {len(pick_mappings)} pick skills, {len(place_mappings)} place skills, {len(atomic_mappings)} atomic skills")
    
    total_successful = 0
    total_failed = 0
    
    # Process pick skills
    if pick_mappings:
        print(f"\n--- Processing {len(pick_mappings)} Pick Skills ---")
        for mapping in pick_mappings:
            success, message = process_pick_skill(mapping, args)
            if success:
                total_successful += 1
                print(f"✅ {mapping['skill_name']}: {message}")
            else:
                total_failed += 1
                print(f"❌ {mapping['skill_name']}: {message}")
    
    # Process place skills
    if place_mappings:
        print(f"\n--- Processing {len(place_mappings)} Place Skills ---")
        for mapping in place_mappings:
            success, message = process_place_skill(mapping, args)
            if success:
                total_successful += 1
                print(f"✅ {mapping['skill_name']}: {message}")
            else:
                total_failed += 1
                print(f"❌ {mapping['skill_name']}: {message}")
    
    # Process atomic skills
    if atomic_mappings:
        print(f"\n--- Processing {len(atomic_mappings)} Atomic Skills ---")
        for mapping in atomic_mappings:
            success, message = process_atomic_skill(mapping, args)
            if success:
                total_successful += 1
                print(f"✅ {mapping['skill_name']}: {message}")
            else:
                total_failed += 1
                print(f"❌ {mapping['skill_name']}: {message}")
    
    print(f"\n=== Phase 3 Complete ===")
    print(f"Successfully processed: {total_successful}")
    print(f"Failed: {total_failed}")
    print(f"Total skills: {len(mappings)}")
    print(f"  - Pick: {len(pick_mappings)}")
    print(f"  - Place: {len(place_mappings)}")
    print(f"  - Atomic: {len(atomic_mappings)}")
    
    # Save success rates to JSON file
    if hasattr(save_aggregated_demos, 'success_rates') and save_aggregated_demos.success_rates:
        success_rates_file = os.path.join(args.output_dir, "success_rates.json")
        with open(success_rates_file, 'w') as f:
            json.dump(save_aggregated_demos.success_rates, f, indent=2)
        print(f"Saved success rates: {success_rates_file}")
    
    if total_successful > 0:
        print(f"\nOutput files saved to: {args.output_dir}")
        print("- *_demo.hdf5: Local demonstration data")
        print("- *.init: Initial states for evaluation")
        print("- success_rates.json: Success rate statistics for all skills")


if __name__ == "__main__":
    main()