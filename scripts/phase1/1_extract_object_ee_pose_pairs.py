#!/usr/bin/env python3
"""
Extract object-EE pose pairs from focused skills with proper MuJoCo object names.

This script processes .init files from the 18 specific skills identified in 
skills_objects_information.txt and extracts pose pairs between objects and the 
end-effector using accurate MuJoCo object names from targeted inspection.

Skills covered:
- 3 Long horizon tasks: cooking_preparation_setup, complete_kitchen_organization, 
  cooking_cleanup_and_storage
- Each containing 6-8 atomic skills for pick/place/atomic interactions

Output: Combined JSON files containing object-EE pose pairs for Phase 1 pipeline.
"""

import os
import sys
import pickle
import numpy as np
import json
import re
import h5py
import csv
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

# Add project paths
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')

# LIBERO imports
from libero.libero.envs import OffScreenRenderEnv
from experiments.robot.libero.libero_utils import get_libero_env

# Local imports
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase1')

# Import get_object_pose function
import importlib.util
spec_detector = importlib.util.spec_from_file_location("contact_detector", "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase1/3_contact_object_detector.py")
contact_detector = importlib.util.module_from_spec(spec_detector)
spec_detector.loader.exec_module(contact_detector)
get_object_pose = contact_detector.get_object_pose

# Load MuJoCo object mappings
with open('/mnt/arc/yygx/pkgs_baselines/openvla-oft/prompts/from_ai/targeted_mujoco_objects.json', 'r') as f:
    MUJOCO_MAPPINGS = json.load(f)


def detect_target_object(env, skill_name: str, ee_body_name: str = "gripper0_eef") -> Optional[str]:
    """
    Detect the target object for pose extraction based on skill type and name.
    
    Args:
        env: Environment object
        skill_name: Name of the skill being executed
        ee_body_name: Name of the end-effector body
        
    Returns:
        Name of the target object, or None if not found
    """
    skill_lower = skill_name.lower()
    
    print(f"    🔍 Skill type analysis: {skill_name}")
    
    # Determine skill type and use appropriate detection strategy
    if '_place' in skill_lower or skill_lower.startswith('place'):
        print("    📦 Place skill detected - using name-based target detection")
        return _detect_place_target_from_name(env, skill_name)
    # TODO: shall swap atomic to else statement
    elif any(action in skill_lower for action in ['open', 'close', 'turn']):
        print("    🔧 Atomic skill detected - finding main object (not handle)")
        return _detect_atomic_target_from_name(env, skill_name)
    else:
        print("    🤏 Pick skill detected - using contact detection")
        return _detect_contact_target(env, ee_body_name)


def _detect_place_target_from_name(env, skill_name: str) -> Optional[str]:
    """
    For place skills, extract target object/container from skill name.
    E.g., 'put_frying_pan_under_cabinet_shelf' -> find cabinet shelf
    """
    skill_lower = skill_name.lower()
    model = env.sim.model
    
    # Extract target container/surface from skill name
    target_keywords = []
    
    print(f"      📋 Analyzing place skill: {skill_name}")
    
    # Common placement targets with their keywords
    if 'cabinet' in skill_lower:
        if 'shelf' in skill_lower:
            target_keywords = ['cabinet', 'shelf']
        elif 'drawer' in skill_lower:
            if 'top' in skill_lower:
                target_keywords = ['cabinet', 'drawer', 'top']
            elif 'bottom' in skill_lower:
                target_keywords = ['cabinet', 'drawer', 'bottom']
            else:
                target_keywords = ['cabinet', 'drawer']
        else:
            target_keywords = ['cabinet']
    elif 'stove' in skill_lower:
        target_keywords = ['stove']
    elif 'plate' in skill_lower and not any(x in skill_lower for x in ['black_plate', 'white_plate']):
        target_keywords = ['plate']
    elif 'wine_rack' in skill_lower:
        target_keywords = ['wine', 'rack']
    elif 'microwave' in skill_lower:
        target_keywords = ['microwave']
    else:
        # Try to extract from "on_the_X" or "in_the_X" patterns
        patterns = [r'on_the_([a-z_]+)', r'in_the_([a-z_]+)', r'under_the_([a-z_]+)', r'on_top_of_the_([a-z_]+)']
        for pattern in patterns:
            match = re.search(pattern, skill_lower)
            if match:
                target_keywords = match.group(1).split('_')
                break
    
    if not target_keywords:
        print(f"      ⚠️  Could not extract target keywords from place skill: {skill_name}")
        return None
    
    print(f"      🎯 Target keywords: {target_keywords}")
    
    # Find matching object in simulation
    best_match = None
    best_score = 0
    candidates = []
    
    for i in range(model.nbody):
        body_name = model.body_id2name(i)
        if body_name:
            body_lower = body_name.lower()
            # Count how many target keywords are present
            score = sum(1 for keyword in target_keywords if keyword in body_lower)
            if score > 0:
                candidates.append((body_name, score, body_lower))
                if score > best_score:
                    best_match = body_name
                    best_score = score
    
    print(f"      📊 Found {len(candidates)} candidates:")
    for name, score, lower in candidates[:5]:  # Show top 5
        print(f"        - {name} (score: {score}/{len(target_keywords)})")
    
    if best_match:
        print(f"      ✅ Place target detected: {best_match}")
        return best_match
    else:
        print(f"      ⚠️  No matching place target found for keywords: {target_keywords}")
        return None


def _detect_atomic_target_from_name(env, skill_name: str) -> Optional[str]:
    """
    For atomic skills, find the main object (not handle) from skill name.
    E.g., 'open_top_drawer_of_cabinet' -> find the drawer or cabinet
    """
    skill_lower = skill_name.lower()
    model = env.sim.model
    
    # Extract main object keywords
    target_keywords = []
    
    print(f"      📋 Analyzing atomic skill: {skill_name}")
    
    if 'drawer' in skill_lower:
        if 'top' in skill_lower:
            target_keywords = ['drawer', 'top'] 
        elif 'bottom' in skill_lower:
            target_keywords = ['drawer', 'bottom']
        else:
            target_keywords = ['drawer']
    elif 'microwave' in skill_lower:
        target_keywords = ['microwave']
    elif 'cabinet' in skill_lower:
        target_keywords = ['cabinet']
    elif 'stove' in skill_lower:
        target_keywords = ['stove']
    else:
        # Extract from action patterns
        patterns = [r'open_(?:the_)?([a-z_]+)', r'close_(?:the_)?([a-z_]+)', r'turn_\w+_(?:the_)?([a-z_]+)']
        for pattern in patterns:
            match = re.search(pattern, skill_lower)
            if match:
                target_keywords = match.group(1).split('_')
                # Remove common words
                target_keywords = [kw for kw in target_keywords if kw not in ['of', 'the']]
                break
    
    if not target_keywords:
        print(f"      ⚠️  Could not extract target keywords from atomic skill: {skill_name}")
        return None
    
    print(f"      🎯 Target keywords: {target_keywords}")
    
    # Find matching object in simulation, preferring main objects over handles
    best_match = None
    best_score = 0
    candidates = []
    
    for i in range(model.nbody):
        body_name = model.body_id2name(i)
        if body_name:
            body_lower = body_name.lower()
            
            # Skip handles and small parts for main object detection
            if any(handle_word in body_lower for handle_word in ['handle', 'knob', 'button']):
                continue
                
            # Count how many target keywords are present
            score = sum(1 for keyword in target_keywords if keyword in body_lower)
            if score > 0:
                candidates.append((body_name, score, body_lower))
                if score > best_score:
                    best_match = body_name
                    best_score = score
    
    print(f"      📊 Found {len(candidates)} candidates (excluding handles):")
    for name, score, lower in candidates[:5]:  # Show top 5
        print(f"        - {name} (score: {score}/{len(target_keywords)})")
    
    if best_match:
        print(f"      ✅ Atomic target detected: {best_match}")
        return best_match
    else:
        print(f"      ⚠️  No matching atomic target found for keywords: {target_keywords}")
        # Fallback: try contact detection
        print(f"      🔄 Falling back to contact detection...")
        return _detect_contact_target(env, "gripper0_eef")


def _detect_contact_target(env, ee_body_name: str) -> Optional[str]:
    """
    Detect target object using contact detection (for pick skills).
    """
    # Method 1: Check for actual MuJoCo contacts
    contact_object = _detect_contact_from_mujoco(env, ee_body_name)
    if contact_object is not None:
        return contact_object
    
    # Method 2: Fallback to closest object detection
    return _detect_closest_manipulable_object(env, ee_body_name)


def _detect_contact_from_mujoco(env, ee_body_name: str) -> Optional[str]:
    """
    Detect contact using actual MuJoCo contact data.
    """
    model = env.sim.model
    data = env.sim.data
    
    # Get end-effector geom IDs
    ee_geom_ids = []
    for i in range(model.ngeom):
        geom_name = model.geom_id2name(i)
        if geom_name and ee_body_name.lower() in geom_name.lower():
            ee_geom_ids.append(i)
    
    if not ee_geom_ids:
        return None
    
    # Check all contacts
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_id = contact.geom1
        geom2_id = contact.geom2
        
        # Check if either geom is part of the end-effector
        if geom1_id in ee_geom_ids or geom2_id in ee_geom_ids:
            # Find the other geom (the object)
            other_geom_id = geom2_id if geom1_id in ee_geom_ids else geom1_id
            other_geom_name = model.geom_id2name(other_geom_id)
            
            if other_geom_name:
                # Extract object name from geom name
                object_keywords = ['bowl', 'plate', 'mug', 'bottle', 'pan', 'pot', 'pudding', 'ketchup']
                for keyword in object_keywords:
                    if keyword in other_geom_name.lower():
                        print(f"      ✅ Contact detected with: {other_geom_name}")
                        return other_geom_name
    
    return None


def _detect_closest_manipulable_object(env, ee_body_name: str) -> Optional[str]:
    """
    Fallback: Find closest manipulable object to end-effector.
    """
    model = env.sim.model
    data = env.sim.data
    
    # Get end-effector position
    ee_body_id = None
    for i in range(model.nbody):
        body_name = model.body_id2name(i)
        if body_name and ee_body_name.lower() in body_name.lower():
            ee_body_id = i
            break
    
    if ee_body_id is None:
        return None
    
    ee_pos = data.body_xpos[ee_body_id]
    
    # Find closest manipulable object
    min_distance = float('inf')
    closest_object = None
    
    object_keywords = ['bowl', 'plate', 'mug', 'bottle', 'pan', 'pot', 'pudding', 'ketchup']
    
    for i in range(model.nbody):
        body_name = model.body_id2name(i)
        if body_name and any(keyword in body_name.lower() for keyword in object_keywords):
            obj_pos = data.body_xpos[i]
            distance = np.linalg.norm(ee_pos - obj_pos)
            if distance < min_distance:
                min_distance = distance
                closest_object = body_name
    
    if closest_object:
        print(f"      ✅ Closest object detected: {closest_object} (distance: {min_distance:.3f})")
    
    return closest_object


def extract_pose_pairs_from_init(init_file_path: str, output_dir: str) -> bool:
    """
    Extract object-EE pose pairs from a single .init file using skill-aware detection.
    
    Args:
        init_file_path: Path to the .init file
        output_dir: Directory to save extracted pose pairs
        
    Returns:
        True if extraction successful, False otherwise
    """
    try:
        # Extract skill name from filename
        skill_name = Path(init_file_path).stem
        
        print(f"\n🔍 Processing skill: {skill_name}")
        
        # Load initial states from .init file
        with open(init_file_path, 'rb') as f:
            initial_states = pickle.load(f)
        
        if not initial_states or len(initial_states) == 0:
            print(f"  ⚠️  No initial states found in {init_file_path}")
            return False
        
        print(f"  📊 Found {len(initial_states)} initial states")
        
        # Process each initial state
        pose_pairs = []
        
        for state_idx, initial_state in enumerate(initial_states):
            if state_idx >= 5:  # Limit to first 5 states for efficiency
                break
                
            print(f"  🔄 Processing state {state_idx + 1}/{min(len(initial_states), 5)}...")
            
            try:
                # Load corresponding BDDL file from atomic_skills directory
                bddl_file = f"/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss/libero/libero/bddl_files/atomic_skills/{skill_name}.bddl"
                
                if not os.path.exists(bddl_file):
                    print(f"    ⚠️  BDDL file not found: {bddl_file}")
                    continue
                
                # Initialize environment
                env_args = {
                    'bddl_file_name': bddl_file,
                    'camera_heights': 256,
                    'camera_widths': 256
                }
                
                env = OffScreenRenderEnv(**env_args)
                env.reset()
                
                # FIXED: Skip first 9 dims (joint+gripper) and use only simulation state (84-dim)
                sim_state_only = initial_state[9:]
                env.sim.set_state_from_flattened(sim_state_only)
                env.sim.forward()
                
                # Detect target object using skill-aware detection
                target_obj = detect_target_object(env, skill_name)
                
                if target_obj is None:
                    print(f"    ⚠️  No target object detected in state {state_idx}")
                    env.close()
                    continue
                
                print(f"    🎯 Target object: {target_obj}")
                
                # Get object pose
                obj_pos, obj_quat = get_object_pose(env, target_obj)
                if obj_pos is None:
                    print(f"    ⚠️  Could not get pose for object: {target_obj}")
                    env.close()
                    continue
                
                # Get end-effector pose from simulation data directly
                data = env.sim.data
                model = env.sim.model
                
                # Find end-effector body
                ee_body_id = None
                possible_ee_names = ["gripper0_eef", "robot0_eef", "gripper_eef", "eef"]
                for ee_name in possible_ee_names:
                    try:
                        if ee_name in [model.body_id2name(i) for i in range(model.nbody)]:
                            ee_body_id = model.body_name2id(ee_name)
                            break
                    except:
                        continue
                
                if ee_body_id is None:
                    print(f"    ⚠️  End-effector not found")
                    env.close()
                    continue
                
                ee_pos = data.body_xpos[ee_body_id].copy()
                ee_quat = data.body_xquat[ee_body_id].copy()
                
                # Store pose pair
                pose_pair = {
                    'skill_name': skill_name,
                    'object_name': target_obj,
                    'object_pos': obj_pos.tolist(),
                    'object_quat': obj_quat.tolist(),
                    'ee_pos': ee_pos.tolist(),
                    'ee_quat': ee_quat.tolist(),
                    'state_index': state_idx
                }
                
                pose_pairs.append(pose_pair)
                print(f"    ✅ Extracted pose pair for {target_obj}")
                
                env.close()
                
            except Exception as e:
                print(f"    ❌ Error processing state {state_idx}: {e}")
                continue
        
        if not pose_pairs:
            print(f"  ❌ No valid pose pairs extracted for {skill_name}")
            return False
        
        # Save pose pairs
        output_file = os.path.join(output_dir, f"{skill_name}_pose_pairs.json")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(pose_pairs, f, indent=2)
        
        print(f"  ✅ Saved {len(pose_pairs)} pose pairs to {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {init_file_path}: {e}")
        return False


def load_skills_mapping() -> Dict[str, str]:
    """
    Load skills mapping from CSV file.
    
    Returns:
        Dictionary mapping init filename to target object name
    """
    csv_file = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/prompts/for_ai/skills_objects_mapping_updated.csv"
    mapping = {}
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                init_files = row['init_files'].strip()
                object_name = row['object'].strip()
                
                # Parse multiple init files separated by ';'
                for init_file in init_files.split(';'):
                    init_file = init_file.strip()
                    if init_file:
                        mapping[init_file] = object_name
        
        print(f"📋 Loaded mapping for {len(mapping)} init files from CSV")
        return mapping
        
    except Exception as e:
        print(f"❌ Error loading CSV mapping: {e}")
        return {}


def extract_focused_pose_pairs(init_file_path: str, target_object: str, output_dir: str) -> Tuple[bool, int, float]:
    """
    Extract pose pairs for focused skills using pre-defined target object.
    
    Args:
        init_file_path: Path to the .init file
        target_object: Pre-defined target object name from CSV
        output_dir: Directory to save extracted pose pairs
        
    Returns:
        (success, num_pairs, avg_distance) tuple
    """
    try:
        skill_name = Path(init_file_path).stem
        print(f"\n🎯 FOCUSED: Processing {skill_name}")
        print(f"  🎯 Target object (from CSV): {target_object}")
        
        # Load initial states
        with open(init_file_path, 'rb') as f:
            initial_states = pickle.load(f)
        
        if not initial_states:
            print(f"  ❌ No initial states found")
            return False, 0, 0.0
        
        print(f"  📊 Processing {len(initial_states)} initial states")
        
        # Process each initial state
        pose_pairs = []
        distances = []
        
        for state_idx, initial_state in enumerate(initial_states):
            if state_idx >= 3:  # Process up to 3 states for focused mode (faster)
                break
                
            print(f"  🔄 State {state_idx + 1}/{min(len(initial_states), 3)}...", end=' ')
            
            try:
                # Load BDDL file
                bddl_file = f"/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss/libero/libero/bddl_files/atomic_skills/{skill_name}.bddl"
                
                if not os.path.exists(bddl_file):
                    print(f"❌ BDDL not found")
                    continue
                
                # Initialize environment
                env_args = {
                    'bddl_file_name': bddl_file,
                    'camera_heights': 256,
                    'camera_widths': 256
                }
                
                env = OffScreenRenderEnv(**env_args)
                env.reset()
                
                # FIXED: Skip first 9 dims (joint+gripper) and use only simulation state (84-dim)
                sim_state_only = initial_state[9:]
                env.sim.set_state_from_flattened(sim_state_only)
                env.sim.forward()
                
                # SMART OBJECT DETECTION: Try multiple methods
                actual_target_object = None
                obj_pose_result = None
                
                # Method 1: Try exact name from CSV
                obj_pose_result = get_object_pose(env, target_object)
                if obj_pose_result is not None:
                    actual_target_object = target_object
                    print(f"✅ Found exact: {target_object}")
                else:
                    print(f"❌ Exact name '{target_object}' not found")
                    
                    # Method 2: Try fuzzy pattern matching (e.g., akita_black_bowl_*_main)
                    model = env.sim.model
                    available_objects = []
                    for i in range(model.nbody):
                        body_name = model.body_id2name(i)
                        if body_name:
                            available_objects.append(body_name)
                    
                    # Create fuzzy pattern from target_object
                    import re
                    # Convert "akita_black_bowl_1_main" -> "akita_black_bowl_*_main"
                    fuzzy_pattern = re.sub(r'_\d+_', '_.*_', target_object)
                    fuzzy_regex = fuzzy_pattern.replace('*', r'\d+')
                    
                    print(f"    🔍 Trying fuzzy pattern: {fuzzy_pattern}")
                    fuzzy_matches = []
                    for obj_name in available_objects:
                        if re.match(fuzzy_regex, obj_name):
                            fuzzy_matches.append(obj_name)
                    
                    if fuzzy_matches:
                        # Try first fuzzy match
                        fuzzy_target = fuzzy_matches[0]
                        obj_pose_result = get_object_pose(env, fuzzy_target)
                        if obj_pose_result is not None:
                            actual_target_object = fuzzy_target
                            print(f"    ✅ Found fuzzy match: {fuzzy_target}")
                        else:
                            print(f"    ❌ Fuzzy match failed: {fuzzy_target}")
                    else:
                        print(f"    ❌ No fuzzy matches found")
                    
                    # Method 3: Fallback to skill-based detection
                    if obj_pose_result is None:
                        print(f"    🔍 Trying skill-based detection...")
                        skill_name = Path(init_file_path).stem
                        detected_object = detect_target_object(env, skill_name)
                        if detected_object:
                            obj_pose_result = get_object_pose(env, detected_object)
                            if obj_pose_result is not None:
                                actual_target_object = detected_object
                                print(f"    ✅ Found via detection: {detected_object}")
                            else:
                                print(f"    ❌ Detection failed: {detected_object}")
                
                # Final check
                if obj_pose_result is None:
                    print(f"    ❌ All methods failed!")
                    print(f"    🔍 Available objects in scene:")
                    for obj in [o for o in available_objects if any(kw in o.lower() for kw in ['bowl', 'plate', 'mug', 'bottle', 'pan', 'pot'])][:5]:
                        print(f"      - {obj}")
                    env.close()
                    continue
                
                obj_pos, obj_quat = obj_pose_result
                
                # Get end-effector pose from simulation data directly
                data = env.sim.data
                model = env.sim.model
                
                # Find end-effector body
                ee_body_id = None
                possible_ee_names = ["gripper0_eef", "robot0_eef", "gripper_eef", "eef"]
                for ee_name in possible_ee_names:
                    try:
                        if ee_name in [model.body_id2name(i) for i in range(model.nbody)]:
                            ee_body_id = model.body_name2id(ee_name)
                            break
                    except:
                        continue
                
                if ee_body_id is None:
                    print(f"❌ EE not found")
                    env.close()
                    continue
                
                ee_pos = data.body_xpos[ee_body_id].copy()
                ee_quat = data.body_xquat[ee_body_id].copy()
                
                # Calculate distance
                distance = np.linalg.norm(ee_pos - obj_pos)
                distances.append(distance)
                
                # Create pose pair
                pose_pair = {
                    'init_state_id': state_idx,
                    'object_pose': {
                        'object_name': target_object,
                        'position': obj_pos.tolist(),
                        'quaternion': obj_quat.tolist()
                    },
                    'ee_pose': {
                        'position': ee_pos.tolist(),
                        'quaternion': ee_quat.tolist()
                    },
                    'distance': float(distance)
                }
                
                pose_pairs.append(pose_pair)
                print(f"✅ d={distance:.3f}m")
                
                env.close()
                
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        if not pose_pairs:
            print(f"  ❌ No valid pose pairs extracted")
            return False, 0, 0.0
        
        # Calculate statistics
        avg_distance = np.mean(distances)
        num_pairs = len(pose_pairs)
        
        # Save pose pairs as pickle file
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{skill_name}_pose_pairs.pkl")
        
        with open(output_file, 'wb') as f:
            pickle.dump(pose_pairs, f)
        
        print(f"  ✅ Saved {num_pairs} pairs, avg distance: {avg_distance:.3f}m")
        print(f"  💾 Output: {output_file}")
        
        return True, num_pairs, avg_distance
        
    except Exception as e:
        print(f"❌ Error processing {init_file_path}: {e}")
        return False, 0, 0.0


def debug_target_detection(init_file_path: str) -> Optional[str]:
    """
    Debug function to test target object detection for a single .init file.
    
    Args:
        init_file_path: Path to the .init file
        
    Returns:
        Detected target object name, or None if failed
    """
    try:
        # Extract skill name from filename
        skill_name = Path(init_file_path).stem
        
        print(f"\n🐛 DEBUG: Testing {skill_name}")
        
        # Load first initial state only
        with open(init_file_path, 'rb') as f:
            initial_states = pickle.load(f)
        
        if not initial_states or len(initial_states) == 0:
            print(f"  ❌ No initial states found")
            return None
        
        print(f"  📊 Using first state from {len(initial_states)} available states")
        
        try:
            # Load corresponding BDDL file from atomic_skills directory
            bddl_file = f"/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss/libero/libero/bddl_files/atomic_skills/{skill_name}.bddl"
            
            if not os.path.exists(bddl_file):
                print(f"  ❌ BDDL file not found: {bddl_file}")
                return None
            
            # Initialize environment
            env_args = {
                'bddl_file_name': bddl_file,
                'camera_heights': 256,
                'camera_widths': 256
            }
            
            env = OffScreenRenderEnv(**env_args)
            env.reset()
            
            # FIXED: Skip first 9 dims (joint+gripper) and use only simulation state (84-dim)
            sim_state_only = initial_states[0][9:]
            env.sim.set_state_from_flattened(sim_state_only)
            env.sim.forward()
            
            # Test target object detection
            target_obj = detect_target_object(env, skill_name)
            
            env.close()
            
            if target_obj:
                print(f"  ✅ TARGET DETECTED: {target_obj}")
                return target_obj
            else:
                print(f"  ❌ NO TARGET DETECTED")
                return None
                
        except Exception as e:
            print(f"  ❌ Error during detection: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to debug {init_file_path}: {e}")
        return None


def main():
    """Main function to process all .init files."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract object-EE pose pairs or debug target detection')
    parser.add_argument('--debug', action='store_true', 
                        help='Debug mode: only test target object detection')
    parser.add_argument('--focused', action='store_true',
                        help='Focused mode: only process skills of interest from CSV mapping')
    parser.add_argument('--focused_debug', type=str,
                        help='Focused debug mode: only process one specific init file (e.g., KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl_pick.init)')
    args = parser.parse_args()
    
    # Input directory containing .init files
    input_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_local_demos"
    
    # Output directory for pose pairs
    output_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/object_ee_pose_pairs"
    focused_output_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_local_demos/ee_target_obj_pairs/only_skills_of_interest"
    debug_output_file = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/debug_target_detection.json"
    
    if args.debug:
        print("🐛 DEBUG MODE: Testing target object detection only")
        print(f"📁 Input directory: {input_dir}")
        print(f"💾 Debug results will be saved to: {debug_output_file}")
    elif args.focused_debug:
        print(f"🔍 FOCUSED DEBUG MODE: Processing single file: {args.focused_debug}")
        print(f"📁 Input directory: {input_dir}")
        print(f"💾 Output directory: {focused_output_dir}")
    elif args.focused:
        print("🎯 FOCUSED MODE: Processing skills of interest only")
        print(f"📁 Input directory: {input_dir}")
        print(f"💾 Output directory: {focused_output_dir}")
    else:
        print("🚀 Starting object-EE pose pairs extraction with skill-aware detection")
        print(f"📁 Input directory: {input_dir}")
        print(f"💾 Output directory: {output_dir}")
    
    # Find all .init files
    init_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.init'):
                init_files.append(os.path.join(root, file))
    
    print(f"\n📊 Found {len(init_files)} .init files to process")
    
    if not init_files:
        print("❌ No .init files found!")
        return
    
    if args.debug:
        # Debug mode: test target detection only
        debug_results = {}
        successful = 0
        failed = 0
        
        for i, init_file in enumerate(init_files):
            print(f"\n{'='*80}")
            print(f"🐛 DEBUG {i+1}/{len(init_files)}: {os.path.basename(init_file)}")
            
            try:
                detected_object = debug_target_detection(init_file)
                init_basename = os.path.basename(init_file)
                debug_results[init_basename] = detected_object
                
                if detected_object:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"❌ Failed to debug {init_file}: {e}")
                init_basename = os.path.basename(init_file)
                debug_results[init_basename] = None
                failed += 1
        
        # Save debug results
        os.makedirs(os.path.dirname(debug_output_file), exist_ok=True)
        with open(debug_output_file, 'w') as f:
            json.dump(debug_results, f, indent=2, sort_keys=True)
        
        # Debug Summary
        print(f"\n{'='*80}")
        print("🐛 DEBUG SUMMARY")
        print(f"✅ Successful detections: {successful}")
        print(f"❌ Failed detections: {failed}")
        print(f"📈 Detection rate: {successful/(successful+failed)*100:.1f}%")
        print(f"💾 Debug results saved to: {debug_output_file}")
        
    elif args.focused:
        # Focused mode: process only skills of interest
        skills_mapping = load_skills_mapping()
        if not skills_mapping:
            print("❌ No skills mapping loaded, cannot proceed with focused mode")
            return
        
        # Filter init files to only those in our mapping
        focused_init_files = []
        for init_file in init_files:
            init_basename = os.path.basename(init_file)
            if init_basename in skills_mapping:
                focused_init_files.append(init_file)
        
        print(f"\n🎯 Found {len(focused_init_files)} files of interest (out of {len(init_files)} total)")
        
        if not focused_init_files:
            print("❌ No files of interest found!")
            return
        
        # Process focused skills
        skill_stats = {}
        total_pairs = 0
        successful_skills = 0
        failed_skills = 0
        
        for i, init_file in enumerate(focused_init_files):
            init_basename = os.path.basename(init_file)
            target_object = skills_mapping[init_basename]
            
            print(f"\n{'='*80}")
            print(f"🎯 FOCUSED {i+1}/{len(focused_init_files)}: {init_basename}")
            
            try:
                success, num_pairs, avg_distance = extract_focused_pose_pairs(
                    init_file, target_object, focused_output_dir
                )
                
                skill_name = Path(init_file).stem
                if success:
                    skill_stats[skill_name] = {
                        'pairs': num_pairs,
                        'avg_distance': avg_distance,
                        'target_object': target_object
                    }
                    total_pairs += num_pairs
                    successful_skills += 1
                else:
                    failed_skills += 1
                    
            except Exception as e:
                print(f"❌ Failed to process {init_file}: {e}")
                failed_skills += 1
        
        # Print focused statistics
        print(f"\n{'='*80}")
        print("🎯 FOCUSED MODE STATISTICS")
        print(f"✅ Successful skills: {successful_skills}")
        print(f"❌ Failed skills: {failed_skills}")
        print(f"📈 Success rate: {successful_skills/(successful_skills+failed_skills)*100:.1f}%")
        print(f"📊 Total pose pairs: {total_pairs}")
        
        print(f"\n📋 PER-SKILL STATISTICS:")
        print(f"{'Skill Name':<60} {'Pairs':<8} {'Avg Dist':<12} {'Target Object'}")
        print("="*100)
        
        for skill_name, stats in sorted(skill_stats.items()):
            pairs = stats['pairs']
            distance = stats['avg_distance']
            obj = stats['target_object']
            print(f"{skill_name:<60} {pairs:<8} {distance:<12.3f} {obj}")
        
        if successful_skills > 0:
            avg_pairs_per_skill = total_pairs / successful_skills
            overall_avg_distance = np.mean([stats['avg_distance'] for stats in skill_stats.values()])
            print(f"\n📊 OVERALL AVERAGES:")
            print(f"  Average pairs per skill: {avg_pairs_per_skill:.1f}")
            print(f"  Overall average EE-object distance: {overall_avg_distance:.3f}m")
            print(f"\n💾 Pose pairs saved to: {focused_output_dir}")
            print("🎉 Ready for Phase 1 pipeline!")
        
    elif args.focused_debug:
        # Focused debug mode: process only the specified init file
        skills_mapping = load_skills_mapping()
        if not skills_mapping:
            print("❌ No skills mapping loaded, cannot proceed with focused debug mode")
            return
        
        target_init_file = args.focused_debug
        if target_init_file not in skills_mapping:
            print(f"❌ Init file '{target_init_file}' not found in CSV mapping")
            print(f"💡 Available files in CSV: {list(skills_mapping.keys())[:5]}...")
            return
        
        # Find the actual init file path
        found_init_path = None
        for init_file in init_files:
            if os.path.basename(init_file) == target_init_file:
                found_init_path = init_file
                break
        
        if not found_init_path:
            print(f"❌ Init file '{target_init_file}' not found in filesystem")
            print(f"📁 Searched in: {input_dir}")
            return
        
        target_object = skills_mapping[target_init_file]
        
        print(f"\n🔍 FOCUSED DEBUG: Processing single file")
        print(f"  📁 File: {target_init_file}")
        print(f"  🎯 Target object (from CSV): {target_object}")
        print(f"  📍 Full path: {found_init_path}")
        
        # Process the single file with detailed output
        try:
            success, num_pairs, avg_distance = extract_focused_pose_pairs(
                found_init_path, target_object, focused_output_dir
            )
            
            print(f"\n🔍 FOCUSED DEBUG RESULTS:")
            print(f"  ✅ Success: {success}")
            print(f"  📊 Pairs extracted: {num_pairs}")
            print(f"  📏 Average distance: {avg_distance:.3f}m" if success else "  📏 No distance data")
            
            if success:
                print(f"  💾 Output saved to: {focused_output_dir}")
                print("  🎉 Single file debug completed successfully!")
            else:
                print("  ❌ Single file debug failed!")
                
        except Exception as e:
            print(f"❌ Error during focused debug: {e}")
            import traceback
            traceback.print_exc()
        
    else:
        # Normal mode: full extraction
        successful = 0
        failed = 0
        
        for i, init_file in enumerate(init_files):
            print(f"\n{'='*60}")
            print(f"📁 Processing file {i+1}/{len(init_files)}: {os.path.basename(init_file)}")
            
            try:
                success = extract_pose_pairs_from_init(init_file, output_dir)
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ Failed to process {init_file}: {e}")
                failed += 1
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 EXTRACTION SUMMARY")
        print(f"✅ Successfully processed: {successful} files")
        print(f"❌ Failed to process: {failed} files")
        print(f"📈 Success rate: {successful/(successful+failed)*100:.1f}%")
        
        if successful > 0:
            print(f"\n💾 Pose pairs saved to: {output_dir}")
            print("🎉 Ready for GT pose calculation!")


if __name__ == "__main__":
    main()