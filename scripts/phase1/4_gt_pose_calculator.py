#!/usr/bin/env python3
"""
GT Pose Calculator for Long Horizon Pipeline.

This module loads object-EE pose pairs from learned demonstrations and calculates
target end-effector poses in current scene coordinates. Handles object instance
tracking to support multiple objects of same type (e.g., stove1 vs stove2).

Key Functions:
- calculate_gt_local_pose(): Main pipeline function - returns target EE pose
- GTLocalPoseCalculator: Main class for pose calculation
"""

import os
import sys
import pickle
import numpy as np
import json
import random
import csv
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# Add project paths
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')


class ObjectInstanceTracker:
    """Tracks which object instances are used for each skill execution."""
    
    def __init__(self, csv_file: str = None):
        self.skill_object_usage = {}
        self.used_objects = set()
        self.skill_execution_count = {}  # Track how many times each skill has been executed
        self.used_pattern_instances = {}  # Track how many times each pattern has been used
        self.csv_mappings = self._load_csv_mappings(csv_file)
    
    def _load_csv_mappings(self, csv_file: str = None) -> Dict[str, str]:
        """Load skill-to-object mappings from CSV file."""
        if csv_file is None:
            csv_file = '/mnt/arc/yygx/pkgs_baselines/openvla-oft/prompts/for_ai/skills_objects_mapping_updated.csv'
        
        mappings = {}
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    skill_name = row['skill_name'].strip().strip('"')  # Remove quotes
                    target_object = row['object'].strip()
                    if skill_name and target_object:
                        mappings[skill_name] = target_object
            print(f"📋 Loaded {len(mappings)} skill-to-object mappings from CSV")
        except Exception as e:
            print(f"❌ Error loading CSV mappings: {e}")
        
        return mappings
    
    def get_next_target_object(self, skill_language: str, available_objects: List[str]) -> Optional[str]:
        """
        Get next available object for a skill using CSV mappings with sequential occurrence tracking.
        
        Args:
            skill_language: Language description like "place frying pan on the stove"
            available_objects: List of available objects in scene
            
        Returns:
            Next target object name or None if no suitable object found
        """
        # First try CSV mapping with sequential occurrence tracking
        csv_target = self._get_target_from_csv_mapping(skill_language, available_objects)
        if csv_target:
            return csv_target
        # Fall back to original pattern matching for skills not in CSV
        return self._get_target_from_pattern_matching(skill_language, available_objects)
    
    def _get_target_from_csv_mapping(self, skill_language: str, available_objects: List[str]) -> Optional[str]:
        """Get target object using CSV mappings with sequential occurrence handling."""
        if not self.csv_mappings:
            return None
        
        # Extract base skill name (remove trailing numbers and extra spaces)
        skill_base = skill_language.strip()
        
        # First try exact match (no count increment yet)
        if skill_base in self.csv_mappings:
            target_pattern = self.csv_mappings[skill_base]
            
            # Find next available instance of this pattern
            selected_obj = self._find_available_instance(target_pattern, available_objects)
            if selected_obj:
                print(f"🎯 Selected {selected_obj} for skill: {skill_language} (CSV mapping: {skill_base} → {target_pattern})")
                return selected_obj
        
        # If no exact match, try to find similar skills with numbers and use sequential logic
        # Increment execution count for this skill base
        if skill_base not in self.skill_execution_count:
            self.skill_execution_count[skill_base] = 0
        self.skill_execution_count[skill_base] += 1
        
        occurrence = self.skill_execution_count[skill_base]
        
        # Try to find CSV entry with occurrence number
        csv_skill_with_number = f"{skill_base} {occurrence}"
        
        if csv_skill_with_number in self.csv_mappings:
            target_pattern = self.csv_mappings[csv_skill_with_number]
            
            # Find next available instance of this pattern
            selected_obj = self._find_available_instance(target_pattern, available_objects)
            if selected_obj:
                print(f"🎯 Selected {selected_obj} for skill: {skill_language} (CSV mapping: {csv_skill_with_number} → {target_pattern})")
                return selected_obj
            
            print(f"⚠️  CSV target '{target_pattern}' not found in available objects for skill: {csv_skill_with_number}")
            return None
        
        # If sequential numbering didn't work, try to find any similar skill in CSV with fuzzy matching
        best_match = self._find_fuzzy_csv_match(skill_base, occurrence)
        if best_match:
            csv_skill_name, target_pattern = best_match
            
            # Find next available instance of this pattern
            selected_obj = self._find_available_instance(target_pattern, available_objects)
            if selected_obj:
                print(f"🎯 Selected {selected_obj} for skill: {skill_language} (CSV fuzzy match: {csv_skill_name} → {target_pattern})")
                return selected_obj
            
            print(f"⚠️  CSV target '{target_pattern}' not found in available objects for skill: {csv_skill_name}")
        
        return None
    
    def _find_fuzzy_csv_match(self, skill_base: str, occurrence: int) -> Optional[Tuple[str, str]]:
        """Find fuzzy match for skill in CSV, considering occurrence count."""
        # Look for skills that start with the base skill name
        candidates = []
        for csv_skill in self.csv_mappings.keys():
            if csv_skill.startswith(skill_base):
                # Extract number from skill name if present
                import re
                match = re.search(r' (\d+)$', csv_skill)
                if match:
                    csv_number = int(match.group(1))
                    candidates.append((csv_number, csv_skill, self.csv_mappings[csv_skill]))
        
        # Sort by number and pick the one matching our occurrence
        candidates.sort(key=lambda x: x[0])
        
        if candidates and len(candidates) >= occurrence:
            # Use the occurrence-th match (1-indexed)
            _, csv_skill_name, target_pattern = candidates[occurrence - 1]
            return csv_skill_name, target_pattern
        
        # If we don't have enough candidates, use the last one available
        if candidates:
            _, csv_skill_name, target_pattern = candidates[-1]
            return csv_skill_name, target_pattern
        
        return None
    
    def _find_available_instance(self, target_pattern: str, available_objects: List[str]) -> Optional[str]:
        """Find next available instance of target pattern, rotating through instances."""
        # Find all objects that match this pattern
        matching_objects = []
        for obj in available_objects:
            if self._object_matches_pattern_exact(obj, target_pattern):
                matching_objects.append(obj)
        
        if not matching_objects:
            return None
        
        # Sort to ensure consistent ordering (plate_1_main, plate_2_main, etc.)
        matching_objects.sort()
        
        # Track usage count for this pattern
        if target_pattern not in self.used_pattern_instances:
            self.used_pattern_instances[target_pattern] = 0
        
        # Get the next instance based on usage count
        instance_index = self.used_pattern_instances[target_pattern] % len(matching_objects)
        selected_object = matching_objects[instance_index]
        
        # Increment usage count for next time
        self.used_pattern_instances[target_pattern] += 1
        
        print(f"🔄 Pattern '{target_pattern}' usage #{self.used_pattern_instances[target_pattern]}: selected '{selected_object}' from {matching_objects}")
        
        return selected_object
    
    def _object_matches_pattern_exact(self, obj_name: str, pattern: str) -> bool:
        """Check if object name matches CSV pattern (allowing for number variations)."""
        # Direct match
        if obj_name == pattern:
            return True
        
        # Pattern matching with number wildcards
        # Convert pattern like "plate_1_main" to regex "plate_.*_main"
        import re
        regex_pattern = re.sub(r'_\d+_', '_.*_', pattern)
        if re.match(regex_pattern, obj_name):
            return True
        
        return False
    
    def _get_target_from_pattern_matching(self, skill_language: str, available_objects: List[str]) -> Optional[str]:
        """Fallback method using original pattern matching logic."""
        # Extract object type from skill language
        skill_lower = skill_language.lower()
        
        # Define object type patterns
        object_patterns = {
            'stove': ['flat_stove', 'stove'],
            'plate': ['plate'],
            'black_bowl': ['akita_black_bowl', 'black_bowl'],  # Specific black bowl
            'white_bowl': ['white_bowl'],  # Specific white bowl  
            'bowl': ['bowl'],  # Generic bowl fallback
            'cabinet_top': ['cabinet_top'], 
            'cabinet_bottom': ['cabinet_bottom'],
            'cabinet': ['cabinet', 'wooden_cabinet'],  # For drawer operations
            'drawer': ['drawer', 'cabinet'],  # Drawers are part of cabinets
            'microwave': ['microwave'],
            'moka_pot': ['moka_pot', 'pot'],
            'frying_pan': ['frypan', 'frying_pan', 'chefmate', 'pan'],
            'wine_bottle': ['wine_bottle', 'bottle'],
            'ketchup': ['ketchup', 'bottle']  # For ketchup placement
        }
        
        # Find matching object type (prioritize specific bowl types)
        target_object_type = None
        
        # Check for specific bowl types first
        if 'black bowl' in skill_lower:
            target_object_type = 'black_bowl'
        elif 'white bowl' in skill_lower:
            target_object_type = 'white_bowl'
        else:
            # Check other patterns
            for obj_type, patterns in object_patterns.items():
                if any(pattern in skill_lower for pattern in patterns):
                    target_object_type = obj_type
                    break
        
        if target_object_type is None:
            print(f"❌ Could not determine object type from skill: {skill_language}")
            return None
        
        # Find available objects of this type that haven't been used
        candidates = []
        for obj in available_objects:
            obj_lower = obj.lower()
            if any(pattern in obj_lower for pattern in object_patterns[target_object_type]):
                if obj not in self.used_objects:
                    candidates.append(obj)
        
        if not candidates:
            print(f"❌ No available {target_object_type} objects found")
            return None
        
        # Select first available candidate
        selected_object = candidates[0]
        
        print(f"🎯 Selected {selected_object} for skill: {skill_language} (pattern matching fallback)")
        return selected_object


class GTLocalPoseCalculator:
    """GT local pose calculator using learned object-EE relationships."""
    
    def __init__(self, pose_pairs_dir: str, csv_file: str = None):
        """
        Initialize the GT pose calculator.
        
        Args:
            pose_pairs_dir: Directory containing .pkl files with pose pairs
            csv_file: Path to skills_objects_mapping_updated.csv (optional)
        """
        self.pose_pairs_dir = pose_pairs_dir
        self.pose_pairs_cache = {}
        self.object_tracker = ObjectInstanceTracker(csv_file)
        self._load_all_pose_pairs()
    
    def _load_all_pose_pairs(self):
        """Load all pose pair files into memory for faster access."""
        if not os.path.exists(self.pose_pairs_dir):
            print(f"WARNING: Pose pairs directory not found: {self.pose_pairs_dir}")
            return
        
        pkl_files = [f for f in os.listdir(self.pose_pairs_dir) if f.endswith('_pose_pairs.pkl')]
        
        print(f"📁 Loading pose pairs from {len(pkl_files)} files...")
        
        for pkl_file in pkl_files:
            skill_name = pkl_file.replace('_pose_pairs.pkl', '')
            pkl_path = os.path.join(self.pose_pairs_dir, pkl_file)
            
            try:
                with open(pkl_path, 'rb') as f:
                    pose_pairs = pickle.load(f)
                
                if pose_pairs:
                    self.pose_pairs_cache[skill_name] = pose_pairs
                    print(f"  ✅ {skill_name}: {len(pose_pairs)} pose pairs")
                else:
                    print(f"  ⚠️  {skill_name}: empty file")
                    
            except Exception as e:
                print(f"  ❌ {skill_name}: {e}")
        
        print(f"✅ Loaded pose pairs for {len(self.pose_pairs_cache)} skills")
    
    def _find_matching_skill_data(self, skill_language: str) -> Optional[List]:
        """
        Find pose pairs data that matches the skill language description.
        
        Args:
            skill_language: Language description like "pick moka pot"
            
        Returns:
            List of pose pairs or None if not found
        """
        skill_clean = skill_language.lower().strip().replace(' ', '_')
        
        # Try exact match first
        if skill_clean in self.pose_pairs_cache:
            print(f"🎯 Using exact match pkl file: {skill_clean}_pose_pairs.pkl")
            return self.pose_pairs_cache[skill_clean]
        
        # Improved matching with action and object type priority
        skill_normalized = skill_language.lower()
        
        # Extract action type and object info
        action_type = None
        if any(word in skill_normalized for word in ['pick', 'grasp']):
            action_type = 'pick'
        elif any(word in skill_normalized for word in ['place', 'put']):
            action_type = 'place' 
        elif any(word in skill_normalized for word in ['open']):
            action_type = 'open'
        elif any(word in skill_normalized for word in ['close']):
            action_type = 'close'
        
        # Find best match with priority scoring
        best_match = None
        best_score = 0
        
        for cached_skill, pose_pairs in self.pose_pairs_cache.items():
            cached_clean = cached_skill.lower().replace('_', ' ')
            
            skill_words = set(skill_normalized.split())
            cached_words = set(cached_clean.split())
            
            # Calculate match score
            common_words = skill_words.intersection(cached_words)
            base_score = len(common_words)
            
            # Bonus points for action type match
            if action_type and action_type in cached_clean:
                base_score += 10  # High priority for action match
            
            # Penalty for action mismatch (e.g., pick vs place)
            if action_type == 'pick' and 'place' in cached_clean:
                base_score -= 5
            elif action_type == 'place' and 'pick' in cached_clean:
                base_score -= 5
            
            # Bonus for object type match
            if 'black bowl' in skill_normalized and 'black bowl' in cached_clean:
                base_score += 3
            elif 'white bowl' in skill_normalized and 'white bowl' in cached_clean:
                base_score += 3
            elif 'plate' in skill_normalized and 'plate' in cached_clean:
                base_score += 2
            elif 'drawer' in skill_normalized and 'drawer' in cached_clean:
                base_score += 3
            elif 'cabinet' in skill_normalized and 'cabinet' in cached_clean:
                base_score += 2
            
            # Only consider matches with reasonable overlap
            if base_score > best_score and len(common_words) >= min(2, len(skill_words)):
                best_match = (cached_skill, pose_pairs)
                best_score = base_score
        
        if best_match:
            cached_skill, pose_pairs = best_match
            print(f"🎯 Using fallback match pkl file: {cached_skill}_pose_pairs.pkl")
            print(f"📊 Matched '{skill_language}' → '{cached_skill}' ({len(pose_pairs)} pairs)")
            return pose_pairs
        
        print(f"❌ No pose pairs found for skill: {skill_language}")
        print(f"💡 Available skills: {list(self.pose_pairs_cache.keys())[:5]}...")
        return None
    
    def calculate_gt_local_pose(self, skill_language: str, env, target_object: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate target end-effector pose for a skill given current object pose.
        
        Args:
            skill_language: Language description like "pick moka pot"
            env: LIBERO environment with MuJoCo simulation
            target_object: Name of target object in scene
            
        Returns:
            Tuple of (target_ee_position, target_ee_quaternion) or None if calculation fails
        """
        print(f"🎯 Calculating GT pose for: {skill_language} → {target_object}")
        
        # Step 1: Find matching skill data
        pose_pairs = self._find_matching_skill_data(skill_language)
        if pose_pairs is None:
            return None
        
        # Step 2: Get current object pose in scene
        # Import dynamically to avoid circular imports
        import importlib.util
        spec = importlib.util.spec_from_file_location("detector", "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase1/3_contact_object_detector.py")
        detector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(detector)
        
        target_obj_pos, target_obj_quat = detector.get_object_pose(env, target_object)
        
        if target_obj_pos is None:
            print(f"❌ Could not find object {target_object} in scene")
            return None
        
        # Convert current object pose to 6D format for logging
        target_obj_rot = R.from_quat(target_obj_quat)
        target_obj_axis = target_obj_rot.as_rotvec()
        
        print(f"1️⃣ Target object name: {target_object}")
        
        # Step 3: Randomly sample a learned pose pair
        random_pair = random.choice(pose_pairs)
        learned_obj_pos = np.array(random_pair['object_pose']['position'])
        learned_obj_quat = np.array(random_pair['object_pose']['quaternion'])
        learned_ee_pos = np.array(random_pair['ee_pose']['position'])
        learned_ee_quat = np.array(random_pair['ee_pose']['quaternion'])
        
        # Convert learned poses to 6D format for logging
        learned_obj_rot = R.from_quat(learned_obj_quat)
        learned_obj_axis = learned_obj_rot.as_rotvec()
        learned_ee_rot = R.from_quat(learned_ee_quat)
        learned_ee_axis = learned_ee_rot.as_rotvec()
        
        
        # Step 4: Calculate relative transformation from learned data
        # EE pose relative to object in learned demonstration
        learned_obj_rot = R.from_quat(learned_obj_quat)
        learned_ee_rot = R.from_quat(learned_ee_quat)
        
        # Relative position (in object frame)
        relative_pos = learned_obj_rot.inv().apply(learned_ee_pos - learned_obj_pos)
        
        # Relative rotation (in object frame)  
        relative_rot = learned_obj_rot.inv() * learned_ee_rot
        
        # Step 5: Transform to current scene coordinates
        current_obj_rot = R.from_quat(target_obj_quat)
        
        # Target EE position in world frame
        target_ee_pos = target_obj_pos + current_obj_rot.apply(relative_pos)
        
        # Target EE orientation in world frame
        target_ee_rot = current_obj_rot * relative_rot
        target_ee_quat = target_ee_rot.as_quat()
        
        # Convert calculated EE pose to 6D format for logging
        target_ee_axis = target_ee_rot.as_rotvec()
        
        final_distance = np.linalg.norm(target_ee_pos - target_obj_pos)
        print(f"3️⃣ Desired robot pose (obs): pos=[{target_ee_pos[0]:.4f}, {target_ee_pos[1]:.4f}, {target_ee_pos[2]:.4f}], axis=[{target_ee_axis[0]:.4f}, {target_ee_axis[1]:.4f}, {target_ee_axis[2]:.4f}]")
        
        return target_ee_pos, target_ee_quat


def create_pose_calculator(pose_pairs_dir: str = None, csv_file: str = None) -> GTLocalPoseCalculator:
    """
    Create GT pose calculator with default or specified directory.
    
    Args:
        pose_pairs_dir: Directory containing pose pairs (optional, uses default)
        csv_file: Path to skills_objects_mapping_updated.csv (optional, uses default)
        
    Returns:
        Initialized GTLocalPoseCalculator
    """
    if pose_pairs_dir is None:
        pose_pairs_dir = '/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_local_demos/ee_target_obj_pairs/simple_extraction_obs_pose'
    
    return GTLocalPoseCalculator(pose_pairs_dir, csv_file)


def calculate_gt_local_pose(skill_language: str, env, target_object: str = None, 
                           available_objects: List[str] = None, calculator: GTLocalPoseCalculator = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Main pipeline function: Calculate target EE pose for skill execution.
    
    This is the function called by execute_long_horizon_pipeline.py.
    
    Args:
        skill_language: Language description like "pick moka pot"
        env: LIBERO environment with MuJoCo simulation
        target_object: Specific target object (optional, will be determined automatically)
        available_objects: List of available objects in scene (optional)
        calculator: Pre-initialized calculator (optional, will create if None)
        
    Returns:
        Tuple of (target_ee_position, target_ee_quaternion) or None if calculation fails
    """
    # Create calculator if not provided
    if calculator is None:
        calculator = create_pose_calculator()
    
    # Determine target object if not specified
    if target_object is None:
        if available_objects is None:
            # Get available objects from scene
            import importlib.util
            spec = importlib.util.spec_from_file_location("detector", "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase1/3_contact_object_detector.py")
            detector = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(detector)
            available_objects = detector.get_all_manipulable_objects(env)
        
        target_object = calculator.object_tracker.get_next_target_object(skill_language, available_objects)
        
        if target_object is None:
            return None
    
    # Calculate pose
    return calculator.calculate_gt_local_pose(skill_language, env, target_object)


def main():
    """Test the GT pose calculator."""
    print("🎯 Testing GT Pose Calculator")
    print("=" * 50)
    
    # Test with Cooking Preparation Setup task
    import importlib.util
    spec = importlib.util.spec_from_file_location("detector", "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase1/3_contact_object_detector.py")
    detector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detector)
    
    task_name = "Cooking Preparation Setup"
    print(f"🎯 Testing with task: {task_name}")
    
    # Create test environment
    env = detector.create_test_environment(task_name)
    if env is None:
        return
    
    try:
        # Get available objects
        available_objects = detector.get_all_manipulable_objects(env)
        print(f"📋 Available objects: {available_objects}")
        
        # Create pose calculator
        calculator = create_pose_calculator()
        
        # Test skills from Cooking Preparation Setup
        test_skills = [
            "pick moka pot",
            "place moka pot on the stove",
            "turn on the stove",
            "pick frying pan",
            "place frying pan on the stove",
            "open the microwave"
        ]
        
        print(f"\n🧪 Testing pose calculation for {len(test_skills)} skills:")
        print("-" * 60)
        
        for i, skill in enumerate(test_skills, 1):
            print(f"\n{i}. {skill}")
            target_pose = calculate_gt_local_pose(skill, env, available_objects=available_objects, calculator=calculator)
            
            if target_pose is not None:
                pos, quat = target_pose
                print(f"   ✅ Success: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            else:
                print(f"   ❌ Failed to calculate pose")
        
        env.close()
        print(f"\n🎉 Testing completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        if env:
            env.close()


if __name__ == "__main__":
    main()