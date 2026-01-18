#!/usr/bin/env python3
"""
GT Pose Calculator for Long Horizon Pipeline (Cleaned).

Simplified version that directly uses target_object from unified config.
All CSV loading, fuzzy matching, and object instance tracking removed.
"""

import os
import sys
import pickle
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# Add project paths
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')

from scripts.phase3.pipeline.config.config_loader import get_cached_config


class GTLocalPoseCalculator:
    """
    Calculates target EE pose from stored object-EE pose pairs.

    Simplified to work with direct target_object names from unified config.
    """

    def __init__(self, pose_pairs_dir: str = None):
        """
        Initialize pose calculator.

        Args:
            pose_pairs_dir: Directory containing object-EE pose pairs
        """
        self.pose_pairs_dir = pose_pairs_dir
        self.pose_pairs_cache = {}  # Cache loaded pose pairs

        if pose_pairs_dir is None:
            self.pose_pairs_dir = "datasets/hdf5_datasets/atomic_local_demos_augmented_pick_10_place_15_atomic_15/ee_target_obj_pairs/simple_extraction_obs_pose"

    def load_pose_pairs(self, skill_language: str) -> dict:
        """
        Load object-EE pose pairs for a skill.

        Args:
            skill_language: Skill description (e.g., "pick black bowl")

        Returns:
            Dictionary mapping object names to list of (obj_pos, obj_quat, ee_pos, ee_quat) tuples
        """
        # Check cache first
        if skill_language in self.pose_pairs_cache:
            return self.pose_pairs_cache[skill_language]

        # Convert skill language to filename format (spaces to underscores)
        skill_filename = skill_language.replace(" ", "_")

        # Build path to pose pairs file
        base_dir = Path(self.pose_pairs_dir)

        # Try different file formats
        pose_pairs = None

        # Try pickle format first
        pickle_file = base_dir / f"{skill_filename}_pose_pairs.pkl"
        print(f"🔍 Searching for pose pairs in: {pickle_file}")
        if pickle_file.exists():
            try:
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)

                # Convert list format to dictionary format
                if isinstance(data, list):
                    pose_pairs = {}
                    for entry in data:
                        obj_name = entry['object_pose']['object_name']
                        obj_pos = entry['object_pose']['position']
                        obj_quat = entry['object_pose']['quaternion']
                        ee_pos = entry['ee_pose']['position']
                        ee_quat = entry['ee_pose']['quaternion']

                        if obj_name not in pose_pairs:
                            pose_pairs[obj_name] = []

                        pose_pairs[obj_name].append((
                            np.array(obj_pos),
                            np.array(obj_quat),
                            np.array(ee_pos),
                            np.array(ee_quat)
                        ))
                elif isinstance(data, dict):
                    pose_pairs = data
                else:
                    raise ValueError(f"Unexpected data type: {type(data)}")

                print(f"✅ Loaded pose pairs from: {pickle_file}")
                print(f"   Objects: {list(pose_pairs.keys())}")
            except Exception as e:
                print(f"❌ Error loading pickle: {e}")

        # Try HDF5 format (if pickle failed or doesn't exist)
        if pose_pairs is None:
            import h5py
            hdf5_file = base_dir / f"{skill_filename}_pose_pairs.hdf5"
            if hdf5_file.exists():
                try:
                    pose_pairs = {}
                    with h5py.File(hdf5_file, 'r') as f:
                        for obj_name in f.keys():
                            grp = f[obj_name]
                            pairs = []
                            for i in range(len(grp['obj_pos'])):
                                pairs.append((
                                    grp['obj_pos'][i],
                                    grp['obj_quat'][i],
                                    grp['ee_pos'][i],
                                    grp['ee_quat'][i]
                                ))
                            pose_pairs[obj_name] = pairs
                    print(f"✅ Loaded pose pairs from: {hdf5_file}")
                except Exception as e:
                    print(f"❌ Error loading HDF5: {e}")

        if pose_pairs is None:
            print(f"❌ No pose pairs found for skill: {skill_language}")
            print(f"   Searched for: {pickle_file} or {hdf5_file}")
            return {}

        # Cache the result
        self.pose_pairs_cache[skill_language] = pose_pairs
        return pose_pairs

    def calculate_ee_pose(self, skill_language: str, target_object: str, env) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate target EE pose for a skill.

        Args:
            skill_language: Skill description (e.g., "pick black bowl")
            target_object: MuJoCo object name (e.g., "akita_black_bowl_1_main")
            env: LIBERO environment

        Returns:
            Tuple of (target_ee_pos, target_ee_quat) or None if failed
        """
        # Load pose pairs for this skill
        pose_pairs = self.load_pose_pairs(skill_language)

        if not pose_pairs:
            print(f"❌ No pose pairs loaded for skill: {skill_language}")
            return None

        # Check if target object has pose pairs, if not use the first available object
        # (typically there's only one object in the pickle file)
        if target_object not in pose_pairs:
            available_objects = list(pose_pairs.keys())
            if not available_objects:
                print(f"❌ No objects in pose pairs")
                return None

            # Use first available object for relative transform
            pkl_object = available_objects[0]
        else:
            pkl_object = target_object

        # Get current object pose from environment (using target_object from env)
        try:
            obj_pos_current = env.sim.data.get_body_xpos(target_object).copy()
            obj_quat_current = env.sim.data.get_body_xquat(target_object).copy()  # [w, x, y, z]
        except Exception as e:
            print(f"❌ Failed to get object pose from env: {e}")
            return None

        # Get a random pose pair (using pkl_object from pose pairs)
        pairs = pose_pairs[pkl_object]
        if not pairs:
            print(f"❌ No pose pairs for object: {target_object}")
            return None

        # Use random pair to add variability
        obj_pos_stored, obj_quat_stored, ee_pos_stored, ee_quat_stored = pairs[np.random.randint(len(pairs))]

        # Calculate relative transform from object to EE in stored demo
        # T_ee = T_obj * T_obj_to_ee
        # Where T_obj_to_ee is the relative transform we want to preserve

        # Convert quaternions to rotation matrices
        # obj_quat_stored is [w, x, y, z] (wxyz - MuJoCo format), convert to [x, y, z, w] for scipy
        R_obj_stored = R.from_quat([obj_quat_stored[1], obj_quat_stored[2], obj_quat_stored[3], obj_quat_stored[0]])
        # ee_quat_stored is [x, y, z, w] (xyzw - robosuite format), use directly
        R_ee_stored = R.from_quat(ee_quat_stored)

        # Relative rotation: R_obj_to_ee = R_obj^(-1) * R_ee
        R_obj_to_ee = R_obj_stored.inv() * R_ee_stored

        # Relative position in object frame
        pos_relative = R_obj_stored.inv().apply(ee_pos_stored - obj_pos_stored)

        # Apply relative transform to current object pose
        R_obj_current = R.from_quat([obj_quat_current[1], obj_quat_current[2], obj_quat_current[3], obj_quat_current[0]])

        # Calculate target EE pose
        target_ee_rot = R_obj_current * R_obj_to_ee
        target_ee_pos = obj_pos_current + R_obj_current.apply(pos_relative)

        # Convert rotation back to quaternion [w, x, y, z]
        target_ee_quat_xyzw = target_ee_rot.as_quat()  # [x, y, z, w]
        target_ee_quat = np.array([target_ee_quat_xyzw[3], target_ee_quat_xyzw[0],
                                   target_ee_quat_xyzw[1], target_ee_quat_xyzw[2]])  # [w, x, y, z]

        print(f"✅ Calculated target EE pose:")
        print(f"   Object: {target_object}")
        print(f"   Obj pos: {obj_pos_current}")
        print(f"   EE pos: {target_ee_pos}")

        return target_ee_pos, target_ee_quat


def create_pose_calculator(pose_pairs_dir: str = None) -> GTLocalPoseCalculator:
    """Factory function to create pose calculator."""
    return GTLocalPoseCalculator(pose_pairs_dir=pose_pairs_dir)


def calculate_gt_local_pose(skill: str, env, pose_pairs_dir: str = None) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Main pipeline function: Calculate target EE pose for a skill.

    Simplified version that uses target_object and language directly from unified config.

    Args:
        skill: Skill name from task sequence (e.g., "pick black bowl")
        env: LIBERO environment
        pose_pairs_dir: Directory containing pose pairs

    Returns:
        Tuple of (target_ee_pos, target_ee_quat, vla_language) or None if failed
        - target_ee_pos: Target end-effector position
        - target_ee_quat: Target end-effector quaternion
        - vla_language: Clean language for VLA execution (from "language" attribute)
    """
    # Get skill data from unified config
    config = get_cached_config()
    skill_mappings = config.get("skill_mappings", {})

    skill_lower = skill.lower().strip()

    if skill_lower not in skill_mappings:
        print(f"❌ Skill '{skill}' not found in unified config")
        return None

    skill_data = skill_mappings[skill_lower]
    target_object = skill_data.get("target_object")
    vla_language = skill_data.get("language")

    if not target_object:
        print(f"❌ No target_object defined for skill: {skill}")
        return None

    if not vla_language:
        print(f"⚠️  No 'language' attribute for skill: {skill}, using skill name as fallback")
        vla_language = skill_lower

    print(f"🎯 Target object: {target_object}")
    print(f"🗣️  VLA language: {vla_language}")

    # Verify object exists in scene
    try:
        obj_pos = env.sim.data.get_body_xpos(target_object)
        print(f"✅ Object found in scene at: {obj_pos}")
    except Exception as e:
        print(f"❌ Object '{target_object}' not found in scene: {e}")
        return None

    # Create pose calculator and compute target pose
    # Use vla_language (e.g., "pick black bowl") to find pose pairs, not skill name (e.g., "pick black bowl 1")
    calculator = create_pose_calculator(pose_pairs_dir=pose_pairs_dir)
    target_pose = calculator.calculate_ee_pose(vla_language, target_object, env)

    if target_pose is None:
        return None

    target_ee_pos, target_ee_quat = target_pose
    return target_ee_pos, target_ee_quat, vla_language, target_object


def main():
    """Test function."""
    print("GT Pose Calculator (Cleaned)")
    print("This module calculates target EE poses using:")
    print("  1. Skill → target_object (from unified config)")
    print("  2. Object-EE pose pairs (from demonstrations)")
    print("  3. Current object pose (from environment)")


if __name__ == "__main__":
    main()
