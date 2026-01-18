#!/usr/bin/env python3
"""
Above-pose calculator for augmented demo generation.

Calculates "above object" EE poses for starting augmented demonstrations.

Example usage (target_object auto-inferred from BDDL):
python scripts/phase3/pipeline/utils/above_pose_calculator.py --bddl_file externals/boss/libero/libero/bddl_files/atomic_skills/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_pick.bddl

python scripts/phase3/pipeline/utils/above_pose_calculator.py \
    --bddl_file externals/boss/libero/libero/bddl_files/atomic_skills/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet.bddl

python scripts/phase3/pipeline/utils/above_pose_calculator.py \
    --bddl_file externals/boss/libero/libero/bddl_files/atomic_skills/KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_place.bddl

python scripts/phase3/pipeline/utils/above_pose_calculator.py \
    --bddl_file externals/boss/libero/libero/bddl_files/atomic_skills/KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet_place.bddl

python scripts/phase3/pipeline/utils/above_pose_calculator.py \
    --bddl_file externals/boss/libero/libero/bddl_files/atomic_skills/KITCHEN_SCENE3_put_the_moka_pot_on_the_stove_place.bddl

python scripts/phase3/pipeline/utils/above_pose_calculator.py \
    --bddl_file externals/boss/libero/libero/bddl_files/atomic_skills/KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet.bddl

python scripts/phase3/pipeline/utils/above_pose_calculator.py \
    --bddl_file externals/boss/libero/libero/bddl_files/atomic_skills/KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove_place.bddl
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from typing import Tuple, Optional
from pathlib import Path
import cv2

# Add project paths
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')

from libero.libero.envs import OffScreenRenderEnv
from scripts.phase3.pipeline.motion_planning.mplib.planner import MPlibMotionPlanner
from scripts.phase3.pipeline.motion_planning.mplib.planner_core import get_object_bounding_box
from scipy.spatial.transform import Rotation as R

# Import get_object_pose from contact detector
import importlib.util
spec = importlib.util.spec_from_file_location("contact_detector",
    "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase2/3_phase2_contact_object_detector.py")
contact_detector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contact_detector)
get_object_pose = contact_detector.get_object_pose


# Hardcoded rest orientation (downward-facing, extracted from env after reset)
# Format: [w, x, y, z] - MuJoCo convention (matches move_to_pose expected format)
# REST_QUAT = np.array([0.999597, 0.000246, -0.028400, -0.000007]) # xyzw
REST_QUAT = np.array([-6.99529601e-06, 9.99596605e-01, 2.46212834e-04, -2.84001205e-02])  # wxyz

# Path to skill config
SKILL_CONFIG_PATH = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase3/pipeline/config/skill_config.json"


def load_skill_config():
    """Load skill config from JSON."""
    with open(SKILL_CONFIG_PATH, 'r') as f:
        return json.load(f)


def infer_target_object_from_bddl(bddl_file: str) -> Optional[Tuple[str, str, list]]:
    """
    Infer target object candidates and skill type from BDDL file using skill_config.json.

    Args:
        bddl_file: Path to BDDL file

    Returns:
        (target_object_list, skill_type, skill_name) tuple if found, None otherwise
        target_object_list: List of possible target objects to try (from skill_config.json)
    """
    # Get skill_config
    skill_config = load_skill_config()
    bddl_filename = Path(bddl_file).name

    # Find matching skill in config
    for skill_name, skill_info in skill_config.items():
        if bddl_filename in skill_info['bddl_files']:
            target_objects = skill_info['target_object']  # Now a list
            skill_type = skill_info['skill_type']
            return target_objects, skill_type, skill_name

    # If not found in config, parse BDDL directly as fallback
    with open(bddl_file, 'r') as f:
        content = f.read()

    obj_interest_match = re.search(r'\(:obj_of_interest\s+(.*?)\)', content, re.DOTALL)
    if not obj_interest_match:
        print(f"⚠️  Could not find skill or obj_of_interest in BDDL file")
        return None

    obj_of_interest = obj_interest_match.group(1).strip().split()[0]
    target_objects = [f"{obj_of_interest}_main"]

    # Infer skill_type from language
    language_match = re.search(r'\(:language\s+(.*?)\)', content)
    if language_match:
        language = language_match.group(1).strip()
        if language.startswith('pick'):
            skill_type = 'pick'
        elif language.startswith('place'):
            skill_type = 'place'
        else:
            skill_type = 'other'
    else:
        skill_type = 'other'

    return target_objects, skill_type, "unknown_skill"


def calculate_above_pose(obj_pos: np.ndarray,
                         above_height: float = 0.10,
                         shift: bool = False,
                         xy_range: float = 0.04,
                         z_range: float = 0.02,
                         ori_range: float = 30.0,
                         bbox_z: float = 0.0,
                         z_only_positive: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[dict]]:
    """
    Calculate EE pose above object.

    Args:
        obj_pos: Object position [x, y, z] (center of object)
        above_height: Height above object's top surface (default: 0.10m = 10cm)
        shift: Whether to apply random shift (default: False for first iteration)
        xy_range: XY shift range (default: 0.04m = 4cm)
        z_range: Z shift range (default: 0.02m = 2cm)
        ori_range: Orientation shift range in degrees (default: 30.0, so shift from -30 to +30)
        bbox_z: Z dimension of object's bounding box (default: 0.0)
        z_only_positive: If True, only apply positive z shift with no xy/ori shift (default: False)

    Returns:
        (above_pos, above_quat, shift_info) tuple where:
        - above_quat is REST_QUAT (possibly shifted)
        - shift_info is None if shift=False, or dict with 'xy_shift', 'z_shift', 'ori_shift' if shift=True
    """
    # Calculate above point: obj center z + half bbox height + above_height
    # This positions EE at above_height above the object's top surface
    above_pt = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + bbox_z + above_height])
    above_quat = REST_QUAT.copy()
    shift_info = None

    # Apply shift if requested
    if shift:
        if z_only_positive:
            # Mode: Only positive z shift (no xy shift, no orientation shift)
            xy_shift = np.array([0.0, 0.0])
            z_shift = np.random.uniform(0, z_range * 3)  # Random value in (0, z_range * 3]
            above_pt[2] += z_shift
            ori_angle_deg = 0.0

            # Store shift info
            shift_info = {
                'xy_shift': xy_shift.tolist(),  # [0, 0]
                'z_shift': float(z_shift),
                'ori_shift': float(ori_angle_deg),  # 0.0
                'z_only_positive': True
            }
        else:
            # Standard mode: Random shift for xy, z, and orientation
            # Position shift
            xy_shift = np.random.uniform(-xy_range, xy_range, size=2)
            z_shift = np.random.uniform(-z_range, z_range)
            above_pt[0] += xy_shift[0]
            above_pt[1] += xy_shift[1]
            above_pt[2] += z_shift

            # Orientation shift: rotate around z-axis (gripper rotation)
            # REST_QUAT is in wxyz format, convert to scipy Rotation (xyzw format)
            rest_quat_xyzw = np.array([REST_QUAT[1], REST_QUAT[2], REST_QUAT[3], REST_QUAT[0]])
            rest_rot = R.from_quat(rest_quat_xyzw)

            # Generate random rotation angle around z-axis (in degrees)
            ori_angle_deg = np.random.uniform(-ori_range, ori_range)
            ori_angle_rad = np.deg2rad(ori_angle_deg)

            # Create rotation around z-axis
            z_axis_rot = R.from_rotvec([0, 0, ori_angle_rad])

            # Apply rotation: new_rot = z_axis_rot * rest_rot
            shifted_rot = z_axis_rot * rest_rot

            # Convert back to quaternion (xyzw format) and then to wxyz format
            shifted_quat_xyzw = shifted_rot.as_quat()
            above_quat = np.array([shifted_quat_xyzw[3], shifted_quat_xyzw[0], shifted_quat_xyzw[1], shifted_quat_xyzw[2]])

            # Store shift info
            shift_info = {
                'xy_shift': xy_shift.tolist(),  # [x_shift, y_shift]
                'z_shift': float(z_shift),
                'ori_shift': float(ori_angle_deg),  # in degrees
                'z_only_positive': False
            }

    return above_pt, above_quat, shift_info


def visualize_above_pose(bddl_file: str,
                         target_object: Optional[str] = None,
                         above_height: float = 0.10,
                         shift: bool = False,
                         output_dir: str = None) -> bool:
    """
    Visualize above pose by moving to it and saving images.

    Args:
        bddl_file: Path to BDDL file
        target_object: Target object name (optional, will be inferred if not provided)
        above_height: Height above object
        shift: Whether to shift the above pose
        output_dir: Directory to save images (default: scripts/phase3/pipeline/outputs/images/)

    Returns:
        True if successful
    """
    if output_dir is None:
        output_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase3/pipeline/outputs/images"

    skill_name = Path(bddl_file).stem
    skill_output_dir = os.path.join(output_dir, skill_name)
    os.makedirs(skill_output_dir, exist_ok=True)

    # Infer target object candidates if not provided
    if target_object is None:
        result = infer_target_object_from_bddl(bddl_file)
        if result is None:
            print(f"❌ Could not infer target object from BDDL file")
            return False
        target_object_list, skill_type, skill_name_inferred = result
        print(f"✅ Inferred skill: {skill_name_inferred} (skill_type: {skill_type})")
        print(f"   Target object candidates: {target_object_list}")
    else:
        # User provided a specific target object
        target_object_list = [target_object]
        skill_type = "unknown"

    print(f"\n🎯 Visualizing above pose for {skill_name}")
    print(f"   BDDL: {bddl_file}")
    print(f"   Above height: {above_height}m")
    print(f"   Shift: {shift}")

    # Initialize environment
    env_args = {
        'bddl_file_name': bddl_file,
        'camera_heights': 256,
        'camera_widths': 256
    }
    env = OffScreenRenderEnv(**env_args)
    env.reset()

    # Try each target object in the list until one is found
    obj_pos = None
    obj_quat = None
    target_object = None
    for candidate in target_object_list:
        obj_pos, obj_quat = get_object_pose(env, candidate)
        if obj_pos is not None:
            target_object = candidate
            print(f"   ✅ Found target object: {target_object}")
            break

    if obj_pos is None:
        print(f"❌ None of the target object candidates {target_object_list} found in environment")
        env.close()
        return False

    # Get bounding box and apply special handling
    bbox_info = get_object_bounding_box(env, target_object)
    bbox_z = bbox_info["aabb"][2] if bbox_info else 0.0
    if "wooden_cabinet" in target_object and "open" in bddl_file:
        obj_pos, obj_quat = get_object_pose(env, "wooden_cabinet_1_cabinet_bottom")
        above_height = 0.30
        bbox_z = 0.0
    elif "wooden_cabinet" in target_object:
        above_height = 0.20
    elif "stove" in target_object and "place" in bddl_file:
        above_height = 0.20
        bbox_z = 0.0
        obj_pos[0] += 0.10
    elif "microwave" in target_object or "stove" in target_object:
        above_height = 0.20
        bbox_z = 0.0
    

    print(f"   Object position: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
    print(f"   Object bbox z-dimension: {bbox_z:.3f}m")

    # Calculate above pose (LIBERO frame) 
    above_pos_libero, above_quat_libero, _ = calculate_above_pose(
        obj_pos, above_height, shift, bbox_z=bbox_z
    )

    print(f"   Above pose (LIBERO): [{above_pos_libero[0]:.3f}, {above_pos_libero[1]:.3f}, {above_pos_libero[2]:.3f}]")

    # Initialize MPlib planner
    motion_planner = MPlibMotionPlanner(env, collision_aware=True, verbose=False)

    # Move to above pose
    print(f"   Moving to above pose...")
    success, [observations, _, _, _] = motion_planner.move_to_pose(
        above_pos_libero,
        above_quat_libero,
        position_threshold=0.02,
        orientation_threshold=0.524,
        skill_type='pick',
        grasped_object_name=None
    )

    if not success:
        print(f"❌ Failed to reach above pose")
        env.close()
        return False

    print(f"✅ Reached above pose")

    # Save images
    if len(observations) > 0:
        final_obs = observations[-1]

        # Save agentview
        if 'agentview_image' in final_obs:
            agentview_path = os.path.join(skill_output_dir, f"above_pose_agentview_h{above_height:.2f}_shift{shift}.png")
            agentview_image = cv2.rotate(final_obs['agentview_image'], cv2.ROTATE_180)
            plt.imsave(agentview_path, agentview_image)
            print(f"   💾 Saved: {agentview_path}")

        # Save wrist view if available
        if 'robot0_eye_in_hand_image' in final_obs:
            wrist_path = os.path.join(skill_output_dir, f"above_pose_wrist_h{above_height:.2f}_shift{shift}.png")
            wrist_image = cv2.rotate(final_obs['robot0_eye_in_hand_image'], cv2.ROTATE_180)
            plt.imsave(wrist_path, wrist_image)
            print(f"   💾 Saved: {wrist_path}")

    env.close()
    return True


def main():
    """CLI for testing above pose calculation and visualization."""
    import argparse

    parser = argparse.ArgumentParser(description='Above-pose calculator and visualizer')
    parser.add_argument('--bddl_file', type=str, required=True,
                       help='Path to BDDL file')
    parser.add_argument('--target_object', type=str, default=None,
                       help='Target object name (optional, will be inferred from BDDL if not provided)')
    parser.add_argument('--above_height', type=float, default=0.10,
                       help='Height above object (default: 0.10m)')
    parser.add_argument('--shift', action='store_true',
                       help='Apply random shift to above pose')
    parser.add_argument('--xy_range', type=float, default=0.04,
                       help='XY shift range (default: 0.04m = 4cm)')
    parser.add_argument('--z_range', type=float, default=0.02,
                       help='Z shift range (default: 0.02m = 2cm)')
    parser.add_argument('--ori_range', type=float, default=30.0,
                       help='Orientation shift range in degrees (default: 30.0, so shift from -30 to +30)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for images')
    args = parser.parse_args()

    success = visualize_above_pose(
        args.bddl_file,
        args.target_object,
        args.above_height,
        args.shift,
        args.output_dir
    )

    if success:
        print(f"\n✅ Visualization completed successfully")
    else:
        print(f"\n❌ Visualization failed")


if __name__ == "__main__":
    main()
