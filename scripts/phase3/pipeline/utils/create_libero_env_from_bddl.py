#!/usr/bin/env python3
"""
Script to create a Libero environment from a BDDL file and save an image.

This script loads a specific BDDL file and creates a Libero environment,
then captures and saves an image of the initial scene.

Usage:
    python create_libero_env_from_bddl.py
"""

import os
import sys
from PIL import Image
from scipy.spatial.transform import Rotation as R
import numpy as np
from scripts.phase3.pipeline.motion_planning.mplib.action_utils import get_controller_robot_pose
import robosuite.utils.transform_utils as T

def create_libero_env_from_bddl(bddl_file_path, output_image_path="./libero_scene.png"):
    """
    Create a Libero environment from a BDDL file and save an image.
    
    Args:
        bddl_file_path (str): Path to the BDDL file
        output_image_path (str): Path where to save the image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Import Libero components
        from libero.libero.envs import OffScreenRenderEnv
        
        print(f"📋 Loading BDDL file: {bddl_file_path}")
        
        # Check if BDDL file exists
        if not os.path.exists(bddl_file_path):
            print(f"❌ BDDL file not found: {bddl_file_path}")
            return False
        
        # Create environment arguments
        env_args = {
            "bddl_file_name": bddl_file_path,
            "camera_heights": 256,
            "camera_widths": 256,
            "camera_depths": True,
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": True,
            "use_camera_obs": True,
            "control_freq": 20,
            "camera_names": ["agentview", "robot0_eye_in_hand"]
        }
        
        print("🔧 Creating Libero environment...")
        env = OffScreenRenderEnv(**env_args)
        
        # Set seed for reproducibility
        env.seed(42)
        
        print("🔄 Resetting environment...")
        obs = env.reset()
        
        # Print all MuJoCo names in the scene
        print("\n" + "="*60)
        print("MUJOCO NAMES IN SCENE")
        print("="*60)
        model = env.sim.model
        
        # Collect and print body names
        body_names = []
        for i in range(model.nbody):
            body_name = model.body_id2name(i)
            if body_name:
                body_names.append(body_name)
        print(f"\n📦 Bodies ({len(body_names)}):")
        for name in sorted(body_names):
            print(f"  - {name}")
        
        # Collect and print geom names
        geom_names = []
        for i in range(model.ngeom):
            geom_name = model.geom_id2name(i)
            if geom_name:
                geom_names.append(geom_name)
        print(f"\n🔷 Geoms ({len(geom_names)}):")
        for name in sorted(geom_names):
            print(f"  - {name}")
        
        # Collect and print joint names
        joint_names = []
        for i in range(model.njnt):
            joint_name = model.joint_id2name(i)
            if joint_name:
                joint_names.append(joint_name)
        print(f"\n🔗 Joints ({len(joint_names)}):")
        for name in sorted(joint_names):
            print(f"  - {name}")
        
        # Collect and print site names
        site_names = []
        for i in range(model.nsite):
            site_name = model.site_id2name(i)
            if site_name:
                site_names.append(site_name)
        if site_names:
            print(f"\n📍 Sites ({len(site_names)}):")
            for name in sorted(site_names):
                print(f"  - {name}")
        
        # Collect and print actuator names
        actuator_names = []
        for i in range(model.nu):
            actuator_name = model.actuator_id2name(i)
            if actuator_name:
                actuator_names.append(actuator_name)
        if actuator_names:
            print(f"\n⚙️  Actuators ({len(actuator_names)}):")
            for name in sorted(actuator_names):
                print(f"  - {name}")
        
        # Collect and print sensor names
        sensor_names = []
        for i in range(model.nsensor):
            sensor_name = model.sensor_id2name(i)
            if sensor_name:
                sensor_names.append(sensor_name)
        if sensor_names:
            print(f"\n📡 Sensors ({len(sensor_names)}):")
            for name in sorted(sensor_names):
                print(f"  - {name}")
        
        print("="*60 + "\n")
        
        # print(obs.keys())
        # print(obs["robot0_eef_pos"])
        print(obs["robot0_eef_quat"])
        # print(np.degrees(R.from_quat(obs["robot0_eef_quat"]).as_rotvec()))

        # # Get reset pose using controller
        # reset_pos, reset_rot_mat = get_controller_robot_pose(env, "right")
        # reset_quat = T.mat2quat(reset_rot_mat)  # [x, y, z, w] - scipy format
        # print(f"Reset pose from get_controller_robot_pose:")
        # print(f"  Position: {reset_pos}")
        # print(f"  Quaternion [x,y,z,w]: {reset_quat}")
        # print(np.degrees(R.from_quat(reset_quat).as_rotvec()))
        # exit(0)
        # Run 5 dummy actions to let objects stabilize
        print("🔄 Running 5 dummy actions...")
        for i in range(5):
            obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])

        # print(obs["robot0_eef_quat"])
        # exit(0)
        
        
        
        # Extract the agent view image
        print("📸 Capturing scene image...")
        agent_img = obs["agentview_image"]
        
        # Rotate image 180 degrees to match training preprocessing
        agent_img = agent_img[::-1, ::-1]
        
        # Save the image
        print(f"💾 Saving image to: {output_image_path}")
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        Image.fromarray(agent_img).save(output_image_path)
        
        print(f"✅ Successfully saved image: {output_image_path}")
        print(f"   Image shape: {agent_img.shape}")
        
        # Clean up
        env.close()
        print("🧹 Environment closed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating environment: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    # Path to the BDDL file
    bddl_file_path = "externals/boss/libero/libero/bddl_files/long_horizon_tasks_v0/LONG_HORIZON_complete_kitchen_organization.bddl"
    bddl_file_path = "externals/boss/libero/libero/bddl_files/long_horizon_tasks_v0/LONG_HORIZON_cooking_preparation_setup.bddl"
    bddl_file_path = "externals/boss/libero/libero/bddl_files/long_horizon_tasks_v1/LONG_HORIZON_complete_kitchen_organization.bddl"
    bddl_file_path = "externals/boss/libero/libero/bddl_files/long_horizon_tasks_v1/LONG_HORIZON_organize_table.bddl"
    bddl_file_path = "externals/boss/libero/libero/bddl_files/long_horizon_tasks_v1/LONG_HORIZON_cooking_preparation_setup.bddl"
    # bddl_file_path = "externals/boss/libero/libero/bddl_files/long_horizon_tasks_v1/LONG_HORIZON_complete_kitchen_organization.bddl"
    bddl_file_path = "externals/boss/libero/libero/bddl_files/long_horizon_tasks_v1/LONG_HORIZON_cooking_preparation_setup.bddl"
    bddl_file_path = "externals/boss/libero/libero/bddl_files/long_horizon_tasks_v1/LONG_HORIZON_complete_kitchen_organization.bddl"
    # bddl_file_path = "externals/boss/libero/libero/bddl_files/long_horizon_tasks_v1/LONG_HORIZON_organize_table.bddl"
    bddl_file_path = "externals/boss/libero/libero/bddl_files/long_horizon_tasks_v1/LONG_HORIZON_organize_table.bddl"

    bddl_file_path = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss/libero/libero/bddl_files/long_horizon_tasks_libero_long/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl"

    # Output image path
    output_image_path = "./scripts/phase3/pipeline/outputs/images/test_new_env/scene_3.png"
    
    print("="*60)
    print("LIBERO ENVIRONMENT CREATION FROM BDDL")
    print("="*60)
    print(f"BDDL File: {bddl_file_path}")
    print(f"Output Image: {output_image_path}")
    print("="*60)
    
    # Create environment and save image
    success = create_libero_env_from_bddl(bddl_file_path, output_image_path)
    
    if success:
        print("\n🎉 Script completed successfully!")
        print(f"📁 Image saved at: {os.path.abspath(output_image_path)}")
    else:
        print("\n❌ Script failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
