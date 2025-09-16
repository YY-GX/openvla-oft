#!/usr/bin/env python3
"""
Script to capture initial scene images for the 3 long-horizon task BDDL files.

This script:
1. Initializes LIBERO environment for each long-horizon task
2. Resets the environment to show the initial state
3. Captures and saves the initial scene image from both camera views
4. Saves images with descriptive filenames

Created for Phase 1 of long-horizon task development.
"""

import sys
import os
sys.path.append('externals/boss')

import numpy as np
from PIL import Image
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env

def capture_initial_scene_images(task_suite_name="long_horizon_tasks_v0", output_dir="./initial_scene_images"):
    """
    Capture initial scene images for all tasks in the given benchmark suite.
    
    Args:
        task_suite_name (str): Name of the benchmark suite
        output_dir (str): Directory to save captured images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Capturing Initial Scene Images for {task_suite_name.upper()} ===")
    
    try:
        # Load task suite
        task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
        print(f"Found {len(task_suite.tasks)} tasks in {task_suite_name}")
        
        # Process each task
        for i, task in enumerate(task_suite.tasks):
            print(f"\n--- Processing Task {i+1}/{len(task_suite.tasks)}: {task.name} ---")
            
            try:
                # Initialize environment
                env, _ = get_libero_env(task, model_family="openvla", resolution=256)
                
                # Reset to initial state
                obs = env.reset()
                
                # Extract images
                agentview_image = obs['agentview_image']  # (256, 256, 3) RGB
                wrist_image = obs['robot0_eye_in_hand_image']  # (256, 256, 3) RGB
                
                # Save agentview image
                agentview_filename = f"{task.name}_initial_agentview.png"
                agentview_path = os.path.join(output_dir, agentview_filename)
                Image.fromarray(agentview_image).save(agentview_path)
                
                # Save wrist camera image  
                wrist_filename = f"{task.name}_initial_wrist.png"
                wrist_path = os.path.join(output_dir, wrist_filename)
                Image.fromarray(wrist_image).save(wrist_path)
                
                print(f"✅ Saved images:")
                print(f"   Agent view: {agentview_path}")
                print(f"   Wrist view: {wrist_path}")
                
                # Print initial state information
                print(f"   Image resolution: {agentview_image.shape}")
                print(f"   End-effector pos: {obs['robot0_eef_pos']}")
                print(f"   Joint positions: {obs['robot0_joint_pos']}")
                
                # Close environment
                env.close()
                
            except Exception as e:
                print(f"❌ Error processing {task.name}: {e}")
                continue
        
        print(f"\n=== Scene Image Capture Complete ===")
        print(f"Images saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error loading benchmark {task_suite_name}: {e}")
        return False
    
    return True

def print_task_details(task_suite_name="long_horizon_tasks_v0"):
    """Print detailed information about tasks in the suite."""
    try:
        task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
        
        print(f"\n=== Task Details for {task_suite_name.upper()} ===")
        for i, task in enumerate(task_suite.tasks):
            print(f"\nTask {i+1}: {task.name}")
            print(f"  Language: {task.language}")
            print(f"  Problem: {task.problem}")
            print(f"  Problem folder: {task.problem_folder}")
            print(f"  BDDL file: {task.bddl_file}")
            print(f"  Init states file: {task.init_states_file}")
            
            # Print BDDL file path
            bddl_path = os.path.join(
                "externals/boss/libero/libero/bddl_files",
                task.problem_folder,
                task.bddl_file
            )
            exists = "✅" if os.path.exists(bddl_path) else "❌"
            print(f"  BDDL path: {bddl_path} {exists}")
    
    except Exception as e:
        print(f"❌ Error loading task details: {e}")

def main():
    """Main execution function."""
    print("🚀 Phase 1: Long-Horizon Task Initial Scene Capture")
    print("=" * 60)
    
    # First, print task details to verify setup
    print_task_details("long_horizon_tasks_v0")
    
    # Capture initial scene images
    success = capture_initial_scene_images(
        task_suite_name="long_horizon_tasks_v0",
        output_dir="./scripts/phase1/initial_scene_images"
    )
    
    if success:
        print("\n✅ All initial scene images captured successfully!")
        print("Next steps:")
        print("1. Review the captured images to verify scene setup")
        print("2. Proceed with object-centric pose extraction (Phase 1 Step 2)")
    else:
        print("\n❌ Some errors occurred during image capture")
        print("Please review the error messages above")

if __name__ == "__main__":
    main()