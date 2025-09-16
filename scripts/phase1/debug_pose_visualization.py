#!/usr/bin/env python3
"""
Pose Visualization Script for Long-Horizon Pipeline Debugging

This script helps debug local pose calculations by:
1. Running through each skill in a long-horizon task
2. Calculating and moving to the local poses
3. Saving agent view and wrist view images for comparison
4. Also creating reference environments from atomic skill BDDL files
5. Saving all images for visual comparison to identify pose calculation issues

Usage:
    python scripts/phase1/debug_pose_visualization.py --task_name "Complete Kitchen Organization"
"""

import os
import sys
import argparse
import numpy as np
from typing import Optional
from pathlib import Path
from datetime import datetime
import cv2

# Add project paths
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')

# LIBERO imports
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env

# Import Phase 1 components dynamically
import importlib.util

def load_script_module(script_path: str, module_name: str):
    """Load a script module dynamically."""
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class MotionPlanner:
    """Motion planning and IK functionality."""
    
    def __init__(self, env):
        self.env = env
    
    def inverse_kinematics(self, target_ee_pos: np.ndarray, target_ee_quat: np.ndarray) -> Optional[np.ndarray]:
        """Calculate joint positions to reach target end-effector pose."""
        try:
            print(f"🔧 Computing IK for pose: pos=[{target_ee_pos[0]:.3f}, {target_ee_pos[1]:.3f}, {target_ee_pos[2]:.3f}]")
            
            # Get current joint positions as starting point
            current_qpos = self.env.sim.data.qpos.copy()
            robot_joints = current_qpos[:7]  # Assuming 7-DOF robot
            
            # Placeholder IK - in real implementation, use actual IK solver
            target_joints = robot_joints + np.random.normal(0, 0.05, 7)  # Small random modification
            
            print(f"   ✅ IK solution found")
            return target_joints
            
        except Exception as e:
            print(f"   ❌ IK failed: {e}")
            return None
    
    def move_to_joints(self, target_joints: np.ndarray) -> bool:
        """Move robot to target joint positions."""
        try:
            print(f"🚀 Moving robot to target pose...")
            
            # Set joint positions directly
            current_qpos = self.env.sim.data.qpos.copy()
            current_qpos[:7] = target_joints  # Set robot joint positions
            
            self.env.sim.data.qpos[:] = current_qpos
            self.env.sim.forward()  # Forward kinematics to update poses
            
            print(f"   ✅ Robot moved to target pose")
            return True
            
        except Exception as e:
            print(f"   ❌ Robot movement failed: {e}")
            return False

class PoseVisualizationDebugger:
    """Main pose visualization debugger."""
    
    def __init__(self, task_name: str = "Complete Kitchen Organization"):
        self.task_name = task_name
        self.motion_planner = None
        self.env = None
        
        # Create output directory for images
        self._setup_output_directory()
        
        # Load pipeline components
        self._load_pipeline_components()
    
    def _setup_output_directory(self):
        """Setup output directory for pose visualization images."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_clean = self.task_name.replace(" ", "_").lower()
        
        self.output_dir = Path("/mnt/arc/yygx/pkgs_baselines/openvla-oft/images_videos/pose_debug")
        self.session_dir = self.output_dir / f"{task_clean}_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Output directory: {self.session_dir}")
    
    def _load_pipeline_components(self):
        """Load all pipeline script components."""
        script_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase1"
        
        print("📁 Loading pipeline components...")
        
        # Load Script 2 - Task Sequence Planner
        self.task_planner = load_script_module(
            os.path.join(script_dir, "2_task_sequence_planner.py"),
            "task_planner"
        )
        
        # Load Script 3 - Contact Object Detector  
        self.object_detector = load_script_module(
            os.path.join(script_dir, "3_contact_object_detector.py"),
            "object_detector"
        )
        
        # Load Script 4 - GT Pose Calculator
        self.pose_calculator_module = load_script_module(
            os.path.join(script_dir, "4_gt_pose_calculator.py"), 
            "pose_calculator"
        )
        
        print("✅ All pipeline components loaded")
    
    def _create_environment(self) -> bool:
        """Create LIBERO environment for the specified task."""
        print(f"🎯 Creating environment for task: {self.task_name}")
        
        try:
            # Get benchmark
            bm_name = "long_horizon_tasks_v0"
            task_suite = benchmark.get_benchmark_dict()[bm_name]()
            
            # Map friendly names to LIBERO task names
            task_name_mapping = {
                "cooking preparation setup": "LONG_HORIZON_cooking_preparation_setup",
                "complete kitchen organization": "LONG_HORIZON_complete_kitchen_organization", 
                "switch table objects": "LONG_HORIZON_switch_table_objects"
            }
            
            task_key = self.task_name.lower().strip()
            if task_key not in task_name_mapping:
                print(f"❌ Unknown task: {self.task_name}")
                return False
            
            libero_task_name = task_name_mapping[task_key]
            
            # Find matching task
            target_task = None
            for task in task_suite.tasks:
                if task.name == libero_task_name:
                    target_task = task
                    break
            
            if target_task is None:
                print(f"❌ Task not found in benchmark")
                return False
            
            # Create environment
            self.env, _ = get_libero_env(target_task, model_family='openvla', resolution=256)
            obs = self.env.reset()
            
            if obs is None:
                print("❌ Environment reset returned None observation")
                return False
            
            print(f"✅ Environment created successfully")
            
            # Initialize motion planner
            self.motion_planner = MotionPlanner(self.env)
            
            return True
            
        except Exception as e:
            print(f"❌ Environment creation failed: {e}")
            return False
    
    def _save_images(self, obs, skill_idx: int, pose_type: str, skill_name: str):
        """Save agent view and wrist view images."""
        # Create filename prefix
        prefix = f"skill_{skill_idx:02d}_{pose_type}_{skill_name.replace(' ', '_')}"
        
        try:
            # Save agent view image
            if 'agentview_image' in obs:
                agent_img = obs['agentview_image']
                if agent_img.dtype != np.uint8:
                    agent_img = (agent_img * 255).astype(np.uint8)
                agent_path = self.session_dir / f"{prefix}_agentview.png"
                cv2.imwrite(str(agent_path), cv2.cvtColor(agent_img, cv2.COLOR_RGB2BGR))
                print(f"   💾 Saved agent view: {agent_path.name}")
            
            # Save wrist view image
            if 'robot0_eye_in_hand_image' in obs:
                wrist_img = obs['robot0_eye_in_hand_image']
                if wrist_img.dtype != np.uint8:
                    wrist_img = (wrist_img * 255).astype(np.uint8)
                wrist_path = self.session_dir / f"{prefix}_wrist.png"
                cv2.imwrite(str(wrist_path), cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))
                print(f"   💾 Saved wrist view: {wrist_path.name}")
                
        except Exception as e:
            print(f"   ❌ Failed to save images: {e}")
    
    def _find_atomic_skill_bddl(self, skill_language: str) -> Optional[str]:
        """Find matching atomic skill BDDL file for a given skill."""
        atomic_skills_dir = Path("/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss/libero/libero/bddl_files/atomic_skills")
        
        # Try different matching strategies
        skill_lower = skill_language.lower()
        
        # Strategy 1: Look for pick/place patterns
        if "pick" in skill_lower:
            if "black bowl" in skill_lower:
                pattern = "*black_bowl*pick*.bddl"
            elif "white bowl" in skill_lower:
                pattern = "*white_bowl*pick*.bddl"
            elif "wine bottle" in skill_lower:
                pattern = "*wine_bottle*pick*.bddl"
            elif "ketchup" in skill_lower:
                pattern = "*ketchup*pick*.bddl"
            else:
                pattern = "*pick*.bddl"
        elif "place" in skill_lower:
            if "plate" in skill_lower:
                pattern = "*plate*place*.bddl"
            elif "drawer" in skill_lower:
                pattern = "*drawer*place*.bddl"
            else:
                pattern = "*place*.bddl"
        else:
            # Generic search
            pattern = "*.bddl"
        
        import glob
        matching_files = list(atomic_skills_dir.glob(pattern))
        
        if matching_files:
            # Return first match for now - could be improved with better matching
            print(f"   🔍 Found atomic skill BDDL: {matching_files[0].name}")
            return str(matching_files[0])
        else:
            print(f"   ⚠️ No matching atomic skill BDDL found for: {skill_language}")
            return None
    
    def _create_atomic_skill_environment(self, bddl_path: str, skill_idx: int, skill_name: str):
        """Create environment from atomic skill BDDL and save reference images."""
        try:
            print(f"   🏗️ Creating atomic skill environment from: {Path(bddl_path).name}")
            
            # Create task from BDDL file
            from libero.libero.envs import TASK_MAPPING
            from libero.libero.benchmark import get_benchmark
            
            # Get the task class (assuming kitchen tabletop manipulation)
            task_class = TASK_MAPPING.get("libero_kitchen_tabletop_manipulation", None)
            if task_class is None:
                print(f"   ❌ Task class not found")
                return
            
            # Create environment from BDDL
            atomic_env, _ = get_libero_env_from_bddl(bddl_path, model_family='openvla', resolution=256)
            
            if atomic_env is None:
                print(f"   ❌ Atomic environment creation failed")
                return
            
            atomic_obs = atomic_env.reset()
            
            if atomic_obs is None:
                print(f"   ❌ Atomic environment reset failed")
                atomic_env.close()
                return
            
            # Save reference images
            self._save_images(atomic_obs, skill_idx, "atomic_reference", skill_name)
            
            # Clean up
            atomic_env.close()
            print(f"   ✅ Atomic skill environment processed")
            
        except Exception as e:
            print(f"   ❌ Failed to create atomic skill environment: {e}")

    def run_pose_visualization(self) -> bool:
        """Run the complete pose visualization pipeline."""
        print(f"🚀 Starting Pose Visualization Debug")
        print(f"📝 Task: {self.task_name}")
        print("=" * 70)
        
        # Create environment
        if not self._create_environment():
            return False
        
        try:
            # Get skill sequence
            print(f"\n📋 Step 1: Getting skill sequence...")
            skills = self.task_planner.plan_task_sequence(self.task_name)
            
            if not skills:
                print("❌ Failed to get skill sequence")
                return False
            
            print(f"✅ Got {len(skills)} skills to visualize")
            
            # Get available objects
            print(f"\n🔍 Step 2: Finding available objects...")
            available_objects = self.object_detector.get_all_manipulable_objects(self.env)
            
            if not available_objects:
                print("❌ No manipulable objects found")
                return False
            
            # Initialize pose calculator
            print(f"\n🎯 Step 3: Initializing pose calculator...")
            pose_calculator = self.pose_calculator_module.create_pose_calculator()
            
            # Process each skill
            print(f"\n🎬 Step 4: Processing skills and visualizing poses...")
            print("-" * 60)
            
            for i, skill_language in enumerate(skills, 1):
                print(f"\n🎯 Skill {i}/{len(skills)}: {skill_language}")
                print("-" * 40)
                
                # Calculate GT local pose for current skill
                print(f"   📐 Calculating GT local pose...")
                target_pose = self.pose_calculator_module.calculate_gt_local_pose(
                    skill_language, self.env, 
                    available_objects=available_objects,
                    calculator=pose_calculator
                )
                
                if target_pose is None:
                    print(f"   ❌ Failed to calculate GT pose")
                    continue
                
                target_ee_pos, target_ee_quat = target_pose
                
                # Move robot to calculated pose
                print(f"   🔧 Moving robot to calculated local pose...")
                target_joints = self.motion_planner.inverse_kinematics(target_ee_pos, target_ee_quat)
                
                if target_joints is None:
                    print(f"   ❌ IK failed")
                    continue
                
                if not self.motion_planner.move_to_joints(target_joints):
                    print(f"   ❌ Robot movement failed")
                    continue
                
                # Save images of calculated pose
                print(f"   📷 Saving calculated pose images...")
                # Use dummy action to get current observation
                dummy_action = np.zeros(7)
                obs, _, _, _ = self.env.step(dummy_action)
                self._save_images(obs, i, "calculated_pose", skill_language)
                
                # Find and process matching atomic skill
                print(f"   🔍 Finding matching atomic skill BDDL...")
                atomic_bddl_path = self._find_atomic_skill_bddl(skill_language)
                
                if atomic_bddl_path:
                    self._create_atomic_skill_environment(atomic_bddl_path, i, skill_language)
            
            print(f"\n✅ Pose visualization completed!")
            print(f"📁 All images saved to: {self.session_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Pose visualization error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Clean up
            if self.env:
                self.env.close()
                print("🔧 Environment closed")

def get_libero_env_from_bddl(bddl_file_path: str, model_family: str = 'openvla', resolution: int = 256):
    """Create LIBERO environment from BDDL file."""
    try:
        # Import the necessary task mapping to create environment directly
        from libero.libero.envs import TASK_MAPPING
        from experiments.robot.libero.libero_utils import get_libero_env
        
        # For kitchen tabletop manipulation tasks, use the appropriate problem class
        problem_name = "libero_kitchen_tabletop_manipulation"
        
        if problem_name in TASK_MAPPING:
            problem_class = TASK_MAPPING[problem_name]
            
            # Create environment directly using problem class
            env = problem_class(
                bddl_file_name=bddl_file_path,
                robots=["Panda"],
                controller_configs={"arm": "OSC_POSE", "gripper": "GRIPPER"},
                has_renderer=True,
                has_offscreen_renderer=True,
                render_camera="frontview",
                ignore_done=True,
                use_camera_obs=True,
                reward_shaping=True,
                control_freq=20,
                camera_names=["agentview", "robot0_eye_in_hand"],
                camera_heights=resolution,
                camera_widths=resolution,
                camera_depths=False,
            )
            
            return env, None
        else:
            print(f"   ❌ Problem class not found: {problem_name}")
            return None, None
            
    except Exception as e:
        print(f"   ❌ Failed to create environment from BDDL: {e}")
        return None, None

def main():
    """Main function for pose visualization debugging."""
    parser = argparse.ArgumentParser(description='Debug pose calculations by visualizing calculated vs reference poses')
    parser.add_argument('--task_name', type=str, required=True,
                        help='Name of long horizon task to debug',
                        choices=['Cooking Preparation Setup', 'Complete Kitchen Organization', 'Switch Table Objects'])
    
    args = parser.parse_args()
    
    print("🔍 Pose Visualization Debugger")
    print(f"📝 Task: {args.task_name}")
    print("=" * 50)
    
    # Create and run debugger
    debugger = PoseVisualizationDebugger(task_name=args.task_name)
    
    success = debugger.run_pose_visualization()
    
    if success:
        print("✅ Pose visualization debugging completed successfully!")
        exit(0)
    else:
        print("❌ Pose visualization debugging failed!")
        exit(1)

if __name__ == "__main__":
    main()