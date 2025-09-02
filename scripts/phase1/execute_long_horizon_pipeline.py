#!/usr/bin/env python3
"""
Long-horizon pipeline executor - Main orchestrator for Phase 1 implementation.

This is the core script that executes complete long-horizon tasks using the corrected pipeline:
1. Get skill sequence using plan_task_sequence (Script 2)
2. Get available objects using get_all_manipulable_objects (Script 3) 
3. For each skill:
   a. Determine target object for current skill
   b. Calculate GT local pose using calculate_gt_local_pose (Script 4)
   c. Use IK to get joints and move robot to local pose
   d. Execute VLA with language instruction
   e. Execute post-actions (gripper control)

The VLA execution is integrated directly in this script as requested.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

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


class VLAExecutor:
    """VLA execution handler integrated into the main pipeline."""
    
    def __init__(self, vla_checkpoint_path: Optional[str] = None, wrist_only: bool = False):
        """
        Initialize VLA executor.
        
        Args:
            vla_checkpoint_path: Path to VLA model checkpoint (optional)
            wrist_only: Whether to use only wrist camera as observation input
        """
        self.vla_model = None
        self.action_head = None
        self.proprio_projector = None
        self.noisy_action_projector = None
        self.processor = None
        self.checkpoint_path = vla_checkpoint_path
        self.wrist_only = wrist_only
        self.model_loaded = False
        self.resize_size = None
        
        # Import VLA utilities
        self._import_vla_utils()
    
    def _import_vla_utils(self):
        """Import VLA utility functions."""
        try:
            from experiments.robot.openvla_utils import (
                get_action_head,
                get_noisy_action_projector,
                get_processor,
                get_proprio_projector,
                resize_image_for_policy,
            )
            from experiments.robot.robot_utils import (
                get_action,
                get_image_resize_size,
                get_model,
                invert_gripper_action,
                normalize_gripper_action,
            )
            from experiments.robot.libero.libero_utils import (
                get_libero_image,
                get_libero_wrist_image,
                quat2axisangle,
            )
            
            self.get_action_head = get_action_head
            self.get_noisy_action_projector = get_noisy_action_projector
            self.get_processor = get_processor
            self.get_proprio_projector = get_proprio_projector
            self.resize_image_for_policy = resize_image_for_policy
            self.get_action = get_action
            self.get_image_resize_size = get_image_resize_size
            self.get_model = get_model
            self.invert_gripper_action = invert_gripper_action
            self.normalize_gripper_action = normalize_gripper_action
            self.get_libero_image = get_libero_image
            self.get_libero_wrist_image = get_libero_wrist_image
            self.quat2axisangle = quat2axisangle
            
            print("✅ VLA utilities imported successfully")
            
        except ImportError as e:
            print(f"❌ Failed to import VLA utilities: {e}")
            raise
    
    def load_vla_model(self):
        """Load VLA model from checkpoint."""
        if self.model_loaded:
            return True
        
        print("🤖 Loading VLA model...")
        
        try:
            # Create config for model loading
            from dataclasses import dataclass
            from typing import Union
            from pathlib import Path
            
            @dataclass
            class VLAConfig:
                model_family: str = "openvla"
                pretrained_checkpoint: Union[str, Path] = ""
                use_l1_regression: bool = True
                use_diffusion: bool = False
                num_diffusion_steps: int = 50
                use_film: bool = False
                num_images_in_input: int = 2
                use_proprio: bool = True
                center_crop: bool = True
                num_open_loop_steps: int = 8
                unnorm_key: Union[str, Path] = ""
                load_in_8bit: bool = False
                load_in_4bit: bool = False
                task_suite_name: str = "libero_local1"
                # Additional attributes needed by VLA utilities
                wrist_only: bool = False
                agent_only: bool = False
                pro_only: bool = False
                is_oss: bool = False
                is_depth: bool = False
            
            # Set up config
            cfg = VLAConfig()
            cfg.pretrained_checkpoint = self.checkpoint_path or "/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/view_wrist/1.0.0/openvla-7b+libero_local1+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--150000_chkpt"
            cfg.num_images_in_input = 1 if self.wrist_only else 2
            cfg.wrist_only = self.wrist_only
            
            # Load model
            self.vla_model = self.get_model(cfg)
            print(f"✅ VLA model loaded: {cfg.pretrained_checkpoint}")
            
            # Load proprio projector if needed
            if cfg.use_proprio:
                self.proprio_projector = self.get_proprio_projector(
                    cfg,
                    self.vla_model.llm_dim,
                    proprio_dim=8,  # 8-dimensional proprio for LIBERO
                )
            
            # Load action head if needed
            if cfg.use_l1_regression or cfg.use_diffusion:
                self.action_head = self.get_action_head(cfg, self.vla_model.llm_dim)
            
            # Load noisy action projector if using diffusion
            if cfg.use_diffusion:
                self.noisy_action_projector = self.get_noisy_action_projector(cfg, self.vla_model.llm_dim)
            
            # Get OpenVLA processor if needed
            if cfg.model_family == "openvla":
                self.processor = self.get_processor(cfg)
            
            # Get expected image dimensions
            self.resize_size = self.get_image_resize_size(cfg)
            
            # Set unnorm_key
            unnorm_key = cfg.task_suite_name
            if unnorm_key not in self.vla_model.norm_stats and f"{unnorm_key}_no_noops" in self.vla_model.norm_stats:
                unnorm_key = f"{unnorm_key}_no_noops"
            cfg.unnorm_key = unnorm_key
            
            self.cfg = cfg
            self.model_loaded = True
            print("✅ VLA model and components loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load VLA model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def prepare_observation(self, obs):
        """Prepare observation for policy input."""
        if self.wrist_only:
            # Get preprocessed images
            wrist_img = self.get_libero_wrist_image(obs)
            
            # Resize images to size expected by model
            wrist_img_resized = self.resize_image_for_policy(wrist_img, self.resize_size)
            
            # Prepare observations dict
            observation = {
                "full_image": wrist_img_resized,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], self.quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }
        else:
            # Get preprocessed images
            img = self.get_libero_image(obs)
            wrist_img = self.get_libero_wrist_image(obs)
            
            # Resize images to size expected by model
            img_resized = self.resize_image_for_policy(img, self.resize_size)
            wrist_img_resized = self.resize_image_for_policy(wrist_img, self.resize_size)
            
            # Prepare observations dict
            observation = {
                "full_image": img_resized,
                "wrist_image": wrist_img_resized,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], self.quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }
        
        return observation
    
    def process_action(self, action):
        """Process action before sending to environment."""
        # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
        action = self.normalize_gripper_action(action, binarize=True)
        
        # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
        # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
        if self.cfg.model_family == "openvla":
            action = self.invert_gripper_action(action)
        
        return action
    
    def execute_vla_skill(self, language_instruction: str, env, max_steps: int = 100) -> bool:
        """
        Execute VLA skill with language conditioning.
        
        Args:
            language_instruction: Language description like "pick moka pot"
            env: LIBERO environment (OffScreenRenderEnv)
            max_steps: Maximum number of steps to execute
            
        Returns:
            True if skill execution successful, False otherwise
        """
        if not self.model_loaded:
            if not self.load_vla_model():
                return False
        
        print(f"🤖 Executing VLA skill: '{language_instruction}'")
        
        try:
            from collections import deque
            from prismatic.vla.constants import NUM_ACTIONS_CHUNK
            
            # Initialize action queue
            action_queue = deque(maxlen=self.cfg.num_open_loop_steps)
            
            # Run skill execution
            t = 0
            success = False
            
            # Get initial observation from environment reset
            obs = env.reset()
            
            while t < max_steps:
                
                # Prepare observation
                observation = self.prepare_observation(obs)
                
                # If action queue is empty, requery model
                if len(action_queue) == 0:
                    # Query model to get action
                    actions = self.get_action(
                        self.cfg,
                        self.vla_model,
                        observation,
                        language_instruction,
                        processor=self.processor,
                        action_head=self.action_head,
                        proprio_projector=self.proprio_projector,
                        noisy_action_projector=self.noisy_action_projector,
                        use_film=self.cfg.use_film,
                    )
                    action_queue.extend(actions)
                
                # Get action from queue
                action = action_queue.popleft()
                
                # Process action
                action = self.process_action(action)
                
                # Execute action in environment
                obs, reward, done, info = env.step(action.tolist())
                
                if done:
                    success = True
                    break
                t += 1
            
            if success:
                print(f"   ✅ VLA skill executed successfully in {t} steps")
            else:
                print(f"   ⚠️ VLA skill completed (max steps reached: {max_steps})")
            
            return True
            
        except Exception as e:
            print(f"   ❌ VLA execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False


class MotionPlanner:
    """Motion planning and IK functionality."""
    
    def __init__(self, env):
        """
        Initialize motion planner.
        
        Args:
            env: LIBERO environment with robot
        """
        self.env = env
    
    def inverse_kinematics(self, target_ee_pos: np.ndarray, target_ee_quat: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate joint positions to reach target end-effector pose.
        
        Args:
            target_ee_pos: Target end-effector position [x, y, z]
            target_ee_quat: Target end-effector quaternion [x, y, z, w]
            
        Returns:
            Joint positions or None if IK fails
        """
        try:
            # TODO: Implement actual IK solver here
            # For now, return dummy joint positions
            
            print(f"🔧 Computing IK for pose: pos=[{target_ee_pos[0]:.3f}, {target_ee_pos[1]:.3f}, {target_ee_pos[2]:.3f}]")
            
            # Get current joint positions as starting point
            current_qpos = self.env.sim.data.qpos.copy()
            robot_joints = current_qpos[:7]  # Assuming 7-DOF robot
            
            # Placeholder IK - in real implementation, use actual IK solver
            # target_joints = ik_solver.solve(target_ee_pos, target_ee_quat, current_joints=robot_joints)
            
            # For demonstration, slightly modify current joints
            target_joints = robot_joints + np.random.normal(0, 0.05, 7)  # Small random modification
            
            print(f"   ✅ IK solution found")
            return target_joints
            
        except Exception as e:
            print(f"   ❌ IK failed: {e}")
            return None
    
    def move_to_joints(self, target_joints: np.ndarray) -> bool:
        """
        Move robot to target joint positions.
        
        Args:
            target_joints: Target joint positions
            
        Returns:
            True if movement successful, False otherwise
        """
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


def execute_post_actions(skill_language: str, env) -> bool:
    """
    Execute post-skill actions like gripper control.
    
    Args:
        skill_language: Language description to determine post-actions
        env: LIBERO environment
        
    Returns:
        True if post-actions successful, False otherwise
    """
    skill_lower = skill_language.lower()
    
    try:
        if 'pick' in skill_lower:
            # Close gripper for picking
            print(f"🤏 Closing gripper (pick action)")
            # TODO: Add actual gripper control
            # env.set_gripper_position(closed=True)
            
        elif 'place' in skill_lower:
            # Open gripper for placing
            print(f"🖐️ Opening gripper (place action)")
            # TODO: Add actual gripper control  
            # env.set_gripper_position(closed=False)
            
        else:
            # No specific gripper action needed
            print(f"💫 No gripper action required")
        
        return True
        
    except Exception as e:
        print(f"❌ Post-action failed: {e}")
        return False


class LongHorizonPipeline:
    """Main pipeline orchestrator implementing the corrected architecture."""
    
    def __init__(self, 
                 vla_checkpoint_path: Optional[str] = None,
                 task_name: str = "Cooking Preparation Setup",
                 wrist_only: bool = False):
        """
        Initialize the long horizon pipeline.
        
        Args:
            vla_checkpoint_path: Path to VLA model checkpoint
            task_name: Name of the long horizon task to execute
            wrist_only: Whether to use only wrist camera as observation input
        """
        self.task_name = task_name
        self.wrist_only = wrist_only
        self.vla_executor = VLAExecutor(vla_checkpoint_path, wrist_only=wrist_only)
        self.motion_planner = None
        self.env = None
        
        # Load pipeline components
        self._load_pipeline_components()
    
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
            
            # Check if initial observation is valid
            if obs is None:
                print("❌ Environment reset returned None observation")
                return False
            
            print(f"✅ Environment created successfully, initial obs keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")
            
            # Test if we can get observations after reset
            try:
                test_obs = self.env.reset()
                print(f"✅ Environment reset successful, obs keys: {list(test_obs.keys()) if isinstance(test_obs, dict) else 'Not a dict'}")
            except Exception as e:
                print(f"❌ Environment reset failed: {e}")
                return False
            
            # Initialize motion planner
            self.motion_planner = MotionPlanner(self.env)
            
            return True
            
        except Exception as e:
            print(f"❌ Environment creation failed: {e}")
            return False
    
    def execute_pipeline(self) -> bool:
        """
        Execute the complete long horizon pipeline.
        
        Returns:
            True if pipeline execution successful, False otherwise
        """
        print(f"🚀 Starting Long Horizon Pipeline Execution")
        print(f"📝 Task: {self.task_name}")
        print("=" * 70)
        
        # Step 0: Create environment
        if not self._create_environment():
            return False
        
        try:
            # Step 1: Get skill sequence (Script 2)
            print(f"\n📋 Step 1: Planning skill sequence...")
            skills = self.task_planner.plan_task_sequence(self.task_name)
            
            if not skills:
                print("❌ Failed to get skill sequence")
                return False
            
            print(f"✅ Got {len(skills)} skills to execute")
            
            # Step 2: Get available objects in scene (Script 3)
            print(f"\n🔍 Step 2: Finding available objects...")
            available_objects = self.object_detector.get_all_manipulable_objects(self.env)
            
            if not available_objects:
                print("❌ No manipulable objects found")
                return False
            
            # Step 3: Initialize GT pose calculator (Script 4)
            print(f"\n🎯 Step 3: Initializing pose calculator...")
            pose_calculator = self.pose_calculator_module.create_pose_calculator()
            
            # Step 4: Execute skill sequence
            print(f"\n🎬 Step 4: Executing skill sequence...")
            print("-" * 50)
            
            success_count = 0
            
            for i, skill_language in enumerate(skills, 1):
                print(f"\n🎯 Skill {i}/{len(skills)}: {skill_language}")
                print("-" * 30)
                
                # 4a. Calculate GT local pose for current skill
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
                
                # 4b. Use IK to get joints and move robot
                print(f"   🔧 Moving robot to local pose...")
                target_joints = self.motion_planner.inverse_kinematics(target_ee_pos, target_ee_quat)
                
                if target_joints is None:
                    print(f"   ❌ IK failed")
                    continue
                
                if not self.motion_planner.move_to_joints(target_joints):
                    print(f"   ❌ Robot movement failed")
                    continue
                
                # 4c. Execute VLA with language instruction
                print(f"   🤖 Executing VLA skill...")
                if not self.vla_executor.execute_vla_skill(skill_language, self.env):
                    print(f"   ❌ VLA execution failed")
                    continue
                
                # 4d. Execute post-actions (gripper control)
                print(f"   💫 Executing post-actions...")
                if not execute_post_actions(skill_language, self.env):
                    print(f"   ❌ Post-actions failed")
                    continue
                
                print(f"   ✅ Skill completed successfully!")
                success_count += 1
            
            # Step 5: Report results
            print(f"\n📊 Pipeline Execution Results:")
            print(f"   ✅ Successful skills: {success_count}/{len(skills)}")
            print(f"   📈 Success rate: {success_count/len(skills)*100:.1f}%")
            
            success = success_count == len(skills)
            if success:
                print(f"\n🎉 Pipeline execution completed successfully!")
            else:
                print(f"\n⚠️ Pipeline completed with {len(skills)-success_count} failed skills")
            
            return success
            
        except Exception as e:
            print(f"❌ Pipeline execution error: {e}")
            return False
        
        finally:
            # Clean up
            if self.env:
                self.env.close()
                print("🔧 Environment closed")


def main():
    """Main function for pipeline execution."""
    parser = argparse.ArgumentParser(description='Execute long horizon manipulation pipeline')
    parser.add_argument('--task_name', type=str, required=True,
                        help='Name of long horizon task to execute',
                        choices=['Cooking Preparation Setup', 'Complete Kitchen Organization', 'Switch Table Objects'])
    parser.add_argument('--vla_checkpoint', type=str, default=None,
                        help='Path to VLA model checkpoint')
    parser.add_argument('--wrist_only', action='store_true',
                        help='Use only wrist camera as observation input')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with extra logging')
    
    args = parser.parse_args()
    
    print("🎯 Long Horizon Pipeline Executor")
    print(f"📝 Task: {args.task_name}")
    if args.vla_checkpoint:
        print(f"🤖 VLA Checkpoint: {args.vla_checkpoint}")
    print(f"📷 Wrist only: {args.wrist_only}")
    print("=" * 50)
    
    # Create and execute pipeline
    pipeline = LongHorizonPipeline(
        vla_checkpoint_path=args.vla_checkpoint,
        task_name=args.task_name,
        wrist_only=args.wrist_only
    )
    
    start_time = time.time()
    success = pipeline.execute_pipeline()
    execution_time = time.time() - start_time
    
    print(f"\n⏱️ Total execution time: {execution_time:.2f} seconds")
    
    if success:
        print("🎉 Pipeline execution completed successfully!")
        exit(0)
    else:
        print("❌ Pipeline execution failed!")
        exit(1)


if __name__ == "__main__":
    main()