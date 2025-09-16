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
from collections import deque
import cv2
from scipy.spatial.transform import Rotation as R

# Add project paths
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')

# LIBERO imports
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env

# # TracIK imports
# # from utils.tracik_tools import solve_ik, pose6d_to_matrix
# try:
#     from tracik_tools import solve_ik, pose6d_to_matrix
# except ImportError:
#     # Fallback: try the original import
#     try:
#         from utils.tracik_tools import solve_ik, pose6d_to_matrix
#     except ImportError as e:
#         print(f"Import error: {e}")
#         print(f"Script dir: {script_dir}")
#         print(f"Repo root: {repo_root}")
#         print(f"Utils tracik path: {utils_tracik_path}")
#         print(f"Utils path exists: {os.path.exists(os.path.join(repo_root, 'utils', 'tracik_tools.py'))}")
#         print(f"Current sys.path: {sys.path[:7]}")
#         raise
from scripts.phase2.utils.motion_planner import MotionPlanner

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
    
    def _check_unnorm_key(self, cfg, model):
        """Check that the model contains the action un-normalization key."""
        # Initialize unnorm_key
        unnorm_key = cfg.task_suite_name

        # For atomic_skills, the model was trained with "libero_atomic_skills" key
        if unnorm_key == "atomic_skills":
            unnorm_key = "libero_atomic_skills"

        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"

        if unnorm_key not in model.norm_stats:
            print(f"❌ Action un-norm key '{unnorm_key}' not found in VLA norm_stats!")
            print(f"Available keys: {list(model.norm_stats.keys())}")
            raise KeyError(f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!")

        # Set the unnorm_key in cfg
        cfg.unnorm_key = unnorm_key
        print(f"✅ Set unnorm_key: {unnorm_key}")
    
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
                task_suite_name: str = "atomic_skills"
                # Additional attributes needed by VLA utilities
                wrist_only: bool = False
                agent_only: bool = False
                pro_only: bool = False
                is_oss: bool = False
                is_depth: bool = False
            
            # Set up config
            cfg = VLAConfig()
            # cfg.pretrained_checkpoint = self.checkpoint_path or "/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/view_wrist/1.0.0/openvla-7b+libero_local1+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--150000_chkpt"
            cfg.pretrained_checkpoint = self.checkpoint_path or "/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/atomic_skills/1.0.1/openvla-7b+libero_atomic_skills+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--200000_chkpt"
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
            
            # Set unnorm_key with proper mapping
            self._check_unnorm_key(cfg, self.vla_model)
            
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
        """Prepare observation for policy input - matches atomic skills evaluation."""
        # Always use wrist-only since atomic skills model was trained with wrist cam only
        # Get preprocessed images - wrist cam only
        wrist_img = self.get_libero_wrist_image(obs)

        # Resize images to size expected by model
        wrist_img_resized = self.resize_image_for_policy(wrist_img, self.resize_size)

        # Prepare observations dict - matches atomic skills evaluation
        observation = {
            "full_image": wrist_img_resized,
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
    
    def execute_vla_skill(self, language_instruction: str, env, max_steps: int = 200) -> bool:
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
        print(f"📝 VLA Task Description: '{language_instruction}'")
        
        try:
            from collections import deque
            from prismatic.vla.constants import NUM_ACTIONS_CHUNK
            
            # Initialize action queue
            action_queue = deque(maxlen=self.cfg.num_open_loop_steps)
            
            # Run skill execution
            t = 0
            
            dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])  # 6-DOF + gripper open
            obs, _, _, _ = env.step(dummy_action)
            
            # Save debug image: initial image before VLA execution
            if hasattr(self, 'pipeline'):
                self.pipeline._save_debug_image(obs, f"4_before_vla_execution_{language_instruction.replace(' ', '_')}")
            
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
                
                # Record frame during VLA execution (if video recording is enabled)
                if hasattr(self, 'pipeline') and hasattr(self.pipeline, '_record_frame'):
                    self.pipeline._record_frame(obs)
                
                if done:
                    print(f"   ⚠️ Environment episode terminated at step {t}/{max_steps}")
                    break
                t += 1
            
            # Check actual goal condition instead of just time/done status
            goal_achieved = self.check_skill_success(env, language_instruction)
            
            if goal_achieved:
                print(f"   ✅ VLA skill executed successfully in {t} steps (goal achieved)")
                return True
            elif t < max_steps:
                print(f"   ⚠️ VLA skill completed early in {t} steps (environment terminated, goal achieved: {goal_achieved})")
                return goal_achieved
            else:
                print(f"   ❌ VLA skill failed (max steps: {max_steps}, steps taken: {t}, goal achieved: {goal_achieved})")
                return False
            
        except Exception as e:
            print(f"   ❌ VLA execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_skill_success(self, env, language_instruction):
        """
        Check if the skill was actually successful by evaluating goal conditions.
        This should be implemented based on the environment's success criteria.
        """
        try:
            # For LIBERO environments, check if the success condition is met
            if hasattr(env, 'env') and hasattr(env.env, '_check_success'):
                return env.env._check_success()
            elif hasattr(env, '_check_success'):
                return env._check_success()
            else:
                # Fallback: assume success if no explicit check available
                # This maintains backward compatibility
                print(f"   ⚠️ Warning: No success check available for environment")
                return True
        except Exception as e:
            print(f"   ⚠️ Warning: Error checking success condition: {e}")
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
    

    if 'pick' in skill_lower:
        # Close gripper for picking
        print(f"🤏 Closing gripper (pick action)")
        # Create gripper close action (gripper value = 1.0 = closed)
        gripper_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # Only gripper closes
        for i in range(5):  # Execute gripper action multiple times to ensure closure
            try:
                obs, _, done, _ = env.step(gripper_action)
                if done:
                    print(f"   ⚠️ Episode terminated during gripper action {i+1}/5")
                    print(f"   🔄 Resetting environment...")
                    env.reset()
                    return False
            except Exception as e:
                print(f"   ❌ Gripper action {i+1}/5 failed: {e}")
                return False
        
    elif 'place' in skill_lower:
        # Open gripper for placing
        print(f"🖐️ Opening gripper (place action)")
        # Create gripper open action (gripper value = -1.0 = open)
        gripper_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])  # Only gripper opens
        for i in range(5):  # Execute gripper action multiple times to ensure opening
            try:
                obs, _, done, _ = env.step(gripper_action)
                if done:
                    print(f"   ⚠️ Episode terminated during gripper action {i+1}/5")
                    print(f"   🔄 Resetting environment...")
                    env.reset()
                    return False
            except Exception as e:
                print(f"   ❌ Gripper action {i+1}/5 failed: {e}")
                return False
        
    else:
        # No specific gripper action needed
        print(f"💫 No gripper action required")
    
    return True
        



class LongHorizonPipeline:
    """Main pipeline orchestrator implementing the corrected architecture."""
    
    def __init__(self, 
                 vla_checkpoint_path: Optional[str] = None,
                 task_name: str = "Cooking Preparation Setup",
                 wrist_only: bool = False,
                 visualize_poses: bool = False,
                 ee_backward_offset: float = 0.0,
                 motion_planner_method: str = "cartesian_linear",
                 continue_on_failure: bool = False,
                 motion_planner_steps: int = 400,
                 motion_planner_pos_gain: float = 5.0,
                 motion_planner_ori_gain: float = 5.0):
        """
        Initialize the long horizon pipeline.
        
        Args:
            vla_checkpoint_path: Path to VLA model checkpoint
            task_name: Name of the long horizon task to execute
            wrist_only: Whether to use only wrist camera as observation input
            visualize_poses: Whether to run in pose visualization mode (no VLA execution)
            ee_backward_offset: Backward offset in meters along EE facing direction to avoid collision
            motion_planner_method: Motion planner method ('ik_setjoint' or 'cartesian_linear')
            continue_on_failure: Whether to continue pipeline execution even if individual skills fail
            motion_planner_steps: Number of interpolation steps for cartesian_linear motion planner
            motion_planner_pos_gain: Position gain for cartesian_linear motion planner
            motion_planner_ori_gain: Orientation gain for cartesian_linear motion planner
        """
        self.task_name = task_name
        self.wrist_only = wrist_only
        self.visualize_poses = visualize_poses
        self.ee_backward_offset = ee_backward_offset
        self.motion_planner_method = motion_planner_method
        self.continue_on_failure = continue_on_failure
        self.motion_planner_steps = motion_planner_steps
        self.motion_planner_pos_gain = motion_planner_pos_gain
        self.motion_planner_ori_gain = motion_planner_ori_gain
        self.vla_executor = VLAExecutor(vla_checkpoint_path, wrist_only=wrist_only)
        self.vla_executor.pipeline = self  # Pass pipeline reference for video recording
        self.motion_planner = None
        self.env = None
        
        # Video recording setup
        self.video_writer = None
        self.video_filename = None  # Initialize video_filename attribute
        self.record_video = not visualize_poses  # Disable video recording in visualization mode
        
        # Pose visualization setup
        self.collected_poses = []  # Store (skill_name, ee_pos, ee_quat) tuples
        self._setup_video_recording()
        
        # Load pipeline components
        self._load_pipeline_components()
    
    def _apply_ee_backward_offset(self, target_ee_pos: np.ndarray, target_ee_quat: np.ndarray) -> tuple:
        """
        Apply backward offset to EE pose along its facing direction.
        
        Args:
            target_ee_pos: Original target EE position [x, y, z]
            target_ee_quat: Original target EE quaternion [x, y, z, w]
        
        Returns:
            Tuple of (offset_pos, unchanged_quat)
        """
        if self.ee_backward_offset <= 0.0:
            return target_ee_pos, target_ee_quat
        
        # Convert quaternion to rotation matrix to get facing direction
        from scipy.spatial.transform import Rotation as R
        rotation = R.from_quat(target_ee_quat)
        
        # Get the forward direction vector (typically -Z in robot frame)
        # For most robots, the forward/facing direction is the negative Z axis
        forward_direction = rotation.apply([0, 0, -1])  # -Z direction in world frame
        
        # Apply backward offset (same direction as forward to move backward)
        backward_offset = forward_direction * self.ee_backward_offset
        offset_pos = target_ee_pos + backward_offset
        
        print(f"   📐 Applied {self.ee_backward_offset:.3f}m backward offset: {target_ee_pos} → {offset_pos}")
        
        return offset_pos, target_ee_quat
    
    def _setup_video_recording(self):
        """Setup video recording directories and parameters."""
        # Create appropriate directory based on mode
        if self.visualize_poses:
            # Create pose visualization directory
            output_dir = Path("/mnt/arc/yygx/pkgs_baselines/openvla-oft/images_videos/pose_visualization")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_clean = self.task_name.replace(" ", "_").lower()
            self.pose_images_dir = output_dir / f"{task_clean}_{timestamp}"
            self.pose_images_dir.mkdir(parents=True, exist_ok=True)
            print(f"📷 Pose images will be saved to: {self.pose_images_dir}")
        elif self.record_video:
            # Create video directory
            video_dir = Path("/mnt/arc/yygx/pkgs_baselines/openvla-oft/images_videos/long_horizon_recordings")
            video_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_clean = self.task_name.replace(" ", "_").lower()
            self.video_filename = video_dir / f"long_horizon_{task_clean}_{timestamp}.mp4"
            print(f"📹 Video recording will be saved to: {self.video_filename}")
    
    def _start_video_recording(self, width=640, height=480, fps=30):
        """Start video recording."""
        if not self.record_video:
            return
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(str(self.video_filename), fourcc, fps, (width, height))
        print(f"🎬 Started video recording at {fps} FPS")
    
    def _record_frame(self, obs):
        """Record a single frame to video."""
        if not self.record_video or self.video_writer is None:
            return
            
        try:
            # Get agent view image from observation
            if 'agentview_image' in obs:
                # Convert from RGB to BGR for OpenCV
                frame = obs['agentview_image']
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.video_writer.write(frame_bgr)
        except Exception as e:
            print(f"⚠️ Warning: Failed to record frame: {e}")
    
    def _stop_video_recording(self):
        """Stop video recording and save file."""
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"🎬 Video recording saved: {self.video_filename}")
            self.video_writer = None
    
    def _save_debug_image(self, obs, image_name: str):
        """Save debug image to tmp_imgs directory."""
        try:
            import os
            debug_dir = Path("/mnt/arc/yygx/pkgs_baselines/openvla-oft/images_videos/long_horizon_recordings/tmp_imgs")
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Get agent view image from observation
            if 'agentview_image' in obs:
                frame = obs['agentview_image']
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                
                # Convert from RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Save image
                image_path = debug_dir / f"{image_name}.png"
                cv2.imwrite(str(image_path), frame_bgr)
                print(f"🖼️ Debug image saved: {image_name}.png")
        except Exception as e:
            print(f"⚠️ Warning: Failed to save debug image: {e}")
    
    def _save_motion_planner_images(self, obs, skill_idx: int, skill_name: str):
        """Save agent view and wrist camera images after motion planner finishes to middle_imgs directory."""
        try:
            # Create middle_imgs directory
            middle_imgs_dir = Path("/mnt/arc/yygx/pkgs_baselines/openvla-oft/images_videos/long_horizon_recordings/middle_imgs")
            middle_imgs_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename prefix
            prefix = f"skill_{skill_idx:02d}_{skill_name.replace(' ', '_')}_after_motion_planner"
            
            # Save agent view image
            if 'agentview_image' in obs:
                agent_img = obs['agentview_image']
                if agent_img.dtype != np.uint8:
                    agent_img = (agent_img * 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                agent_img_bgr = cv2.cvtColor(agent_img, cv2.COLOR_RGB2BGR)
                
                # Save the agent view image
                agent_path = middle_imgs_dir / f"{prefix}_agentview.png"
                cv2.imwrite(str(agent_path), agent_img_bgr)
                print(f"   💾 Saved agent view: {prefix}_agentview.png")
            
            # Save wrist camera image
            if 'robot0_eye_in_hand_image' in obs:
                wrist_img = obs['robot0_eye_in_hand_image']
                if wrist_img.dtype != np.uint8:
                    wrist_img = (wrist_img * 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                wrist_img_bgr = cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR)
                
                # Save the wrist view image
                wrist_path = middle_imgs_dir / f"{prefix}_wrist.png"
                cv2.imwrite(str(wrist_path), wrist_img_bgr)
                print(f"   💾 Saved wrist view: {prefix}_wrist.png")
        
        except Exception as e:
            print(f"⚠️ Warning: Failed to save motion planner images: {e}")
    
    def _draw_coordinate_axes(self, img: np.ndarray, origin_2d: tuple, axis_length: int = 50, label: str = "") -> np.ndarray:
        """
        Draw RGB coordinate axes on image.
        
        Args:
            img: Image array (H, W, 3)
            origin_2d: (x, y) pixel coordinates of origin
            axis_length: Length of each axis in pixels
            label: Optional label for the coordinate system
            
        Returns:
            Modified image with coordinate axes drawn
        """
        try:
            img_copy = img.copy()
            ox, oy = origin_2d
            
            # Make sure coordinates are within image bounds
            h, w = img.shape[:2]
            if ox < 0 or ox >= w or oy < 0 or oy >= h:
                return img_copy
            
            # Draw X axis (Red)
            end_x = (min(ox + axis_length, w - 1), oy)
            cv2.arrowedLine(img_copy, (ox, oy), end_x, (0, 0, 255), 3, tipLength=0.3)  # BGR format
            cv2.putText(img_copy, 'X', (end_x[0] + 5, end_x[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw Y axis (Green)  
            end_y = (ox, max(oy - axis_length, 0))  # Y-axis points up (negative in image coordinates)
            cv2.arrowedLine(img_copy, (ox, oy), end_y, (0, 255, 0), 3, tipLength=0.3)  # BGR format
            cv2.putText(img_copy, 'Y', (end_y[0] + 5, end_y[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw Z axis (Blue) - approximate as diagonal since we can't show true 3D
            end_z = (max(ox - int(axis_length * 0.7), 0), max(oy - int(axis_length * 0.7), 0))
            cv2.arrowedLine(img_copy, (ox, oy), end_z, (255, 0, 0), 3, tipLength=0.3)  # BGR format
            cv2.putText(img_copy, 'Z', (end_z[0] - 15, end_z[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Add label if provided
            if label:
                cv2.putText(img_copy, label, (ox - 10, oy + axis_length + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # Add black outline for better visibility
                cv2.putText(img_copy, label, (ox - 10, oy + axis_length + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            
            return img_copy
            
        except Exception as e:
            print(f"   ⚠️  Failed to draw coordinate axes: {e}")
            return img

    def _project_3d_to_2d(self, point_3d: np.ndarray, camera_name: str = "agentview") -> tuple:
        """
        Project 3D world coordinates to 2D image coordinates.
        This is a simplified projection - in practice you'd need camera intrinsics.
        
        Args:
            point_3d: 3D point [x, y, z] in world coordinates
            camera_name: Name of the camera
            
        Returns:
            (x, y) pixel coordinates
        """
        try:
            # Get camera pose (simplified approach)
            model = self.env.sim.model
            data = self.env.sim.data
            
            # Find camera
            cam_id = None
            for i in range(model.ncam):
                if model.camera_id2name(i) == camera_name:
                    cam_id = i
                    break
            
            if cam_id is None:
                # Fallback to simple projection
                # Assume camera looking down from above
                x_2d = int(128 + point_3d[0] * 100)  # Scale and center
                y_2d = int(128 - point_3d[1] * 100)  # Flip Y for image coordinates
                return (max(0, min(x_2d, 255)), max(0, min(y_2d, 255)))
            
            # Get camera position and orientation
            cam_pos = data.cam_xpos[cam_id]
            cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
            
            # Transform point to camera frame
            point_cam = cam_mat.T @ (point_3d - cam_pos)
            
            # Simple perspective projection (assuming known intrinsics)
            if abs(point_cam[2]) < 1e-6:  # Avoid division by zero
                return (128, 128)
                
            # Project to image plane (simplified)
            focal_length = 200  # Approximate focal length in pixels
            x_2d = int(128 + focal_length * point_cam[0] / point_cam[2])
            y_2d = int(128 - focal_length * point_cam[1] / point_cam[2])  # Flip Y
            
            # Clamp to image bounds
            x_2d = max(0, min(x_2d, 255))
            y_2d = max(0, min(y_2d, 255))
            
            return (x_2d, y_2d)
            
        except Exception as e:
            print(f"   ⚠️  Failed to project 3D point: {e}")
            # Fallback to center of image
            return (128, 128)

    def _save_pose_images(self, obs, skill_idx: int, skill_name: str):
        """Save pose visualization images with coordinate system overlays."""
        if not self.visualize_poses:
            return
            
        try:
            # Create filename prefix
            prefix = f"skill_{skill_idx:02d}_{skill_name.replace(' ', '_')}"
            
            # Save agent view image with coordinate systems
            if 'agentview_image' in obs:
                agent_img = obs['agentview_image']
                if agent_img.dtype != np.uint8:
                    agent_img = (agent_img * 255).astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                agent_img_bgr = cv2.cvtColor(agent_img, cv2.COLOR_RGB2BGR)
                
                
                # Save the annotated image
                agent_path = self.pose_images_dir / f"{prefix}_agentview.png"
                cv2.imwrite(str(agent_path), agent_img_bgr)
                print(f"   💾 Saved agent view with coords: {agent_path.name}")
            
            # Save wrist view image (no coordinate overlay for wrist view)
            if 'robot0_eye_in_hand_image' in obs:
                wrist_img = obs['robot0_eye_in_hand_image']
                if wrist_img.dtype != np.uint8:
                    wrist_img = (wrist_img * 255).astype(np.uint8)
                wrist_path = self.pose_images_dir / f"{prefix}_wrist.png"
                cv2.imwrite(str(wrist_path), cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))
                print(f"   💾 Saved wrist view: {wrist_path.name}")
                
        except Exception as e:
            print(f"   ❌ Failed to save pose images: {e}")
    
    def _save_collected_poses(self):
        """Save collected poses to numpy file."""
        if not self.visualize_poses or not self.collected_poses:
            return
            
        try:
            poses_file = self.pose_images_dir / "local_ee_poses.npy"
            np.save(str(poses_file), self.collected_poses)
            print(f"📦 Saved {len(self.collected_poses)} poses to: {poses_file.name}")
            
        except Exception as e:
            print(f"❌ Failed to save poses: {e}")
    
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
                "switch table objects": "LONG_HORIZON_switch_table_objects",
                "pick white bowl": "LONG_HORIZON_pick_white_bowl",
                "pick black bowl": "LONG_HORIZON_pick_black_bowl"
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
            
            # Create environment with extended horizon for long horizon tasks
            self.env, _ = get_libero_env(target_task, model_family='openvla', resolution=256, horizon=10000)
            
            # Set pipeline reference on environment for motion planner access
            self.env.pipeline = self
            obs = self.env.reset()
            
            # Save debug image: reset position
            self._save_debug_image(obs, "1_reset_position")

            if self.record_video:
                self._record_frame(obs)
            
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
            self.motion_planner = MotionPlanner(self.env, method=self.motion_planner_method, 
                                              num_steps=self.motion_planner_steps,
                                              pos_gain=self.motion_planner_pos_gain,
                                              ori_gain=self.motion_planner_ori_gain)
            
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
        
        # Step 0.5: Stabilize objects with dummy actions
        print(f"\n🔧 Stabilizing objects in environment...")
        dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])  # 6-DOF + gripper open
        for i in range(5):
            obs, _, _, _ = self.env.step(dummy_action)
            if self.record_video:
                self._record_frame(obs)
        
        # Save debug image: after 5 dummy actions
        self._save_debug_image(obs, "2_after_5_dummy_actions")
        print(f"✅ Environment stabilized with 5 dummy actions")
        
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
            
            # Step 4: Start video recording and execute skill sequence
            print(f"\n🎬 Step 4: Executing skill sequence...")
            print("-" * 50)
            
            # Start video recording
            if self.record_video:
                if 'agentview_image' in obs:
                    height, width = obs['agentview_image'].shape[:2]
                    self._start_video_recording(width=width, height=height, fps=20)
            
            success_count = 0
            individual_skill_results = []
            
            for i, skill_language in enumerate(skills, 1):
                print(f"\n🎯 Skill {i}/{len(skills)}: {skill_language}")
                print("-" * 30)
                
                # Get current robot pose (obs system)
                current_ee_pos, current_ee_quat = self.motion_planner._get_ee_pose_from_simulation()
                if current_ee_pos is not None and current_ee_quat is not None:
                    current_ee_rot = R.from_quat([current_ee_quat[1], current_ee_quat[2], current_ee_quat[3], current_ee_quat[0]])
                    current_ee_axis = current_ee_rot.as_rotvec()
                    print(f"2️⃣ Current robot pose (obs): pos=[{current_ee_pos[0]:.4f}, {current_ee_pos[1]:.4f}, {current_ee_pos[2]:.4f}], axis=[{current_ee_axis[0]:.4f}, {current_ee_axis[1]:.4f}, {current_ee_axis[2]:.4f}]")
                
                # 4a. Calculate GT local pose for current skill
                target_pose = self.pose_calculator_module.calculate_gt_local_pose(
                    skill_language, self.env, 
                    available_objects=available_objects,
                    calculator=pose_calculator
                )
                
                if target_pose is None:
                    print(f"4️⃣ Motion planner: Failed (no target pose)")
                    if self.visualize_poses:
                        self.collected_poses.append({
                            'skill_index': i,
                            'skill_name': skill_language,
                            'ee_pos': None,
                            'ee_quat': None,
                            'status': 'pose_calculation_failed'
                        })
                    if not self.continue_on_failure:
                        print(f"   🛑 Terminating entire pipeline due to pose calculation failure")
                        return False
                    else:
                        print(f"   ⚠️ Continuing pipeline despite pose calculation failure")
                        continue
                
                target_ee_pos, target_ee_quat = target_pose
                
                # Apply backward offset if enabled
                offset_ee_pos, offset_ee_quat = self._apply_ee_backward_offset(target_ee_pos, target_ee_quat)
                
                # 4b. Move robot to offset pose using configured motion planner
                move_success, obs = self.motion_planner.move_to_pose(offset_ee_pos, offset_ee_quat)
                if move_success:
                    # Save debug image: after set joint
                    self._save_debug_image(obs, f"3_after_set_joint_skill_{i}")
                    # Save agent view and wrist view images after motion planner finishes
                    self._save_motion_planner_images(obs, i, skill_language)
                if self.record_video:
                    self._record_frame(obs)
                if not move_success:
                    print(f"4️⃣ Motion planner: Failed (movement)")
                    if not self.continue_on_failure:
                        print(f"   🛑 Terminating entire pipeline due to motion planner failure")
                        return False
                    else:
                        print(f"   ⚠️ Continuing pipeline despite motion planner failure")
                        continue
                
                
                print(f"4️⃣ Motion planner: Success")
                # exit(0)
                
                # If in pose visualization mode, save images and collect poses
                if self.visualize_poses:
                    self.collected_poses.append({
                        'skill_index': i,
                        'skill_name': skill_language,
                        'ee_pos': offset_ee_pos.copy(),  # Store the offset pose that was actually used
                        'ee_quat': offset_ee_quat.copy(),
                        'original_ee_pos': target_ee_pos.copy(),  # Keep original for reference
                        'original_ee_quat': target_ee_quat.copy(),
                        'status': 'success'
                    })
                    # Use dummy action to get current observation
                    dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])  # 6-DOF + gripper open
                    obs, _, _, _ = self.env.step(dummy_action)
                    self._save_pose_images(obs, i, skill_language)
                    skill_success = True  # Mark as success for visualization mode
                    vla_success = True
                    post_success = True
                else:
                    skill_success = False
                    
                    # 4c. Execute VLA with language instruction
                    print(f"   🤖 Executing VLA skill...")
                    print(f"   📝 VLA Task Description: '{skill_language}'")
                    vla_success = self.vla_executor.execute_vla_skill(skill_language, self.env)
                
                if vla_success and not self.visualize_poses:
                    # 4d. Execute post-actions (gripper control)
                    print(f"   💫 Executing post-actions...")
                    post_success = execute_post_actions(skill_language, self.env)
                    
                    if post_success:
                        skill_success = True
                        print(f"   ✅ Skill completed successfully!")
                        success_count += 1
                    else:
                        print(f"   ❌ Post-actions failed")
                        if not self.continue_on_failure:
                            print(f"   🛑 Terminating entire pipeline due to post-actions failure")
                            return False
                        else:
                            print(f"   ⚠️ Continuing pipeline despite post-actions failure")
                            skill_success = False
                elif vla_success and self.visualize_poses:
                    # In visualization mode, skip post-actions and mark as successful
                    skill_success = True
                    print(f"   ✅ Pose visualization completed!")
                    success_count += 1
                else:
                    print(f"   ❌ VLA execution failed")
                    if not self.continue_on_failure:
                        print(f"   🛑 Terminating entire pipeline due to VLA execution failure")
                        return False
                    else:
                        print(f"   ⚠️ Continuing pipeline despite VLA execution failure")
                        skill_success = False
                
                # Track individual skill result
                individual_skill_results.append({
                    "skill": skill_language,
                    "success": skill_success,
                    "vla_success": vla_success,
                    "post_actions_success": post_success if vla_success else False
                })
            
            # Step 5: Stop video recording and check overall task success
            if self.record_video:
                self._stop_video_recording()
            
            # Check if overall long-horizon task was successful
            overall_task_success = False
            try:
                if hasattr(self.env, 'env') and hasattr(self.env.env, '_check_success'):
                    overall_task_success = self.env.env._check_success()
                elif hasattr(self.env, '_check_success'):
                    overall_task_success = self.env._check_success()
                print(f"🎯 Overall task goal achieved: {overall_task_success}")
            except Exception as e:
                print(f"⚠️ Could not check overall task success: {e}")
            
            # Step 6: Report detailed results
            print(f"\n📊 Detailed Pipeline Execution Results:")
            print(f"=" * 60)
            
            # Individual skill results
            print(f"\n📋 Individual Skill Results:")
            for i, result in enumerate(individual_skill_results, 1):
                status = "✅" if result["success"] else "❌"
                vla_status = "✅" if result["vla_success"] else "❌"
                post_status = "✅" if result["post_actions_success"] else "❌"
                print(f"   {i}. {result['skill']}: {status}")
                print(f"      - VLA execution: {vla_status}")
                print(f"      - Post-actions: {post_status}")
            
            # Summary statistics
            skill_success_rate = success_count / len(skills) * 100
            print(f"\n📈 Summary:")
            print(f"   🎯 Overall task success: {'✅ YES' if overall_task_success else '❌ NO'}")
            print(f"   🔧 Individual skills completed: {success_count}/{len(skills)} ({skill_success_rate:.1f}%)")
            
            # Determine final result
            final_success = overall_task_success
            if final_success:
                print(f"\n🎉 LONG-HORIZON TASK COMPLETED SUCCESSFULLY!")
            else:
                print(f"\n⚠️ Long-horizon task failed (goal not achieved)")
                if success_count > 0:
                    print(f"   Note: {success_count} individual skills completed successfully")
            
            # Save collected poses in visualization mode
            self._save_collected_poses()
            
            return final_success
            
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
                        choices=['Cooking Preparation Setup', 'Complete Kitchen Organization', 'Switch Table Objects', 'Pick White Bowl', 'Pick Black Bowl'])
    parser.add_argument('--vla_checkpoint', type=str, default=None,
                        help='Path to VLA model checkpoint')
    parser.add_argument('--wrist_only', action='store_true', default=True,
                        help='Use only wrist camera as observation input (default: True)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with extra logging')
    parser.add_argument('--visualize_poses', action='store_true',
                        help='Visualize local poses mode: only calculate poses, move robot, and save images (no VLA execution)')
    parser.add_argument('--ee_backward_offset', type=float, default=0.0,
                        help='Backward offset in meters along EE facing direction to avoid object collision (default: 0.0)')
    parser.add_argument('--motion_planner_method', type=str, default='cartesian_linear',
                        choices=['ik_setjoint', 'cartesian_linear'],
                        help='Motion planner method: ik_setjoint (IK+setjoint) or cartesian_linear (Cartesian Linear Interpolator) (default: cartesian_linear)')
    parser.add_argument('--continue_on_failure', action='store_true', default=False,
                        help='Continue pipeline execution even if individual skills fail (default: False - terminate on first failure)')
    parser.add_argument('--motion_planner_steps', type=int, default=400,
                        help='Number of interpolation steps for cartesian_linear motion planner (default: 400)')
    parser.add_argument('--motion_planner_pos_gain', type=float, default=5.0,
                        help='Position gain for cartesian_linear motion planner (default: 5.0)')
    parser.add_argument('--motion_planner_ori_gain', type=float, default=5.0,
                        help='Orientation gain for cartesian_linear motion planner (default: 5.0)')
    
    args = parser.parse_args()
    
    print("🎯 Long Horizon Pipeline Executor")
    print(f"📝 Task: {args.task_name}")
    if args.vla_checkpoint:
        print(f"🤖 VLA Checkpoint: {args.vla_checkpoint}")
    print(f"📷 Wrist only: {args.wrist_only}")
    print(f"🔄 Continue on failure: {args.continue_on_failure}")
    print(f"🔧 Motion planner steps: {args.motion_planner_steps}")
    print(f"⚙️ Motion planner gains: pos={args.motion_planner_pos_gain}, ori={args.motion_planner_ori_gain}")
    print("=" * 50)
    
    # Create and execute pipeline
    pipeline = LongHorizonPipeline(
        vla_checkpoint_path=args.vla_checkpoint,
        task_name=args.task_name,
        wrist_only=args.wrist_only,
        visualize_poses=args.visualize_poses,
        ee_backward_offset=args.ee_backward_offset,
        motion_planner_method=args.motion_planner_method,
        continue_on_failure=args.continue_on_failure,
        motion_planner_steps=args.motion_planner_steps,
        motion_planner_pos_gain=args.motion_planner_pos_gain,
        motion_planner_ori_gain=args.motion_planner_ori_gain
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