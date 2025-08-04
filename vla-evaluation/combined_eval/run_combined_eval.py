#!/usr/bin/env python3
"""
run_combined_eval.py

Combined evaluation script that uses PoseVLM to predict local EE poses from LIBERO reset images,
solves IK to reach those poses, and then uses OpenVLA-OFT for task completion.

Pipeline:
LIBERO Task → Reset Image + Language → PoseVLM → Local EE Pose → IK → Joint Positions → OpenVLA-OFT → Task Success

Usage:
    python vla-evaluation/combined_eval/run_combined_eval.py \
        --pose_vlm_checkpoint runs/pose_vlm/1.0.0/.../vla--90000_checkpoint.pt \
        --openvla_checkpoint runs/view_wrist/1.0.3/.../50000_chkpt \
        --task_suite_name "boss_44" \
        --num_episodes_per_task 20 \
        --max_ik_attempts 5 \
        --save_debug_images True
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import argparse
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# LIBERO imports
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image

# PoseVLM imports
from prismatic.models.vlms.pose_vlm import PoseVLM
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from transformers import AutoModelForVision2Seq

# OpenVLA imports
from experiments.robot.openvla_utils import get_vla, get_vla_action
from experiments.robot.robot_utils import get_action, get_image_resize_size, set_seed_everywhere

# IK tools
import sys
sys.path.append('../shared')
from tracik_tools import solve_ik, pose6d_to_matrix

# Import PoseFinetuneConfig early to make it available for pickle
import sys
sys.path.append('vla-scripts')
try:
    from finetune_pose import PoseFinetuneConfig
except ImportError:
    # Fallback if import fails
    class PoseFinetuneConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

# Import OpenVLA eval config
from experiments.robot.libero.run_libero_eval_local import GenerateConfig


class CombinedEvaluator:
    """
    Combined evaluator that uses PoseVLM + IK + OpenVLA-OFT for LIBERO task evaluation.
    """
    
    def __init__(
        self,
        pose_vlm_checkpoint: str,
        openvla_checkpoint: str,
        task_suite_name: str = "boss_44",
        num_episodes_per_task: int = 20,
        max_ik_attempts: int = 5,
        save_debug_images: bool = True,
        debug_image_dir: str = "runs/eval_results/combined_eval/debug_images",
        device: str = "cuda",
        seed: int = 42,
        use_pose_vlm_smaller_split_tasks_only: bool = False,
        wrist_only: bool = True,
        third_person_view_only: bool = False,
        both_views: bool = False,
        debug_single_skill: bool = False,
        debug_skill_name: str = "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
        compare_training_image: bool = False,
        training_data_root: str = "../../datasets/local_pairs_datasets",
    ):
        """
        Initialize the combined evaluator.
        
        Args:
            pose_vlm_checkpoint: Path to PoseVLM checkpoint
            openvla_checkpoint: Path to OpenVLA-OFT checkpoint
            task_suite_name: LIBERO task suite name
            num_episodes_per_task: Number of episodes per task
            max_ik_attempts: Maximum IK attempts per pose prediction
            save_debug_images: Whether to save debug images
            debug_image_dir: Directory to save debug images
            device: Device to run evaluation on
            seed: Random seed for reproducibility
        """
        self.pose_vlm_checkpoint = pose_vlm_checkpoint
        self.openvla_checkpoint = openvla_checkpoint
        self.task_suite_name = task_suite_name
        self.num_episodes_per_task = num_episodes_per_task
        self.max_ik_attempts = max_ik_attempts
        self.save_debug_images = save_debug_images
        self.debug_image_dir = Path(debug_image_dir)
        self.device = device
        self.seed = seed
        self.use_pose_vlm_smaller_split_tasks_only = use_pose_vlm_smaller_split_tasks_only
        
        # Debug parameters
        self.debug_single_skill = debug_single_skill
        self.debug_skill_name = debug_skill_name
        self.compare_training_image = compare_training_image
        self.training_data_root = training_data_root
        
        # View selection parameters
        self.wrist_only = wrist_only
        self.third_person_view_only = third_person_view_only
        self.both_views = both_views
        
        # Validate view selection
        view_count = sum([wrist_only, third_person_view_only, both_views])
        if view_count != 1:
            raise ValueError("Exactly one view option must be selected: wrist_only, third_person_view_only, or both_views")
        
        # Define the 5 tasks that are in both smaller_splits and BOSS_44 (intersection)
        self.pose_vlm_smaller_split_tasks = [
            "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet",
            "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet",
            "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet",
            "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
            "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack"
        ]
        
        # Set random seed
        set_seed_everywhere(seed)
        
        # Load models
        self._load_pose_vlm()
        self._load_openvla()
        self._load_libero_suite()
        
        # Load training data if comparison is enabled
        if self.compare_training_image:
            self._load_training_data()
        
        # Create debug image directory
        if self.save_debug_images:
            self.debug_image_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_pose_vlm(self):
        """Load PoseVLM model."""
        print("Loading PoseVLM model...")
        

        
        # Load checkpoint
        checkpoint = torch.load(self.pose_vlm_checkpoint, map_location="cpu")
        self.config = checkpoint.get("config", None)
        
        if self.config is None:
            # Fallback: create a simple config
            class SimpleConfig:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
            
            self.config = SimpleConfig(
                pose_head_type="gmm",
                gmm_num_components=3,
                pose_dim=6,
                num_pose_tokens=6,
                num_images_in_input=1,
                max_length=512,
            )
        
        # Load processor - use default for now to avoid loading issues
        print("Loading processor...")
        self.pose_processor = PrismaticProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

        # Load base VLA model (matches training script)
        base_vla = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # Wrap LLM backbone
        class OpenVLAWrapper(torch.nn.Module):
            def __init__(self, language_model, processor):
                super().__init__()
                self.llm = language_model
                self.tokenizer = processor.tokenizer
                self.config = language_model.config
                self.embed_dim = language_model.config.hidden_size
            def forward(self, **kwargs):
                return self.llm(**kwargs)
            def get_tokenizer(self):
                return self.tokenizer

        llm_backbone = OpenVLAWrapper(base_vla.language_model, self.pose_processor)

        # Construct PoseVLM (matches training)
        self.pose_vlm = PoseVLM(
            model_id=f"pose_vlm_{self.config.pose_head_type}",
            vision_backbone=base_vla.vision_backbone,
            llm_backbone=llm_backbone,
            pose_head_type=self.config.pose_head_type,
            pose_dim=self.config.pose_dim,
            num_pose_tokens=self.config.num_pose_tokens,
            gmm_num_components=self.config.gmm_num_components,
            enable_mixed_precision_training=True,
        )

        # Load checkpoint weights
        state_dict = checkpoint["model"]
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_key = key[7:]
            else:
                new_key = key
            new_state_dict[new_key] = value
        self.pose_vlm.load_state_dict(new_state_dict, strict=False)
        # Move PoseVLM to the same device as OpenVLA for consistency
        self.pose_vlm.to(self.device)
        self.pose_vlm.eval()
        print("PoseVLM loaded successfully!")
    
    def _load_openvla(self):
        """Load OpenVLA-OFT model."""
        print("Loading OpenVLA-OFT model...")
        
        # Create config for OpenVLA (following run_libero_eval.py pattern)
        from experiments.robot.robot_utils import get_model
        from experiments.robot.openvla_utils import get_action_head, get_noisy_action_projector, get_processor, get_proprio_projector
        
        # Create config similar to run_libero_eval.py
        class OpenVLAConfig:
            def __init__(self, checkpoint_path, task_suite_name):
                self.model_family = "openvla"
                self.pretrained_checkpoint = checkpoint_path
                self.use_l1_regression = True
                self.use_diffusion = False
                self.use_film = False
                self.num_images_in_input = 1  # Wrist-only, so only 1 image
                self.use_proprio = True
                self.load_in_8bit = False
                self.load_in_4bit = False
                self.task_suite_name = task_suite_name
                self.wrist_only = True
                self.is_depth = False
                self.center_crop = True
                # Use a default key that should exist in the normalization stats
                self.unnorm_key = "libero44_local"
                self.num_open_loop_steps = 1
        
        config = OpenVLAConfig(self.openvla_checkpoint, self.task_suite_name)
        
        # Initialize model following run_libero_eval.py pattern
        self.openvla_model = get_model(config)
        
        # Load proprio projector if needed
        self.proprio_projector = None
        if config.use_proprio:
            self.proprio_projector = get_proprio_projector(
                config,
                self.openvla_model.llm_dim,
                proprio_dim=8,  # 8-dimensional proprio for LIBERO
            )

        # Load action head if needed
        self.action_head = None
        if config.use_l1_regression or config.use_diffusion:
            self.action_head = get_action_head(config, self.openvla_model.llm_dim)
            # Move to device (get_action_head already handles bfloat16 conversion)
            self.action_head = self.action_head.to(self.device)

        # Load noisy action projector if using diffusion
        self.noisy_action_projector = None
        if config.use_diffusion:
            self.noisy_action_projector = get_noisy_action_projector(config, self.openvla_model.llm_dim)

        # Get OpenVLA processor if needed
        self.processor = None
        if config.model_family == "openvla":
            self.processor = get_processor(config)
        
        self.openvla_config = config
        print("OpenVLA-OFT loaded successfully!")
    
    def _load_libero_suite(self):
        """Load LIBERO task suite."""
        print(f"Loading LIBERO task suite: {self.task_suite_name}")
        
        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[self.task_suite_name]()
        
        # Filter tasks if using PoseVLM smaller split tasks only or debug single skill
        if self.debug_single_skill:
            print(f"DEBUG MODE: Using single skill: {self.debug_skill_name}")
            original_tasks = self.task_suite.tasks
            filtered_tasks = []
            
            for task in original_tasks:
                if task.name == self.debug_skill_name:
                    filtered_tasks.append(task)
                    print(f"  ✓ Included: {task.name}")
                else:
                    print(f"  ✗ Excluded: {task.name}")
            
            # Replace task list
            self.task_suite.tasks = filtered_tasks
            self.num_tasks = len(filtered_tasks)
            print(f"Debug mode: Using {self.num_tasks} task")
        elif self.use_pose_vlm_smaller_split_tasks_only:
            print("Filtering to PoseVLM smaller split training tasks only...")
            original_tasks = self.task_suite.tasks
            filtered_tasks = []
            
            for task in original_tasks:
                if task.name in self.pose_vlm_smaller_split_tasks:
                    filtered_tasks.append(task)
                    print(f"  ✓ Included: {task.name}")
                else:
                    print(f"  ✗ Excluded: {task.name}")
            
            # Replace task list
            self.task_suite.tasks = filtered_tasks
            self.num_tasks = len(filtered_tasks)
            print(f"Filtered to {self.num_tasks} tasks")
        else:
            self.num_tasks = self.task_suite.n_tasks
            print(f"Using all {self.num_tasks} tasks from {self.task_suite_name}")
    
    def _load_training_data(self):
        """Load training dataset for image comparison."""
        print("Loading training dataset for image comparison...")
        
        # Import necessary modules
        from prismatic.vla.datasets.pose_dataset import create_pose_dataset
        import pandas as pd
        import os
        
        # Load training dataset
        self.training_dataset = create_pose_dataset(
            data_root=self.training_data_root,
            split="train",
            num_images_in_input=1,
            use_image_augmentation=False,
            tokenizer_name=self.pose_processor.tokenizer.name_or_path,
            splits_folder="smaller_splits",
            use_pose_normalization=True,
        )
        
        # Load annotation
        annotation_path = os.path.join(self.training_data_root, "train_annotation.csv")
        if not os.path.exists(annotation_path):
            annotation_path = os.path.join(self.training_data_root, "smaller_splits", "train_annotation.csv")
        
        self.training_annotation_df = pd.read_csv(annotation_path)
        print(f"Loaded training dataset with {len(self.training_dataset)} samples")
        print(f"Loaded training annotation with {len(self.training_annotation_df)} entries")
    
    def _get_training_sample_for_task(self, task_name: str) -> Optional[tuple]:
        """
        Get a training sample for a given task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Tuple of (sample, sample_idx) or None if not found
        """
        if not self.compare_training_image:
            return None
        
        # Find samples for this task
        task_samples = self.training_annotation_df[
            self.training_annotation_df['task_name'] == task_name
        ]
        
        if len(task_samples) == 0:
            print(f"    [Warning] No training samples found for task: {task_name}")
            return None
        
        # Get the first sample
        sample_idx = task_samples.iloc[0].name
        sample = self.training_dataset[sample_idx]
        
        print(f"    Found training sample {sample_idx} for task: {task_name}")
        return sample, sample_idx
    
    def _predict_pose_with_retry(self, image: np.ndarray, language: str, is_training_image: bool = False) -> Optional[np.ndarray]:
        """
        Predict pose with retry logic for IK failures.
        
        Args:
            image: Image (reset image from LIBERO environment or training image)
            language: Task language description
            is_training_image: Whether this is a training image (already processed)
            
        Returns:
            Predicted pose (6D) or None if all attempts failed
        """
        for attempt in range(self.max_ik_attempts):
            # Prepare input for PoseVLM
            if is_training_image:
                # Training image is already processed, just convert to tensor
                inputs = {
                    "pixel_values": torch.from_numpy(image).unsqueeze(0).to(self.device, dtype=torch.bfloat16),
                    "input_ids": torch.tensor([[1, 2, 3, 4, 5]]).to(self.device),  # Placeholder
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]).to(self.device),  # Placeholder
                }
            else:
                # Reset image needs processing
                from PIL import Image
                pil_image = Image.fromarray(image)
                pil_image = pil_image.resize((224, 224))  # PoseVLM expects 224x224
                
                # Prepare text input
                prompt = f"In: What action should the robot take to {language.lower()}?\nOut:"
                
                # Process inputs
                inputs = self.pose_processor(prompt, pil_image).to(self.device, dtype=torch.bfloat16)
            
            # Predict pose
            with torch.no_grad():
                predictions = self.pose_vlm.predict_pose(
                    images=inputs["pixel_values"],
                    text=inputs["input_ids"],
                    text_attention_mask=inputs["attention_mask"],
                    num_samples=1,
                )
                
                if self.config.pose_head_type == "gmm":
                    sampled_poses = predictions['sampled_poses'].squeeze(1)
                    weights = predictions['weights']
                    first_token_weights = weights[:, 0, :]
                    first_token_poses = sampled_poses[:, 0, :, :]
                    best_component_indices = torch.argmax(first_token_weights, dim=-1)
                    pred_pose_tensor = first_token_poses[0, best_component_indices[0]]
                else:
                    pred_pose_tensor = predictions['predicted_poses'][0]
                
                # Convert to numpy
                if pred_pose_tensor.dtype == torch.bfloat16:
                    pred_pose = pred_pose_tensor.to(torch.float32).cpu().numpy()
                else:
                    pred_pose = pred_pose_tensor.cpu().numpy()
            
            # Apply pose offset (same as in visualize_vlm_pose_predictions.py)
            pose_offset = np.array([0.667, 0.0, -0.81, 0.0, 0.0, 0.0])
            pred_pose_offset = pred_pose + pose_offset
            
            # Try IK
            pred_pose_mat = pose6d_to_matrix(pred_pose_offset)
            joint_pos = solve_ik(pred_pose_mat)
            
            if joint_pos is not None:
                print(f"    ✓ IK succeeded on attempt {attempt + 1}")
                return pred_pose_offset
            else:
                print(f"    ✗ IK failed on attempt {attempt + 1}")
        
        print(f"    ✗ All {self.max_ik_attempts} IK attempts failed")
        return None
    
    def _predict_pose_from_training_sample(self, sample: dict) -> Optional[np.ndarray]:
        """
        Predict pose from a training sample.
        
        Args:
            sample: Training sample dictionary
            
        Returns:
            Predicted pose (6D) or None if failed
        """
        # Prepare input batch for VLM
        batch = {}
        for k in ["pixel_values", "input_ids", "attention_mask"]:
            v = sample[k]
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0).to(self.device)
            elif isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(v).unsqueeze(0).to(self.device)
            elif np.isscalar(v):
                batch[k] = torch.tensor([[v]], device=self.device)
            else:
                batch[k] = torch.tensor(v, device=self.device).unsqueeze(0)
        
        # Predict pose
        with torch.no_grad():
            predictions = self.pose_vlm.predict_pose(
                images=batch["pixel_values"],
                text=batch["input_ids"],
                text_attention_mask=batch["attention_mask"],
                num_samples=1,
            )
            
            if self.config.pose_head_type == "gmm":
                sampled_poses = predictions['sampled_poses'].squeeze(1)
                weights = predictions['weights']
                first_token_weights = weights[:, 0, :]
                first_token_poses = sampled_poses[:, 0, :, :]
                best_component_indices = torch.argmax(first_token_weights, dim=-1)
                pred_pose_tensor = first_token_poses[0, best_component_indices[0]]
            else:
                pred_pose_tensor = predictions['predicted_poses'][0]
            
            # Convert to numpy
            if pred_pose_tensor.dtype == torch.bfloat16:
                pred_pose = pred_pose_tensor.to(torch.float32).cpu().numpy()
            else:
                pred_pose = pred_pose_tensor.cpu().numpy()
        
        # Apply pose offset
        pose_offset = np.array([0.667, 0.0, -0.81, 0.0, 0.0, 0.0])
        pred_pose_offset = pred_pose + pose_offset
        
        return pred_pose_offset
    
    def _save_debug_image(self, env, task_name: str, episode_idx: int, image_type: str):
        """Save debug image."""
        if not self.save_debug_images:
            return
        
        # Get observation - try different methods
        try:
            obs = env.get_observation()
        except AttributeError:
            try:
                obs = env.env._get_observations()
            except AttributeError:
                # Fallback: use the observation from reset
                obs = env.reset()
        img = obs["agentview_image"][::-1, ::-1]  # Rotate 180 degrees
        
        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)
        
        # Save image
        import cv2
        task_name_clean = task_name.replace(" ", "_").replace("/", "_")
        filename = f"{task_name_clean}_ep{episode_idx}_{image_type}.png"
        filepath = self.debug_image_dir / filename
        cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"    Saved debug image: {filepath}")
    
    def _record_openvla_episode_video(self, env, task_name: str, episode_idx: int, task_description: str):
        """Record video of OpenVLA-OFT episode execution."""
        if not self.save_debug_images:
            return False
        
        import cv2
        import time
        
        # Video settings
        fps = 10  # 10 frames per second
        task_name_clean = task_name.replace(" ", "_").replace("/", "_")
        video_filename = f"{task_name_clean}_ep{episode_idx}_openvla_execution.mp4"
        video_filepath = self.debug_image_dir / video_filename
        
        print(f"    Recording OpenVLA-OFT execution video...")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = None
        
        try:
            # Run the existing OpenVLA episode but record video
            success = self._run_openvla_episode_with_video_recording(
                env, task_description, video_filepath, fourcc, fps
            )
            
            return success
            
        except Exception as e:
            print(f"    Error recording video: {e}")
            return False
        finally:
            if video_writer is not None:
                video_writer.release()
    
    def _run_openvla_episode_with_video_recording(self, env, task_description: str, video_filepath, fourcc, fps):
        """Run OpenVLA-OFT episode with video recording using existing logic."""
        import cv2
        import time
        
        video_writer = None
        frame_count = 0
        max_frames = 600  # 60 seconds at 10 fps
        
        # Use the existing OpenVLA episode logic but add video recording
        max_steps = 200
        step = 0
        
        while step < max_steps and frame_count < max_frames:
            # Get observation
            try:
                obs = env.get_observation()
            except AttributeError:
                try:
                    obs = env.env._get_observations()
                except AttributeError:
                    obs = env.reset()
            
            # Get image for video
            img = obs["agentview_image"][::-1, ::-1]  # Rotate 180 degrees
            
            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                else:
                    img = img.clip(0, 255).astype(np.uint8)
            
            # Initialize video writer on first frame
            if video_writer is None:
                height, width = img.shape[:2]
                video_writer = cv2.VideoWriter(str(video_filepath), fourcc, fps, (width, height))
                print(f"    Started video recording: {video_filepath}")
            
            # Write frame to video
            video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            frame_count += 1
            
            # Use existing OpenVLA episode logic
            success = self._run_single_openvla_step(env, obs, task_description)
            
            # Check for task completion
            if success:
                print(f"    Episode completed after {step + 1} steps")
                if video_writer is not None:
                    video_writer.release()
                return True
            
            step += 1
            
            # Small delay to control video frame rate
            time.sleep(1.0 / fps)
        
        # Episode timeout
        print(f"    Episode timeout after {step} steps")
        if video_writer is not None:
            video_writer.release()
        return False
    
    def _run_single_openvla_step(self, env, obs, task_description: str):
        """Run a single OpenVLA-OFT step using existing logic."""
        try:
            # Use the existing OpenVLA episode logic
            return self._run_openvla_episode(env, task_description)
        except Exception as e:
            print(f"    Error in OpenVLA step: {e}")
            return False
    
    def _run_openvla_episode(self, env, task_description: str, record_video: bool = False, 
                            video_filepath: str = None) -> bool:
        """
        Run OpenVLA-OFT episode.
        
        Args:
            env: LIBERO environment
            task_description: Task language description
            record_video: Whether to record video
            video_filepath: Path to save video
            
        Returns:
            True if task completed successfully, False otherwise
        """
        import cv2
        import time
        
        # Get image resize size
        resize_size = get_image_resize_size(self.openvla_config)
        
        # Initialize video recording if requested
        video_writer = None
        if record_video and video_filepath:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            print(f"    Starting video recording: {video_filepath}")
        
        # Run episode
        max_steps = 400  # Same as in run_libero_eval_local.py
        success = False
        
        for step in range(max_steps):
            # Get observation - try different methods
            try:
                obs = env.get_observation()
            except AttributeError:
                try:
                    obs = env.env._get_observations()
                except AttributeError:
                    # Fallback: use the observation from reset
                    obs = env.reset()
            
            # Record video frame if requested
            if record_video and video_filepath:
                img = obs["agentview_image"][::-1, ::-1]  # Rotate 180 degrees
                
                # Convert to uint8 if needed
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).clip(0, 255).astype(np.uint8)
                    else:
                        img = img.clip(0, 255).astype(np.uint8)
                
                # Initialize video writer on first frame
                if video_writer is None:
                    height, width = img.shape[:2]
                    video_writer = cv2.VideoWriter(str(video_filepath), fourcc, 10, (width, height))
                
                # Write frame to video
                video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # Prepare observation for OpenVLA - default to wrist-only
            wrist_img = obs["robot0_eye_in_hand_image"][::-1, ::-1]  # Rotate 180 degrees
            wrist_img_resized = self._resize_image_for_policy(wrist_img, resize_size)
            
            # Use the same proprioceptive state format as in the training
            proprio_state = np.concatenate([
                obs["robot0_eef_pos"],  # 3D position
                self._quat2axisangle(obs["robot0_eef_quat"]),  # 3D orientation (axis-angle)
                obs["robot0_gripper_qpos"][:1]  # Only gripper position, not width
            ])
            
            observation = {
                "full_image": wrist_img_resized,
                "state": proprio_state,
            }
            
            # Get action from OpenVLA
            actions = get_action(
                self.openvla_config,
                self.openvla_model,
                observation,
                task_description,
                processor=self.processor,
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                noisy_action_projector=self.noisy_action_projector,
            )
            
            # Process action
            action = actions[0]  # Take first action
            action = self._normalize_gripper_action(action, binarize=True)
            if self.openvla_config.model_family == "openvla":
                action = self._invert_gripper_action(action)
            
            # Execute action
            obs, reward, done, info = env.step(action.tolist())
            
            if done:
                success = True
                break
        
        # Clean up video writer
        if video_writer is not None:
            video_writer.release()
        
        return success
    
    def _resize_image_for_policy(self, img: np.ndarray, resize_size) -> np.ndarray:
        """Resize image for policy input."""
        from PIL import Image
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((resize_size, resize_size))
        return np.array(pil_img)
    
    def _quat2axisangle(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to axis-angle representation."""
        # Simplified conversion - in practice, you'd use a proper quaternion library
        return quat  # Placeholder
    
    def _normalize_gripper_action(self, action: np.ndarray, binarize: bool = True) -> np.ndarray:
        """Normalize gripper action."""
        normalized_action = action.copy()
        orig_low, orig_high = 0.0, 1.0
        normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1
        
        if binarize:
            normalized_action[..., -1] = np.sign(normalized_action[..., -1])
        
        return normalized_action
    
    def _invert_gripper_action(self, action: np.ndarray) -> np.ndarray:
        """Invert gripper action."""
        inverted_action = action.copy()
        inverted_action[..., -1] *= -1.0
        return inverted_action
    
    def evaluate_task(self, task_id: int) -> Dict[str, Any]:
        """
        Evaluate a single task.
        
        Args:
            task_id: Task ID in the suite
            
        Returns:
            Dictionary with evaluation results
        """
        task = self.task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        
        print(f"\n{'='*20} Evaluating Task {task_id+1}/{self.num_tasks}: {task_name} {'='*20}")
        
        # Initialize metrics
        total_episodes = 0
        ik_successes = 0
        vla_successes = 0
        overall_successes = 0
        debug_images_saved = []
        
        for episode in range(self.num_episodes_per_task):
            print(f"  Episode {episode + 1}/{self.num_episodes_per_task}")
            
            # Create environment
            env, _ = get_libero_env(task, model_family="openvla", resolution=256)
            
            # Reset environment and get initial state
            obs = env.reset()
            reset_image = obs["agentview_image"][::-1, ::-1]  # Rotate 180 degrees
            
            # Save reset image
            if self.save_debug_images:
                self._save_debug_image(env, task_name, episode, "reset")
                debug_images_saved.append(f"{task_name}_ep{episode}_reset.png")
            
            # Get training sample for comparison (if enabled)
            training_result = None
            if self.compare_training_image:
                training_result = self._get_training_sample_for_task(task_name)
                if training_result is not None:
                    training_sample, sample_idx = training_result
                    print(f"    Using training sample {sample_idx} for comparison")
            
            # Predict pose with retry (reset image)
            predicted_pose = self._predict_pose_with_retry(reset_image, task_description)
            
            if predicted_pose is not None:
                ik_successes += 1
                print(f"    ✓ IK succeeded (reset image)")
                
                # Convert pose to joint positions and set robot state
                pose_mat = pose6d_to_matrix(predicted_pose)
                joint_pos = solve_ik(pose_mat)
                
                # Set robot joint positions
                env.env.robots[0].set_robot_joint_positions(joint_pos)
                env.env.sim.forward()
                if hasattr(env.env, '_update_observables'):
                    env.env._update_observables(force=True)
                
                # Stabilize objects with dummy actions (wait for objects to settle)
                print(f"    Stabilizing objects after IK positioning...")
                for _ in range(5):
                    # Execute dummy action (zero action)
                    dummy_action = np.zeros(7)
                    obs, reward, done, info = env.step(dummy_action)
                    
                    # Small delay to let objects settle
                    time.sleep(0.1)
                
                print(f"    Objects stabilized after IK positioning")
                
                # Save IK result image (reset image)
                if self.save_debug_images:
                    self._save_debug_image(env, task_name, episode, "ik_result_reset")
                    debug_images_saved.append(f"{task_name}_ep{episode}_ik_result_reset.png")
                
                # Test training sample IK if available
                if training_result is not None:
                    print(f"    Testing training sample IK...")
                    
                    # Predict pose from training sample
                    training_predicted_pose = self._predict_pose_from_training_sample(training_sample)
                    
                    if training_predicted_pose is not None:
                        print(f"    ✓ IK succeeded (training sample)")
                        
                        # Convert training pose to joint positions
                        training_pose_mat = pose6d_to_matrix(training_predicted_pose)
                        training_joint_pos = solve_ik(training_pose_mat)
                        
                        # Set robot joint positions for training pose
                        env.env.robots[0].set_robot_joint_positions(training_joint_pos)
                        env.env.sim.forward()
                        if hasattr(env.env, '_update_observables'):
                            env.env._update_observables(force=True)
                        
                        # Stabilize objects after training IK positioning
                        print(f"    Stabilizing objects after training IK positioning...")
                        for _ in range(5):
                            # Execute dummy action (zero action)
                            dummy_action = np.zeros(7)
                            obs, reward, done, info = env.step(dummy_action)
                            
                            # Small delay to let objects settle
                            time.sleep(0.1)
                        
                        print(f"    Objects stabilized after training IK positioning")
                        
                        # Save IK result image (training sample)
                        if self.save_debug_images:
                            self._save_debug_image(env, task_name, episode, "ik_result_training")
                            debug_images_saved.append(f"{task_name}_ep{episode}_ik_result_training.png")
                        
                        # Save initial state image (which IK result is used for OpenVLA)
                        if self.save_debug_images:
                            self._save_debug_image(env, task_name, episode, "initial_state_for_openvla")
                            debug_images_saved.append(f"{task_name}_ep{episode}_initial_state_for_openvla.png")
                    else:
                        print(f"    ✗ IK failed (training sample)")
                
                # Choose initial state for OpenVLA evaluation
                if training_result is not None and training_predicted_pose is not None:
                    # Use training image IK as initial state (potentially better)
                    print(f"    Using training image IK as initial state for OpenVLA-OFT")
                    initial_joint_pos = training_joint_pos
                else:
                    # Fallback to reset image IK
                    print(f"    Using reset image IK as initial state for OpenVLA-OFT")
                    initial_joint_pos = joint_pos
                
                # Set robot joint positions for OpenVLA evaluation
                env.env.robots[0].set_robot_joint_positions(initial_joint_pos)
                env.env.sim.forward()
                if hasattr(env.env, '_update_observables'):
                    env.env._update_observables(force=True)
                
                # Stabilize objects with dummy actions (wait for objects to settle)
                print(f"    Stabilizing objects with dummy actions...")
                for _ in range(5):
                    # Execute dummy action (zero action)
                    dummy_action = np.zeros(7)
                    obs, reward, done, info = env.step(dummy_action)
                    
                    # Small delay to let objects settle
                    time.sleep(0.1)
                
                print(f"    Objects stabilized, proceeding with OpenVLA-OFT")
                
                # Run OpenVLA-OFT episode with video recording
                video_filepath = self.debug_image_dir / f"{task_name.replace(' ', '_').replace('/', '_')}_ep{episode}_openvla_execution.mp4"
                vla_success = self._run_openvla_episode(env, task_description, record_video=True, video_filepath=str(video_filepath))
                
                if vla_success:
                    vla_successes += 1
                    overall_successes += 1
                    print(f"    ✓ OpenVLA-OFT succeeded")
                    
                    # Save final result image
                    if self.save_debug_images:
                        self._save_debug_image(env, task_name, episode, "final_success")
                        debug_images_saved.append(f"{task_name}_ep{episode}_final_success.png")
                else:
                    print(f"    ✗ OpenVLA-OFT failed")
                    
                    # Save failure image
                    if self.save_debug_images:
                        self._save_debug_image(env, task_name, episode, "final_failure")
                        debug_images_saved.append(f"{task_name}_ep{episode}_final_failure.png")
            else:
                print(f"    ✗ IK failed after all attempts")
            
            total_episodes += 1
            env.close()
        
        # Calculate metrics
        ik_success_rate = ik_successes / total_episodes if total_episodes > 0 else 0
        vla_success_rate = vla_successes / ik_successes if ik_successes > 0 else 0
        overall_success_rate = overall_successes / total_episodes if total_episodes > 0 else 0
        ik_failure_rate = 1 - ik_success_rate
        vla_failure_rate = 1 - vla_success_rate if ik_successes > 0 else 0
        
        results = {
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'total_episodes': total_episodes,
            'ik_successes': ik_successes,
            'vla_successes': vla_successes,
            'overall_successes': overall_successes,
            'ik_success_rate': ik_success_rate,
            'vla_success_rate': vla_success_rate,
            'overall_success_rate': overall_success_rate,
            'ik_failure_rate': ik_failure_rate,
            'vla_failure_rate': vla_failure_rate,
            'debug_images_saved': debug_images_saved,
        }
        
        print(f"  Results:")
        print(f"    IK Success Rate: {ik_success_rate:.3f} ({ik_successes}/{total_episodes})")
        print(f"    VLA Success Rate: {vla_success_rate:.3f} ({vla_successes}/{ik_successes})")
        print(f"    Overall Success Rate: {overall_success_rate:.3f} ({overall_successes}/{total_episodes})")
        
        return results
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation on all tasks."""
        print("Starting combined evaluation...")
        print(f"PoseVLM checkpoint: {self.pose_vlm_checkpoint}")
        print(f"OpenVLA checkpoint: {self.openvla_checkpoint}")
        print(f"Task suite: {self.task_suite_name}")
        print(f"Episodes per task: {self.num_episodes_per_task}")
        print(f"Max IK attempts: {self.max_ik_attempts}")
        
        all_results = []
        
        for task_id in tqdm(range(self.num_tasks), desc="Evaluating tasks"):
            task_results = self.evaluate_task(task_id)
            all_results.append(task_results)
        
        # Aggregate results
        total_episodes = sum(r['total_episodes'] for r in all_results)
        total_ik_successes = sum(r['ik_successes'] for r in all_results)
        total_vla_successes = sum(r['vla_successes'] for r in all_results)
        total_overall_successes = sum(r['overall_successes'] for r in all_results)
        
        overall_metrics = {
            'total_episodes': total_episodes,
            'total_ik_successes': total_ik_successes,
            'total_vla_successes': total_vla_successes,
            'total_overall_successes': total_overall_successes,
            'overall_ik_success_rate': total_ik_successes / total_episodes if total_episodes > 0 else 0,
            'overall_vla_success_rate': total_vla_successes / total_ik_successes if total_ik_successes > 0 else 0,
            'overall_success_rate': total_overall_successes / total_episodes if total_episodes > 0 else 0,
            'overall_ik_failure_rate': 1 - (total_ik_successes / total_episodes) if total_episodes > 0 else 1,
            'overall_vla_failure_rate': 1 - (total_vla_successes / total_ik_successes) if total_ik_successes > 0 else 1,
        }
        
        # Print final results
        print("\n" + "="*80)
        print("COMBINED EVALUATION RESULTS")
        print("="*80)
        print(f"Total Episodes: {total_episodes}")
        print(f"IK Success Rate: {overall_metrics['overall_ik_success_rate']:.3f} ({total_ik_successes}/{total_episodes})")
        print(f"VLA Success Rate: {overall_metrics['overall_vla_success_rate']:.3f} ({total_vla_successes}/{total_ik_successes})")
        print(f"Overall Success Rate: {overall_metrics['overall_success_rate']:.3f} ({total_overall_successes}/{total_episodes})")
        print(f"IK Failure Rate: {overall_metrics['overall_ik_failure_rate']:.3f}")
        print(f"VLA Failure Rate: {overall_metrics['overall_vla_failure_rate']:.3f}")
        
        # Per-task results
        print(f"\nPer-task results:")
        for result in all_results:
            print(f"  {result['task_name']}: IK={result['ik_success_rate']:.3f}, VLA={result['vla_success_rate']:.3f}, Overall={result['overall_success_rate']:.3f}")
        
        final_results = {
            'overall_metrics': overall_metrics,
            'per_task_results': all_results,
            'evaluation_config': {
                'pose_vlm_checkpoint': self.pose_vlm_checkpoint,
                'openvla_checkpoint': self.openvla_checkpoint,
                'task_suite_name': self.task_suite_name,
                'num_episodes_per_task': self.num_episodes_per_task,
                'max_ik_attempts': self.max_ik_attempts,
                'seed': self.seed,
                'debug_single_skill': self.debug_single_skill,
                'debug_skill_name': self.debug_skill_name,
                'compare_training_image': self.compare_training_image,
                'training_data_root': self.training_data_root,
            }
        }
        
        return final_results


def main():
    parser = argparse.ArgumentParser(description="Combined PoseVLM + OpenVLA-OFT evaluation")
    parser.add_argument(
        "--pose_vlm_checkpoint",
        type=str,
        required=True,
        help="Path to PoseVLM checkpoint"
    )
    parser.add_argument(
        "--openvla_checkpoint",
        type=str,
        required=True,
        help="Path to OpenVLA-OFT checkpoint"
    )
    parser.add_argument(
        "--task_suite_name",
        type=str,
        default="boss_44",
        help="LIBERO task suite name"
    )
    parser.add_argument(
        "--num_episodes_per_task",
        type=int,
        default=20,
        help="Number of episodes per task"
    )
    parser.add_argument(
        "--max_ik_attempts",
        type=int,
        default=5,
        help="Maximum IK attempts per pose prediction"
    )
    parser.add_argument(
        "--save_debug_images",
        action="store_true",
        default=True,
        help="Whether to save debug images"
    )
    parser.add_argument(
        "--debug_image_dir",
        type=str,
        default="runs/eval_results/combined_eval/debug_images",
        help="Directory to save debug images"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--use_pose_vlm_smaller_split_tasks_only",
        action="store_true",
        help="Only evaluate on the 6 tasks used for PoseVLM smaller split training"
    )
    parser.add_argument(
        "--wrist_only",
        action="store_true",
        help="Use only wrist camera view for OpenVLA-OFT"
    )
    parser.add_argument(
        "--third_person_view_only",
        action="store_true",
        help="Use only third-person camera view for OpenVLA-OFT"
    )
    parser.add_argument(
        "--both_views",
        action="store_true",
        help="Use both wrist and third-person camera views for OpenVLA-OFT"
    )
    parser.add_argument(
        "--debug_single_skill",
        action="store_true",
        help="Debug mode: evaluate only a single skill"
    )
    parser.add_argument(
        "--debug_skill_name",
        type=str,
        default="KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
        help="Name of the skill to debug (when debug_single_skill is True)"
    )
    parser.add_argument(
        "--compare_training_image",
        action="store_true",
        help="Compare IK results from reset image vs training image"
    )
    parser.add_argument(
        "--training_data_root",
        type=str,
        default="../../datasets/local_pairs_datasets",
        help="Root directory of training dataset for comparison"
    )
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = CombinedEvaluator(
        pose_vlm_checkpoint=args.pose_vlm_checkpoint,
        openvla_checkpoint=args.openvla_checkpoint,
        task_suite_name=args.task_suite_name,
        num_episodes_per_task=args.num_episodes_per_task,
        max_ik_attempts=args.max_ik_attempts,
        save_debug_images=args.save_debug_images,
        debug_image_dir=args.debug_image_dir,
        device=args.device,
        seed=args.seed,
        use_pose_vlm_smaller_split_tasks_only=args.use_pose_vlm_smaller_split_tasks_only,
        wrist_only=args.wrist_only,
        third_person_view_only=args.third_person_view_only,
        both_views=args.both_views,
        debug_single_skill=args.debug_single_skill,
        debug_skill_name=args.debug_skill_name,
        compare_training_image=args.compare_training_image,
        training_data_root=args.training_data_root,
    )
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Save results
    results_dir = Path("runs/eval_results/combined_eval")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"combined_eval_results_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main() 