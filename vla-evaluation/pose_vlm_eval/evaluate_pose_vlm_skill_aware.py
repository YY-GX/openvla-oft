#!/usr/bin/env python3
"""
evaluate_pose_vlm_skill_aware.py

Skill-aware evaluation script for PoseVLM that correctly handles the multi-pose nature
of the dataset. For each overview image, it considers all valid poses from the same
skill (language_description) as correct targets.

Key differences from training validation:
1. Uses skill-aware minimum distance instead of direct L1 loss
2. Compares against all valid poses in the same skill (across all demos)
3. Supports sampling for faster evaluation

Usage:
    python vla-scripts/evaluate_pose_vlm_skill_aware.py \
        --checkpoint_path runs/pose_vlm/1.0.0/openvla-7b+pose_dataset+b8+lr-0.0005+pose_gmm_c3+lora-r32+dropout-0.0--image_aug--pose_aug--gmm_pose_head--3_components--pose_aug/vla--90000_checkpoint.pt \
        --data_root datasets/local_pairs_datasets \
        --batch_size 8 \
        --num_workers 4 \
        --sample_size 1000 \
        --num_runs 3
"""

import os
import argparse
import time
import json
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Import necessary modules
from prismatic.vla.datasets.pose_dataset import create_pose_dataset
from prismatic.util.data_utils import PaddedCollatorForPosePrediction
from prismatic.models.vlms.pose_vlm import PoseVLM
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from experiments.robot.openvla_utils import update_auto_map, check_model_logic_mismatch

# Import the config class from the training script
import sys
sys.path.append('vla-scripts')
from finetune_pose import PoseFinetuneConfig


class SkillAwareEvaluator:
    """
    Skill-aware evaluator that considers all valid poses within each skill (language_description).
    For each overview image, it considers all valid poses from the same skill as correct targets.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        data_root: str,
        device: str = "cuda",
        batch_size: int = 8,
        num_workers: int = 4,
        max_length: int = 512,
        sample_size: int = None,  # Number of samples to evaluate (None = full dataset)
        num_runs: int = 1,  # Number of runs for averaging results
        splits_folder: str = "splits",
        sample_indices_file: Optional[str] = None,
        split: str = "train",
    ):
        """
        Initialize the evaluator.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint
            data_root: Root directory containing the dataset
            device: Device to run evaluation on
            batch_size: Batch size for evaluation
            num_workers: Number of workers for data loading
            max_length: Maximum sequence length for text
            sample_size: Number of samples to evaluate (None = full dataset)
            num_runs: Number of runs for averaging results
        """
        self.checkpoint_path = checkpoint_path
        self.data_root = data_root
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.sample_size = sample_size
        self.num_runs = num_runs
        self.splits_folder = splits_folder
        self.sample_indices_file = sample_indices_file
        self.split = split
        
        # Load model and data
        self._load_model()
        self._load_data()
        
    def _load_model(self):
        """Load the trained model from checkpoint."""
        print("Loading model...")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
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
        self.processor = PrismaticProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

        # Load base VLA model (matches training script)
        from transformers import AutoModelForVision2Seq
        base_vla = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # Wrap LLM backbone
        class OpenVLAWrapper(nn.Module):
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

        llm_backbone = OpenVLAWrapper(base_vla.language_model, self.processor)

        # Construct PoseVLM (matches training)
        self.model = PoseVLM(
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
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
        
    def _load_data(self):
        """Load validation and test datasets."""
        print("Loading datasets...")
        # Create datasets - EXACTLY like training validation
        self.train_dataset = create_pose_dataset(
            data_root=self.data_root,
            split="train",
            num_images_in_input=self.config.num_images_in_input,
            use_image_augmentation=False,  # No augmentation for default evaluation
            tokenizer_name=self.processor.tokenizer.name_or_path,
            use_pose_normalization=True,  # Enable pose normalization for evaluation
        )
        self.train_dataset_aug = create_pose_dataset(
            data_root=self.data_root,
            split="train",
            num_images_in_input=self.config.num_images_in_input,
            use_image_augmentation=True,  # Augmentation enabled for augmented evaluation
            tokenizer_name=self.processor.tokenizer.name_or_path,
            use_pose_normalization=True,  # Enable pose normalization for evaluation
        )
        self.val_dataset = create_pose_dataset(
            data_root=self.data_root,
            split="val",
            num_images_in_input=self.config.num_images_in_input,
            use_image_augmentation=False,  # No augmentation for evaluation (same as training)
            tokenizer_name=self.processor.tokenizer.name_or_path,
            use_pose_normalization=True,  # Enable pose normalization for evaluation
        )
        self.test_dataset = create_pose_dataset(
            data_root=self.data_root,
            split="test",
            num_images_in_input=self.config.num_images_in_input,
            use_image_augmentation=False,  # No augmentation for evaluation (same as training)
            tokenizer_name=self.processor.tokenizer.name_or_path,
            use_pose_normalization=True,  # Enable pose normalization for evaluation
        )
        # Create collator - EXACTLY like training
        self.collator = PaddedCollatorForPosePrediction(
            model_max_length=self.max_length,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            padding_side=self.processor.tokenizer.padding_side,
        )
        print(f"Training dataset: {len(self.train_dataset)} samples (no aug)")
        print(f"Training dataset: {len(self.train_dataset_aug)} samples (with aug)")
        print(f"Validation dataset: {len(self.val_dataset)} samples")
        print(f"Test dataset: {len(self.test_dataset)} samples")
        
    def _load_annotation_data(self, split: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load annotation data and poses for a given split."""
        # Load annotation - try data_root first, then data_root/splits
        annotation_path = os.path.join(self.data_root, f"{split}_annotation.csv")
        if not os.path.exists(annotation_path):
            # Fall back to the original splits subdirectory
            annotation_path = os.path.join(self.data_root, self.splits_folder, f"{split}_annotation.csv")
            if not os.path.exists(annotation_path):
                raise FileNotFoundError(f"Annotation file not found in {self.data_root} or {self.data_root}/{self.splits_folder}")
        annotation_df = pd.read_csv(annotation_path)
        
        # Load poses
        poses_path = os.path.join(self.data_root, "local_poses.npy")
        poses = np.load(poses_path)
        
        return annotation_df, poses
        
    def _group_by_skill(self, annotation_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Group annotation data by skill (language_description)."""
        skill_groups = defaultdict(list)
        
        for _, row in annotation_df.iterrows():
            skill_key = row['language_description']
            skill_groups[skill_key].append(row.to_dict())
            
        return skill_groups
        
    def _get_valid_poses_for_skill(self, skill_pairs: List[Dict], poses: np.ndarray) -> np.ndarray:
        """Get all valid poses for a skill."""
        unique_pose_indices = list(set(pair['ee_pose_idx'] for pair in skill_pairs))
        valid_poses = poses[unique_pose_indices]
        return valid_poses
        
    def _compute_skill_aware_metrics(
        self, 
        predicted_pose: torch.Tensor, 
        valid_poses: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute skill-aware metrics by considering all valid poses for the skill.
        
        Args:
            predicted_pose: Model prediction (6D pose)
            valid_poses: All valid poses from the same skill (N, 6)
            
        Returns:
            Dictionary with metrics
        """
        # Convert to numpy for easier computation
        if isinstance(predicted_pose, torch.Tensor):
            # Convert bfloat16 to float32 first, then to numpy
            if predicted_pose.dtype == torch.bfloat16:
                predicted_pose = predicted_pose.to(torch.float32)
            predicted_pose = predicted_pose.detach().cpu().numpy()
        
        # Compute distances to all valid poses (use L2 norm like demo-aware evaluation)
        distances = np.linalg.norm(predicted_pose - valid_poses, axis=1)
        
        # Position error (first 3 dimensions)
        pos_distances = np.linalg.norm(predicted_pose[:3] - valid_poses[:, :3], axis=1)
        
        # Orientation error (last 3 dimensions)
        ori_distances = np.linalg.norm(predicted_pose[3:] - valid_poses[:, 3:], axis=1)
        
        # Get minimum distances
        min_distance = np.min(distances)
        min_pos_distance = np.min(pos_distances)
        min_ori_distance = np.min(ori_distances)
        
        return {
            'min_pose_error': min_distance,
            'min_position_error': min_pos_distance,
            'min_orientation_error': min_ori_distance,
        }
    
    def _compute_skill_aware_metrics_single(
        self, 
        predicted_pose: torch.Tensor, 
        target_pose: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute skill-aware metrics for single pose comparison.
        
        Args:
            predicted_pose: Model prediction (6D pose)
            target_pose: Target pose (6D pose) - same coordinate frame as training
        
        Returns:
            Dictionary with metrics
        """
        # Convert to numpy for easier computation
        if isinstance(predicted_pose, torch.Tensor):
            if predicted_pose.dtype == torch.bfloat16:
                predicted_pose = predicted_pose.to(torch.float32)
            predicted_pose = predicted_pose.detach().cpu().numpy()
        
        if isinstance(target_pose, torch.Tensor):
            if target_pose.dtype == torch.bfloat16:
                target_pose = target_pose.to(torch.float32)
            target_pose = target_pose.detach().cpu().numpy()
        
        # Compute L1 distances (same as training validation)
        distance = np.mean(np.abs(predicted_pose - target_pose))
        
        # Position error (first 3 dimensions)
        pos_distance = np.mean(np.abs(predicted_pose[:3] - target_pose[:3]))
        
        # Orientation error (last 3 dimensions)
        ori_distance = np.mean(np.abs(predicted_pose[3:] - target_pose[3:]))
        
        return {
            'min_pose_error': distance,
            'min_position_error': pos_distance,
            'min_orientation_error': ori_distance,
        }
        
    def _run_forward_pass(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Run a forward pass through the model - EXACTLY like training validation.
        
        Args:
            batch: Input batch
            
        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: Loss and metrics
        """
        # Extract batch components and ensure bfloat16 dtype - EXACTLY like training
        images = batch["pixel_values"].to(self.device)
        text = batch["input_ids"].to(self.device)
        text_attention_mask = batch["attention_mask"].to(self.device)
        target_poses = batch["pose_targets"].to(self.device)
        
        # Convert to bfloat16 like training
        images = images.to(torch.bfloat16)
        target_poses = target_poses.to(torch.bfloat16)
        
        # Compute loss using PoseVLM's built-in method - EXACTLY like training
        loss = self.model.compute_loss(
            images=images,
            text=text,
            text_attention_mask=text_attention_mask,
            target_poses=target_poses,
        )
        
        # Compute additional metrics - EXACTLY like training
        with torch.no_grad():
            # Get predictions for metrics - EXACTLY like training
            predictions = self.model.predict_pose(
                images=images,
                text=text,
                text_attention_mask=text_attention_mask,
                num_samples=1,
            )
            
            if self.config.pose_head_type == "gmm":
                # EXACTLY like training validation
                sampled_poses = predictions['sampled_poses'].squeeze(1)  # (batch_size, num_pose_tokens, num_components, pose_dim)
                weights = predictions['weights']  # (batch_size, num_pose_tokens, num_components)
                
                # Use the first pose token for metrics (same as training)
                first_token_weights = weights[:, 0, :]  # (batch_size, num_components)
                first_token_poses = sampled_poses[:, 0, :, :]  # (batch_size, num_components, pose_dim)
                
                # Select the component with highest weight for metrics (same as training)
                best_component_indices = torch.argmax(first_token_weights, dim=-1)  # (batch_size,)
                
                # Select the best component for each batch item
                batch_size = first_token_poses.shape[0]
                selected_poses = []
                for b in range(batch_size):
                    selected_poses.append(first_token_poses[b, best_component_indices[b]])  # (pose_dim,)
                selected_poses = torch.stack(selected_poses)  # (batch_size, pose_dim)
                
                # Ensure target_poses has the right shape for broadcasting
                if target_poses.dim() == 1:
                    target_poses = target_poses.unsqueeze(0)  # (1, pose_dim)
                
                # Ensure both tensors have the same batch size
                if selected_poses.shape[0] != target_poses.shape[0]:
                    if selected_poses.shape[0] == 1:
                        selected_poses = selected_poses.expand(target_poses.shape[0], -1)
                    elif target_poses.shape[0] == 1:
                        target_poses = target_poses.expand(selected_poses.shape[0], -1)
                    else:
                        raise ValueError(f"Cannot broadcast shapes: {selected_poses.shape} vs {target_poses.shape}")
                
                # Training validation uses direct L1 loss
                l1_error = torch.mean(torch.abs(selected_poses - target_poses))
                
                # Compute component weights entropy (diversity measure)
                weights = predictions['weights']
                entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1).mean()
                
                metrics = {
                    "loss": loss.item(),
                    "l1_error": l1_error.item(),
                    "entropy": entropy.item(),
                }
            else:
                predicted_poses = predictions['predicted_poses']
                l1_error = torch.mean(torch.abs(predicted_poses - target_poses))
                metrics = {
                    "loss": loss.item(),
                    "l1_error": l1_error.item(),
                }
        
        return loss, metrics
        
    def evaluate_split(self, split: str, use_aug: bool = False) -> Dict[str, Any]:
        """
        Evaluate model on a specific split with skill-aware metrics.
        Args:
            split: Dataset split ('train', 'val', or 'test')
            use_aug: If True and split=='train', use augmented training set
        Returns:
            Dictionary with evaluation results
        """
        print(f"\nEvaluating {split} split{' (with augmentation)' if use_aug else ''}...")
        # Load annotation data
        annotation_df, poses = self._load_annotation_data(split)
        skill_groups = self._group_by_skill(annotation_df)
        # Get dataset
        if split == "train":
            dataset = self.train_dataset_aug if use_aug else self.train_dataset
        elif split == "val":
            dataset = self.val_dataset
        else:  # test
            dataset = self.test_dataset
        # Create subset if sampling is enabled
        if self.sample_size is not None and self.sample_size < len(dataset):
            all_results = []
            model_info = {
                'checkpoint_path': self.checkpoint_path,
                'pose_head_type': getattr(self.config, 'pose_head_type', 'unknown'),
                'gmm_num_components': getattr(self.config, 'gmm_num_components', 'unknown'),
                'pose_dim': getattr(self.config, 'pose_dim', 'unknown'),
            }
            eval_config = {
                'sample_size': self.sample_size,
                'num_runs': self.num_runs,
                'batch_size': self.batch_size,
            }
            for run in range(self.num_runs):
                print(f"  Run {run + 1}/{self.num_runs}...")
                random.seed(run)  # Different seed for each run
                indices = random.sample(range(len(dataset)), self.sample_size)
                subset = Subset(dataset, indices)
                dataloader = DataLoader(
                    subset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    collate_fn=self.collator,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
                run_results = self._evaluate_dataloader(dataloader, skill_groups, poses, split, f"run_{run}")
                all_results.append(run_results)
                # Print milestone results after each run
                print(f"\nMilestone results after run {run + 1}/{self.num_runs}:")
                self.print_results({split: run_results, 'model_info': model_info, 'evaluation_config': eval_config})
            results = self._aggregate_runs(all_results, split)
        else:
            # If sample_indices_file is provided, use those indices
            if self.sample_indices_file is not None:
                import json
                with open(self.sample_indices_file, 'r') as f:
                    all_indices = json.load(f)
                indices = all_indices[self.split]
                subset = Subset(dataset, indices)
                dataloader = DataLoader(
                    subset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    collate_fn=self.collator,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
                results = self._evaluate_dataloader(dataloader, skill_groups, poses, split, "sampled_indices")
                return results
            else:
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    collate_fn=self.collator,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
                results = self._evaluate_dataloader(dataloader, skill_groups, poses, split, "full")
        return results
        
    def _evaluate_dataloader(self, dataloader: DataLoader, skill_groups: Dict, poses: np.ndarray, split: str, run_name: str) -> Dict[str, Any]:
        """Evaluate a specific dataloader."""
        # Initialize metrics
        all_metrics = []
        skill_metrics = defaultdict(list)
        training_metrics = {
            "loss": [],
            "l1_error": [],
            "entropy": [] if self.config.pose_head_type == "gmm" else None,
        }
        
        # Evaluation loop - EXACTLY like training validation
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {split} ({run_name})")):
                # Run forward pass - EXACTLY like training validation
                loss, metrics = self._run_forward_pass(batch)
                
                # Store training metrics (same as training validation)
                training_metrics["loss"].append(metrics["loss"])
                training_metrics["l1_error"].append(metrics["l1_error"])
                if "entropy" in metrics:
                    training_metrics["entropy"].append(metrics["entropy"])
                
                # For skill-aware evaluation, we need to compute minimum distance to all valid poses in the same skill
                # Get predictions for skill-aware metrics
                images = batch["pixel_values"].to(self.device)
                text = batch["input_ids"].to(self.device)
                text_attention_mask = batch["attention_mask"].to(self.device)
                target_poses = batch["pose_targets"].to(self.device)  # Use the same processed poses as training
                
                if self.config.pose_head_type == "gmm":
                    predictions = self.model.predict_pose(
                        images=images,
                        text=text,
                        text_attention_mask=text_attention_mask,
                        num_samples=1,
                    )
                    # Use the same pose selection as training validation
                    sampled_poses = predictions['sampled_poses'].squeeze(1)
                    weights = predictions['weights']
                    
                    first_token_weights = weights[:, 0, :]
                    first_token_poses = sampled_poses[:, 0, :, :]
                    best_component_indices = torch.argmax(first_token_weights, dim=-1)
                    
                    batch_size = first_token_poses.shape[0]
                    predicted_poses = []
                    for b in range(batch_size):
                        predicted_poses.append(first_token_poses[b, best_component_indices[b]])
                    predicted_poses = torch.stack(predicted_poses)
                else:
                    predictions = self.model.predict_pose(
                        images=images,
                        text=text,
                        text_attention_mask=text_attention_mask,
                    )
                    predicted_poses = predictions['predicted_poses']
                
                # Process each sample in the batch for skill-aware metrics
                overview_indices = batch["overview_image_indices"]
                ee_pose_indices = batch["ee_pose_indices"]
                
                for i in range(len(predicted_poses)):
                    overview_idx = overview_indices[i].item()
                    ee_pose_idx = ee_pose_indices[i].item()
                    
                    # Find the skill this sample belongs to
                    skill_key = None
                    for key, pairs in skill_groups.items():
                        for pair in pairs:
                            if (pair['overview_image_idx'] == overview_idx and 
                                pair['ee_pose_idx'] == ee_pose_idx):
                                skill_key = key
                                break
                        if skill_key:
                            break
                    
                    if skill_key is None:
                        continue
                    
                    # For skill-aware evaluation, compare against all valid poses in the same skill
                    skill_pairs = skill_groups[skill_key]
                    valid_poses = self._get_valid_poses_for_skill(skill_pairs, poses)
                    
                    # Compute skill-aware metrics (minimum distance to any valid pose in the skill)
                    predicted_pose = predicted_poses[i]
                    metrics = self._compute_skill_aware_metrics(predicted_pose, valid_poses)
                    
                    # Store metrics
                    all_metrics.append(metrics)
                    skill_metrics[skill_key].append(metrics)
        
        # Aggregate results
        results = self._aggregate_metrics(all_metrics, skill_metrics, training_metrics, split, run_name)
        
        return results
        
    def _aggregate_metrics(
        self, 
        all_metrics: List[Dict[str, float]], 
        skill_metrics: Dict[str, List[Dict[str, float]]],
        training_metrics: Dict[str, List[float]],
        split: str,
        run_name: str
    ) -> Dict[str, Any]:
        """Aggregate metrics across all samples and skills."""
        
        # Skill-aware metrics
        overall_metrics = {
            'mean_min_pose_error': np.mean([m['min_pose_error'] for m in all_metrics]),
            'std_min_pose_error': np.std([m['min_pose_error'] for m in all_metrics]),
            'mean_min_position_error': np.mean([m['min_position_error'] for m in all_metrics]),
            'std_min_position_error': np.std([m['min_position_error'] for m in all_metrics]),
            'mean_min_orientation_error': np.mean([m['min_orientation_error'] for m in all_metrics]),
            'std_min_orientation_error': np.std([m['min_orientation_error'] for m in all_metrics]),
        }
        
        # Training validation metrics (same as training)
        training_overall = {
            'mean_loss': np.mean(training_metrics["loss"]),
            'std_loss': np.std(training_metrics["loss"]),
            'mean_l1_error': np.mean(training_metrics["l1_error"]),
            'std_l1_error': np.std(training_metrics["l1_error"]),
        }
        
        if training_metrics["entropy"] is not None:
            training_overall.update({
                'mean_entropy': np.mean(training_metrics["entropy"]),
                'std_entropy': np.std(training_metrics["entropy"]),
            })
        
        # Per-skill metrics
        skill_summary = {}
        for skill_key, metrics in skill_metrics.items():
            skill_summary[skill_key] = {
                'num_samples': len(metrics),
                'mean_min_pose_error': np.mean([m['min_pose_error'] for m in metrics]),
                'mean_min_position_error': np.mean([m['min_position_error'] for m in metrics]),
                'mean_min_orientation_error': np.mean([m['min_orientation_error'] for m in metrics]),
            }
        
        # Percentiles
        pose_errors = [m['min_pose_error'] for m in all_metrics]
        pos_errors = [m['min_position_error'] for m in all_metrics]
        ori_errors = [m['min_orientation_error'] for m in all_metrics]
        
        percentiles = {
            'pose_error_50th': np.percentile(pose_errors, 50),
            'pose_error_75th': np.percentile(pose_errors, 75),
            'pose_error_90th': np.percentile(pose_errors, 90),
            'pose_error_95th': np.percentile(pose_errors, 95),
            'position_error_50th': np.percentile(pos_errors, 50),
            'position_error_75th': np.percentile(pos_errors, 75),
            'position_error_90th': np.percentile(pos_errors, 90),
            'position_error_95th': np.percentile(pos_errors, 95),
            'orientation_error_50th': np.percentile(ori_errors, 50),
            'orientation_error_75th': np.percentile(ori_errors, 75),
            'orientation_error_90th': np.percentile(ori_errors, 90),
            'orientation_error_95th': np.percentile(ori_errors, 95),
        }
        
        results = {
            'split': split,
            'run_name': run_name,
            'num_samples': len(all_metrics),
            'num_skills': len(skill_metrics),
            'overall_metrics': overall_metrics,
            'training_metrics': training_overall,
            'percentiles': percentiles,
            'skill_summary': skill_summary,
        }
        
        return results
        
    def _aggregate_runs(self, all_results: List[Dict[str, Any]], split: str) -> Dict[str, Any]:
        """Aggregate results across multiple runs."""
        # Extract metrics from all runs
        skill_metrics_list = [r['overall_metrics'] for r in all_results]
        training_metrics_list = [r['training_metrics'] for r in all_results]
        
        # Compute mean and std across runs
        skill_metrics_avg = {}
        skill_metrics_std = {}
        for key in skill_metrics_list[0].keys():
            values = [m[key] for m in skill_metrics_list]
            skill_metrics_avg[key] = np.mean(values)
            skill_metrics_std[key] = np.std(values)
        
        training_metrics_avg = {}
        training_metrics_std = {}
        for key in training_metrics_list[0].keys():
            values = [m[key] for m in training_metrics_list]
            training_metrics_avg[key] = np.mean(values)
            training_metrics_std[key] = np.std(values)
        
        # Use the first result as template and update metrics
        result = all_results[0].copy()
        result['split'] = split
        result['run_name'] = f"avg_over_{self.num_runs}_runs"
        result['overall_metrics'] = skill_metrics_avg
        result['training_metrics'] = training_metrics_avg
        result['overall_metrics_std'] = skill_metrics_std
        result['training_metrics_std'] = training_metrics_std
        result['num_runs'] = self.num_runs
        
        return result
        
    def run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation on training, validation, and test splits."""
        print("Starting skill-aware evaluation...")
        # Prepare model_info dict for all print_results calls
        model_info = {
            'checkpoint_path': self.checkpoint_path,
            'pose_head_type': getattr(self.config, 'pose_head_type', 'unknown'),
            'gmm_num_components': getattr(self.config, 'gmm_num_components', 'unknown'),
            'pose_dim': getattr(self.config, 'pose_dim', 'unknown'),
        }
        eval_config = {
            'sample_size': self.sample_size,
            'num_runs': self.num_runs,
            'batch_size': self.batch_size,
        }
        # Evaluate training set (no augmentation)
        print("\nEvaluating TRAINING SET (no augmentation)...")
        train_results = self.evaluate_split("train")
        self.print_results({
            'training': train_results,
            'model_info': model_info,
            'evaluation_config': eval_config,
        })
        # Evaluate training set (with augmentation)
        print("\nEvaluating TRAINING SET (with augmentation)...")
        train_results_aug = self.evaluate_split("train", use_aug=True)
        self.print_results({
            'training': train_results_aug,
            'model_info': model_info,
            'evaluation_config': eval_config,
        })
        # Evaluate validation set
        print("\nEvaluating VALIDATION SET...")
        val_results = self.evaluate_split("val")
        self.print_results({
            'validation': val_results,
            'model_info': model_info,
            'evaluation_config': eval_config,
        })
        # Evaluate test set
        print("\nEvaluating TEST SET...")
        test_results = self.evaluate_split("test")
        self.print_results({
            'test': test_results,
            'model_info': model_info,
            'evaluation_config': eval_config,
        })
        # Combine results
        all_results = {
            'training': train_results,
            'training_aug': train_results_aug,
            'validation': val_results,
            'test': test_results,
            'model_info': model_info,
            'evaluation_config': eval_config,
        }
        return all_results
        
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a readable format."""
        print("\n" + "="*80)
        print("SKILL-AWARE POSEVLM EVALUATION RESULTS")
        print("="*80)
        # Model info
        model_info = results['model_info']
        eval_config = results['evaluation_config']
        print(f"Model: {model_info['pose_head_type']} pose head")
        print(f"GMM Components: {model_info['gmm_num_components']}")
        print(f"Pose Dimension: {model_info['pose_dim']}")
        print(f"Checkpoint: {model_info['checkpoint_path']}")
        print(f"Sample Size: {eval_config['sample_size']}")
        print(f"Number of Runs: {eval_config['num_runs']}")
        # Results for each split
        for split_name, split_results in results.items():
            if split_name not in ['model_info', 'evaluation_config']:
                overall = split_results['overall_metrics']
                print(f"\n{split_name.upper()} SUMMARY:")
                print(f"  Pose Error:        {overall['mean_min_pose_error']:.4f} ± {overall['std_min_pose_error']:.4f}")
                print(f"  Position Error:    {overall['mean_min_position_error']:.4f} ± {overall['std_min_position_error']:.4f}")
                print(f"  Orientation Error: {overall['mean_min_orientation_error']:.4f} ± {overall['std_min_orientation_error']:.4f}")
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Skill-aware evaluation of PoseVLM")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing the dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = full dataset)"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of runs for averaging results"
    )
    parser.add_argument(
        "--splits_folder",
        type=str,
        default="splits",
        help="Folder name for annotation splits (default: splits)"
    )
    parser.add_argument(
        "--sample_indices_file",
        type=str,
        default=None,
        help="Path to a file containing sample indices to use for evaluation (one per line)"
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Which split to evaluate: train, val, or test"
    )
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = SkillAwareEvaluator(
        checkpoint_path=args.checkpoint_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_size=args.sample_size,
        num_runs=args.num_runs,
        splits_folder=args.splits_folder,
        sample_indices_file=args.sample_indices_file,
        split=args.split,
    )
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(checkpoint_dir, f"evaluation_results_skill_wise_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main() 