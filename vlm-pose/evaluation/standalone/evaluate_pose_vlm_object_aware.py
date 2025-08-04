#!/usr/bin/env python3
"""
evaluate_pose_vlm_object_aware.py

Object-aware evaluation script for PoseVLM that compares predicted EE pose-to-object distances
against the average distance between skill-wise EE poses and target objects.

Key differences from skill-aware evaluation:
1. Uses object distance metric instead of pose-to-pose distance
2. Compares predicted EE-to-object distance against skill-wise average EE-to-object distance
3. Requires target object poses to be extracted first

Usage:
    python utils/evaluate_pose_vlm_object_aware.py \
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
sys.path.append('../../..')
from finetune_pose import PoseFinetuneConfig


class ObjectAwareEvaluator:
    """
    Object-aware evaluator that compares predicted EE pose-to-object distances
    against skill-wise average EE-to-object distances.
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
        sampled_target_object_poses_dir: str = "runs/eval_results/vlm_pose_generator/",
        is_train_aug_eval: bool = False,
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
        self.sampled_target_object_poses_dir = sampled_target_object_poses_dir
        self.is_train_aug_eval = is_train_aug_eval
        
        # Load model and data
        self._load_model()
        self._load_data()
        self._load_target_object_data()
        
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
        
    def _load_target_object_data(self):
        """Load target object poses and annotations."""
        # Fast path: if sample_indices_file is provided and sampled_target_object_poses_{split}.npy exists, use it
        if self.sample_indices_file is not None:
            sampled_npy_path = os.path.join(self.sampled_target_object_poses_dir, f"sampled_target_object_poses_{self.split}.npy")
            if os.path.exists(sampled_npy_path):
                print(f"[Fast path] Using pre-generated sampled target object poses from {sampled_npy_path}")
                self.target_object_poses = np.load(sampled_npy_path)
                self.target_object_df = None  # Not needed
                self.skill_demo_to_target_pose = None  # Not needed
                self.use_sampled_target_object_poses = True
                return
        print("Loading target object data...")
        
        # Load target object poses and annotations
        poses_path = os.path.join(self.data_root, "target_object_poses.npy")
        csv_path = os.path.join(self.data_root, "target_object_poses.csv")
        
        if not os.path.exists(poses_path) or not os.path.exists(csv_path):
            raise FileNotFoundError("Target object poses not found. Run extract_target_object_poses.py first.")
        
        self.target_object_poses = np.load(poses_path)
        self.target_object_df = pd.read_csv(csv_path)
        
        # Create mapping from (skill, demo) to target object pose
        self.skill_demo_to_target_pose = {}
        for _, row in self.target_object_df.iterrows():
            skill_name = row['language_description']
            demo_idx = row['source_demo_idx']
            pose_idx = row['pose_index']
            key = (skill_name, demo_idx)
            self.skill_demo_to_target_pose[key] = self.target_object_poses[pose_idx]
        
        self.use_sampled_target_object_poses = False
        print(f"Loaded {len(self.target_object_poses)} target object poses")
        print(f"Created mapping for {len(self.skill_demo_to_target_pose)} skill-demo pairs")
        
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
        
    def _get_target_object_pose_for_skill_demo(self, skill_name: str, demo_idx: str) -> Optional[np.ndarray]:
        """Get target object pose for a specific skill-demo pair."""
        key = (skill_name, demo_idx)
        return self.skill_demo_to_target_pose.get(key, None)

    def _get_target_object_pose_for_sampled_idx(self, idx: int) -> np.ndarray:
        # Only valid if use_sampled_target_object_poses is True
        return self.target_object_poses[idx]
        
    def _compute_object_distance_metrics(
        self, 
        predicted_pose: torch.Tensor, 
        target_object_pose: np.ndarray,
        skill_ee_poses: np.ndarray,
        skill_target_object_poses: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute object distance metrics.
        
        Args:
            predicted_pose: Model prediction (6D pose)
            target_object_pose: Target object pose for this sample (6D pose)
            skill_ee_poses: All EE poses in the same skill (N, 6)
            skill_target_object_poses: All target object poses in the same skill (M, 6)
            
        Returns:
            Dictionary with metrics
        """
        # Convert to numpy for easier computation
        if isinstance(predicted_pose, torch.Tensor):
            if predicted_pose.dtype == torch.bfloat16:
                predicted_pose = predicted_pose.to(torch.float32)
            predicted_pose = predicted_pose.detach().cpu().numpy()
        
        # 1. Distance between predicted EE pose and target object pose
        pred_to_obj_distance = np.linalg.norm(predicted_pose - target_object_pose)
        pred_to_obj_pos_distance = np.linalg.norm(predicted_pose[:3] - target_object_pose[:3])
        pred_to_obj_ori_distance = np.linalg.norm(predicted_pose[3:] - target_object_pose[3:])
        
        # 2. Average distance between skill-wise EE poses and target object poses
        skill_distances = []
        skill_pos_distances = []
        skill_ori_distances = []
        
        for ee_pose in skill_ee_poses:
            for obj_pose in skill_target_object_poses:
                distance = np.linalg.norm(ee_pose - obj_pose)
                pos_distance = np.linalg.norm(ee_pose[:3] - obj_pose[:3])
                ori_distance = np.linalg.norm(ee_pose[3:] - obj_pose[3:])
                
                skill_distances.append(distance)
                skill_pos_distances.append(pos_distance)
                skill_ori_distances.append(ori_distance)
        
        avg_skill_distance = np.mean(skill_distances) if skill_distances else 0.0
        avg_skill_pos_distance = np.mean(skill_pos_distances) if skill_pos_distances else 0.0
        avg_skill_ori_distance = np.mean(skill_ori_distances) if skill_ori_distances else 0.0
        
        # 3. Comparison metrics
        distance_ratio = pred_to_obj_distance / avg_skill_distance if avg_skill_distance > 0 else 1.0
        pos_distance_ratio = pred_to_obj_pos_distance / avg_skill_pos_distance if avg_skill_pos_distance > 0 else 1.0
        ori_distance_ratio = pred_to_obj_ori_distance / avg_skill_ori_distance if avg_skill_ori_distance > 0 else 1.0
        
        return {
            'pred_to_obj_distance': pred_to_obj_distance,
            'pred_to_obj_pos_distance': pred_to_obj_pos_distance,
            'pred_to_obj_ori_distance': pred_to_obj_ori_distance,
            'avg_skill_distance': avg_skill_distance,
            'avg_skill_pos_distance': avg_skill_pos_distance,
            'avg_skill_ori_distance': avg_skill_ori_distance,
            'distance_ratio': distance_ratio,
            'pos_distance_ratio': pos_distance_ratio,
            'ori_distance_ratio': ori_distance_ratio,
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
        Evaluate model on a specific split with object-aware metrics.
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
                
                # For object-aware evaluation, we need to compute object distance metrics
                # Get predictions for object-aware metrics
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
                
                # Process each sample in the batch for object-aware metrics
                overview_indices = batch["overview_image_indices"]
                ee_pose_indices = batch["ee_pose_indices"]
                
                for i in range(len(predicted_poses)):
                    overview_idx = overview_indices[i].item()
                    ee_pose_idx = ee_pose_indices[i].item()
                    
                    if self.use_sampled_target_object_poses:
                        # Use fast path: index in batch is index in sampled_target_object_poses
                        target_object_pose = self._get_target_object_pose_for_sampled_idx(batch_idx * self.batch_size + i)
                        skill_key = None
                        sample_info = None
                        # Optionally, you could try to get skill_key/sample_info if needed, but for fast eval, just skip
                    else:
                        lookup_key = (overview_idx, ee_pose_idx)
                        if lookup_key in sample_lookup:
                            skill_key, sample_info = sample_lookup[lookup_key]
                        else:
                            continue
                        skill_name = sample_info['language_description']
                        demo_idx = sample_info['source_demo_idx']
                        target_object_pose = self._get_target_object_pose_for_skill_demo(skill_name, demo_idx)
                        if target_object_pose is None:
                            continue
                    
                    # Get all EE poses and target object poses for this skill
                    if not self.use_sampled_target_object_poses and skill_key is not None:
                        skill_pairs = skill_groups[skill_key]
                        skill_ee_pose_indices = [pair['ee_pose_idx'] for pair in skill_pairs]
                        skill_ee_poses = poses[skill_ee_pose_indices]
                        # Get all target object poses for this skill
                        skill_target_object_poses = []
                        for pair in skill_pairs:
                            obj_pose = self._get_target_object_pose_for_skill_demo(
                                pair['language_description'], pair['source_demo_idx']
                            )
                            if obj_pose is not None:
                                skill_target_object_poses.append(obj_pose)
                        if not skill_target_object_poses:
                            continue
                    else:
                        # For fast path, just use the batch's EE poses and all sampled target object poses
                        skill_ee_poses = poses[batch_idx * self.batch_size + i:batch_idx * self.batch_size + i + 1]
                        skill_target_object_poses = self.target_object_poses
                    
                    # Compute object-aware metrics
                    predicted_pose = predicted_poses[i]
                    metrics = self._compute_object_distance_metrics(
                        predicted_pose, target_object_pose, skill_ee_poses, skill_target_object_poses
                    )
                    
                    # Store metrics
                    all_metrics.append(metrics)
                    if not self.use_sampled_target_object_poses and skill_key is not None:
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
        
        # Object-aware metrics
        overall_metrics = {
            'mean_pred_to_obj_distance': np.mean([m['pred_to_obj_distance'] for m in all_metrics]),
            'std_pred_to_obj_distance': np.std([m['pred_to_obj_distance'] for m in all_metrics]),
            'mean_pred_to_obj_pos_distance': np.mean([m['pred_to_obj_pos_distance'] for m in all_metrics]),
            'std_pred_to_obj_pos_distance': np.std([m['pred_to_obj_pos_distance'] for m in all_metrics]),
            'mean_pred_to_obj_ori_distance': np.mean([m['pred_to_obj_ori_distance'] for m in all_metrics]),
            'std_pred_to_obj_ori_distance': np.std([m['pred_to_obj_ori_distance'] for m in all_metrics]),
            'mean_avg_skill_distance': np.mean([m['avg_skill_distance'] for m in all_metrics]),
            'std_avg_skill_distance': np.std([m['avg_skill_distance'] for m in all_metrics]),
            'mean_avg_skill_pos_distance': np.mean([m['avg_skill_pos_distance'] for m in all_metrics]),
            'std_avg_skill_pos_distance': np.std([m['avg_skill_pos_distance'] for m in all_metrics]),
            'mean_avg_skill_ori_distance': np.mean([m['avg_skill_ori_distance'] for m in all_metrics]),
            'std_avg_skill_ori_distance': np.std([m['avg_skill_ori_distance'] for m in all_metrics]),
            'mean_distance_ratio': np.mean([m['distance_ratio'] for m in all_metrics]),
            'std_distance_ratio': np.std([m['distance_ratio'] for m in all_metrics]),
            'mean_pos_distance_ratio': np.mean([m['pos_distance_ratio'] for m in all_metrics]),
            'std_pos_distance_ratio': np.std([m['pos_distance_ratio'] for m in all_metrics]),
            'mean_ori_distance_ratio': np.mean([m['ori_distance_ratio'] for m in all_metrics]),
            'std_ori_distance_ratio': np.std([m['ori_distance_ratio'] for m in all_metrics]),
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
                'mean_pred_to_obj_distance': np.mean([m['pred_to_obj_distance'] for m in metrics]),
                'mean_avg_skill_distance': np.mean([m['avg_skill_distance'] for m in metrics]),
                'mean_distance_ratio': np.mean([m['distance_ratio'] for m in metrics]),
                'mean_pos_distance_ratio': np.mean([m['pos_distance_ratio'] for m in metrics]),
                'mean_ori_distance_ratio': np.mean([m['ori_distance_ratio'] for m in metrics]),
            }
        
        # Percentiles
        pred_to_obj_distances = [m['pred_to_obj_distance'] for m in all_metrics]
        distance_ratios = [m['distance_ratio'] for m in all_metrics]
        pos_distance_ratios = [m['pos_distance_ratio'] for m in all_metrics]
        ori_distance_ratios = [m['ori_distance_ratio'] for m in all_metrics]
        
        percentiles = {
            'pred_to_obj_distance_50th': np.percentile(pred_to_obj_distances, 50),
            'pred_to_obj_distance_75th': np.percentile(pred_to_obj_distances, 75),
            'pred_to_obj_distance_90th': np.percentile(pred_to_obj_distances, 90),
            'pred_to_obj_distance_95th': np.percentile(pred_to_obj_distances, 95),
            'distance_ratio_50th': np.percentile(distance_ratios, 50),
            'distance_ratio_75th': np.percentile(distance_ratios, 75),
            'distance_ratio_90th': np.percentile(distance_ratios, 90),
            'distance_ratio_95th': np.percentile(distance_ratios, 95),
            'pos_distance_ratio_50th': np.percentile(pos_distance_ratios, 50),
            'pos_distance_ratio_75th': np.percentile(pos_distance_ratios, 75),
            'pos_distance_ratio_90th': np.percentile(pos_distance_ratios, 90),
            'pos_distance_ratio_95th': np.percentile(pos_distance_ratios, 95),
            'ori_distance_ratio_50th': np.percentile(ori_distance_ratios, 50),
            'ori_distance_ratio_75th': np.percentile(ori_distance_ratios, 75),
            'ori_distance_ratio_90th': np.percentile(ori_distance_ratios, 90),
            'ori_distance_ratio_95th': np.percentile(ori_distance_ratios, 95),
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
        object_metrics_list = [r['overall_metrics'] for r in all_results]
        training_metrics_list = [r['training_metrics'] for r in all_results]
        
        # Compute mean and std across runs
        object_metrics_avg = {}
        object_metrics_std = {}
        for key in object_metrics_list[0].keys():
            values = [m[key] for m in object_metrics_list]
            object_metrics_avg[key] = np.mean(values)
            object_metrics_std[key] = np.std(values)
        
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
        result['overall_metrics'] = object_metrics_avg
        result['training_metrics'] = training_metrics_avg
        result['overall_metrics_std'] = object_metrics_std
        result['training_metrics_std'] = training_metrics_std
        result['num_runs'] = self.num_runs
        
        return result
        
    def run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation on training, validation, and test splits."""
        print("Starting object-aware evaluation...")
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
        results = {}
        # If using fast path (sampled target object poses), only evaluate the specified split
        if hasattr(self, 'use_sampled_target_object_poses') and self.use_sampled_target_object_poses:
            print(f"[Fast path] Only evaluating split: {self.split}")
            split_results = self.evaluate_split(self.split)
            results[self.split] = split_results
        else:
            # Evaluate training set (no augmentation)
            print("\nEvaluating TRAINING SET (no augmentation)...")
            train_results = self.evaluate_split("train")
            results['training'] = train_results
            # Evaluate training set (with augmentation) if enabled
            if self.is_train_aug_eval:
                print("\nEvaluating TRAINING SET (with augmentation)...")
                train_results_aug = self.evaluate_split("train", use_aug=True)
                results['training_aug'] = train_results_aug
            # Evaluate validation set
            print("\nEvaluating VALIDATION SET...")
            val_results = self.evaluate_split("val")
            results['validation'] = val_results
            # Evaluate test set
            print("\nEvaluating TEST SET...")
            test_results = self.evaluate_split("test")
            results['test'] = test_results
        # Add model info/config
        results['model_info'] = model_info
        results['evaluation_config'] = eval_config
        return results
        
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results in a readable format."""
        print("\n" + "="*80)
        print("OBJECT-AWARE POSEVLM EVALUATION RESULTS")
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
        # Only print the evaluated split in fast path
        splits_to_print = [k for k in results.keys() if k not in ['model_info', 'evaluation_config']]
        # List of all expected keys and their pretty names
        metric_keys = [
            ('mean_pred_to_obj_distance', 'Pred-to-Obj Distance'),
            ('std_pred_to_obj_distance', 'Pred-to-Obj Distance (std)'),
            ('mean_pred_to_obj_pos_distance', 'Pred-to-Obj Pos Distance'),
            ('std_pred_to_obj_pos_distance', 'Pred-to-Obj Pos Distance (std)'),
            ('mean_pred_to_obj_ori_distance', 'Pred-to-Obj Ori Distance'),
            ('std_pred_to_obj_ori_distance', 'Pred-to-Obj Ori Distance (std)'),
            ('mean_avg_skill_distance', 'Avg Skill Distance'),
            ('std_avg_skill_distance', 'Avg Skill Distance (std)'),
            ('mean_avg_skill_pos_distance', 'Avg Skill Pos Distance'),
            ('std_avg_skill_pos_distance', 'Avg Skill Pos Distance (std)'),
            ('mean_avg_skill_ori_distance', 'Avg Skill Ori Distance'),
            ('std_avg_skill_ori_distance', 'Avg Skill Ori Distance (std)'),
            ('mean_distance_ratio', 'Distance Ratio'),
            ('std_distance_ratio', 'Distance Ratio (std)'),
            ('mean_pos_distance_ratio', 'Pos Distance Ratio'),
            ('std_pos_distance_ratio', 'Pos Distance Ratio (std)'),
            ('mean_ori_distance_ratio', 'Ori Distance Ratio'),
            ('std_ori_distance_ratio', 'Ori Distance Ratio (std)'),
        ]
        for split_name in splits_to_print:
            split_results = results[split_name]
            overall = split_results['overall_metrics']
            print(f"\n{split_name.upper()} SUMMARY:")
            for k, pretty in metric_keys:
                if k in overall:
                    print(f"  {pretty}: {overall[k]:.4f}")
                else:
                    print(f"  [Warning] {pretty} (key: {k}) is missing in results.")
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Object-aware evaluation of PoseVLM")
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
    parser.add_argument(
        "--sampled_target_object_poses_dir",
        type=str,
        default="runs/eval_results/vlm_pose_generator/",
        help="Directory containing sampled_target_object_poses_{split}.npy for fast evaluation (optional)"
    )
    parser.add_argument(
        "--is_train_aug_eval",
        action="store_true",
        default=False,
        help="If set, also evaluate training set with augmentation (default: False)"
    )
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ObjectAwareEvaluator(
        checkpoint_path=args.checkpoint_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_size=args.sample_size,
        num_runs=args.num_runs,
        splits_folder=args.splits_folder,
        sample_indices_file=args.sample_indices_file,
        split=args.split,
        sampled_target_object_poses_dir=args.sampled_target_object_poses_dir,
        is_train_aug_eval=args.is_train_aug_eval,
    )
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    results_dir = "runs/eval_results/vlm_pose_generator/eval_obj_dist_metric"
    os.makedirs(results_dir, exist_ok=True)
    
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_dir, f"evaluation_results_object_aware_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main() 