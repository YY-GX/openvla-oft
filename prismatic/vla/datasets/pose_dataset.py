"""
pose_dataset.py

Dataset for pose prediction from image + language pairs.
Follows original codebase patterns and inherits from Dataset.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoTokenizer

from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class PoseDataset(Dataset):
    """
    Dataset for pose prediction, following original codebase patterns.
    
    Loads overview images + language descriptions + 6D poses for training.
    Supports single or multiple images like the original codebase.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_images_in_input: int = 1,
        max_length: int = 512,
        image_size: int = 224,
        use_image_augmentation: bool = False,
        pose_augmentation: Optional[Any] = None,
        tokenizer_name: str = "microsoft/DialoGPT-medium",
        **kwargs
    ):
        """
        Initialize pose dataset.
        
        Args:
            data_root: Root directory containing data
            split: Dataset split ('train', 'val', 'test')
            num_images_in_input: Number of images per sample (1 or 2)
            max_length: Maximum sequence length for language
            image_size: Size to resize images to
            use_image_augmentation: Whether to use image augmentation
            pose_augmentation: Pose augmentation object
            tokenizer_name: Name of tokenizer to use
        """
        super().__init__()
        
        self.data_root = data_root
        self.split = split
        self.num_images_in_input = num_images_in_input
        self.max_length = max_length
        self.image_size = image_size
        self.use_image_augmentation = use_image_augmentation
        self.pose_augmentation = pose_augmentation
        
        # Load annotation data
        annotation_path = os.path.join(data_root, "splits", f"{split}_annotation.csv")
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
        self.annotation_df = pd.read_csv(annotation_path)
        overwatch.info(f"Loaded {len(self.annotation_df)} samples for {split} split")
        
        # Load poses
        poses_path = os.path.join(data_root, "local_poses.npy")
        if not os.path.exists(poses_path):
            raise FileNotFoundError(f"Poses file not found: {poses_path}")
        
        self.poses = np.load(poses_path)
        overwatch.info(f"Loaded {len(self.poses)} poses")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set up image directory
        self.image_dir = os.path.join(data_root, "3rd_imgs")
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # Validate data consistency
        self._validate_data()
        
        # Set up image transforms (following original patterns)
        self._setup_transforms()
    
    def _validate_data(self):
        """Validate data consistency."""
        # Check that all pose indices are valid
        max_pose_idx = len(self.poses) - 1
        invalid_poses = self.annotation_df[self.annotation_df['ee_pose_idx'] > max_pose_idx]
        if len(invalid_poses) > 0:
            raise ValueError(f"Found {len(invalid_poses)} samples with invalid pose indices")
        
        # Check that all image indices are valid
        # Get the maximum image index from the CSV
        max_image_idx = self.annotation_df['overview_image_idx'].max()
        
        # Check if the maximum image index is within the range of available images
        # Images are named 00000.jpg to 156326.jpg (156327 total images)
        available_image_count = len([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        
        if max_image_idx >= available_image_count:
            raise ValueError(f"Found image indices up to {max_image_idx}, but only {available_image_count} images available")
        
        overwatch.info("Data validation passed")
    
    def _setup_transforms(self):
        """Set up image transforms following original patterns."""
        from torchvision import transforms
        
        # Basic transforms (following original patterns)
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Image augmentation (following original patterns)
        if self.use_image_augmentation:
            self.aug_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.aug_transform = self.transform
    
    def _load_image(self, image_idx: int) -> torch.Tensor:
        """Load and preprocess image."""
        image_path = os.path.join(self.image_dir, f"{image_idx:05d}.jpg")
        image = Image.open(image_path).convert('RGB')
        
        # Apply augmentation if training
        if self.split == "train" and self.use_image_augmentation:
            image = self.aug_transform(image)
        else:
            image = self.transform(image)
        
        return image
    
    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text following original patterns."""
        # Simple prompt format (can be extended)
        prompt = f"Given this image, predict the end-effector pose for: {text}"
        
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }
    
    def _get_pose_target(self, pose_idx: int) -> torch.Tensor:
        """Get pose target and apply augmentation if needed."""
        pose = torch.tensor(self.poses[pose_idx], dtype=torch.float32)
        
        # Apply pose augmentation if available and training
        if self.split == "train" and self.pose_augmentation is not None:
            pose = self.pose_augmentation(pose)
        
        return pose
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset."""
        sample = self.annotation_df.iloc[idx]
        
        try:
            # Load image(s)
            if self.num_images_in_input == 1:
                image = self._load_image(sample['overview_image_idx'])
                pixel_values = image.unsqueeze(0)  # Add batch dimension
            elif self.num_images_in_input == 2:
                # For 2 images, we could load the same image twice or implement different logic
                # For now, just duplicate the image
                image = self._load_image(sample['overview_image_idx'])
                pixel_values = torch.stack([image, image])
            else:
                raise ValueError(f"Unsupported num_images_in_input: {self.num_images_in_input}")
            
            # Tokenize text
            text_encoding = self._tokenize_text(sample['language_description'])
            
            # Get pose target
            pose_target = self._get_pose_target(sample['ee_pose_idx'])
            
            # Create sample
            sample_dict = {
                "pixel_values": pixel_values,
                "input_ids": text_encoding["input_ids"],
                "attention_mask": text_encoding["attention_mask"],
                "pose_targets": pose_target,
                "language_description": sample['language_description'],
                "overview_image_idx": sample['overview_image_idx'],
                "ee_pose_idx": sample['ee_pose_idx']
            }
            
            return sample_dict
            
        except Exception as e:
            overwatch.warning(f"Error loading sample {idx}: {e}")
            # Return a dummy sample in case of error
            dummy_image = torch.zeros(3, self.image_size, self.image_size)
            dummy_input_ids = torch.zeros(self.max_length, dtype=torch.long)
            dummy_attention_mask = torch.zeros(self.max_length, dtype=torch.long)
            dummy_pose = torch.zeros(self.poses.shape[1] if len(self.poses.shape) > 1 else 1)
            
            return {
                "pixel_values": dummy_image.unsqueeze(0),
                "input_ids": dummy_input_ids,
                "attention_mask": dummy_attention_mask,
                "pose_targets": dummy_pose,
                "language_description": "",
                "overview_image_idx": -1,
                "ee_pose_idx": -1
            }
    
    def __len__(self):
        """Return dataset length."""
        return len(self.annotation_df)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "split": self.split,
            "num_samples": len(self.annotation_df),
            "num_images_in_input": self.num_images_in_input,
            "unique_skills": self.annotation_df['language_description'].nunique(),
            "pose_dim": self.poses.shape[1] if len(self.poses.shape) > 1 else 1,
            "image_size": self.image_size,
            "max_length": self.max_length
        }
        
        # Add pose statistics
        if len(self.poses.shape) > 1:
            stats.update({
                "pose_mean": self.poses.mean(axis=0).tolist(),
                "pose_std": self.poses.std(axis=0).tolist(),
                "pose_min": self.poses.min(axis=0).tolist(),
                "pose_max": self.poses.max(axis=0).tolist()
            })
        
        return stats


def create_pose_dataset(
    data_root: str,
    split: str = "train",
    num_images_in_input: int = 1,
    **kwargs
) -> PoseDataset:
    """
    Factory function to create pose dataset.
    
    Args:
        data_root: Root directory containing data
        split: Dataset split ('train', 'val', 'test')
        num_images_in_input: Number of images per sample
        **kwargs: Additional arguments passed to PoseDataset
    
    Returns:
        PoseDataset instance
    """
    return PoseDataset(
        data_root=data_root,
        split=split,
        num_images_in_input=num_images_in_input,
        **kwargs
    ) 
