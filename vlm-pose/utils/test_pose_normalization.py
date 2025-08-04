#!/usr/bin/env python3
"""
test_pose_normalization.py

Test script to verify that pose normalization is working correctly.
"""

import sys
sys.path.append('..')

import numpy as np
import torch
from prismatic.vla.datasets.pose_dataset import create_pose_dataset

def test_pose_normalization():
    """Test pose normalization functionality."""
    print("Testing pose normalization...")
    
    # Create a dataset with normalization enabled
    dataset = create_pose_dataset(
        data_root="datasets/local_pairs_datasets",
        split="train",
        num_images_in_input=1,
        use_image_augmentation=False,
        tokenizer_name="microsoft/DialoGPT-medium",
        splits_folder="smaller_splits",
        use_pose_normalization=True,
        max_samples=10,  # Limit for testing
    )
    
    # Test the normalize_pose method directly
    print("\nTesting normalize_pose method directly:")
    
    # Test cases with angles outside [-π, π]
    test_poses = [
        np.array([0, 0, 0, 0, 0, 0]),  # All zeros
        np.array([0, 0, 0, np.pi, 0, 0]),  # π
        np.array([0, 0, 0, -np.pi, 0, 0]),  # -π
        np.array([0, 0, 0, 2*np.pi, 0, 0]),  # 2π
        np.array([0, 0, 0, -2*np.pi, 0, 0]),  # -2π
        np.array([0, 0, 0, 3*np.pi, 0, 0]),  # 3π
        np.array([0, 0, 0, -3*np.pi, 0, 0]),  # -3π
        np.array([0, 0, 0, 0, 4.5, 0]),  # ry = 4.5 (should become ~-1.78)
        np.array([0, 0, 0, 0, -4.5, 0]),  # ry = -4.5 (should become ~1.78)
    ]
    
    for i, pose in enumerate(test_poses):
        normalized = dataset.normalize_pose(pose)
        print(f"Test {i+1}:")
        print(f"  Original: {pose}")
        print(f"  Normalized: {normalized}")
        print(f"  Orientation range: [{normalized[3:].min():.3f}, {normalized[3:].max():.3f}]")
        print()
    
    # Test with actual dataset samples
    print("Testing with actual dataset samples:")
    original_ranges = []
    normalized_ranges = []
    
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        pose_target = sample['pose_targets']
        
        # Get the original pose from the dataset
        pose_idx = dataset.annotation_df.iloc[i]['ee_pose_idx']
        original_pose = dataset.poses[pose_idx]
        
        original_ranges.append(original_pose[3:])  # orientation part
        normalized_ranges.append(pose_target.numpy()[3:])  # orientation part
        
        print(f"Sample {i+1}:")
        print(f"  Original orientation: {original_pose[3:]}")
        print(f"  Normalized orientation: {pose_target.numpy()[3:]}")
        print()
    
    # Check ranges
    original_ranges = np.array(original_ranges)
    normalized_ranges = np.array(normalized_ranges)
    
    print("Range analysis:")
    print(f"Original orientation ranges:")
    print(f"  Min: {original_ranges.min(axis=0)}")
    print(f"  Max: {original_ranges.max(axis=0)}")
    print(f"  Mean: {original_ranges.mean(axis=0)}")
    print(f"  Std: {original_ranges.std(axis=0)}")
    
    print(f"\nNormalized orientation ranges:")
    print(f"  Min: {normalized_ranges.min(axis=0)}")
    print(f"  Max: {normalized_ranges.max(axis=0)}")
    print(f"  Mean: {normalized_ranges.mean(axis=0)}")
    print(f"  Std: {normalized_ranges.std(axis=0)}")
    
    # Verify that normalized angles are within [-π, π]
    assert np.all(normalized_ranges >= -np.pi), "Normalized angles should be >= -π"
    assert np.all(normalized_ranges <= np.pi), "Normalized angles should be <= π"
    print("\n✅ All normalized angles are within [-π, π] range!")
    
    # Test without normalization
    print("\nTesting without normalization:")
    dataset_no_norm = create_pose_dataset(
        data_root="datasets/local_pairs_datasets",
        split="train",
        num_images_in_input=1,
        use_image_augmentation=False,
        tokenizer_name="microsoft/DialoGPT-medium",
        splits_folder="smaller_splits",
        use_pose_normalization=False,
        max_samples=5,
    )
    
    for i in range(min(3, len(dataset_no_norm))):
        sample = dataset_no_norm[i]
        pose_target = sample['pose_targets']
        pose_idx = dataset_no_norm.annotation_df.iloc[i]['ee_pose_idx']
        original_pose = dataset_no_norm.poses[pose_idx]
        
        print(f"Sample {i+1} (no normalization):")
        print(f"  Original orientation: {original_pose[3:]}")
        print(f"  Target orientation: {pose_target.numpy()[3:]}")
        print(f"  Are they equal? {np.allclose(original_pose[3:], pose_target.numpy()[3:])}")
        print()

if __name__ == "__main__":
    test_pose_normalization() 