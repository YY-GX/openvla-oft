#!/usr/bin/env python3
"""
test_phase1.py

Test script for Phase 1 components: data split, pose dataset, and pose collator.
Has two modes: self-contained (only data split) and full (all components).
"""

import os
import sys
import argparse
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Only import what we have implemented
try:
    from utils.split_pose_data import split_by_language_description
    SPLIT_AVAILABLE = True
except ImportError:
    SPLIT_AVAILABLE = False
    print("Warning: split_pose_data module not available")

# Try to import modules that may not be implemented yet
try:
    import torch
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader
    from prismatic.vla.datasets.pose_dataset import PoseDataset, create_pose_dataset
    from prismatic.util.data_utils import PaddedCollatorForPosePrediction
    FULL_MODULES_AVAILABLE = True
except ImportError as e:
    FULL_MODULES_AVAILABLE = False
    print(f"Warning: Some modules not available: {e}")


def test_data_split():
    """Test data splitting functionality."""
    print("=" * 50)
    print("Testing Data Split Functionality")
    print("=" * 50)
    
    if not SPLIT_AVAILABLE:
        print("❌ split_pose_data module not available")
        return False
    
    # Check if annotation file exists
    annotation_path = "datasets/local_pairs_datasets/annotation.csv"
    if not os.path.exists(annotation_path):
        print(f"❌ Annotation file not found: {annotation_path}")
        print("Please run the data extraction script first.")
        return False
    
    # Create output directory
    output_dir = "datasets/local_pairs_datasets/splits"
    
    try:
        # Run data split
        split_config = split_by_language_description(
            annotation_csv_path=annotation_path,
            output_dir=output_dir,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            min_samples_per_skill=5,
            random_seed=42
        )
        
        # Verify output files
        expected_files = [
            "train_annotation.csv",
            "val_annotation.csv", 
            "test_annotation.csv",
            "split_config.json"
        ]
        
        for filename in expected_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                print(f"✅ {filename} created successfully")
            else:
                print(f"❌ {filename} not found")
                return False
        
        # Load and verify split files
        train_df = pd.read_csv(os.path.join(output_dir, "train_annotation.csv"))
        val_df = pd.read_csv(os.path.join(output_dir, "val_annotation.csv"))
        test_df = pd.read_csv(os.path.join(output_dir, "test_annotation.csv"))
        
        print(f"✅ Train samples: {len(train_df)}")
        print(f"✅ Val samples: {len(val_df)}")
        print(f"✅ Test samples: {len(test_df)}")
        
        # Check for overlap
        train_skills = set(train_df['language_description'].unique())
        val_skills = set(val_df['language_description'].unique())
        test_skills = set(test_df['language_description'].unique())
        
        overlap = train_skills & val_skills & test_skills
        if len(overlap) == 0:
            print("✅ No skill overlap between splits")
        else:
            print(f"❌ Found overlap: {overlap}")
            return False
        
        # Print some statistics
        print(f"✅ Total skills: {len(train_skills) + len(val_skills) + len(test_skills)}")
        print(f"✅ Train skills: {len(train_skills)}")
        print(f"✅ Val skills: {len(val_skills)}")
        print(f"✅ Test skills: {len(test_skills)}")
        
        print("✅ Data split test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data split test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pose_dataset():
    """Test pose dataset functionality."""
    print("\n" + "=" * 50)
    print("Testing Pose Dataset Functionality")
    print("=" * 50)
    
    if not FULL_MODULES_AVAILABLE:
        print("❌ Full modules not available, skipping pose dataset test")
        return False
    
    # Check if split files exist
    data_root = "datasets/local_pairs_datasets"
    train_annotation = os.path.join(data_root, "splits", "train_annotation.csv")
    
    if not os.path.exists(train_annotation):
        print(f"❌ Train annotation not found: {train_annotation}")
        print("Please run data split first.")
        return False
    
    try:
        # Test single image dataset
        print("Testing single image dataset...")
        dataset = PoseDataset(
            data_root=data_root,
            split="train",
            num_images_in_input=1,
            max_length=512,
            image_size=224,
            use_image_augmentation=False
        )
        
        # Get dataset statistics
        stats = dataset.get_statistics()
        print(f"✅ Dataset stats: {stats}")
        
        # Test iteration
        sample_count = 0
        for sample in dataset:
            if sample_count >= 3:  # Test first 3 samples
                break
            
            print(f"✅ Sample {sample_count + 1}:")
            print(f"  - Pixel values shape: {sample['pixel_values'].shape}")
            print(f"  - Input IDs shape: {sample['input_ids'].shape}")
            print(f"  - Attention mask shape: {sample['attention_mask'].shape}")
            print(f"  - Pose targets shape: {sample['pose_targets'].shape}")
            print(f"  - Language: {sample['language_description'][:50]}...")
            
            sample_count += 1
        
        # Test two image dataset
        print("\nTesting two image dataset...")
        dataset_2img = PoseDataset(
            data_root=data_root,
            split="train",
            num_images_in_input=2,
            max_length=512,
            image_size=224,
            use_image_augmentation=False
        )
        
        for sample in dataset_2img:
            print(f"✅ Two image sample:")
            print(f"  - Pixel values shape: {sample['pixel_values'].shape}")
            break
        
        print("✅ Pose dataset test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Pose dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pose_collator():
    """Test pose collator functionality."""
    print("\n" + "=" * 50)
    print("Testing Pose Collator Functionality")
    print("=" * 50)
    
    if not FULL_MODULES_AVAILABLE:
        print("❌ Full modules not available, skipping pose collator test")
        return False
    
    try:
        # Create collator
        collator = PaddedCollatorForPosePrediction(
            model_max_length=512,
            pad_token_id=0,
            padding_side="right"
        )
        
        # Create dummy samples
        dummy_samples = []
        for i in range(3):
            sample = {
                "pixel_values": torch.randn(3, 224, 224),
                "input_ids": torch.randint(0, 1000, (50 + i * 10,)),  # Variable length
                "attention_mask": torch.ones(50 + i * 10),
                "pose_targets": torch.randn(6),  # 6D pose
                "language_description": f"test skill {i}",
                "overview_image_idx": i,
                "ee_pose_idx": i
            }
            dummy_samples.append(sample)
        
        # Test collation
        batch = collator(dummy_samples)
        
        print("✅ Collated batch:")
        print(f"  - Pixel values shape: {batch['pixel_values'].shape}")
        print(f"  - Input IDs shape: {batch['input_ids'].shape}")
        print(f"  - Attention mask shape: {batch['attention_mask'].shape}")
        print(f"  - Pose targets shape: {batch['pose_targets'].shape}")
        print(f"  - Language descriptions: {batch['language_descriptions']}")
        
        # Verify shapes
        expected_batch_size = 3
        assert batch['pixel_values'].shape[0] == expected_batch_size, "Wrong batch size"
        assert batch['input_ids'].shape[0] == expected_batch_size, "Wrong batch size"
        assert batch['pose_targets'].shape[0] == expected_batch_size, "Wrong batch size"
        
        print("✅ Pose collator test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Pose collator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end():
    """Test end-to-end data loading pipeline."""
    print("\n" + "=" * 50)
    print("Testing End-to-End Data Loading Pipeline")
    print("=" * 50)
    
    if not FULL_MODULES_AVAILABLE:
        print("❌ Full modules not available, skipping end-to-end test")
        return False
    
    try:
        # Create dataset
        data_root = "datasets/local_pairs_datasets"
        dataset = PoseDataset(
            data_root=data_root,
            split="train",
            num_images_in_input=1,
            max_length=512,
            image_size=224,
            use_image_augmentation=False
        )
        
        # Create collator
        collator = PaddedCollatorForPosePrediction(
            model_max_length=512,
            pad_token_id=0,
            padding_side="right"
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=collator,
            num_workers=0  # Use 0 for testing
        )
        
        # Test iteration
        batch_count = 0
        for batch in dataloader:
            if batch_count >= 2:  # Test first 2 batches
                break
            
            print(f"✅ Batch {batch_count + 1}:")
            print(f"  - Pixel values shape: {batch['pixel_values'].shape}")
            print(f"  - Input IDs shape: {batch['input_ids'].shape}")
            print(f"  - Attention mask shape: {batch['attention_mask'].shape}")
            print(f"  - Pose targets shape: {batch['pose_targets'].shape}")
            print(f"  - Batch size: {batch['pixel_values'].shape[0]}")
            
            batch_count += 1
        
        print("✅ End-to-end test passed!")
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_split_config():
    """Test that split configuration is valid."""
    print("\n" + "=" * 50)
    print("Testing Split Configuration")
    print("=" * 50)
    
    config_path = "datasets/local_pairs_datasets/splits/split_config.json"
    if not os.path.exists(config_path):
        print(f"❌ Split config not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check required fields
        required_fields = ['train_skills', 'val_skills', 'test_skills', 'statistics']
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field: {field}")
                return False
        
        # Check statistics
        stats = config['statistics']
        print(f"✅ Total skills: {stats['total_skills']}")
        print(f"✅ Train skills: {stats['train_skills']}")
        print(f"✅ Val skills: {stats['val_skills']}")
        print(f"✅ Test skills: {stats['test_skills']}")
        print(f"✅ Total samples: {stats['total_samples']}")
        print(f"✅ Train samples: {stats['train_samples']}")
        print(f"✅ Val samples: {stats['val_samples']}")
        print(f"✅ Test samples: {stats['test_samples']}")
        
        # Check for overlap
        train_skills = set(config['train_skills'])
        val_skills = set(config['val_skills'])
        test_skills = set(config['test_skills'])
        
        overlap = train_skills & val_skills & test_skills
        if len(overlap) == 0:
            print("✅ No skill overlap between splits")
        else:
            print(f"❌ Found overlap: {overlap}")
            return False
        
        print("✅ Split configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Split configuration test failed: {e}")
        return False


def main():
    """Run Phase 1 tests based on available modules."""
    parser = argparse.ArgumentParser(description="Test Phase 1 components")
    parser.add_argument("--mode", choices=["self-contained", "full"], default="auto",
                       help="Test mode: self-contained (only data split) or full (all components)")
    parser.add_argument("--skip-split", action="store_true", help="Skip data split test")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset test")
    parser.add_argument("--skip-collator", action="store_true", help="Skip collator test")
    parser.add_argument("--skip-e2e", action="store_true", help="Skip end-to-end test")
    parser.add_argument("--skip-config", action="store_true", help="Skip split config test")
    
    args = parser.parse_args()
    
    # Auto-detect mode based on available modules
    if args.mode == "auto":
        if FULL_MODULES_AVAILABLE:
            args.mode = "full"
        else:
            args.mode = "self-contained"
    
    print("🚀 Starting Phase 1 Tests")
    print(f"Mode: {args.mode}")
    print(f"Split module available: {SPLIT_AVAILABLE}")
    print(f"Full modules available: {FULL_MODULES_AVAILABLE}")
    
    if args.mode == "self-contained":
        print("Running self-contained tests (data split only)")
    else:
        print("Running full tests (all components)")
    
    results = []
    
    # Test data split (always available if implemented)
    if not args.skip_split and SPLIT_AVAILABLE:
        results.append(("Data Split", test_data_split()))
    
    # Test split configuration (always available if split was run)
    if not args.skip_config:
        results.append(("Split Config", test_split_config()))
    
    # Test other components only in full mode
    if args.mode == "full":
        if not args.skip_dataset and FULL_MODULES_AVAILABLE:
            results.append(("Pose Dataset", test_pose_dataset()))
        
        if not args.skip_collator and FULL_MODULES_AVAILABLE:
            results.append(("Pose Collator", test_pose_collator()))
        
        if not args.skip_e2e and FULL_MODULES_AVAILABLE:
            results.append(("End-to-End", test_end_to_end()))
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        if args.mode == "self-contained":
            print("🎉 Self-contained tests passed! Data split is working correctly.")
            print("💡 Run with --mode full when all modules are implemented.")
        else:
            print("🎉 All Phase 1 tests passed! Ready for Phase 2.")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main()) 