"""
split_pose_data.py

Split pose data by language description to ensure each skill appears in only one split.
This prevents data leakage between train/val/test sets.
"""

import pandas as pd
import numpy as np
import os
import argparse
from collections import defaultdict
import json


def split_by_language_description(
    annotation_csv_path, 
    output_dir, 
    train_ratio=0.7, 
    val_ratio=0.15, 
    test_ratio=0.15,
    min_samples_per_skill=5,
    random_seed=42
):
    """
    Split data based on unique language descriptions.
    
    Args:
        annotation_csv_path: Path to annotation.csv
        output_dir: Directory to save split files
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set  
        test_ratio: Ratio for test set
        min_samples_per_skill: Minimum samples required per skill
        random_seed: Random seed for reproducibility
    
    Returns:
        dict: Statistics about the split
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Load annotation data
    print(f"Loading annotation data from {annotation_csv_path}")
    df = pd.read_csv(annotation_csv_path)
    
    # Group by language description
    skill_groups = df.groupby('language_description')
    
    # Filter skills with enough samples
    valid_skills = []
    for skill_name, skill_data in skill_groups:
        if len(skill_data) >= min_samples_per_skill:
            valid_skills.append(skill_name)
        else:
            print(f"Warning: Skill '{skill_name}' has only {len(skill_data)} samples, skipping")
    
    print(f"Found {len(valid_skills)} valid skills with >= {min_samples_per_skill} samples each")
    
    # Shuffle skills for random split
    np.random.shuffle(valid_skills)
    
    # Calculate split indices
    n_skills = len(valid_skills)
    n_train = int(n_skills * train_ratio)
    n_val = int(n_skills * val_ratio)
    n_test = n_skills - n_train - n_val  # Handle rounding
    
    # Split skills
    train_skills = valid_skills[:n_train]
    val_skills = valid_skills[n_train:n_train + n_val]
    test_skills = valid_skills[n_train + n_val:]
    
    # Create split dataframes
    train_df = df[df['language_description'].isin(train_skills)]
    val_df = df[df['language_description'].isin(val_skills)]
    test_df = df[df['language_description'].isin(test_skills)]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save split files
    train_path = os.path.join(output_dir, 'train_annotation.csv')
    val_path = os.path.join(output_dir, 'val_annotation.csv')
    test_path = os.path.join(output_dir, 'test_annotation.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Save split configuration
    split_config = {
        'train_skills': train_skills,
        'val_skills': val_skills,
        'test_skills': test_skills,
        'split_ratios': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio},
        'min_samples_per_skill': min_samples_per_skill,
        'random_seed': random_seed,
        'statistics': {
            'total_skills': n_skills,
            'train_skills': len(train_skills),
            'val_skills': len(val_skills), 
            'test_skills': len(test_skills),
            'total_samples': len(df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df)
        }
    }
    
    config_path = os.path.join(output_dir, 'split_config.json')
    with open(config_path, 'w') as f:
        json.dump(split_config, f, indent=2)
    
    # Print statistics
    print(f"\nSplit Statistics:")
    print(f"  Total skills: {n_skills}")
    print(f"  Train skills: {len(train_skills)} ({len(train_df)} samples)")
    print(f"  Val skills: {len(val_skills)} ({len(val_df)} samples)")
    print(f"  Test skills: {len(test_skills)} ({len(test_df)} samples)")
    print(f"\nFiles saved to {output_dir}:")
    print(f"  - train_annotation.csv")
    print(f"  - val_annotation.csv") 
    print(f"  - test_annotation.csv")
    print(f"  - split_config.json")
    
    return split_config


def main():
    parser = argparse.ArgumentParser(description="Split pose data by language description")
    parser.add_argument("--annotation_csv", type=str, 
                       default="datasets/local_pairs_datasets/annotation.csv",
                       help="Path to annotation.csv")
    parser.add_argument("--output_dir", type=str,
                       default="datasets/local_pairs_datasets/splits",
                       help="Output directory for split files")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                       help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                       help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15,
                       help="Test set ratio")
    parser.add_argument("--min_samples", type=int, default=5,
                       help="Minimum samples per skill")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Error: Ratios sum to {total_ratio}, must equal 1.0")
        return
    
    # Run split
    split_config = split_by_language_description(
        args.annotation_csv,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.min_samples,
        args.random_seed
    )
    
    print(f"\nSplit completed successfully!")


if __name__ == "__main__":
    main() 