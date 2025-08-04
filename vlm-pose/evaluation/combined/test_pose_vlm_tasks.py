#!/usr/bin/env python3
"""
test_pose_vlm_tasks.py

Test script for the combined PoseVLM + IK + OpenVLA-OFT pipeline 
specifically on the 6 tasks used for PoseVLM training.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
import argparse
from typing import Dict, Any
import sys

# Add project root to path for PoseFinetuneConfig import
sys.path.append('../../..')

# Import PoseFinetuneConfig to avoid AttributeError
try:
    from finetune_pose import PoseFinetuneConfig
except ImportError:
    # Fallback if import fails
    class PoseFinetuneConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

# Import the combined evaluator
from run_combined_eval import CombinedEvaluator


def test_pose_vlm_tasks(
    pose_vlm_checkpoint: str,
    openvla_checkpoint: str,
    num_episodes: int = 5,
    max_ik_attempts: int = 3,
    save_debug_images: bool = True,
    debug_image_dir: str = "runs/eval_results/combined_eval/pose_vlm_tasks_debug",
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Test the combined pipeline on the 6 PoseVLM training tasks.
    
    Args:
        pose_vlm_checkpoint: Path to PoseVLM checkpoint
        openvla_checkpoint: Path to OpenVLA-OFT checkpoint
        num_episodes: Number of episodes per task
        max_ik_attempts: Maximum IK attempts
        save_debug_images: Whether to save debug images
        debug_image_dir: Directory to save debug images
        device: Device to use
        seed: Random seed
        
    Returns:
        Dictionary containing evaluation results
    """
    
    print("Testing Combined Pipeline on PoseVLM Training Tasks")
    print("=" * 60)
    print(f"PoseVLM checkpoint: {pose_vlm_checkpoint}")
    print(f"OpenVLA checkpoint: {openvla_checkpoint}")
    print(f"Episodes per task: {num_episodes}")
    print(f"Max IK attempts: {max_ik_attempts}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print()
    
    # Create evaluator with PoseVLM tasks only
    evaluator = CombinedEvaluator(
        pose_vlm_checkpoint=pose_vlm_checkpoint,
        openvla_checkpoint=openvla_checkpoint,
        task_suite_name="boss_44",
        num_episodes_per_task=num_episodes,
        max_ik_attempts=max_ik_attempts,
        save_debug_images=save_debug_images,
        debug_image_dir=debug_image_dir,
        device=device,
        seed=seed,
        use_pose_vlm_smaller_split_tasks_only=True,  # This is the key difference
    )
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total tasks evaluated: {results['total_tasks']}")
    print(f"Total episodes: {results['total_episodes']}")
    print(f"Overall success rate: {results['overall_success_rate']:.2%}")
    print(f"IK failure rate: {results['ik_failure_rate']:.2%}")
    print(f"VLA failure rate: {results['vla_failure_rate']:.2%}")
    print()
    
    # Print per-task results
    print("Per-Task Results:")
    print("-" * 40)
    for task_name, task_results in results['task_results'].items():
        success_rate = task_results['success_rate']
        ik_failures = task_results['ik_failures']
        vla_failures = task_results['vla_failures']
        total_episodes = task_results['total_episodes']
        
        print(f"{task_name}:")
        print(f"  Success rate: {success_rate:.2%} ({task_results['successful_episodes']}/{total_episodes})")
        print(f"  IK failures: {ik_failures}")
        print(f"  VLA failures: {vla_failures}")
        print()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test combined pipeline on PoseVLM tasks")
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
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes per task"
    )
    parser.add_argument(
        "--max_ik_attempts",
        type=int,
        default=3,
        help="Maximum IK attempts"
    )
    parser.add_argument(
        "--save_debug_images",
        action="store_true",
        default=True,
        help="Save debug images"
    )
    parser.add_argument(
        "--debug_image_dir",
        type=str,
        default="runs/eval_results/combined_eval/pose_vlm_tasks_debug",
        help="Directory to save debug images"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Run test
    results = test_pose_vlm_tasks(
        pose_vlm_checkpoint=args.pose_vlm_checkpoint,
        openvla_checkpoint=args.openvla_checkpoint,
        num_episodes=args.num_episodes,
        max_ik_attempts=args.max_ik_attempts,
        save_debug_images=args.save_debug_images,
        debug_image_dir=args.debug_image_dir,
        device=args.device,
        seed=args.seed,
    )
    
    # Save results
    results_dir = Path("runs/eval_results/combined_eval")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"pose_vlm_tasks_results_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main() 