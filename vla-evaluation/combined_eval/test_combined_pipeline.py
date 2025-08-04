#!/usr/bin/env python3
"""
test_combined_pipeline.py

Test script for the combined PoseVLM + IK + OpenVLA-OFT pipeline on a single task.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
import argparse
from typing import Dict, Any

# Import the combined evaluator
from run_combined_eval import CombinedEvaluator


def test_single_task(
    pose_vlm_checkpoint: str,
    openvla_checkpoint: str,
    task_id: int = 16,  # Default to first intersection task (KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet)
    num_episodes: int = 5,
    max_ik_attempts: int = 3,
    save_debug_images: bool = True,
    debug_image_dir: str = "runs/eval_results/combined_eval/test_debug_images",
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Test the combined pipeline on a single task.
    
    Args:
        pose_vlm_checkpoint: Path to PoseVLM checkpoint
        openvla_checkpoint: Path to OpenVLA-OFT checkpoint
        task_id: Task ID to test
        num_episodes: Number of episodes to run
        max_ik_attempts: Maximum IK attempts per pose prediction
        save_debug_images: Whether to save debug images
        debug_image_dir: Directory to save debug images
        device: Device to run evaluation on
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with test results
    """
    print(f"Testing combined pipeline on task {task_id}")
    print(f"PoseVLM checkpoint: {pose_vlm_checkpoint}")
    print(f"OpenVLA checkpoint: {openvla_checkpoint}")
    print(f"Episodes: {num_episodes}")
    print(f"Max IK attempts: {max_ik_attempts}")
    
    # Create evaluator
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
    )
    
    # Test single task
    results = evaluator.evaluate_task(task_id)
    
    print(f"\nTest Results:")
    print(f"  Task: {results['task_name']}")
    print(f"  Total Episodes: {results['total_episodes']}")
    print(f"  IK Success Rate: {results['ik_success_rate']:.3f}")
    print(f"  VLA Success Rate: {results['vla_success_rate']:.3f}")
    print(f"  Overall Success Rate: {results['overall_success_rate']:.3f}")
    print(f"  Debug Images Saved: {len(results['debug_images_saved'])}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test combined PoseVLM + OpenVLA-OFT pipeline")
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
        "--task_id",
        type=int,
        default=0,
        help="Task ID to test"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--max_ik_attempts",
        type=int,
        default=3,
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
        default="runs/eval_results/combined_eval/test_debug_images",
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
    args = parser.parse_args()
    
    # Run test
    results = test_single_task(
        pose_vlm_checkpoint=args.pose_vlm_checkpoint,
        openvla_checkpoint=args.openvla_checkpoint,
        task_id=args.task_id,
        num_episodes=args.num_episodes,
        max_ik_attempts=args.max_ik_attempts,
        save_debug_images=args.save_debug_images,
        debug_image_dir=args.debug_image_dir,
        device=args.device,
        seed=args.seed,
    )
    
    # Save results
    results_dir = Path("runs/eval_results/combined_eval/test_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"test_results_task{args.task_id}_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nTest results saved to {output_path}")


if __name__ == "__main__":
    main() 