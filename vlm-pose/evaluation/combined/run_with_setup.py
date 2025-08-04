#!/usr/bin/env python3
"""
Script to run combined evaluation with proper environment setup
"""

import sys
import os

# Add project root to path
sys.path.append('../../..')

# Import PoseFinetuneConfig early to make it available for pickle
from finetune_pose import PoseFinetuneConfig

# Now import the test function
from test_combined_pipeline import test_single_task

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_vlm_checkpoint", type=str, required=True)
    parser.add_argument("--openvla_checkpoint", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=16)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_ik_attempts", type=int, default=1)
    
    args = parser.parse_args()
    
    print("Starting combined evaluation with proper environment setup...")
    
    results = test_single_task(
        pose_vlm_checkpoint=args.pose_vlm_checkpoint,
        openvla_checkpoint=args.openvla_checkpoint,
        task_id=args.task_id,
        num_episodes=args.num_episodes,
        max_ik_attempts=args.max_ik_attempts,
    )
    
    print("Test completed successfully!")
    print(f"Results: {results}") 