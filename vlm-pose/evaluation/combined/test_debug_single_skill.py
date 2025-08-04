#!/usr/bin/env python3
"""
test_debug_single_skill.py

Test script to run combined evaluation in debug mode for a single skill.
"""

import sys
import os

# Add project root to path for PoseFinetuneConfig import
sys.path.append('../../..')

# Import PoseFinetuneConfig early to make it available for pickle
try:
    from finetune_pose import PoseFinetuneConfig
except ImportError:
    # Fallback if import fails
    class PoseFinetuneConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

# Add current directory to path
sys.path.append('.')

from run_combined_eval import CombinedEvaluator

def main():
    """Run debug evaluation for single skill."""
    
    # Configuration for debug mode
    pose_vlm_checkpoint = "../../runs/pose_vlm/1.0.0/openvla-7b+pose_dataset+b8+lr-1e-05+pose_simple+lora-r32+dropout-0.0--image_aug--pose_aug--simple_pose_head_pose_aug--smaller_splits/vla--20000_checkpoint.pt"
    openvla_checkpoint = "../../runs/view_wrist/1.0.3/openvla-7b+libero44_local+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt"
    
    print("="*60)
    print("DEBUG MODE: Single Skill Evaluation")
    print("="*60)
    print(f"PoseVLM Checkpoint: {pose_vlm_checkpoint}")
    print(f"OpenVLA Checkpoint: {openvla_checkpoint}")
    print(f"Debug Skill: KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet")
    print(f"Training Image Comparison: ENABLED")
    print(f"Images per episode: 5 (reset, ik_reset, ik_training, initial_state, final)")
    print(f"Video per episode: 1 (OpenVLA-OFT execution)")
    print(f"Initial State: Training Image IK (if available)")
    print("="*60)
    
    # Create evaluator in debug mode
    evaluator = CombinedEvaluator(
        pose_vlm_checkpoint=pose_vlm_checkpoint,
        openvla_checkpoint=openvla_checkpoint,
        task_suite_name="boss_44",
        num_episodes_per_task=5,  # Reduced for debug
        max_ik_attempts=5,
        save_debug_images=True,
        debug_image_dir="runs/eval_results/combined_eval/debug_single_skill",
        device="cuda",
        seed=42,
        use_pose_vlm_smaller_split_tasks_only=False,
        wrist_only=True,
        third_person_view_only=False,
        both_views=False,
        debug_single_skill=True,
        debug_skill_name="KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
        compare_training_image=True,  # Enable training image comparison
        training_data_root="../../datasets/local_pairs_datasets",
    )
    
    # Run evaluation
    print("\nStarting debug evaluation...")
    results = evaluator.run_evaluation()
    
    # Print results
    print("\n" + "="*60)
    print("DEBUG EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Success Rate: {results['overall_metrics']['success_rate']:.2%}")
    print(f"IK Success Rate: {results['overall_metrics']['ik_success_rate']:.2%}")
    print(f"VLA Success Rate: {results['overall_metrics']['vla_success_rate']:.2%}")
    print(f"Total Episodes: {results['overall_metrics']['total_episodes']}")
    print(f"Successful Episodes: {results['overall_metrics']['successful_episodes']}")
    print(f"IK Failures: {results['overall_metrics']['ik_failures']}")
    print(f"VLA Failures: {results['overall_metrics']['vla_failures']}")
    print("="*60)
    
    # Print per-task results
    for task_id, task_results in results['per_task_results'].items():
        task_name = task_results['task_name']
        print(f"\nTask: {task_name}")
        print(f"  Success Rate: {task_results['success_rate']:.2%}")
        print(f"  IK Success Rate: {task_results['ik_success_rate']:.2%}")
        print(f"  VLA Success Rate: {task_results['vla_success_rate']:.2%}")
        print(f"  Episodes: {task_results['total_episodes']}")
        print(f"  Successful: {task_results['successful_episodes']}")
        print(f"  IK Failures: {task_results['ik_failures']}")
        print(f"  VLA Failures: {task_results['vla_failures']}")
    
    print("\nDebug evaluation complete!")
    print(f"Debug images saved to: runs/eval_results/combined_eval/debug_single_skill")

if __name__ == "__main__":
    main() 