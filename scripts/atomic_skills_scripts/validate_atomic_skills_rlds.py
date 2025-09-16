#!/usr/bin/env python3
"""
Validate Atomic Skills RLDS Dataset

This script validates the libero_atomic_skills RLDS dataset by:
1. Checking number of demos per skill
2. Verifying action space (OSC delta format)
3. Printing sample actions to verify format
4. Running assertions to ensure dataset correctness
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from collections import defaultdict
import sys
import os

def analyze_atomic_skills_dataset():
    """Analyze the libero_atomic_skills RLDS dataset."""
    
    print("="*80)
    print("🔍 VALIDATING LIBERO ATOMIC SKILLS RLDS DATASET")
    print("="*80)
    
    # Load dataset
    dataset_name = 'libero_atomic_skills'
    data_dir = '/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/rlds_datasets'
    
    try:
        ds = tfds.load(
            dataset_name,
            data_dir=data_dir,
            split='train',
            shuffle_files=False,
            as_supervised=False
        )
        print(f"✅ Successfully loaded dataset: {dataset_name}")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Analyze dataset
    skill_stats = defaultdict(list)
    total_episodes = 0
    total_steps = 0
    action_samples = []
    language_instructions = set()
    
    print(f"\n📊 ANALYZING DATASET...")
    
    for episode in ds:
        total_episodes += 1
        
        # Extract steps
        steps = episode['steps']
        episode_length = len(steps)
        total_steps += episode_length
        
        # Get language instruction from first step
        first_step = next(iter(steps))
        language_instruction = first_step['language_instruction'].numpy().decode('utf-8')
        language_instructions.add(language_instruction)
        skill_stats[language_instruction].append(episode_length)
        
        # Collect action samples (first 3 episodes only for efficiency)
        if total_episodes <= 3:
            for step_idx, step in enumerate(steps.take(3)):  # First 3 steps
                action = step['action'].numpy()
                action_samples.append({
                    'episode': total_episodes,
                    'step': step_idx,
                    'action': action,
                    'skill': language_instruction
                })
        
        # Progress indicator
        if total_episodes % 50 == 0:
            print(f"  Processed {total_episodes} episodes...")
    
    print(f"✅ Analysis complete!")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Total steps: {total_steps}")
    print(f"  Unique skills: {len(language_instructions)}")
    
    # Print skill statistics
    print(f"\n📈 SKILL STATISTICS:")
    print(f"{'Skill Name':<50} {'Episodes':<10} {'Avg Steps':<12} {'Min Steps':<10} {'Max Steps':<10}")
    print("-" * 95)
    
    total_skill_episodes = 0
    for skill, episode_lengths in sorted(skill_stats.items()):
        num_episodes = len(episode_lengths)
        avg_steps = np.mean(episode_lengths)
        min_steps = min(episode_lengths)
        max_steps = max(episode_lengths)
        total_skill_episodes += num_episodes
        
        skill_short = skill[:47] + "..." if len(skill) > 50 else skill
        print(f"{skill_short:<50} {num_episodes:<10} {avg_steps:<12.1f} {min_steps:<10} {max_steps:<10}")
    
    # Verify totals
    assert total_skill_episodes == total_episodes, f"Episode count mismatch: {total_skill_episodes} vs {total_episodes}"
    
    # Analyze actions
    print(f"\n🎮 ACTION ANALYSIS:")
    print(f"Collected {len(action_samples)} action samples for analysis")
    
    # Check action dimensions and ranges
    all_actions = [sample['action'] for sample in action_samples]
    if all_actions:
        actions_array = np.array(all_actions)
        action_dim = actions_array.shape[1]
        
        print(f"  Action dimension: {action_dim}D")
        print(f"  Action dtype: {actions_array.dtype}")
        print(f"  Action shape: {actions_array.shape}")
        
        # Print sample actions
        print(f"\n🔢 SAMPLE ACTIONS (First 3 episodes, first 3 steps each):")
        print(f"{'Episode':<8} {'Step':<6} {'Skill':<30} {'Action'}")
        print("-" * 100)
        
        for sample in action_samples:
            skill_short = sample['skill'][:27] + "..." if len(sample['skill']) > 30 else sample['skill']
            action_str = np.array2string(sample['action'], precision=3, suppress_small=True, separator=', ')
            print(f"{sample['episode']:<8} {sample['step']:<6} {skill_short:<30} {action_str}")
        
        # Action statistics
        print(f"\n📊 ACTION STATISTICS:")
        print(f"  Action ranges per dimension:")
        for i in range(action_dim):
            dim_min = actions_array[:, i].min()
            dim_max = actions_array[:, i].max()
            dim_mean = actions_array[:, i].mean()
            dim_std = actions_array[:, i].std()
            print(f"    Dim {i}: [{dim_min:8.3f}, {dim_max:8.3f}] mean={dim_mean:7.3f} std={dim_std:6.3f}")
        
        # Verify OSC delta format
        print(f"\n🔍 OSC DELTA ACTION VERIFICATION:")
        
        # OSC delta actions should be roughly in range [-1, 1] for position/orientation deltas
        # Last dimension should be gripper command (typically -1 or +1)
        pos_dims = actions_array[:, :3]  # x, y, z deltas
        ori_dims = actions_array[:, 3:6]  # roll, pitch, yaw deltas  
        gripper_dim = actions_array[:, 6]  # gripper command
        
        pos_range = [pos_dims.min(), pos_dims.max()]
        ori_range = [ori_dims.min(), ori_dims.max()]
        gripper_values = np.unique(gripper_dim)
        
        print(f"  Position deltas (dims 0-2): [{pos_range[0]:.3f}, {pos_range[1]:.3f}]")
        print(f"  Orientation deltas (dims 3-5): [{ori_range[0]:.3f}, {ori_range[1]:.3f}]")
        print(f"  Gripper values (dim 6): {gripper_values}")
        
        # Assertions for OSC delta format
        assert action_dim == 7, f"❌ Expected 7D actions (6D pose delta + gripper), got {action_dim}D"
        print(f"  ✅ Action dimension is correct: {action_dim}D")
        
        assert -2.0 <= pos_range[0] and pos_range[1] <= 2.0, f"❌ Position deltas out of expected range [-2, 2]: {pos_range}"
        print(f"  ✅ Position deltas in reasonable range: {pos_range}")
        
        assert -2.0 <= ori_range[0] and ori_range[1] <= 2.0, f"❌ Orientation deltas out of expected range [-2, 2]: {ori_range}"
        print(f"  ✅ Orientation deltas in reasonable range: {ori_range}")
        
        assert len(gripper_values) <= 10, f"❌ Too many unique gripper values: {len(gripper_values)} (expected <= 10)"
        print(f"  ✅ Gripper values look reasonable: {len(gripper_values)} unique values")
        
        # Check for obvious issues
        has_nan = np.isnan(actions_array).any()
        has_inf = np.isinf(actions_array).any()
        
        assert not has_nan, "❌ Found NaN values in actions"
        assert not has_inf, "❌ Found infinite values in actions"
        print(f"  ✅ No NaN or infinite values found")
        
    # Dataset integrity checks
    print(f"\n🛡️  DATASET INTEGRITY CHECKS:")
    
    # Check that all skills have reasonable number of demos
    min_demos_per_skill = min(len(episodes) for episodes in skill_stats.values())
    max_demos_per_skill = max(len(episodes) for episodes in skill_stats.values())
    
    print(f"  Demo count per skill: {min_demos_per_skill} - {max_demos_per_skill}")
    assert min_demos_per_skill > 0, "❌ Some skills have 0 demos"
    print(f"  ✅ All skills have at least 1 demo")
    
    # Check episode lengths
    all_episode_lengths = [length for lengths in skill_stats.values() for length in lengths]
    min_episode_length = min(all_episode_lengths)
    max_episode_length = max(all_episode_lengths)
    avg_episode_length = np.mean(all_episode_lengths)
    
    print(f"  Episode lengths: {min_episode_length} - {max_episode_length} (avg: {avg_episode_length:.1f})")
    assert min_episode_length > 0, "❌ Found episodes with 0 steps"
    assert max_episode_length < 1000, f"❌ Found unreasonably long episodes: {max_episode_length} steps"
    print(f"  ✅ Episode lengths are reasonable")
    
    # Check language instructions
    assert len(language_instructions) > 1, "❌ Dataset should have multiple different skills"
    print(f"  ✅ Found {len(language_instructions)} different skills")
    
    # Check for empty language instructions
    empty_instructions = [instr for instr in language_instructions if not instr.strip()]
    assert len(empty_instructions) == 0, f"❌ Found {len(empty_instructions)} empty language instructions"
    print(f"  ✅ All language instructions are non-empty")
    
    print(f"\n🎉 VALIDATION COMPLETE!")
    print(f"✅ Dataset appears to be correctly formatted")
    print(f"✅ Actions are in OSC delta format (6D pose delta + gripper)")
    print(f"✅ {total_episodes} episodes across {len(language_instructions)} skills")
    print(f"✅ Total of {total_steps} training steps")
    
    print(f"\n📋 SUMMARY:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Episodes: {total_episodes}")
    print(f"  Skills: {len(language_instructions)}")
    print(f"  Steps: {total_steps}")
    print(f"  Action format: 7D OSC delta (✅ Ready for training)")
    
    return True

def main():
    """Main function."""
    try:
        success = analyze_atomic_skills_dataset()
        if success:
            print(f"\n🚀 READY TO TRAIN!")
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()