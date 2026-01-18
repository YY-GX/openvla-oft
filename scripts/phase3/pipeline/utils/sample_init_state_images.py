#!/usr/bin/env python3
"""
Sample 1 init state per skill from .init files and save the first observation image.

Usage:
    python scripts/phase3/pipeline/utils/sample_init_state_images.py
"""

import sys
import os
import pickle
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from PIL import Image

# Add project paths
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')

from libero.libero.envs import OffScreenRenderEnv


def load_bddl_to_init_mapping() -> Dict:
    """Load BDDL to init mapping."""
    mapping_path = "scripts/phase3/pipeline/config/bddl_to_init_mapping.json"
    with open(mapping_path, 'r') as f:
        return json.load(f)


def find_bddl_from_skill(skill_name: str, mapping: Dict) -> Optional[str]:
    """Find BDDL file from skill name (returns first match)."""
    for bddl_name, init_name in mapping.items():
        if init_name == skill_name:
            return f"{bddl_name}.bddl"
    return None


def infer_skill_type(skill_name: str) -> str:
    """Infer skill type from name."""
    if 'pick' in skill_name.lower():
        return 'pick'
    elif 'place' in skill_name.lower():
        return 'place'
    else:
        return 'other'


def main():
    # Paths
    init_dir = Path("datasets/hdf5_datasets/atomic_above_fewer/all")
    # init_dir = Path("datasets/hdf5_datasets/atomic_above_fewer/v2")
    bddl_dir = Path("externals/boss/libero/libero/bddl_files/atomic_skills")
    output_dir = Path("scripts/phase3/pipeline/outputs/images/demo_gen/dataset_samples_all")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load BDDL mapping
    bddl_mapping = load_bddl_to_init_mapping()
    
    # Find all .init files
    init_files = sorted(list(init_dir.glob("*.init")))
    
    if len(init_files) == 0:
        print(f"❌ No .init files found in {init_dir}")
        return
    
    print(f"📁 Found {len(init_files)} .init files")
    print(f"💾 Output directory: {output_dir}")
    print(f"{'='*80}")
    
    success_count = 0
    failed_count = 0
    
    # Process each .init file
    for init_file in init_files:
        skill_name = init_file.stem  # Get skill name from filename (without .init)
        print(f"\n[{success_count + failed_count + 1}/{len(init_files)}] Processing: {skill_name}")
        
        try:
            # Load init states
            with open(init_file, 'rb') as f:
                init_states = pickle.load(f)
            
            if len(init_states) == 0:
                print(f"  ⚠️  No init states found in {init_file}")
                failed_count += 1
                continue
            
            # Sample first init state (index 0)
            init_state = init_states[0]

            # Detect skill type
            skill_type = infer_skill_type(skill_name)
            print(f"  📊 Loaded {len(init_states)} init states, using first one (type: {skill_type})")

            # Find BDDL file
            bddl_file = find_bddl_from_skill(skill_name, bddl_mapping)
            if bddl_file is None:
                print(f"  ❌ BDDL mapping not found for: {skill_name}")
                failed_count += 1
                continue
            
            bddl_path = bddl_dir / bddl_file
            if not bddl_path.exists():
                print(f"  ❌ BDDL file not found: {bddl_path}")
                failed_count += 1
                continue
            
            print(f"  📄 BDDL: {bddl_file}")
            
            # Create environment
            env_args = {
                'bddl_file_name': str(bddl_path),
                'camera_heights': 256,
                'camera_widths': 256,
                'has_renderer': False,
                'has_offscreen_renderer': True,
                'ignore_done': True,
                'use_camera_obs': True,
                'control_freq': 20,
                'camera_names': ['agentview', 'robot0_eye_in_hand']
            }
            
            env = OffScreenRenderEnv(**env_args)
            env.reset()

            # Use skill_type detected earlier
            is_place = (skill_type == 'place')

            # Handle place skills: disable gravity, set state, close gripper, enable gravity
            # This matches the preprocessing in evaluate_atomic_skills.py
            if is_place:
                old_gravity = env.sim.model.opt.gravity.copy()
                env.sim.model.opt.gravity[:] = 0
                env.sim.forward()

            # Set init state (full sim state)
            env.set_init_state(init_state)
            obs = env.env._get_observations()

            if is_place:
                # Close gripper (5 steps)
                for _ in range(5):
                    obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))

                # Re-enable gravity
                env.sim.model.opt.gravity[:] = old_gravity
                env.sim.forward()
                obs = env.env._get_observations()

            # Get agentview image
            if 'agentview_image' not in obs:
                print(f"  ❌ No agentview_image in observation")
                env.close()
                failed_count += 1
                continue
            
            image = obs['agentview_image']
            
            # Ensure image is in correct format (uint8, 0-255)
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Save image
            image_filename = f"{skill_name}.png"
            image_path = output_dir / image_filename
            
            # Convert to PIL Image and save
            if len(image.shape) == 3:
                img = Image.fromarray(image, 'RGB')
            elif len(image.shape) == 2:
                img = Image.fromarray(image, 'L').convert('RGB')
            else:
                print(f"  ❌ Unexpected image shape: {image.shape}")
                env.close()
                failed_count += 1
                continue
            
            img.save(image_path)
            print(f"  ✅ Saved: {image_filename} (shape: {image.shape})")
            
            env.close()
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ Error processing {skill_name}: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"✅ SUMMARY")
    print(f"{'='*80}")
    print(f"Success: {success_count}/{len(init_files)}")
    print(f"Failed:  {failed_count}/{len(init_files)}")
    print(f"Output:  {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

