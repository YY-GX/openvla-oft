#!/usr/bin/env python3
"""
Simple test script to debug checkpoint loading
"""

import torch
import sys
import os

# Add the path to vla-scripts
sys.path.append('../../vla-scripts')

# Try to import PoseFinetuneConfig
try:
    from finetune_pose import PoseFinetuneConfig
    print("Successfully imported PoseFinetuneConfig")
except ImportError as e:
    print(f"Failed to import PoseFinetuneConfig: {e}")
    # Create a fallback class
    class PoseFinetuneConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    print("Created fallback PoseFinetuneConfig")

# Test loading the checkpoint
checkpoint_path = "../../runs/pose_vlm/1.0.0/openvla-7b+pose_dataset+b8+lr-1e-05+pose_simple+lora-r32+dropout-0.0--image_aug--pose_aug--simple_pose_head_pose_aug--smaller_splits/vla--20000_checkpoint.pt"

print(f"Testing checkpoint loading: {checkpoint_path}")
print(f"Checkpoint exists: {os.path.exists(checkpoint_path)}")

try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print("Successfully loaded checkpoint!")
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    if "config" in checkpoint:
        print(f"Config type: {type(checkpoint['config'])}")
        print(f"Config attributes: {dir(checkpoint['config'])}")
    else:
        print("No config found in checkpoint")
except Exception as e:
    print(f"Failed to load checkpoint: {e}")
    import traceback
    traceback.print_exc() 