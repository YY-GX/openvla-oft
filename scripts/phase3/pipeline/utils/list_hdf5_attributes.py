#!/usr/bin/env python3
"""
List all attributes in HDF5 files with their formats and shapes.
"""

import h5py
import numpy as np
import os
import sys
import json

HDF5_DIR = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_above_fewer/all"


def analyze_hdf5_structure(hdf5_path: str):
    """Analyze the structure of an HDF5 file."""
    structure = {
        'file': os.path.basename(hdf5_path),
        'demos': {}
    }
    
    with h5py.File(hdf5_path, 'r') as h5file:
        if 'data' not in h5file:
            return structure
        
        data_group = h5file['data']
        demo_keys = sorted(data_group.keys())
        
        if not demo_keys:
            return structure
        
        # Analyze first demo in detail
        demo_key = demo_keys[0]
        demo_group = data_group[demo_key]
        
        demo_info = {}
        
        # Top-level datasets
        for key in demo_group.keys():
            if isinstance(demo_group[key], h5py.Dataset):
                dataset = demo_group[key]
                demo_info[key] = {
                    'type': 'dataset',
                    'shape': dataset.shape,
                    'dtype': str(dataset.dtype),
                    'format': 'array'
                }
        
        # Observations group
        if 'obs' in demo_group:
            obs_group = demo_group['obs']
            demo_info['obs'] = {}
            for key in obs_group.keys():
                dataset = obs_group[key]
                demo_info['obs'][key] = {
                    'type': 'dataset',
                    'shape': dataset.shape,
                    'dtype': str(dataset.dtype),
                    'format': 'array'
                }
                
                # Special format detection for orientation-related fields
                if 'ee_ori' in key or 'ee_states' in key:
                    if dataset.shape[-1] == 3:
                        demo_info['obs'][key]['orientation_format'] = 'axis-angle (3D vector)'
                    elif dataset.shape[-1] == 4:
                        demo_info['obs'][key]['orientation_format'] = 'quaternion (4D vector)'
                    elif dataset.shape[-1] == 6:
                        demo_info['obs'][key]['orientation_format'] = 'pose (pos 3D + axis-angle 3D)'
        
        # Metadata group
        if 'metadata' in demo_group:
            meta_group = demo_group['metadata']
            demo_info['metadata'] = {
                'attributes': {},
                'datasets': {}
            }
            
            # Attributes
            for key in meta_group.attrs.keys():
                val = meta_group.attrs[key]
                if isinstance(val, str) and val.startswith('{'):
                    try:
                        val = json.loads(val)
                        demo_info['metadata']['attributes'][key] = {
                            'type': 'dict (JSON string)',
                            'value': val
                        }
                    except:
                        demo_info['metadata']['attributes'][key] = {
                            'type': 'string',
                            'value': str(val)
                        }
                else:
                    demo_info['metadata']['attributes'][key] = {
                        'type': type(val).__name__,
                        'value': val
                    }
            
            # Datasets in metadata
            for key in meta_group.keys():
                dataset = meta_group[key]
                demo_info['metadata']['datasets'][key] = {
                    'type': 'dataset',
                    'shape': dataset.shape if hasattr(dataset, 'shape') else 'scalar',
                    'dtype': str(dataset.dtype) if hasattr(dataset, 'dtype') else 'unknown',
                    'format': 'array' if hasattr(dataset, 'shape') and len(dataset.shape) > 0 else 'scalar'
                }
        
        structure['demos'][demo_key] = demo_info
        
        # Count total demos
        structure['total_demos'] = len(demo_keys)
    
    return structure


def main():
    """Main function to analyze HDF5 files."""
    if not os.path.exists(HDF5_DIR):
        print(f"Error: HDF5 directory does not exist: {HDF5_DIR}")
        sys.exit(1)
    
    hdf5_files = [f for f in os.listdir(HDF5_DIR) if f.endswith('.hdf5')]
    hdf5_files = sorted(hdf5_files)
    
    if not hdf5_files:
        print(f"No HDF5 files found in {HDF5_DIR}")
        sys.exit(1)
    
    # Analyze first file as representative
    hdf5_path = os.path.join(HDF5_DIR, hdf5_files[0])
    print("=" * 80)
    print("HDF5 File Structure Analysis")
    print("=" * 80)
    print(f"Analyzing: {os.path.basename(hdf5_path)}")
    print()
    
    structure = analyze_hdf5_structure(hdf5_path)
    
    if 'demos' not in structure or not structure['demos']:
        print("No demos found in file")
        return
    
    demo_key = list(structure['demos'].keys())[0]
    demo_info = structure['demos'][demo_key]
    
    print(f"Total demos in file: {structure.get('total_demos', 0)}")
    print()
    print("=" * 80)
    print("Top-level Attributes")
    print("=" * 80)
    
    for key, info in demo_info.items():
        if key not in ['obs', 'metadata']:
            print(f"\n{key}:")
            print(f"  Type: {info['type']}")
            print(f"  Shape: {info['shape']}")
            print(f"  Dtype: {info['dtype']}")
            print(f"  Format: {info['format']}")
    
    print()
    print("=" * 80)
    print("Observations (obs/)")
    print("=" * 80)
    
    if 'obs' in demo_info:
        for key, info in demo_info['obs'].items():
            print(f"\n{key}:")
            print(f"  Type: {info['type']}")
            print(f"  Shape: {info['shape']}")
            print(f"  Dtype: {info['dtype']}")
            print(f"  Format: {info['format']}")
            if 'orientation_format' in info:
                print(f"  Orientation Format: {info['orientation_format']}")
            
            # Special note for ee_states
            if key == 'ee_states':
                print(f"  Note: Contains [ee_pos (3D) + ee_ori (axis-angle 3D)] = 6D pose")
            elif key == 'ee_ori':
                print(f"  Note: Axis-angle representation (3D vector)")
            elif key == 'ee_pos':
                print(f"  Note: 3D position (x, y, z)")
    
    print()
    print("=" * 80)
    print("Metadata (metadata/)")
    print("=" * 80)
    
    if 'metadata' in demo_info:
        print("\nAttributes:")
        for key, info in demo_info['metadata']['attributes'].items():
            print(f"  {key}: {info['type']}")
            if 'value' in info and not isinstance(info['value'], dict):
                print(f"    Value: {info['value']}")
        
        print("\nDatasets:")
        for key, info in demo_info['metadata']['datasets'].items():
            print(f"  {key}:")
            print(f"    Type: {info['type']}")
            print(f"    Shape: {info['shape']}")
            print(f"    Dtype: {info['dtype']}")
    
    print()
    print("=" * 80)
    print("Summary: All Attributes and Formats")
    print("=" * 80)
    print()
    print("Top-level datasets:")
    print("  - actions: (T, 7) - float64 - OSC delta commands [delta_pos(3), delta_ori(3), gripper(1)]")
    print("  - dones: (T,) - uint8 - Episode termination flags")
    print("  - rewards: (T,) - uint8 - Reward signals")
    print("  - robot_states: (T, 9) - float32 - First 9 elements of proprio state")
    print("  - states: (T, 84) - float32 - Full MuJoCo simulation state (qpos + qvel)")
    print()
    print("Observations (obs/):")
    print("  - joint_states: (T, 7) - float32 - 7-DOF joint positions")
    print("  - gripper_states: (T, 2) - float32 - 2-element gripper positions")
    print("  - ee_pos: (T, 3) - float32 - 3D end-effector position (x, y, z)")
    print("  - ee_ori: (T, 3) - float32 - Axis-angle orientation (3D vector)")
    print("  - ee_states: (T, 6) - float32 - Combined pose [pos(3) + axis-angle(3)]")
    print("  - agentview_rgb: (T, H, W, 3) - uint8 - Agent viewpoint RGB image")
    print("  - eye_in_hand_rgb: (T, H, W, 3) - uint8 - Wrist camera RGB image")
    print()
    print("Metadata (metadata/):")
    print("  - above_pose: (6,) - float32 - Above pose [pos(3) + axis-angle(3)]")
    print("  - target_timestep: int - Target timestep from original demo")
    print("  - trigger_timestep: int - Timestep when above region was reached")
    print("  - iteration: int - Augmentation iteration (0 = non-shifted, >0 = shifted)")
    print("  - shifted: bool - Whether this demo was shifted")
    print("  - shift_info: dict - Shift parameters (xy_shift, z_shift, ori_shift)")
    print()
    print("=" * 80)
    print("Key Points:")
    print("=" * 80)
    print("1. ee_ori and ee_states use AXIS-ANGLE format (3D vector), NOT Euler angles")
    print("2. Axis-angle is converted from quaternion using quat2axisangle()")
    print("3. states contains full MuJoCo state (qpos + qvel) with quaternions in qpos")
    print("4. All orientation in obs/ is axis-angle format")
    print("=" * 80)


if __name__ == "__main__":
    main()

