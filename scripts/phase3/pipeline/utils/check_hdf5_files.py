#!/usr/bin/env python3
"""
Check all success HDF5 files in atomic_above_fewer/v* directories for corruption or issues.

This script:
1. Finds all HDF5 files in the v* directories
2. Tries to open each file and read basic structure
3. Reports which files have issues
4. Saves results to a JSON file
"""

import os
import json
import h5py
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Base directory containing v* folders
BASE_DIR = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_above_fewer"

# Output directory for results
OUTPUT_DIR = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase3/pipeline/outputs/results"

def check_hdf5_file(file_path: str) -> Tuple[bool, str, Dict]:
    """
    Check if an HDF5 file can be opened and read.
    
    Args:
        file_path: Path to HDF5 file
        
    Returns:
        (is_valid, error_message, file_info) tuple
        - is_valid: True if file is valid, False otherwise
        - error_message: Error message if file is invalid, empty string otherwise
        - file_info: Dictionary with file metadata (keys, shape info, etc.)
    """
    file_info = {
        "file_path": file_path,
        "file_size": 0,
        "keys": [],
        "num_demos": 0,
        "has_obs": False,
        "has_actions": False,
        "has_states": False,
    }
    
    try:
        # Check file size
        if not os.path.exists(file_path):
            return False, "File does not exist", file_info
        
        file_info["file_size"] = os.path.getsize(file_path)
        
        # Try to open the file
        with h5py.File(file_path, 'r') as h5file:
            # Get all top-level keys
            file_info["keys"] = list(h5file.keys())
            
            # Check for expected structure
            if 'data' in h5file:
                data_group = h5file['data']
                file_info["num_demos"] = len(data_group) if hasattr(data_group, '__len__') else 0
                
                # Check first demo if available
                if file_info["num_demos"] > 0:
                    demo_key = list(data_group.keys())[0]
                    demo_group = data_group[demo_key]
                    
                    if 'obs' in demo_group:
                        file_info["has_obs"] = True
                        obs_keys = list(demo_group['obs'].keys())
                        file_info["obs_keys"] = obs_keys
                    
                    if 'actions' in demo_group:
                        file_info["has_actions"] = True
                        actions = demo_group['actions']
                        if hasattr(actions, 'shape'):
                            file_info["actions_shape"] = actions.shape
                    
                    if 'states' in demo_group:
                        file_info["has_states"] = True
                        states = demo_group['states']
                        if hasattr(states, 'shape'):
                            file_info["states_shape"] = states.shape
            
            # Try to read a small portion to ensure file is not corrupted
            # Just accessing the keys should be enough to detect major corruption
            
        return True, "", file_info
        
    except OSError as e:
        error_msg = f"OSError: {str(e)}"
        return False, error_msg, file_info
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        return False, error_msg, file_info


def find_hdf5_files(base_dir: str) -> List[str]:
    """
    Find all HDF5 files in v* directories (excluding failure files).
    
    Args:
        base_dir: Base directory containing v* folders
        
    Returns:
        List of HDF5 file paths
    """
    hdf5_files = []
    
    # Find all v* directories
    v_dirs = glob.glob(os.path.join(base_dir, "v*"))
    v_dirs.sort()
    
    for v_dir in v_dirs:
        # Find all HDF5 files in this directory (excluding _failure.hdf5 files)
        files = glob.glob(os.path.join(v_dir, "*.hdf5"))
        # Filter out failure files
        success_files = [f for f in files if "_failure" not in os.path.basename(f)]
        hdf5_files.extend(success_files)
    
    return sorted(hdf5_files)


def main():
    """Main function to check all HDF5 files and generate report."""
    print("=" * 80)
    print("HDF5 File Checker")
    print("=" * 80)
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all HDF5 files
    print("Finding HDF5 files...")
    hdf5_files = find_hdf5_files(BASE_DIR)
    print(f"Found {len(hdf5_files)} HDF5 files to check")
    print()
    
    # Check each file
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_directory": BASE_DIR,
        "total_files": len(hdf5_files),
        "valid_files": [],
        "invalid_files": [],
        "summary": {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "total_size_bytes": 0,
            "valid_size_bytes": 0,
            "invalid_size_bytes": 0
        }
    }
    
    print("Checking files...")
    for i, file_path in enumerate(hdf5_files, 1):
        file_name = os.path.basename(file_path)
        v_dir = os.path.basename(os.path.dirname(file_path))
        print(f"[{i}/{len(hdf5_files)}] Checking {v_dir}/{file_name}...", end=" ")
        
        is_valid, error_msg, file_info = check_hdf5_file(file_path)
        
        if is_valid:
            print("✓ OK")
            results["valid_files"].append(file_info)
            results["summary"]["valid"] += 1
            results["summary"]["valid_size_bytes"] += file_info["file_size"]
        else:
            print(f"✗ ERROR: {error_msg}")
            file_info["error"] = error_msg
            results["invalid_files"].append(file_info)
            results["summary"]["invalid"] += 1
            results["summary"]["invalid_size_bytes"] += file_info["file_size"]
        
        results["summary"]["total"] += 1
        results["summary"]["total_size_bytes"] += file_info["file_size"]
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total files: {results['summary']['total']}")
    print(f"Valid files: {results['summary']['valid']}")
    print(f"Invalid files: {results['summary']['invalid']}")
    print(f"Total size: {results['summary']['total_size_bytes'] / (1024**3):.2f} GB")
    print(f"Valid size: {results['summary']['valid_size_bytes'] / (1024**3):.2f} GB")
    print(f"Invalid size: {results['summary']['invalid_size_bytes'] / (1024**3):.2f} GB")
    print()
    
    # Save results to JSON
    output_file = os.path.join(OUTPUT_DIR, "hdf5_check_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Print invalid files if any
    if results["invalid_files"]:
        print()
        print("=" * 80)
        print("Invalid Files:")
        print("=" * 80)
        for file_info in results["invalid_files"]:
            file_path = file_info["file_path"]
            v_dir = os.path.basename(os.path.dirname(file_path))
            file_name = os.path.basename(file_path)
            error = file_info.get("error", "Unknown error")
            print(f"  {v_dir}/{file_name}: {error}")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()

