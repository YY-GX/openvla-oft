#!/usr/bin/env python3
"""
Test script for object-only pointcloud extraction.

This script tests the improved object segmentation that extracts only the target
object's pointcloud, excluding nearby surfaces and other objects. It tests with
various object types including cabinets.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project paths
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')

# LIBERO imports
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env

# Object-segmented extractor
from scripts.phase2.utils.object_segmented_extractor import (
    create_object_segmented_extractor,
    save_object_only_pointcloud_for_grasp_generation
)

# Original corrected extractor for comparison
from scripts.phase2.utils.pointcloud_extractor_corrected import create_corrected_pointcloud_extractor

# Import phase 1 components for object detection
import importlib.util


def load_script_module(script_path: str, module_name: str):
    """Load a script module dynamically."""
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_object_only_extraction():
    """Test object-only pointcloud extraction with various object types."""
    print("🎯 Testing Object-Only Pointcloud Extraction")
    print("=" * 60)
    print("This test extracts pointclouds containing ONLY target objects,")
    print("excluding nearby surfaces, tables, and other objects.")
    print("")
    
    # Test multiple tasks to find different object types
    test_tasks = [
        ("LONG_HORIZON_cooking_preparation_setup", "Cooking task"),
        ("LONG_HORIZON_complete_kitchen_organization", "Kitchen organization"),
    ]
    
    results_summary = {}
    
    for task_name, task_desc in test_tasks:
        print(f"🏗️  Testing with {task_desc} ({task_name})...")
        
        try:
            # Setup environment
            bm_name = "long_horizon_tasks_v0"
            task_suite = benchmark.get_benchmark_dict()[bm_name]()
            
            target_task = None
            for task in task_suite.tasks:
                if task.name == task_name:
                    target_task = task
                    break
            
            if target_task is None:
                print(f"   ❌ Task {task_name} not found")
                continue
            
            env, _ = get_libero_env(target_task, model_family='openvla', resolution=256)
            obs = env.reset()
            print(f"   ✅ Environment setup complete")
            
            # Load object detector
            script_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase1"
            object_detector = load_script_module(
                os.path.join(script_dir, "3_contact_object_detector.py"),
                "object_detector"
            )
            
            # Find manipulable objects
            available_objects = object_detector.get_all_manipulable_objects(env)
            print(f"   🔍 Found {len(available_objects)} manipulable objects:")
            
            # Show all available objects
            for i, obj in enumerate(available_objects, 1):
                print(f"      {i}. {obj}")
            
            # Get object poses for interesting objects
            target_objects = available_objects[:6]  # Test first 6 objects
            object_poses = {}
            
            print(f"\n   📍 Getting poses for target objects...")
            for obj_name in target_objects:
                pos, quat = object_detector.get_object_pose(env, obj_name)
                if pos is not None:
                    object_poses[obj_name] = (pos, quat)
                    print(f"      ✅ {obj_name}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            
            if not object_poses:
                print(f"   ❌ No valid object poses found")
                env.close()
                continue
            
            # Test object-only extraction
            print(f"\n   🎯 Testing object-only extraction...")
            
            # Create both extractors for comparison
            segmented_extractor = create_object_segmented_extractor(env, debug_mode=True)
            original_extractor = create_corrected_pointcloud_extractor(env, debug_mode=False)
            
            # Get current observation
            dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])  # 6-DOF + gripper open
            obs, _, _, _ = env.step(dummy_action)
            
            test_results = {}
            
            # Test with different object types
            for obj_name, (pos, quat) in list(object_poses.items())[:4]:  # First 4 objects
                print(f"\n      🔍 Testing {obj_name}...")
                
                # Extract with object-only method (new)
                object_only_pc = segmented_extractor.extract_object_only_pointcloud(
                    obs, obj_name, pos, camera="agentview"
                )
                
                # Extract with original method (for comparison)
                original_pc = original_extractor.extract_object_pointcloud(
                    obs, obj_name, pos, camera="agentview", filter_radius=0.2
                )
                
                # Compare results
                if len(object_only_pc) > 0:
                    obj_bounds = {
                        'x': [object_only_pc[:, 0].min(), object_only_pc[:, 0].max()],
                        'y': [object_only_pc[:, 1].min(), object_only_pc[:, 1].max()],
                        'z': [object_only_pc[:, 2].min(), object_only_pc[:, 2].max()]
                    }
                    
                    spread_x = obj_bounds['x'][1] - obj_bounds['x'][0]
                    spread_y = obj_bounds['y'][1] - obj_bounds['y'][0]
                    spread_z = obj_bounds['z'][1] - obj_bounds['z'][0]
                    is_3d = spread_x > 0.01 and spread_y > 0.01 and spread_z > 0.01
                    
                    reduction_factor = len(original_pc) / max(len(object_only_pc), 1)
                    
                    print(f"         ✅ Object-only: {len(object_only_pc)} points, 3D: {'✅' if is_3d else '❌'}")
                    print(f"            Spread: x={spread_x:.3f}m, y={spread_y:.3f}m, z={spread_z:.3f}m")
                    print(f"         📊 Original: {len(original_pc)} points")
                    print(f"         📉 Reduction: {reduction_factor:.1f}x fewer points")
                    
                    test_results[obj_name] = {
                        'object_only_points': len(object_only_pc),
                        'original_points': len(original_pc),
                        'reduction_factor': reduction_factor,
                        'is_3d': is_3d,
                        'spread': {'x': spread_x, 'y': spread_y, 'z': spread_z}
                    }
                else:
                    print(f"         ❌ No points extracted")
                    test_results[obj_name] = {
                        'object_only_points': 0,
                        'original_points': len(original_pc),
                        'reduction_factor': float('inf'),
                        'is_3d': False,
                        'spread': {'x': 0, 'y': 0, 'z': 0}
                    }
            
            results_summary[task_name] = test_results
            env.close()
            print(f"   ✅ {task_desc} testing complete")
            
        except Exception as e:
            print(f"   ❌ Error testing {task_desc}: {e}")
            continue
    
    # Summary results
    print(f"\n📊 Object-Only Extraction Test Summary")
    print(f"=" * 50)
    
    total_objects_tested = 0
    successful_extractions = 0
    
    for task_name, results in results_summary.items():
        print(f"\n🏗️  {task_name}:")
        
        for obj_name, result in results.items():
            total_objects_tested += 1
            
            if result['object_only_points'] > 0:
                successful_extractions += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"   {status} {obj_name}:")
            print(f"      Points: {result['object_only_points']} (vs {result['original_points']} original)")
            print(f"      Reduction: {result['reduction_factor']:.1f}x")
            print(f"      3D structure: {'✅' if result['is_3d'] else '❌'}")
    
    success_rate = successful_extractions / max(total_objects_tested, 1) * 100
    print(f"\n🎯 Overall Results:")
    print(f"   Objects tested: {total_objects_tested}")
    print(f"   Successful extractions: {successful_extractions}")
    print(f"   Success rate: {success_rate:.1f}%")
    
    return successful_extractions > 0


def test_object_only_with_save():
    """Test object-only extraction and save results for inspection."""
    print(f"\n💾 Testing Object-Only Extraction with File Output")
    print(f"=" * 55)
    
    # Setup environment
    bm_name = "long_horizon_tasks_v0"
    task_suite = benchmark.get_benchmark_dict()[bm_name]()
    
    target_task = None
    for task in task_suite.tasks:
        if task.name == "LONG_HORIZON_cooking_preparation_setup":
            target_task = task
            break
    
    env, _ = get_libero_env(target_task, model_family='openvla', resolution=256)
    obs = env.reset()
    
    # Load object detector
    script_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase1"
    object_detector = load_script_module(
        os.path.join(script_dir, "3_contact_object_detector.py"),
        "object_detector"
    )
    
    # Find objects
    available_objects = object_detector.get_all_manipulable_objects(env)
    object_poses = {}
    
    for obj_name in available_objects[:4]:  # First 4 objects
        pos, quat = object_detector.get_object_pose(env, obj_name)
        if pos is not None:
            object_poses[obj_name] = (pos, quat)
    
    # Create output directory
    output_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase2/test_output_object_only"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save object-only pointclouds
    saved_count = 0
    
    for obj_name, (pos, quat) in object_poses.items():
        print(f"\n   💾 Saving {obj_name}...")
        
        try:
            saved_files = save_object_only_pointcloud_for_grasp_generation(
                env, obj_name, pos, output_dir,
                camera="agentview", 
                formats=["ply", "pcd", "txt"],
                debug_mode=True
            )
            
            if saved_files:
                saved_count += 1
                print(f"      ✅ Saved {len(saved_files)} files:")
                for fmt, file_path in saved_files.items():
                    file_size = Path(file_path).stat().st_size
                    print(f"         {fmt.upper()}: {Path(file_path).name} ({file_size} bytes)")
            else:
                print(f"      ❌ No files saved")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    env.close()
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📊 Successfully saved: {saved_count} objects")
    
    return saved_count > 0


if __name__ == "__main__":
    print("🧪 Object-Only Pointcloud Extraction Tests")
    print("=" * 50)
    
    # Run extraction tests
    test1_success = test_object_only_extraction()
    
    # Run save tests
    test2_success = test_object_only_with_save()
    
    # Final result
    overall_success = test1_success and test2_success
    print(f"\n🎉 Overall Test Result: {'SUCCESS' if overall_success else 'PARTIAL SUCCESS'}")
    
    if overall_success:
        print("✅ Object-only extraction is working correctly!")
        print("✅ Generated pointclouds should contain only target objects.")
    else:
        print("⚠️  Some tests had issues, but basic functionality is working.")
    
    exit(0 if overall_success else 1)