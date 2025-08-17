import sys
import os
sys.path.append('../..')
sys.path.append('../data_preparation/v2_pipeline')

import pytest
import subprocess
import tempfile
import json
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

"""
Test Runner for Dataset Curation Pipeline
This script runs all tests and validates the complete dataset curation pipeline.
"""

class TestPipelineIntegration:
    """Test the complete dataset curation pipeline integration."""
    
    def create_minimal_test_environment(self, tmp_dir):
        """Create a minimal test environment for pipeline testing."""
        # Create single skill tasks file
        tasks_file = os.path.join(tmp_dir, "single_skill_tasks_44.json")
        test_tasks = ["KITCHEN_SCENE1_test_task"]
        with open(tasks_file, 'w') as f:
            json.dump(test_tasks, f)
        
        # Create raw demo directory and file
        raw_demo_dir = os.path.join(tmp_dir, "raw_demos")
        os.makedirs(raw_demo_dir, exist_ok=True)
        
        demo_file = os.path.join(raw_demo_dir, "KITCHEN_SCENE1_test_task_demo.hdf5")
        self.create_test_demo_file(demo_file)
        
        return tasks_file, raw_demo_dir, demo_file
    
    def create_test_demo_file(self, demo_file):
        """Create a test demo HDF5 file."""
        with h5py.File(demo_file, 'w') as f:
            data_group = f.create_group('data')
            
            # Create multiple demos
            for demo_idx in range(3):
                demo_group = data_group.create_group(f'demo_{demo_idx}')
                
                # Create obs group
                obs_group = demo_group.create_group('obs')
                
                # Create mock EE poses
                n_timesteps = 10
                ee_pos = np.random.rand(n_timesteps, 3) * 0.1  # Small workspace
                ee_ori = np.random.rand(n_timesteps, 3) * 0.1  # Small rotations
                obs_group.create_dataset('ee_pos', data=ee_pos)
                obs_group.create_dataset('ee_ori', data=ee_ori)
                
                # Create mock images
                agentview_rgb = np.random.randint(0, 255, (n_timesteps, 64, 64, 3), dtype=np.uint8)
                obs_group.create_dataset('agentview_rgb', data=agentview_rgb)
                
                # Create mock states (required for contact detection)
                states = np.zeros((n_timesteps, 100))  # Placeholder state data
                demo_group.create_dataset('states', data=states)
    
    def test_pipeline_file_structure(self):
        """Test that all pipeline files exist and are importable."""
        # Check main scripts exist
        script_dir = Path(__file__).parent.parent / "data_preparation" / "v2_pipeline"
        
        required_files = [
            "detect_contact_timesteps_v2.py",
            "extract_local_pairs_v2.py", 
            "validate_pose_similarity.py",
            "single_skill_tasks_44.json"
        ]
        
        for filename in required_files:
            file_path = script_dir / filename
            assert file_path.exists(), f"Required file missing: {filename}"
        
        # Test imports
        try:
            import detect_contact_timesteps_v2
            import extract_local_pairs_v2
            import validate_pose_similarity
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import pipeline modules: {e}")
    
    def test_contact_detection_mock_integration(self):
        """Test contact detection with mocked environment (safe for CI)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test environment
            tasks_file, raw_demo_dir, demo_file = self.create_minimal_test_environment(tmp_dir)
            contact_file = os.path.join(tmp_dir, "contact_timesteps.json")
            
            # Import and create a simplified version that doesn't require libero
            from detect_contact_timesteps_v2 import load_hdf5_to_dict, load_single_skill_tasks
            
            # Test data loading functions work
            tasks = load_single_skill_tasks(tasks_file)
            assert len(tasks) == 1
            assert tasks[0] == "KITCHEN_SCENE1_test_task"
            
            demo_data = load_hdf5_to_dict(demo_file)
            assert 'data' in demo_data
            assert len(demo_data['data']) == 3  # 3 demos
            
            # Create mock contact timesteps for next stage
            mock_contact_data = {
                "KITCHEN_SCENE1_test_task": {
                    "demo_0": 5,
                    "demo_1": 6,
                    "demo_2": 4
                }
            }
            with open(contact_file, 'w') as f:
                json.dump(mock_contact_data, f)
            
            # Verify contact file is readable
            from extract_local_pairs_v2 import load_contact_timesteps
            loaded_contacts = load_contact_timesteps(contact_file)
            assert loaded_contacts == mock_contact_data
    
    def test_pose_extraction_integration(self):
        """Test pose extraction integration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test environment
            tasks_file, raw_demo_dir, demo_file = self.create_minimal_test_environment(tmp_dir)
            
            # Create contact timesteps
            contact_file = os.path.join(tmp_dir, "contact_timesteps.json")
            contact_data = {
                "KITCHEN_SCENE1_test_task": {
                    "demo_0": 7,
                    "demo_1": 8,
                    "demo_2": 6
                }
            }
            with open(contact_file, 'w') as f:
                json.dump(contact_data, f)
            
            # Test pose extraction functions
            from extract_local_pairs_v2 import (
                load_contact_timesteps, load_single_skill_tasks, 
                extract_contact_based_poses, parse_language_description
            )
            
            # Test data loading
            contacts = load_contact_timesteps(contact_file)
            tasks = load_single_skill_tasks(tasks_file)
            
            # Test pose extraction with mock demo
            demo_data = {
                'obs': {
                    'ee_pos': np.random.rand(10, 3),
                    'ee_ori': np.random.rand(10, 3)
                }
            }
            
            poses = extract_contact_based_poses(demo_data, 7, [-4, -3, -2])
            assert len(poses) == 3  # 3 poses extracted
            for pose, timestep in poses:
                assert pose.shape == (6,)  # 6D pose
                assert 0 <= timestep < 10  # Valid timestep
            
            # Test language parsing
            lang_desc = parse_language_description("KITCHEN_SCENE1_test_task")
            assert lang_desc == "test task"
    
    def test_validation_integration(self):
        """Test pose validation integration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test poses and annotations
            n_poses = 20
            poses = np.random.rand(n_poses, 6)
            poses_file = os.path.join(tmp_dir, "poses.npy")
            np.save(poses_file, poses)
            
            # Create annotations with multiple skills
            annotations_data = []
            skills = ["skill_1", "skill_2"]
            for i in range(n_poses):
                skill = skills[i % len(skills)]
                annotations_data.append({
                    'ee_pose_idx': i,
                    'language_description': skill,
                    'source_demo_idx': f"demo_{i // 5}",
                    'overview_image_idx': i * 2,
                    'contact_timestep': 10 + i,
                    'pose_timestep': 8 + i
                })
            
            annotations_df = pd.DataFrame(annotations_data)
            annotations_file = os.path.join(tmp_dir, "annotations.csv")
            annotations_df.to_csv(annotations_file, index=False)
            
            # Test statistics analyzer
            from validate_pose_similarity import PoseStatisticsAnalyzer
            analyzer = PoseStatisticsAnalyzer(poses_file, annotations_file)
            
            # Test analysis
            all_stats = analyzer.analyze_all_skills()
            assert len(all_stats) == 2  # 2 skills
            
            for stats in all_stats:
                if 'error' not in stats:
                    assert 'position_mean' in stats
                    assert 'orientation_mean' in stats
                    assert stats['n_poses'] > 0
            
            # Test filtering processor
            from validate_pose_similarity import PoseFilteringProcessor
            processor = PoseFilteringProcessor(poses_file, annotations_file)
            
            filtered_poses, filtered_annotations, filtering_stats = processor.filter_poses_by_similarity(
                position_threshold=1.0,  # Loose threshold for testing
                orientation_threshold=1.0,
                combined_threshold=1.0
            )
            
            assert len(filtered_poses) <= len(poses)
            assert len(filtering_stats) == 2  # 2 skills


class TestScriptExecution:
    """Test script execution and command-line interfaces."""
    
    def test_script_help_options(self):
        """Test that scripts provide help options."""
        script_dir = Path(__file__).parent.parent / "data_preparation" / "v2_pipeline"
        
        scripts = [
            "detect_contact_timesteps_v2.py",
            "extract_local_pairs_v2.py",
            "validate_pose_similarity.py"
        ]
        
        for script in scripts:
            script_path = script_dir / script
            
            # Test help option (should not raise error and should provide usage info)
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path), "--help"], 
                    capture_output=True, 
                    text=True,
                    timeout=10
                )
                # Help should return 0 or usage information
                assert result.returncode == 0 or "usage:" in result.stdout.lower()
            except subprocess.TimeoutExpired:
                pytest.fail(f"Script {script} help option timed out")
            except Exception as e:
                pytest.fail(f"Failed to run help for {script}: {e}")


def run_all_tests():
    """Run all tests in the test suite."""
    test_dir = Path(__file__).parent
    
    print("=== Running Dataset Curation Pipeline Tests ===")
    print(f"Test directory: {test_dir}")
    
    # Run all test files
    test_files = [
        "test_detect_contact_timesteps_v2.py",
        "test_extract_local_pairs_v2.py", 
        "test_validate_pose_similarity.py",
        "test_runner.py"  # This file
    ]
    
    all_passed = True
    
    for test_file in test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            print(f"\n--- Running {test_file} ---")
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pytest", str(test_path), "-v"
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    print(f"✅ {test_file} PASSED")
                else:
                    print(f"❌ {test_file} FAILED")
                    print("STDOUT:", result.stdout)
                    print("STDERR:", result.stderr)
                    all_passed = False
                    
            except subprocess.TimeoutExpired:
                print(f"⏰ {test_file} TIMED OUT")
                all_passed = False
            except Exception as e:
                print(f"💥 {test_file} ERROR: {e}")
                all_passed = False
        else:
            print(f"⚠️  {test_file} NOT FOUND")
    
    print(f"\n=== Test Summary ===")
    if all_passed:
        print("🎉 All tests PASSED!")
        return True
    else:
        print("💔 Some tests FAILED!")
        return False


if __name__ == "__main__":
    # Can be run directly or via pytest
    if len(sys.argv) > 1 and sys.argv[1] == "--run-all":
        success = run_all_tests()
        sys.exit(0 if success else 1)
    else:
        # Run this file's tests via pytest
        pytest.main([__file__, "-v"])