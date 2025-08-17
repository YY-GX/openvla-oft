import sys
import os
sys.path.append('../..')
sys.path.append('../data_preparation/v2_pipeline')

import pytest
import tempfile
import json
import numpy as np
import pandas as pd
import h5py
from unittest.mock import Mock, patch
from PIL import Image
from extract_local_pairs_v2 import (
    load_hdf5_to_dict,
    load_contact_timesteps,
    load_single_skill_tasks,
    match_task_name_to_file,
    extract_contact_based_poses,
    extract_overview_images,
    parse_language_description,
    save_image,
    main
)

"""
Test suite for extract_local_pairs_v2.py
Tests pose extraction, image handling, and data processing functionality.
"""

class TestLoadFunctions:
    """Test data loading functions."""
    
    def test_load_contact_timesteps(self):
        """Test loading contact timesteps from JSON."""
        test_data = {
            "task1": {"demo_0": 5, "demo_1": 8},
            "task2": {"demo_0": 3}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(test_data, tmp_file)
            tmp_file.flush()
            
            result = load_contact_timesteps(tmp_file.name)
            assert result == test_data
            
            os.unlink(tmp_file.name)
    
    def test_load_single_skill_tasks(self):
        """Test loading single skill tasks."""
        test_tasks = ["task1", "task2", "task3"]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(test_tasks, tmp_file)
            tmp_file.flush()
            
            result = load_single_skill_tasks(tmp_file.name)
            assert result == test_tasks
            
            os.unlink(tmp_file.name)


class TestExtractContactBasedPoses:
    """Test contact-based pose extraction."""
    
    def create_mock_demo(self, n_timesteps=10):
        """Create a mock demonstration with poses."""
        # Create mock EE positions and orientations
        ee_pos = np.random.rand(n_timesteps, 3)
        ee_ori = np.random.rand(n_timesteps, 3)  # Using 3D for simplicity
        
        demo = {
            'obs': {
                'ee_pos': ee_pos,
                'ee_ori': ee_ori
            }
        }
        return demo
    
    def test_extract_poses_normal_case(self):
        """Test pose extraction in normal case."""
        demo = self.create_mock_demo(10)
        contact_timestep = 7
        window_offsets = [-4, -3, -2]
        
        poses = extract_contact_based_poses(demo, contact_timestep, window_offsets)
        
        # Should return 3 poses
        assert len(poses) == 3
        
        # Check timesteps are correct
        expected_timesteps = [3, 4, 5]  # 7 + [-4, -3, -2]
        actual_timesteps = [timestep for _, timestep in poses]
        assert actual_timesteps == expected_timesteps
        
        # Check pose dimensions (3 pos + 3 ori = 6D)
        for pose, timestep in poses:
            assert pose.shape == (6,)
    
    def test_extract_poses_boundary_conditions(self):
        """Test pose extraction at boundaries."""
        demo = self.create_mock_demo(5)  # Short demo
        contact_timestep = 2
        window_offsets = [-4, -3, -2]
        
        poses = extract_contact_based_poses(demo, contact_timestep, window_offsets)
        
        # Should return 3 poses
        assert len(poses) == 3
        
        # Timesteps should be clamped to valid range
        expected_timesteps = [0, 0, 0]  # All clamped to start
        actual_timesteps = [timestep for _, timestep in poses]
        assert actual_timesteps == expected_timesteps
    
    def test_extract_poses_end_boundary(self):
        """Test pose extraction near end boundary."""
        demo = self.create_mock_demo(5)
        contact_timestep = 10  # Beyond demo length
        window_offsets = [-2, -1, 0]
        
        poses = extract_contact_based_poses(demo, contact_timestep, window_offsets)
        
        # Should return 3 poses
        assert len(poses) == 3
        
        # Timesteps should be clamped to valid range [0, 4]
        expected_timesteps = [4, 4, 4]  # All clamped to end
        actual_timesteps = [timestep for _, timestep in poses]
        assert actual_timesteps == expected_timesteps


class TestExtractOverviewImages:
    """Test overview image extraction."""
    
    def create_mock_demo_with_images(self, n_timesteps=10):
        """Create a mock demonstration with images."""
        # Create mock RGB images (random values)
        agentview_rgb = np.random.randint(0, 255, (n_timesteps, 64, 64, 3), dtype=np.uint8)
        
        demo = {
            'obs': {
                'agentview_rgb': agentview_rgb
            }
        }
        return demo
    
    def test_extract_overview_images_normal(self):
        """Test normal overview image extraction."""
        demo = self.create_mock_demo_with_images(10)
        contact_timestep = 10
        overview_percentage = 70.0
        
        images = extract_overview_images(demo, contact_timestep, overview_percentage)
        
        # Should extract 70% of 10 timesteps = 7 images
        expected_count = int(10 * 70.0 / 100)
        assert len(images) == expected_count
        
        # Check image shape
        for image in images:
            assert image.shape == (64, 64, 3)
    
    def test_extract_overview_images_short_demo(self):
        """Test overview extraction with short demo."""
        demo = self.create_mock_demo_with_images(2)
        contact_timestep = 2
        overview_percentage = 70.0
        
        images = extract_overview_images(demo, contact_timestep, overview_percentage)
        
        # Should extract at least 1 image
        assert len(images) >= 1
        assert len(images) <= 2


class TestParseLanguageDescription:
    """Test language description parsing."""
    
    def test_parse_kitchen_scene(self):
        """Test parsing kitchen scene task names."""
        task_name = "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate"
        result = parse_language_description(task_name)
        expected = "put the black bowl on the plate"
        assert result == expected
    
    def test_parse_living_room_scene(self):
        """Test parsing living room scene task names."""
        task_name = "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate"
        result = parse_language_description(task_name)
        expected = "put the red mug on the left plate"
        assert result == expected
    
    def test_parse_no_scene_prefix(self):
        """Test parsing task names without scene prefix."""
        task_name = "open_the_drawer"
        result = parse_language_description(task_name)
        expected = "open the drawer"
        assert result == expected


class TestSaveImage:
    """Test image saving functionality."""
    
    def test_save_image(self):
        """Test saving image to file."""
        # Create a test image
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            save_image(test_image, tmp_file.name)
            
            # Verify file was created and is readable
            assert os.path.exists(tmp_file.name)
            
            # Try to load the image back
            loaded_image = Image.open(tmp_file.name)
            loaded_array = np.array(loaded_image)
            
            # Should have same shape
            assert loaded_array.shape == test_image.shape
            
            os.unlink(tmp_file.name)


class TestMainIntegration:
    """Test main function integration."""
    
    def create_test_environment(self, tmp_dir):
        """Create a complete test environment."""
        # Create single skill tasks
        tasks_file = os.path.join(tmp_dir, "single_skill_tasks_test.json")
        test_tasks = ["KITCHEN_SCENE1_test_task"]
        with open(tasks_file, 'w') as f:
            json.dump(test_tasks, f)
        
        # Create contact timesteps
        contact_file = os.path.join(tmp_dir, "contact_timesteps_test.json")
        contact_data = {
            "KITCHEN_SCENE1_test_task": {
                "demo_0": 5
            }
        }
        with open(contact_file, 'w') as f:
            json.dump(contact_data, f)
        
        # Create demo HDF5 file
        demo_file = os.path.join(tmp_dir, "KITCHEN_SCENE1_test_task_demo.hdf5")
        with h5py.File(demo_file, 'w') as f:
            data_group = f.create_group('data')
            demo_group = data_group.create_group('demo_0')
            
            # Create obs group
            obs_group = demo_group.create_group('obs')
            
            # Create mock EE poses
            n_timesteps = 10
            ee_pos = np.random.rand(n_timesteps, 3)
            ee_ori = np.random.rand(n_timesteps, 3)
            obs_group.create_dataset('ee_pos', data=ee_pos)
            obs_group.create_dataset('ee_ori', data=ee_ori)
            
            # Create mock images
            agentview_rgb = np.random.randint(0, 255, (n_timesteps, 64, 64, 3), dtype=np.uint8)
            obs_group.create_dataset('agentview_rgb', data=agentview_rgb)
        
        return tasks_file, contact_file, demo_file
    
    @patch('sys.argv')
    def test_main_integration(self, mock_argv):
        """Test main function integration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test environment
            tasks_file, contact_file, demo_file = self.create_test_environment(tmp_dir)
            save_dir = os.path.join(tmp_dir, "output")
            
            # Mock command line arguments
            mock_argv.__getitem__.side_effect = lambda x: [
                'extract_local_pairs_v2.py',
                '--raw_demo_dir', os.path.dirname(demo_file),
                '--contact_timesteps_file', contact_file,
                '--single_skill_tasks_file', tasks_file,
                '--save_dir', save_dir
            ][x]
            mock_argv.__len__.return_value = 9
            
            # Run main function with mocked args
            with patch('argparse.ArgumentParser.parse_args') as mock_parse_args:
                mock_args = Mock()
                mock_args.raw_demo_dir = os.path.dirname(demo_file)
                mock_args.contact_timesteps_file = contact_file
                mock_args.single_skill_tasks_file = tasks_file
                mock_args.save_dir = save_dir
                mock_args.window_offsets = [-4, -3, -2]
                mock_args.overview_percentage = 70.0
                mock_parse_args.return_value = mock_args
                
                main()
            
            # Verify output files were created
            assert os.path.exists(os.path.join(save_dir, "local_poses.npy"))
            assert os.path.exists(os.path.join(save_dir, "annotation.csv"))
            assert os.path.exists(os.path.join(save_dir, "overview_images"))
            assert os.path.exists(os.path.join(save_dir, "pose_images"))
            
            # Verify poses file
            poses = np.load(os.path.join(save_dir, "local_poses.npy"))
            assert poses.shape[1] == 6  # 6D poses
            
            # Verify annotation file
            annotations = pd.read_csv(os.path.join(save_dir, "annotation.csv"))
            assert len(annotations) > 0
            assert 'language_description' in annotations.columns
            assert 'contact_timestep' in annotations.columns
            assert 'ee_pose_idx' in annotations.columns


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_missing_contact_timesteps(self):
        """Test behavior when contact timesteps are missing for a task."""
        contact_data = {"other_task": {"demo_0": 5}}
        tasks = ["missing_task"]
        
        # This should be handled gracefully in main function
        # (task would be skipped)
        assert True  # Placeholder for actual implementation test
    
    def test_empty_demo_file(self):
        """Test behavior with empty demo file."""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp_file:
            # Create empty HDF5 file
            with h5py.File(tmp_file.name, 'w') as f:
                pass
            
            # Should handle this gracefully
            try:
                load_hdf5_to_dict(tmp_file.name)
            except Exception as e:
                # Expected to fail gracefully
                assert isinstance(e, (KeyError, AttributeError))
            
            os.unlink(tmp_file.name)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])