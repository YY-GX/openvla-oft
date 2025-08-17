import sys
import os
sys.path.append('../..')
sys.path.append('../data_preparation/v2_pipeline')

import pytest
import tempfile
import json
import numpy as np
import h5py
from unittest.mock import Mock, patch, MagicMock
from detect_contact_timesteps_v2 import (
    load_hdf5_to_dict, 
    detect_first_contact_timestep,
    load_single_skill_tasks,
    match_task_name_to_file,
    main
)

"""
Test suite for detect_contact_timesteps_v2.py
Tests contact detection, file matching, and overall functionality.
"""

class TestLoadHDF5ToDict:
    """Test HDF5 loading functionality."""
    
    def test_load_simple_hdf5(self):
        """Test loading a simple HDF5 file."""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp_file:
            # Create a simple HDF5 file
            with h5py.File(tmp_file.name, 'w') as f:
                f.create_dataset('data', data=np.array([1, 2, 3]))
                f.create_group('group1')
                f['group1'].create_dataset('nested_data', data=np.array([4, 5, 6]))
            
            # Load and test
            result = load_hdf5_to_dict(tmp_file.name)
            
            assert 'data' in result
            assert 'group1' in result
            assert 'nested_data' in result['group1']
            np.testing.assert_array_equal(result['data'], [1, 2, 3])
            np.testing.assert_array_equal(result['group1']['nested_data'], [4, 5, 6])
            
            # Cleanup
            os.unlink(tmp_file.name)


class TestDetectFirstContactTimestep:
    """Test contact detection functionality."""
    
    def create_mock_demo(self, n_timesteps=10):
        """Create a mock demonstration with states."""
        demo = {
            'states': [Mock() for _ in range(n_timesteps)]
        }
        return demo
    
    def create_mock_env(self, contacts_per_timestep):
        """
        Create a mock environment with specified contacts.
        
        Args:
            contacts_per_timestep: List where each element is a list of (geom1, geom2) tuples
        """
        mock_env = Mock()
        mock_sim = Mock()
        mock_data = Mock()
        
        # Setup simulation data
        mock_env.sim = mock_sim
        mock_sim.data = mock_data
        
        # Track timestep for set_init_state calls
        timestep_counter = [0]
        
        def mock_set_init_state(state):
            """Mock set_init_state that updates contact data based on timestep."""
            current_timestep = timestep_counter[0]
            if current_timestep < len(contacts_per_timestep):
                contacts = contacts_per_timestep[current_timestep]
                mock_data.ncon = len(contacts)
                
                # Create mock contacts
                mock_contacts = []
                for geom1, geom2 in contacts:
                    contact = Mock()
                    contact.geom1 = geom1
                    contact.geom2 = geom2
                    mock_contacts.append(contact)
                mock_data.contact = mock_contacts
                
                # Mock geom_id2name
                def mock_geom_id2name(geom_id):
                    # Map geom IDs to names for testing
                    geom_names = {
                        0: "robot0_eef",
                        1: "robot0_gripper0_finger0", 
                        2: "object1",
                        3: "object2",
                        4: "floor"
                    }
                    return geom_names.get(geom_id, f"geom_{geom_id}")
                
                mock_sim.model.geom_id2name = mock_geom_id2name
            else:
                mock_data.ncon = 0
                mock_data.contact = []
            
            timestep_counter[0] += 1
        
        mock_env.set_init_state = mock_set_init_state
        return mock_env
    
    @patch('detect_contact_timesteps_v2.get_libero_env')
    @patch('detect_contact_timesteps_v2.benchmark')
    def test_detect_first_contact_found(self, mock_benchmark, mock_get_env):
        """Test detecting first contact when contact exists."""
        # Setup mock data
        demo = self.create_mock_demo(5)
        
        # Create contacts: no contact at t=0,1, contact at t=2,3,4
        contacts_per_timestep = [
            [],  # t=0: no contact
            [],  # t=1: no contact  
            [(0, 2)],  # t=2: robot0_eef contacts object1
            [(0, 2), (1, 3)],  # t=3: multiple contacts
            [(1, 2)]  # t=4: different contact
        ]
        
        mock_env = self.create_mock_env(contacts_per_timestep)
        mock_get_env.return_value = (mock_env, None)
        
        # Mock benchmark
        mock_task_suite = Mock()
        mock_task = Mock()
        mock_task.name = "test_task"
        mock_task_suite.tasks = [mock_task]
        mock_benchmark.get_benchmark_dict.return_value = {"libero_90": lambda: mock_task_suite}
        
        # Test detection
        result = detect_first_contact_timestep(demo, "test_file.hdf5", "demo_0", "test_task")
        
        # Should detect first contact at timestep 2
        assert result == 2
    
    @patch('detect_contact_timesteps_v2.get_libero_env')
    @patch('detect_contact_timesteps_v2.benchmark')
    def test_detect_no_contact(self, mock_benchmark, mock_get_env):
        """Test when no contact is detected."""
        demo = self.create_mock_demo(3)
        
        # No contacts at any timestep
        contacts_per_timestep = [[], [], []]
        
        mock_env = self.create_mock_env(contacts_per_timestep)
        mock_get_env.return_value = (mock_env, None)
        
        # Mock benchmark
        mock_task_suite = Mock()
        mock_task = Mock()
        mock_task.name = "test_task"
        mock_task_suite.tasks = [mock_task]
        mock_benchmark.get_benchmark_dict.return_value = {"libero_90": lambda: mock_task_suite}
        
        # Test detection
        result = detect_first_contact_timestep(demo, "test_file.hdf5", "demo_0", "test_task")
        
        # Should return None
        assert result is None
    
    @patch('detect_contact_timesteps_v2.get_libero_env')
    @patch('detect_contact_timesteps_v2.benchmark')
    def test_detect_contact_error_handling(self, mock_benchmark, mock_get_env):
        """Test error handling in contact detection."""
        demo = self.create_mock_demo(2)
        
        # Make get_libero_env raise an exception
        mock_get_env.side_effect = Exception("Test error")
        
        # Test detection
        result = detect_first_contact_timestep(demo, "test_file.hdf5", "demo_0", "test_task")
        
        # Should return None due to exception
        assert result is None


class TestLoadSingleSkillTasks:
    """Test loading single skill tasks."""
    
    def test_load_valid_json(self):
        """Test loading valid JSON file."""
        test_tasks = ["task1", "task2", "task3"]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(test_tasks, tmp_file)
            tmp_file.flush()
            
            result = load_single_skill_tasks(tmp_file.name)
            assert result == test_tasks
            
            os.unlink(tmp_file.name)


class TestMatchTaskNameToFile:
    """Test task name to file matching."""
    
    def test_exact_match(self):
        """Test exact filename matching."""
        task_name = "KITCHEN_SCENE1_open_drawer"
        demo_files = [
            "/path/to/KITCHEN_SCENE1_open_drawer_demo.hdf5",
            "/path/to/other_task_demo.hdf5"
        ]
        
        result = match_task_name_to_file(task_name, demo_files)
        assert result == "/path/to/KITCHEN_SCENE1_open_drawer_demo.hdf5"
    
    def test_partial_match(self):
        """Test partial filename matching."""
        task_name = "KITCHEN_SCENE1_open_drawer"
        demo_files = [
            "/path/to/KITCHEN_SCENE1_open_drawer_modified.hdf5",
            "/path/to/other_task_demo.hdf5"
        ]
        
        result = match_task_name_to_file(task_name, demo_files)
        assert result == "/path/to/KITCHEN_SCENE1_open_drawer_modified.hdf5"
    
    def test_no_match(self):
        """Test when no matching file is found."""
        task_name = "KITCHEN_SCENE1_open_drawer"
        demo_files = [
            "/path/to/other_task1_demo.hdf5",
            "/path/to/other_task2_demo.hdf5"
        ]
        
        result = match_task_name_to_file(task_name, demo_files)
        assert result is None


class TestMainIntegration:
    """Test main function integration."""
    
    def create_test_data(self, tmp_dir):
        """Create test data files for integration testing."""
        # Create single skill tasks JSON
        tasks_file = os.path.join(tmp_dir, "single_skill_tasks_test.json")
        test_tasks = ["KITCHEN_SCENE1_test_task"]
        with open(tasks_file, 'w') as f:
            json.dump(test_tasks, f)
        
        # Create mock demo HDF5 file
        demo_file = os.path.join(tmp_dir, "KITCHEN_SCENE1_test_task_demo.hdf5")
        with h5py.File(demo_file, 'w') as f:
            data_group = f.create_group('data')
            demo_group = data_group.create_group('demo_0')
            
            # Create mock states (just placeholders)
            demo_group.create_dataset('states', data=np.zeros((5, 10)))
        
        return tasks_file, demo_file
    
    @patch('detect_contact_timesteps_v2.detect_first_contact_timestep')
    @patch('sys.argv')
    def test_main_basic_functionality(self, mock_argv, mock_detect_contact):
        """Test basic functionality of main function."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test data
            tasks_file, demo_file = self.create_test_data(tmp_dir)
            output_file = os.path.join(tmp_dir, "contact_timesteps_test.json")
            
            # Mock command line arguments
            mock_argv.__getitem__.side_effect = lambda x: [
                'detect_contact_timesteps_v2.py',
                '--raw_demo_dir', os.path.dirname(demo_file),
                '--single_skill_tasks_file', tasks_file,
                '--output_file', output_file
            ][x]
            mock_argv.__len__.return_value = 7
            
            # Mock contact detection to return a contact at timestep 3
            mock_detect_contact.return_value = 3
            
            # Run main function (with mocked args)
            with patch('argparse.ArgumentParser.parse_args') as mock_parse_args:
                mock_args = Mock()
                mock_args.raw_demo_dir = os.path.dirname(demo_file)
                mock_args.single_skill_tasks_file = tasks_file
                mock_args.output_file = output_file
                mock_parse_args.return_value = mock_args
                
                main()
            
            # Verify output file was created
            assert os.path.exists(output_file)
            
            # Verify output content
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            assert "KITCHEN_SCENE1_test_task" in results
            assert "demo_0" in results["KITCHEN_SCENE1_test_task"]
            assert results["KITCHEN_SCENE1_test_task"]["demo_0"] == 3


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])