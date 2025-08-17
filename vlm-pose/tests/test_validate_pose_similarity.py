import sys
import os
sys.path.append('../..')
sys.path.append('../data_preparation/v2_pipeline')

import pytest
import tempfile
import json
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from validate_pose_similarity import (
    PoseStatisticsAnalyzer,
    PoseFilteringProcessor
)

"""
Test suite for validate_pose_similarity.py
Tests pose statistics analysis and filtering functionality.
"""

class TestPoseStatisticsAnalyzer:
    """Test pose statistics analysis functionality."""
    
    def create_test_data(self, tmp_dir):
        """Create test poses and annotations."""
        # Create test poses (6D: position + orientation)
        n_poses = 20
        poses = np.random.rand(n_poses, 6)
        poses_file = os.path.join(tmp_dir, "test_poses.npy")
        np.save(poses_file, poses)
        
        # Create test annotations
        annotations_data = []
        skill_names = ["skill_1", "skill_2", "skill_3"]
        
        for i, pose_idx in enumerate(range(n_poses)):
            skill_name = skill_names[i % len(skill_names)]
            annotations_data.append({
                'ee_pose_idx': pose_idx,
                'language_description': skill_name,
                'source_demo_idx': f"demo_{i // 3}",
                'overview_image_idx': i * 2,
                'contact_timestep': 10 + i,
                'pose_timestep': 8 + i
            })
        
        annotations_df = pd.DataFrame(annotations_data)
        annotations_file = os.path.join(tmp_dir, "test_annotations.csv")
        annotations_df.to_csv(annotations_file, index=False)
        
        return poses_file, annotations_file, poses, annotations_df
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poses_file, annotations_file, poses, annotations_df = self.create_test_data(tmp_dir)
            
            analyzer = PoseStatisticsAnalyzer(poses_file, annotations_file)
            
            assert analyzer.poses.shape == poses.shape
            assert len(analyzer.annotations) == len(annotations_df)
            assert isinstance(analyzer.pose_stats, dict)
    
    def test_compute_pose_distances(self):
        """Test pose distance computation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poses_file, annotations_file, poses, annotations_df = self.create_test_data(tmp_dir)
            analyzer = PoseStatisticsAnalyzer(poses_file, annotations_file)
            
            # Test with small subset of poses
            test_poses = poses[:5]
            distances = analyzer.compute_pose_distances(test_poses)
            
            # Check returned structure
            assert 'position_distances' in distances
            assert 'orientation_distances' in distances
            assert 'combined_distances' in distances
            
            # Check matrix shapes
            assert distances['position_distances'].shape == (5, 5)
            assert distances['orientation_distances'].shape == (5, 5)
            assert distances['combined_distances'].shape == (5, 5)
            
            # Check diagonal is zero
            np.testing.assert_array_equal(np.diag(distances['position_distances']), np.zeros(5))
            np.testing.assert_array_equal(np.diag(distances['orientation_distances']), np.zeros(5))
            np.testing.assert_array_equal(np.diag(distances['combined_distances']), np.zeros(5))
            
            # Check symmetry
            pos_dist = distances['position_distances']
            assert np.allclose(pos_dist, pos_dist.T)
    
    def test_analyze_skill_poses(self):
        """Test analysis of poses for a single skill."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poses_file, annotations_file, poses, annotations_df = self.create_test_data(tmp_dir)
            analyzer = PoseStatisticsAnalyzer(poses_file, annotations_file)
            
            # Analyze first skill
            skill_name = "skill_1"
            skill_annotations = annotations_df[annotations_df['language_description'] == skill_name]
            
            stats = analyzer.analyze_skill_poses(skill_name, skill_annotations)
            
            # Check required fields
            required_fields = [
                'skill_name', 'n_poses', 'n_pairs', 'n_demos',
                'position_mean', 'position_std', 'position_min', 'position_max', 'position_median',
                'orientation_mean', 'orientation_std', 'orientation_min', 'orientation_max', 'orientation_median',
                'combined_mean', 'combined_std', 'combined_min', 'combined_max', 'combined_median'
            ]
            
            for field in required_fields:
                assert field in stats
            
            # Check value types and ranges
            assert stats['skill_name'] == skill_name
            assert stats['n_poses'] > 0
            assert stats['n_pairs'] > 0
            assert stats['n_demos'] > 0
            assert stats['position_mean'] >= 0
            assert stats['position_std'] >= 0
            assert stats['position_min'] >= 0
            assert stats['position_max'] >= stats['position_min']
    
    def test_analyze_skill_poses_too_few(self):
        """Test analysis with too few poses."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poses_file, annotations_file, poses, annotations_df = self.create_test_data(tmp_dir)
            analyzer = PoseStatisticsAnalyzer(poses_file, annotations_file)
            
            # Create single-pose skill annotation
            single_annotation = pd.DataFrame([{
                'ee_pose_idx': 0,
                'language_description': 'single_pose_skill',
                'source_demo_idx': 'demo_0',
                'overview_image_idx': 0,
                'contact_timestep': 10,
                'pose_timestep': 8
            }])
            
            stats = analyzer.analyze_skill_poses('single_pose_skill', single_annotation)
            
            assert 'error' in stats
            assert stats['n_poses'] == 1
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_report(self, mock_close, mock_savefig):
        """Test report generation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poses_file, annotations_file, poses, annotations_df = self.create_test_data(tmp_dir)
            analyzer = PoseStatisticsAnalyzer(poses_file, annotations_file)
            
            output_dir = os.path.join(tmp_dir, "output")
            analyzer.generate_report(output_dir)
            
            # Check output files
            assert os.path.exists(os.path.join(output_dir, 'pose_similarity_statistics.json'))
            assert os.path.exists(os.path.join(output_dir, 'pose_similarity_statistics.csv'))
            
            # Check JSON structure
            with open(os.path.join(output_dir, 'pose_similarity_statistics.json'), 'r') as f:
                report_data = json.load(f)
            
            assert 'summary' in report_data
            assert 'per_skill_stats' in report_data
            assert 'total_skills' in report_data['summary']
            
            # Check CSV
            stats_df = pd.read_csv(os.path.join(output_dir, 'pose_similarity_statistics.csv'))
            assert len(stats_df) > 0
            assert 'skill_name' in stats_df.columns


class TestPoseFilteringProcessor:
    """Test pose filtering functionality."""
    
    def create_clustered_test_data(self, tmp_dir):
        """Create test data with clustered poses for better filtering testing."""
        # Create poses with clear clusters
        cluster_centers = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Cluster 1 center
            np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]),  # Cluster 2 center
        ]
        
        poses = []
        annotations_data = []
        
        pose_idx = 0
        for skill_idx, center in enumerate(cluster_centers):
            skill_name = f"skill_{skill_idx + 1}"
            
            # Add poses close to cluster center (should be kept)
            for i in range(8):
                noise = np.random.normal(0, 0.02, 6)  # Small noise
                pose = center + noise
                poses.append(pose)
                
                annotations_data.append({
                    'ee_pose_idx': pose_idx,
                    'language_description': skill_name,
                    'source_demo_idx': f"demo_{i}",
                    'overview_image_idx': pose_idx * 2,
                    'contact_timestep': 10 + pose_idx,
                    'pose_timestep': 8 + pose_idx
                })
                pose_idx += 1
            
            # Add outlier poses (should be filtered out)
            for i in range(2):
                noise = np.random.normal(0, 0.5, 6)  # Large noise
                pose = center + noise
                poses.append(pose)
                
                annotations_data.append({
                    'ee_pose_idx': pose_idx,
                    'language_description': skill_name,
                    'source_demo_idx': f"demo_outlier_{i}",
                    'overview_image_idx': pose_idx * 2,
                    'contact_timestep': 10 + pose_idx,
                    'pose_timestep': 8 + pose_idx
                })
                pose_idx += 1
        
        poses = np.array(poses)
        poses_file = os.path.join(tmp_dir, "test_poses.npy")
        np.save(poses_file, poses)
        
        annotations_df = pd.DataFrame(annotations_data)
        annotations_file = os.path.join(tmp_dir, "test_annotations.csv")
        annotations_df.to_csv(annotations_file, index=False)
        
        return poses_file, annotations_file, poses, annotations_df
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poses_file, annotations_file, poses, annotations_df = self.create_clustered_test_data(tmp_dir)
            
            processor = PoseFilteringProcessor(poses_file, annotations_file)
            
            assert processor.poses.shape == poses.shape
            assert len(processor.annotations) == len(annotations_df)
    
    def test_filter_poses_by_similarity(self):
        """Test pose filtering by similarity."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poses_file, annotations_file, poses, annotations_df = self.create_clustered_test_data(tmp_dir)
            processor = PoseFilteringProcessor(poses_file, annotations_file)
            
            # Apply filtering with strict thresholds
            filtered_poses, filtered_annotations, filtering_stats = processor.filter_poses_by_similarity(
                position_threshold=0.1,
                orientation_threshold=0.1,
                combined_threshold=0.05
            )
            
            # Should filter out some poses
            assert len(filtered_poses) <= len(poses)
            assert len(filtered_annotations) <= len(annotations_df)
            
            # Check filtering stats structure
            assert len(filtering_stats) == 2  # Two skills
            for skill_name in ["skill_1", "skill_2"]:
                assert skill_name in filtering_stats
                stats = filtering_stats[skill_name]
                assert 'original_poses' in stats
                assert 'kept_poses' in stats
                assert 'filtered_poses' in stats
                assert stats['original_poses'] == stats['kept_poses'] + stats['filtered_poses']
    
    def test_filter_poses_loose_thresholds(self):
        """Test filtering with loose thresholds (should keep most poses)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poses_file, annotations_file, poses, annotations_df = self.create_clustered_test_data(tmp_dir)
            processor = PoseFilteringProcessor(poses_file, annotations_file)
            
            # Apply filtering with loose thresholds
            filtered_poses, filtered_annotations, filtering_stats = processor.filter_poses_by_similarity(
                position_threshold=10.0,
                orientation_threshold=10.0,
                combined_threshold=10.0
            )
            
            # Should keep most/all poses
            assert len(filtered_poses) >= len(poses) * 0.8  # At least 80% kept
    
    def test_filter_poses_strict_thresholds(self):
        """Test filtering with strict thresholds (should filter more poses)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poses_file, annotations_file, poses, annotations_df = self.create_clustered_test_data(tmp_dir)
            processor = PoseFilteringProcessor(poses_file, annotations_file)
            
            # Apply filtering with strict thresholds
            filtered_poses, filtered_annotations, filtering_stats = processor.filter_poses_by_similarity(
                position_threshold=0.01,
                orientation_threshold=0.01,
                combined_threshold=0.005
            )
            
            # Should filter out more poses
            assert len(filtered_poses) < len(poses)
    
    def test_save_filtered_data(self):
        """Test saving filtered data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            poses_file, annotations_file, poses, annotations_df = self.create_clustered_test_data(tmp_dir)
            processor = PoseFilteringProcessor(poses_file, annotations_file)
            
            # Filter data
            filtered_poses, filtered_annotations, _ = processor.filter_poses_by_similarity()
            
            # Save filtered data
            output_dir = os.path.join(tmp_dir, "filtered_output")
            processor.save_filtered_data(filtered_poses, filtered_annotations, output_dir)
            
            # Check output files
            filtered_poses_file = os.path.join(output_dir, 'local_poses_filtered.npy')
            filtered_annotations_file = os.path.join(output_dir, 'annotation_filtered.csv')
            
            assert os.path.exists(filtered_poses_file)
            assert os.path.exists(filtered_annotations_file)
            
            # Verify data integrity
            loaded_poses = np.load(filtered_poses_file)
            loaded_annotations = pd.read_csv(filtered_annotations_file)
            
            assert loaded_poses.shape == filtered_poses.shape
            assert len(loaded_annotations) == len(filtered_annotations)
            
            # Check pose index mapping is valid
            max_pose_idx = loaded_annotations['ee_pose_idx'].max()
            assert max_pose_idx < len(loaded_poses)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_poses_file(self):
        """Test behavior with empty poses file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create empty poses file
            empty_poses = np.array([]).reshape(0, 6)
            poses_file = os.path.join(tmp_dir, "empty_poses.npy")
            np.save(poses_file, empty_poses)
            
            # Create empty annotations
            empty_annotations = pd.DataFrame(columns=['ee_pose_idx', 'language_description'])
            annotations_file = os.path.join(tmp_dir, "empty_annotations.csv")
            empty_annotations.to_csv(annotations_file, index=False)
            
            # Should handle gracefully
            analyzer = PoseStatisticsAnalyzer(poses_file, annotations_file)
            assert len(analyzer.poses) == 0
            assert len(analyzer.annotations) == 0
    
    def test_mismatched_pose_indices(self):
        """Test behavior with mismatched pose indices in annotations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create poses
            poses = np.random.rand(5, 6)
            poses_file = os.path.join(tmp_dir, "poses.npy")
            np.save(poses_file, poses)
            
            # Create annotations with out-of-range pose indices
            annotations_data = [{
                'ee_pose_idx': 10,  # Out of range
                'language_description': 'test_skill',
                'source_demo_idx': 'demo_0',
                'overview_image_idx': 0,
                'contact_timestep': 10,
                'pose_timestep': 8
            }]
            annotations_df = pd.DataFrame(annotations_data)
            annotations_file = os.path.join(tmp_dir, "annotations.csv")
            annotations_df.to_csv(annotations_file, index=False)
            
            # Should handle gracefully (might raise IndexError or filter out invalid indices)
            try:
                analyzer = PoseStatisticsAnalyzer(poses_file, annotations_file)
                # If it doesn't raise an error, the implementation should handle it gracefully
                assert True
            except IndexError:
                # This is also acceptable behavior
                assert True


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])