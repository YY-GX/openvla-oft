#!/usr/bin/env python3
"""
Object-Segmented Pointcloud Extractor for Phase 2 Pipeline

This module provides object-specific pointcloud extraction that only extracts
points belonging to the target object, not nearby objects or surfaces.

Key improvements:
1. Object segmentation using depth discontinuities and surface normals
2. Color-based clustering to separate objects
3. Geometric constraints based on object properties
4. Multi-step filtering to isolate target object only

Functions:
- extract_object_only_pointcloud(): Extract only target object points
- segment_pointcloud_by_object(): Advanced object segmentation
- filter_by_geometric_properties(): Use object shape/size constraints
"""

import os
import sys
import numpy as np
import open3d as o3d
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import json
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy import ndimage

# Add project paths
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')

# Import base corrected extractor
from scripts.phase2.utils.pointcloud_extractor_corrected import CorrectedPointcloudExtractor

# Import robosuite transform utilities
import robosuite.utils.transform_utils as T


class ObjectSegmentedExtractor(CorrectedPointcloudExtractor):
    """Extract pointclouds for specific objects only, excluding nearby surfaces and objects."""
    
    def __init__(self, env, debug_mode: bool = False):
        """
        Initialize object-segmented pointcloud extractor.
        
        Args:
            env: LIBERO environment instance
            debug_mode: Enable debug visualizations and logging
        """
        super().__init__(env, debug_mode)
        
        if self.debug_mode:
            print(f"🎯 ObjectSegmentedExtractor initialized")
            print(f"   🔍 Advanced object segmentation enabled")
    
    def get_object_mesh_info(self, object_name: str) -> Dict[str, Any]:
        """
        Get object mesh/geometry information from MuJoCo simulation.
        This helps understand the object's size and shape for better filtering.
        
        Args:
            object_name: Name of the target object
            
        Returns:
            Dictionary with object mesh information
        """
        try:
            # Find object body in MuJoCo
            body_names = [self.env.sim.model.body_id2name(i) for i in range(self.env.sim.model.nbody)]
            
            # Look for object body (may have different naming conventions)
            object_body_id = None
            for i, body_name in enumerate(body_names):
                if object_name in body_name or any(part in body_name for part in object_name.split('_')):
                    object_body_id = i
                    break
            
            if object_body_id is None:
                # Fallback: estimate from object name
                return self._estimate_object_properties(object_name)
            
            # Get object position and approximate size
            object_pos = self.env.sim.data.body_xpos[object_body_id]
            
            # Estimate object size from geometry (simplified)
            geom_ids = []
            for i in range(self.env.sim.model.ngeom):
                if self.env.sim.model.geom_bodyid[i] == object_body_id:
                    geom_ids.append(i)
            
            # Calculate approximate bounding box
            if geom_ids:
                geom_sizes = [self.env.sim.model.geom_size[i] for i in geom_ids]
                avg_size = np.mean(geom_sizes, axis=0)
                max_extent = np.max(avg_size) * 2  # Approximate max dimension
            else:
                max_extent = 0.15  # Default fallback
            
            return {
                'position': object_pos,
                'approximate_size': max_extent,
                'body_id': object_body_id,
                'geom_ids': geom_ids
            }
            
        except Exception as e:
            if self.debug_mode:
                print(f"   ⚠️  Could not get mesh info for {object_name}: {e}")
            return self._estimate_object_properties(object_name)
    
    def _estimate_object_properties(self, object_name: str) -> Dict[str, Any]:
        """Estimate object properties based on name."""
        # Object size estimates based on common kitchen objects
        size_estimates = {
            'moka_pot': 0.12,      # ~12cm
            'frypan': 0.25,        # ~25cm diameter
            'bowl': 0.15,          # ~15cm
            'plate': 0.20,         # ~20cm
            'microwave': 0.40,     # ~40cm
            'cabinet': 0.60,       # ~60cm
            'stove': 0.30,         # ~30cm
            'default': 0.15        # 15cm fallback
        }
        
        estimated_size = size_estimates['default']
        for key, size in size_estimates.items():
            if key in object_name.lower():
                estimated_size = size
                break
        
        return {
            'position': np.array([0, 0, 0]),  # Will be updated with actual position
            'approximate_size': estimated_size,
            'body_id': None,
            'geom_ids': []
        }
    
    def segment_by_surface_normals(self, pointcloud: np.ndarray, colors: np.ndarray = None, 
                                 normal_threshold: float = 0.9) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Segment pointcloud using surface normals to separate different surfaces.
        
        Args:
            pointcloud: Input pointcloud (N, 3)
            colors: Optional color information (N, 3)
            normal_threshold: Threshold for normal similarity
            
        Returns:
            Tuple of (normals, segments) where segments is list of point indices for each segment
        """
        if len(pointcloud) < 100:  # Too few points for normal computation
            return np.zeros((len(pointcloud), 3)), [np.arange(len(pointcloud))]
        
        try:
            # Create Open3D pointcloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud)
            
            # Estimate normals
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
            normals = np.asarray(pcd.normals)
            
            # Cluster points by normal similarity (simplified approach)
            segments = []
            processed = np.zeros(len(pointcloud), dtype=bool)
            
            for i in range(len(pointcloud)):
                if processed[i]:
                    continue
                
                # Find points with similar normals
                normal_similarities = np.dot(normals, normals[i])
                similar_points = np.where(normal_similarities > normal_threshold)[0]
                
                segments.append(similar_points)
                processed[similar_points] = True
            
            if self.debug_mode:
                print(f"   🔍 Normal-based segmentation: {len(segments)} segments")
            
            return normals, segments
            
        except Exception as e:
            if self.debug_mode:
                print(f"   ⚠️  Normal segmentation failed: {e}")
            return np.zeros((len(pointcloud), 3)), [np.arange(len(pointcloud))]
    
    def cluster_by_spatial_proximity(self, pointcloud: np.ndarray, eps: float = 0.02, 
                                   min_samples: int = 10) -> List[np.ndarray]:
        """
        Cluster points by spatial proximity using DBSCAN.
        
        Args:
            pointcloud: Input pointcloud (N, 3)
            eps: DBSCAN epsilon parameter
            min_samples: Minimum samples per cluster
            
        Returns:
            List of point indices for each cluster
        """
        if len(pointcloud) < min_samples:
            return [np.arange(len(pointcloud))]
        
        try:
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pointcloud)
            labels = clustering.labels_
            
            # Group points by cluster
            clusters = []
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                if label != -1:  # Ignore noise points
                    cluster_indices = np.where(labels == label)[0]
                    clusters.append(cluster_indices)
            
            if self.debug_mode:
                noise_points = np.sum(labels == -1)
                print(f"   🔍 Spatial clustering: {len(clusters)} clusters, {noise_points} noise points")
            
            return clusters
            
        except Exception as e:
            if self.debug_mode:
                print(f"   ⚠️  Spatial clustering failed: {e}")
            return [np.arange(len(pointcloud))]
    
    def filter_by_object_constraints(self, pointcloud: np.ndarray, object_name: str, 
                                   object_pos: np.ndarray, object_info: Dict[str, Any]) -> np.ndarray:
        """
        Filter pointcloud using object-specific geometric constraints.
        
        Args:
            pointcloud: Input pointcloud (N, 3)
            object_name: Name of target object
            object_pos: Object center position
            object_info: Object mesh information
            
        Returns:
            Filtered pointcloud containing only likely object points
        """
        if len(pointcloud) == 0:
            return pointcloud
        
        # Distance-based filtering (but more restrictive than before)
        distances = np.linalg.norm(pointcloud - object_pos[np.newaxis, :], axis=1)
        max_distance = object_info['approximate_size'] * 0.8  # More restrictive
        
        distance_mask = distances <= max_distance
        
        # Height-based filtering (objects are typically above the table)
        table_height = 0.87  # Approximate table height in LIBERO
        object_height_tolerance = 0.25  # 25cm above/below object center
        
        height_mask = np.abs(pointcloud[:, 2] - object_pos[2]) <= object_height_tolerance
        
        # Combine filters
        combined_mask = distance_mask & height_mask
        filtered_pointcloud = pointcloud[combined_mask]
        
        if self.debug_mode:
            print(f"   🎯 Object constraints filter: {len(filtered_pointcloud)}/{len(pointcloud)} points")
            print(f"      Distance threshold: {max_distance:.3f}m")
            print(f"      Height tolerance: ±{object_height_tolerance:.3f}m around z={object_pos[2]:.3f}")
        
        return filtered_pointcloud
    
    def extract_object_only_pointcloud(self, obs: Dict[str, Any], object_name: str, 
                                     object_pos: np.ndarray, camera: str = "agentview") -> np.ndarray:
        """
        Extract pointcloud containing only the target object, excluding nearby surfaces and objects.
        
        Args:
            obs: Environment observation dictionary
            object_name: Name of target object to extract
            object_pos: 3D position of object center in world coordinates
            camera: Camera to use ("agentview" or "robot0_eye_in_hand")
            
        Returns:
            Object-only pointcloud as numpy array (N, 3)
        """
        if self.debug_mode:
            print(f"🎯 Extracting object-only pointcloud for {object_name} using {camera}")
        
        try:
            # Get object mesh information
            object_info = self.get_object_mesh_info(object_name)
            object_info['position'] = object_pos  # Update with actual position
            
            # Get depth and RGB images from observation
            if camera == "agentview":
                depth_key = "agentview_depth"
                rgb_key = "agentview_image"
            elif camera == "robot0_eye_in_hand":
                depth_key = "robot0_eye_in_hand_depth"
                rgb_key = "robot0_eye_in_hand_image"
            else:
                raise ValueError(f"Unknown camera: {camera}")
            
            if depth_key not in obs or rgb_key not in obs:
                raise ValueError(f"Required keys {depth_key} or {rgb_key} not found in observations")
            
            depth_image = obs[depth_key]
            rgb_image = obs[rgb_key]
            
            # Convert depth and RGB to pointcloud using corrected method
            full_pointcloud = self.depth_to_pointcloud(depth_image, rgb_image, camera)
            
            if len(full_pointcloud) == 0:
                return np.zeros((0, 3))
            
            # Step 1: Apply object-specific geometric constraints
            constrained_pointcloud = self.filter_by_object_constraints(
                full_pointcloud, object_name, object_pos, object_info
            )
            
            if len(constrained_pointcloud) == 0:
                if self.debug_mode:
                    print(f"   ⚠️  No points after geometric constraints")
                return np.zeros((0, 3))
            
            # Step 2: Spatial clustering to separate distinct objects/surfaces
            clusters = self.cluster_by_spatial_proximity(constrained_pointcloud, eps=0.03, min_samples=20)
            
            if not clusters:
                if self.debug_mode:
                    print(f"   ⚠️  No clusters found")
                return constrained_pointcloud  # Return constrained points if clustering fails
            
            # Step 3: Select cluster closest to object center
            best_cluster = None
            min_center_distance = float('inf')
            
            for cluster_indices in clusters:
                cluster_points = constrained_pointcloud[cluster_indices]
                cluster_center = np.mean(cluster_points, axis=0)
                center_distance = np.linalg.norm(cluster_center - object_pos)
                
                if center_distance < min_center_distance:
                    min_center_distance = center_distance
                    best_cluster = cluster_indices
            
            if best_cluster is not None:
                object_pointcloud = constrained_pointcloud[best_cluster]
            else:
                object_pointcloud = constrained_pointcloud
            
            # Step 4: Final size validation
            if len(object_pointcloud) < 50:  # Too few points
                if self.debug_mode:
                    print(f"   ⚠️  Too few points after clustering ({len(object_pointcloud)}), using constrained set")
                return constrained_pointcloud
            
            if self.debug_mode:
                print(f"   ✅ Object-only extraction complete: {len(object_pointcloud)} points")
                if len(object_pointcloud) > 0:
                    bounds = {
                        'x': [object_pointcloud[:, 0].min(), object_pointcloud[:, 0].max()],
                        'y': [object_pointcloud[:, 1].min(), object_pointcloud[:, 1].max()],
                        'z': [object_pointcloud[:, 2].min(), object_pointcloud[:, 2].max()]
                    }
                    print(f"      Final bounds: x=[{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}]")
                    print(f"                    y=[{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}]")
                    print(f"                    z=[{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}]")
            
            return object_pointcloud
            
        except Exception as e:
            print(f"❌ Error extracting object-only pointcloud for {object_name}: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((0, 3))


def create_object_segmented_extractor(env, debug_mode: bool = False) -> ObjectSegmentedExtractor:
    """
    Factory function to create ObjectSegmentedExtractor instance.
    
    Args:
        env: LIBERO environment
        debug_mode: Enable debug output
        
    Returns:
        ObjectSegmentedExtractor instance
    """
    return ObjectSegmentedExtractor(env, debug_mode=debug_mode)


def save_object_only_pointcloud_for_grasp_generation(
    env, object_name: str, object_pos: np.ndarray, output_dir: str,
    camera: str = "agentview", formats: list = ["ply", "pcd", "txt"], 
    debug_mode: bool = False
) -> Dict[str, str]:
    """
    High-level function to extract and save object-only pointcloud for grasp generation.
    
    Args:
        env: LIBERO environment
        object_name: Target object name  
        object_pos: Object position in world coordinates
        output_dir: Directory to save pointcloud files
        camera: Camera to use for pointcloud extraction
        formats: List of formats to save pointcloud in
        debug_mode: Enable debug output
        
    Returns:
        Dictionary mapping format -> file path
    """
    # Create object-segmented extractor
    extractor = create_object_segmented_extractor(env, debug_mode=debug_mode)
    
    # Get current observation
    dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])  # 6-DOF + gripper open
    obs, _, _, _ = env.step(dummy_action)
    
    # Extract object-only pointcloud
    object_pointcloud = extractor.extract_object_only_pointcloud(
        obs, object_name, object_pos, camera=camera
    )
    
    # Save in multiple formats
    saved_files = {}
    output_path_base = Path(output_dir) / f"{object_name}_{camera}_object_only"
    
    for fmt in formats:
        file_path = f"{output_path_base}.{fmt}"
        success = extractor.save_pointcloud(object_pointcloud, file_path, format=fmt)
        if success:
            saved_files[fmt] = file_path
    
    return saved_files