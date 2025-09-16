#!/usr/bin/env python3
"""
Pointcloud Extractor for Phase 2 Pipeline

This module extracts pointclouds of target objects from LIBERO environment observations.
Key functions:
- extract_object_pointcloud(): Extract pointcloud for a specific object
- depth_to_pointcloud(): Convert depth image to 3D pointcloud 
- get_camera_intrinsics(): Get camera intrinsic parameters from MuJoCo
- filter_object_pointcloud(): Filter pointcloud to contain only target object
- save_pointcloud(): Save pointcloud in formats compatible with AnyGrasp/GraspGen
"""

import os
import sys
import numpy as np
import open3d as o3d
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import json

# Add project paths
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')

# scipy for spatial operations
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


class PointcloudExtractor:
    """Extract pointclouds from LIBERO environment observations."""
    
    def __init__(self, env, debug_mode: bool = False):
        """
        Initialize pointcloud extractor.
        
        Args:
            env: LIBERO environment instance
            debug_mode: Enable debug visualizations and logging
        """
        self.env = env
        self.debug_mode = debug_mode
        
        # Get camera intrinsics for both cameras
        self.agentview_intrinsics = self._get_camera_intrinsics("agentview")
        self.wrist_intrinsics = self._get_camera_intrinsics("robot0_eye_in_hand")
        
        if self.debug_mode:
            print(f"🔧 PointcloudExtractor initialized")
            print(f"   📷 Agent view intrinsics: {self.agentview_intrinsics}")
            print(f"   🤏 Wrist view intrinsics: {self.wrist_intrinsics}")
    
    def _get_camera_intrinsics(self, camera_name: str) -> Dict[str, float]:
        """
        Extract camera intrinsic parameters from MuJoCo simulation.
        
        Args:
            camera_name: Name of camera ("agentview" or "robot0_eye_in_hand")
            
        Returns:
            Dictionary with intrinsic parameters: fx, fy, cx, cy, width, height
        """
        try:
            model = self.env.sim.model
            
            # Find camera ID
            cam_id = None
            for i in range(model.ncam):
                if model.camera_id2name(i) == camera_name:
                    cam_id = i
                    break
            
            if cam_id is None:
                raise ValueError(f"Camera '{camera_name}' not found")
            
            # Get field of view in Y direction (in radians)
            fovy_rad = model.cam_fovy[cam_id] * np.pi / 180.0
            
            # Assume square image (common in LIBERO)
            height = width = 256  # Default LIBERO resolution
            
            # Calculate focal length from field of view
            # fovy = 2 * arctan(height / (2 * fy))
            # fy = height / (2 * tan(fovy/2))
            fy = height / (2.0 * np.tan(fovy_rad / 2.0))
            fx = fy  # Assume square pixels
            
            # Principal point (center of image)
            cx = width / 2.0
            cy = height / 2.0
            
            intrinsics = {
                'fx': fx,
                'fy': fy, 
                'cx': cx,
                'cy': cy,
                'width': width,
                'height': height,
                'fovy_deg': model.cam_fovy[cam_id],
                'fovy_rad': fovy_rad
            }
            
            return intrinsics
            
        except Exception as e:
            print(f"❌ Error getting camera intrinsics for {camera_name}: {e}")
            # Fallback intrinsics for 256x256 images
            return {
                'fx': 200.0, 'fy': 200.0, 
                'cx': 128.0, 'cy': 128.0,
                'width': 256, 'height': 256,
                'fovy_deg': 45.0, 'fovy_rad': 45.0 * np.pi / 180.0
            }
    
    def _get_camera_extrinsics(self, camera_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get camera extrinsic parameters (position and orientation in world frame).
        
        Args:
            camera_name: Name of camera
            
        Returns:
            Tuple of (camera_pos, camera_rot_matrix) in world coordinates
        """
        try:
            model = self.env.sim.model
            data = self.env.sim.data
            
            # Find camera ID
            cam_id = None
            for i in range(model.ncam):
                if model.camera_id2name(i) == camera_name:
                    cam_id = i
                    break
            
            if cam_id is None:
                raise ValueError(f"Camera '{camera_name}' not found")
            
            # Get camera position and orientation
            camera_pos = data.cam_xpos[cam_id].copy()
            camera_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()
            
            return camera_pos, camera_mat
            
        except Exception as e:
            print(f"❌ Error getting camera extrinsics for {camera_name}: {e}")
            # Return identity as fallback
            return np.zeros(3), np.eye(3)
    
    def _convert_normalized_depth_to_meters(self, normalized_depth: np.ndarray, camera_name: str) -> np.ndarray:
        """
        Convert normalized depth values (0-1) to actual depth in meters.
        
        Based on investigation of LIBERO environments, the effective range appears to be
        optimized for table-top robotics scenarios.
        
        Args:
            normalized_depth: Normalized depth values from MuJoCo (0-1)
            camera_name: Camera name (for future camera-specific parameters)
            
        Returns:
            Actual depth values in meters
        """
        # Based on investigation, these parameters give reasonable depth ranges
        # for LIBERO table-top robotics scenarios
        near_plane = 0.5   # 0.5m (50cm from camera)
        far_plane = 3.0    # 3.0m (3m from camera)
        
        # Convert normalized depth to actual depth using linear interpolation
        # depth_meters = near + normalized_depth * (far - near)
        depth_meters = near_plane + normalized_depth * (far_plane - near_plane)
        
        return depth_meters

    def depth_to_pointcloud(self, depth_image: np.ndarray, camera_name: str) -> np.ndarray:
        """
        Convert depth image to 3D pointcloud in world coordinates.
        
        Args:
            depth_image: Depth image array of shape (H, W) or (H, W, 1)
            camera_name: Name of camera used to capture depth
            
        Returns:
            Pointcloud as numpy array of shape (N, 3) in world coordinates
        """
        # Handle single channel depth images
        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]
        
        # Get camera parameters
        if camera_name == "agentview":
            intrinsics = self.agentview_intrinsics
        elif camera_name == "robot0_eye_in_hand":
            intrinsics = self.wrist_intrinsics
        else:
            raise ValueError(f"Unknown camera: {camera_name}")
        
        # Get camera extrinsics
        camera_pos, camera_rot = self._get_camera_extrinsics(camera_name)
        
        # Convert normalized depth to actual depth in meters
        depth_meters = self._convert_normalized_depth_to_meters(depth_image, camera_name)
        
        # Create pixel coordinate arrays
        height, width = depth_image.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u = u.flatten()
        v = v.flatten()
        depth = depth_meters.flatten()
        
        # Filter out invalid depth values (reasonable range for robotics - be more permissive)
        valid_mask = (depth > 0.1) & (depth < 10.0) & np.isfinite(depth)
        u = u[valid_mask]
        v = v[valid_mask]
        depth = depth[valid_mask]
        
        if len(depth) == 0:
            print(f"⚠️ Warning: No valid depth values found")
            return np.zeros((0, 3))
        
        # Convert to normalized camera coordinates
        x_cam = (u - intrinsics['cx']) * depth / intrinsics['fx']
        y_cam = (v - intrinsics['cy']) * depth / intrinsics['fy']
        z_cam = depth
        
        # Stack into camera coordinate points
        points_cam = np.column_stack([x_cam, y_cam, z_cam])
        
        # Transform to world coordinates
        # points_world = camera_rot @ points_cam.T + camera_pos[:, np.newaxis]
        points_world = (camera_rot @ points_cam.T).T + camera_pos
        
        if self.debug_mode:
            print(f"   📊 Generated {len(points_world)} points from {camera_name}")
            print(f"   🔢 Depth range: normalized=[{depth_image.min():.3f}, {depth_image.max():.3f}], meters=[{depth.min():.3f}, {depth.max():.3f}]")
            print(f"   📍 Point cloud bounds: x=[{points_world[:, 0].min():.3f}, {points_world[:, 0].max():.3f}]")
            print(f"                         y=[{points_world[:, 1].min():.3f}, {points_world[:, 1].max():.3f}]") 
            print(f"                         z=[{points_world[:, 2].min():.3f}, {points_world[:, 2].max():.3f}]")
        
        return points_world
    
    def filter_object_pointcloud(self, pointcloud: np.ndarray, object_name: str, 
                                object_pos: np.ndarray, filter_radius: float = 0.15) -> np.ndarray:
        """
        Filter pointcloud to contain only points near the target object.
        
        Args:
            pointcloud: Full scene pointcloud (N, 3)
            object_name: Name of target object
            object_pos: 3D position of target object center
            filter_radius: Radius around object to include points (meters)
            
        Returns:
            Filtered pointcloud containing only object points
        """
        if len(pointcloud) == 0:
            return pointcloud
        
        # Calculate distances from all points to object center
        distances = np.linalg.norm(pointcloud - object_pos[np.newaxis, :], axis=1)
        
        # Filter points within radius
        object_mask = distances <= filter_radius
        object_pointcloud = pointcloud[object_mask]
        
        if self.debug_mode:
            print(f"   🎯 Filtered {len(object_pointcloud)}/{len(pointcloud)} points for {object_name}")
            print(f"   📏 Filter radius: {filter_radius:.3f}m around {object_pos}")
        
        return object_pointcloud
    
    def extract_object_pointcloud(self, obs: Dict[str, Any], object_name: str, 
                                 object_pos: np.ndarray, camera: str = "agentview",
                                 filter_radius: float = 0.15) -> np.ndarray:
        """
        Extract pointcloud for a specific target object from observations.
        
        Args:
            obs: Environment observation dictionary
            object_name: Name of target object to extract
            object_pos: 3D position of object center in world coordinates
            camera: Camera to use ("agentview" or "robot0_eye_in_hand")
            filter_radius: Radius around object to include points
            
        Returns:
            Object pointcloud as numpy array (N, 3)
        """
        if self.debug_mode:
            print(f"🔍 Extracting pointcloud for {object_name} using {camera} camera")
        
        try:
            # Get depth image from observation
            if camera == "agentview":
                depth_key = "agentview_depth"
            elif camera == "robot0_eye_in_hand":
                depth_key = "robot0_eye_in_hand_depth"
            else:
                raise ValueError(f"Unknown camera: {camera}")
            
            if depth_key not in obs:
                raise ValueError(f"Depth image {depth_key} not found in observations")
            
            depth_image = obs[depth_key]
            
            # Convert depth to pointcloud
            full_pointcloud = self.depth_to_pointcloud(depth_image, camera)
            
            # Filter to object region
            object_pointcloud = self.filter_object_pointcloud(
                full_pointcloud, object_name, object_pos, filter_radius
            )
            
            return object_pointcloud
            
        except Exception as e:
            print(f"❌ Error extracting pointcloud for {object_name}: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((0, 3))
    
    def save_pointcloud(self, pointcloud: np.ndarray, output_path: str, 
                       format: str = "ply") -> bool:
        """
        Save pointcloud to file in format compatible with grasp generation methods.
        
        Args:
            pointcloud: Pointcloud array (N, 3)
            output_path: Output file path
            format: Output format ("ply", "pcd", "txt", "npy")
            
        Returns:
            True if saved successfully, False otherwise
        """
        if len(pointcloud) == 0:
            print(f"⚠️ Warning: Empty pointcloud, not saving to {output_path}")
            return False
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "ply":
                # Save as PLY using Open3D
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pointcloud)
                o3d.io.write_point_cloud(str(output_path), pcd)
                
            elif format.lower() == "pcd":
                # Save as PCD using Open3D  
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pointcloud)
                o3d.io.write_point_cloud(str(output_path), pcd)
                
            elif format.lower() == "txt":
                # Save as simple text file (x y z per line)
                np.savetxt(str(output_path), pointcloud, fmt='%.6f')
                
            elif format.lower() == "npy":
                # Save as numpy binary file
                np.save(str(output_path), pointcloud)
                
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            if self.debug_mode:
                print(f"✅ Saved pointcloud ({len(pointcloud)} points) to {output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving pointcloud to {output_path}: {e}")
            return False
    
    def visualize_pointcloud(self, pointcloud: np.ndarray, object_name: str = "object",
                           save_image: bool = True, output_dir: str = "/tmp") -> bool:
        """
        Visualize pointcloud using Open3D (optional for debugging).
        
        Args:
            pointcloud: Pointcloud to visualize (N, 3)
            object_name: Name for visualization window
            save_image: Whether to save visualization image
            output_dir: Directory to save visualization image
            
        Returns:
            True if visualization successful, False otherwise
        """
        if len(pointcloud) == 0:
            print(f"⚠️ Warning: Empty pointcloud, skipping visualization")
            return False
        
        try:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud)
            
            # Color points (optional)
            colors = np.tile([0.7, 0.3, 0.3], (len(pointcloud), 1))  # Red-ish color
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            if save_image:
                # Save visualization image
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False)  # Headless rendering
                vis.add_geometry(pcd)
                
                # Set view point
                vis.get_view_control().set_front([0, 0, -1])
                vis.get_view_control().set_up([0, -1, 0])
                vis.get_view_control().set_zoom(0.8)
                
                # Capture image
                output_path = Path(output_dir) / f"{object_name}_pointcloud.png"
                vis.capture_screen_image(str(output_path))
                vis.destroy_window()
                
                if self.debug_mode:
                    print(f"📸 Saved pointcloud visualization to {output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error visualizing pointcloud: {e}")
            return False


def create_pointcloud_extractor(env, debug_mode: bool = False) -> PointcloudExtractor:
    """
    Factory function to create PointcloudExtractor instance.
    
    Args:
        env: LIBERO environment
        debug_mode: Enable debug output
        
    Returns:
        PointcloudExtractor instance
    """
    return PointcloudExtractor(env, debug_mode=debug_mode)


def save_object_pointcloud_for_grasp_generation(
    env, object_name: str, object_pos: np.ndarray, output_dir: str,
    camera: str = "agentview", filter_radius: float = 0.15,
    formats: list = ["ply", "pcd", "txt"], debug_mode: bool = False
) -> Dict[str, str]:
    """
    High-level function to extract and save object pointcloud for grasp generation.
    
    Args:
        env: LIBERO environment
        object_name: Target object name  
        object_pos: Object position in world coordinates
        output_dir: Directory to save pointcloud files
        camera: Camera to use for pointcloud extraction
        filter_radius: Radius for filtering object points
        formats: List of formats to save pointcloud in
        debug_mode: Enable debug output
        
    Returns:
        Dictionary mapping format -> file path
    """
    # Create extractor
    extractor = create_pointcloud_extractor(env, debug_mode=debug_mode)
    
    # Get current observation
    dummy_action = np.zeros(7)
    obs, _, _, _ = env.step(dummy_action)
    
    # Extract object pointcloud
    object_pointcloud = extractor.extract_object_pointcloud(
        obs, object_name, object_pos, camera=camera, filter_radius=filter_radius
    )
    
    # Save in multiple formats
    saved_files = {}
    output_path_base = Path(output_dir) / f"{object_name}_{camera}"
    
    for fmt in formats:
        file_path = f"{output_path_base}.{fmt}"
        success = extractor.save_pointcloud(object_pointcloud, file_path, format=fmt)
        if success:
            saved_files[fmt] = file_path
    
    # Optional visualization
    if debug_mode:
        extractor.visualize_pointcloud(object_pointcloud, object_name, 
                                     save_image=True, output_dir=output_dir)
    
    return saved_files