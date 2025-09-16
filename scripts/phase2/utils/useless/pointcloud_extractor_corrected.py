#!/usr/bin/env python3
"""
Corrected Pointcloud Extractor for Phase 2 Pipeline

This module provides corrected pointcloud extraction for target objects from LIBERO 
environment observations, based on the proper implementation found in Adapt3R.

Key corrections made:
1. Proper depth conversion using actual MuJoCo near/far parameters
2. Correct camera intrinsic and extrinsic matrix calculation  
3. Proper coordinate system transformation with camera axis correction
4. Use of Open3D for robust RGBD-to-pointcloud conversion

Functions:
- extract_object_pointcloud(): Extract pointcloud for a specific object
- depth_to_pointcloud(): Convert depth image to 3D pointcloud using proper MuJoCo parameters
- get_camera_intrinsics(): Get actual camera intrinsic parameters from MuJoCo
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

# Import robosuite transform utilities (from Adapt3R)
import robosuite.utils.transform_utils as T


def cammat2o3d(cam_mat, width, height):
    """
    Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
    matrix and image width and height (from Adapt3R)
    """
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


class CorrectedPointcloudExtractor:
    """Extract pointclouds from LIBERO environment observations using corrected implementation."""
    
    def __init__(self, env, debug_mode: bool = False):
        """
        Initialize corrected pointcloud extractor.
        
        Args:
            env: LIBERO environment instance
            debug_mode: Enable debug visualizations and logging
        """
        self.env = env
        self.debug_mode = debug_mode
        self.img_width = 256  # Standard LIBERO resolution
        self.img_height = 256
        
        if self.debug_mode:
            print(f"🔧 CorrectedPointcloudExtractor initialized")
            print(f"   📏 Image resolution: {self.img_width}x{self.img_height}")
    
    def depth_im_to_meters(self, depth_im: np.ndarray) -> np.ndarray:
        """
        Convert normalized depth image to actual depth in meters using MuJoCo parameters.
        This is the corrected implementation from Adapt3R.
        
        Args:
            depth_im: Normalized depth image (0-1)
            
        Returns:
            Depth image in meters
        """
        extent = self.env.sim.model.stat.extent
        near = self.env.sim.model.vis.map.znear * extent
        far = self.env.sim.model.vis.map.zfar * extent
        
        # Corrected depth conversion formula from Adapt3R
        depth_im_m = near / (1 - np.array(depth_im) * (1 - near / far))
        
        if self.debug_mode:
            print(f"   📏 MuJoCo depth params: extent={extent:.3f}, near={near:.3f}, far={far:.3f}")
            print(f"   📏 Depth range: normalized=[{np.min(depth_im):.3f}, {np.max(depth_im):.3f}], meters=[{np.min(depth_im_m):.3f}, {np.max(depth_im_m):.3f}]")
        
        return depth_im_m
    
    def get_camera_intrinsic_matrix(self, camera_name: str) -> np.ndarray:
        """
        Obtains camera intrinsic matrix using the corrected method from Adapt3R.
        
        Args:
            camera_name: Name of camera ("agentview" or "robot0_eye_in_hand")
            
        Returns:
            K: 3x3 camera intrinsic matrix
        """
        cam_id = self.env.sim.model.camera_name2id(camera_name)
        fovy = self.env.sim.model.cam_fovy[cam_id]
        f = 0.5 * self.img_height / np.tan(fovy * np.pi / 360)
        K = np.array([[f, 0, self.img_width / 2], 
                      [0, f, self.img_height / 2], 
                      [0, 0, 1]])
        return K
    
    def get_camera_extrinsic_matrix(self, camera_name: str) -> np.ndarray:
        """
        Returns a 4x4 homogenous matrix corresponding to the camera pose in the
        world frame with proper axis correction (from Adapt3R).
        
        Args:
            camera_name: Name of camera
            
        Returns:
            R: 4x4 camera extrinsic matrix
        """
        cam_id = self.env.sim.model.camera_name2id(camera_name)
        camera_pos = self.env.sim.data.cam_xpos[cam_id]
        camera_rot = self.env.sim.data.cam_xmat[cam_id].reshape(3, 3)
        R = T.make_pose(camera_pos, camera_rot)

        # IMPORTANT! This is the camera axis correction from Adapt3R
        camera_axis_correction = np.array(
            [[1.0, 0.0, 0.0, 0.0], 
             [0.0, -1.0, 0.0, 0.0], 
             [0.0, 0.0, -1.0, 0.0], 
             [0.0, 0.0, 0.0, 1.0]]
        )
        R = R @ camera_axis_correction
        return R
    
    def depth_to_pointcloud(self, depth_image: np.ndarray, rgb_image: np.ndarray, camera_name: str) -> np.ndarray:
        """
        Convert depth and RGB images to 3D pointcloud in world coordinates using 
        the corrected Open3D-based method from Adapt3R.
        
        Args:
            depth_image: Depth image array of shape (H, W) or (H, W, 1)
            rgb_image: RGB image array of shape (H, W, 3)
            camera_name: Name of camera used to capture depth
            
        Returns:
            Pointcloud as numpy array of shape (N, 3) in world coordinates
        """
        # Handle single channel depth images
        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]
        
        # Convert normalized depth to meters using corrected method
        depth_meters = self.depth_im_to_meters(depth_image)
        
        # Get camera parameters
        intrinsic_matrix = self.get_camera_intrinsic_matrix(camera_name)
        extrinsic_matrix = self.get_camera_extrinsic_matrix(camera_name)
        
        # Create Open3D RGBD image (following Adapt3R approach)
        rgb_im = o3d.geometry.Image(np.ascontiguousarray(rgb_image[::-1]))  # Flip for Open3D
        depth_im = o3d.geometry.Image(np.clip(np.ascontiguousarray(depth_meters[::-1]), 0, 4))  # Clip depth
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_im, 
            depth_im, 
            convert_rgb_to_intensity=False,
            depth_trunc=5,
            depth_scale=1
        )
        
        # Convert camera intrinsic matrix to Open3D format
        o3d_cam_mat = cammat2o3d(intrinsic_matrix, self.img_width, self.img_height)
        
        # Create pointcloud from RGBD image
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_cam_mat)
        
        # Transform to world coordinates using extrinsic matrix
        transformed_cloud = cloud.transform(extrinsic_matrix)
        
        # Extract points as numpy array
        points = np.asarray(transformed_cloud.points)
        
        if self.debug_mode:
            print(f"   📊 Generated {len(points)} points from {camera_name}")
            if len(points) > 0:
                print(f"   📍 Point cloud bounds: x=[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
                print(f"                         y=[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]") 
                print(f"                         z=[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        
        return points
    
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
        Extract pointcloud for a specific target object from observations using corrected method.
        
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
                # Save visualization image (headless rendering)
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False)
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


def create_corrected_pointcloud_extractor(env, debug_mode: bool = False) -> CorrectedPointcloudExtractor:
    """
    Factory function to create CorrectedPointcloudExtractor instance.
    
    Args:
        env: LIBERO environment
        debug_mode: Enable debug output
        
    Returns:
        CorrectedPointcloudExtractor instance
    """
    return CorrectedPointcloudExtractor(env, debug_mode=debug_mode)


def save_object_pointcloud_for_grasp_generation_corrected(
    env, object_name: str, object_pos: np.ndarray, output_dir: str,
    camera: str = "agentview", filter_radius: float = 0.15,
    formats: list = ["ply", "pcd", "txt"], debug_mode: bool = False
) -> Dict[str, str]:
    """
    High-level function to extract and save object pointcloud for grasp generation using corrected method.
    
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
    # Create corrected extractor
    extractor = create_corrected_pointcloud_extractor(env, debug_mode=debug_mode)
    
    # Get current observation
    dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])  # 6-DOF + gripper open
    obs, _, _, _ = env.step(dummy_action)
    
    # Extract object pointcloud using corrected method
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