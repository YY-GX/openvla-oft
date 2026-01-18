"""
Standalone helpers for generating pointclouds from MuJoCo cameras.

Content:
- depth_im_to_meters / camera intrinsics / extrinsics helpers
- extract_pointcloud_from_camera_adapt3r: per-camera RGBD -> world pointcloud
- extract_object_pointcloud: multi-camera merge + mask filtering for one object
"""

import os
from datetime import datetime

import cv2
import numpy as np
import open3d as o3d
import robosuite.utils.transform_utils as T
from PIL import Image


# ---------------------------- Camera utilities ---------------------------- #
def depth_im_to_meters(depth_im, sim):
    """Convert normalized MuJoCo depth image (0-1) to meters."""
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    return near / (1 - np.array(depth_im) * (1 - near / far))


def get_camera_intrinsic_matrix(camera_name, sim, img_height, img_width):
    """Return 3x3 pinhole intrinsics for a MuJoCo camera."""
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    f = 0.5 * img_height / np.tan(fovy * np.pi / 360)
    return np.array([[f, 0, img_width / 2], [0, f, img_height / 2], [0, 0, 1]])


def get_camera_extrinsic_matrix(camera_name, sim):
    """Return 4x4 camera-to-world pose with axis correction matching Adapt3R."""
    cam_id = sim.model.camera_name2id(camera_name)
    camera_pos = sim.data.cam_xpos[cam_id]
    camera_rot = sim.data.cam_xmat[cam_id].reshape(3, 3)
    pose = T.make_pose(camera_pos, camera_rot)

    # Align axes so +z looks outward along the view direction.
    axis_correction = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    return pose @ axis_correction


def cammat2o3d(cam_mat, width, height):
    """Convert numpy intrinsics to Open3D PinholeCameraIntrinsic."""
    cx = cam_mat[0, 2]
    fx = cam_mat[0, 0]
    cy = cam_mat[1, 2]
    fy = cam_mat[1, 1]
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


# ---------------------------- Pointcloud extraction ---------------------------- #
def extract_pointcloud_from_camera_adapt3r(sim, camera_name, rgb_image, depth_image, img_height, img_width):
    """
    Backproject one camera's RGBD into a world-frame pointcloud using Open3D.

    Returns:
        points_world: (H, W, 3) array aligned with the input image layout.
    """
    depths_m = depth_im_to_meters(depth_image, sim)
    intrinsics = get_camera_intrinsic_matrix(camera_name, sim, img_height, img_width)
    extrinsics = get_camera_extrinsic_matrix(camera_name, sim)

    rgb_im = o3d.geometry.Image(np.ascontiguousarray(rgb_image[::-1]))
    depth_im = o3d.geometry.Image(np.clip(np.ascontiguousarray(depths_m[::-1]), 0, 4))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_im, depth_im, convert_rgb_to_intensity=False, depth_trunc=5, depth_scale=1
    )

    o3d_cam_mat = cammat2o3d(intrinsics, img_width, img_height)
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_cam_mat)
    transformed_cloud = cloud.transform(extrinsics)

    points_world = np.asarray(transformed_cloud.points).reshape(img_height, img_width, 3)
    return points_world[::-1]


def extract_object_pointcloud(
    env,
    object_name,
    camera_names=("agentview", "birdview", "sideview"),
    resolution=512,
    voxel_size=0.01,
    erode_kernel_size=3,
    outlier_std_ratio=2.0,
    save_debug_files=False,
    verbose=False,
):
    """
    Extract a pointcloud for a specific object by merging multiple camera views.

    Pipeline:
    - Render depth + segmentation per camera
    - Convert RGBD to world points (Adapt3R/Open3D)
    - Mask to target object's geoms; optional erosion to drop noisy edges
    - Concatenate across cameras, remove outliers, voxel downsample
    """
    sim = env.sim

    body_id = object_name if isinstance(object_name, int) else sim.model.body_name2id(object_name)

    object_geom_ids = {gid for gid in range(sim.model.ngeom) if sim.model.geom_bodyid[gid] == body_id}
    if len(object_geom_ids) == 0:
        if verbose:
            print(f"   ⚠️  WARNING: No geometries found for object '{object_name}'")
        return np.zeros((0, 3))

    all_points = []
    for camera_name in camera_names:
        try:
            result = sim.render(width=resolution, height=resolution, camera_name=camera_name, depth=True)
            rgb, depth = result if isinstance(result, tuple) else (None, result)
            depth = np.array(depth)
            seg_img = sim.render(width=resolution, height=resolution, camera_name=camera_name, segmentation=True)

            if rgb is not None:
                rgb_image = np.array(rgb)
                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                if len(rgb_image.shape) == 2:
                    rgb_image = np.stack([rgb_image, rgb_image, rgb_image], axis=-1)
            else:
                rgb_image = np.zeros((resolution, resolution, 3), dtype=np.uint8)

            points_world = extract_pointcloud_from_camera_adapt3r(
                sim, camera_name, rgb_image, depth, resolution, resolution
            )
            points_world_flat = points_world.reshape(-1, 3)

            object_mask = np.zeros((resolution, resolution), dtype=bool)
            for geom_id in object_geom_ids:
                object_mask |= seg_img[:, :, 1] == geom_id

            if erode_kernel_size > 0:
                kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
                object_mask = cv2.erode(object_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

            if save_debug_files:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = "scripts/phase3/pipeline/outputs/pointclouds/mplib_debug"
                os.makedirs(output_dir, exist_ok=True)
                seg_output_path = os.path.join(
                    output_dir, f"object_segmentation_{object_name}_{camera_name}_{timestamp}.png"
                )
                seg_image = (object_mask.astype(np.float32) * 255).astype(np.uint8)
                Image.fromarray(seg_image).save(seg_output_path)

            object_points = points_world_flat[object_mask.flatten()]
            if len(object_points) > 0:
                all_points.append(object_points)

        except Exception as e:
            if verbose:
                print(f"   ⚠️  WARNING: Failed to extract from camera '{camera_name}': {e}")
            continue

    if len(all_points) == 0:
        return np.zeros((0, 3))

    merged = np.concatenate(all_points, axis=0)

    if outlier_std_ratio > 0 and len(merged) > 10:
        centroid = merged.mean(axis=0)
        distances = np.linalg.norm(merged - centroid, axis=1)
        mean_dist = distances.mean()
        std_dist = distances.std()
        threshold = mean_dist + std_dist * outlier_std_ratio
        merged = merged[distances <= threshold]

    voxel_indices = np.floor(merged / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    return merged[unique_indices]

