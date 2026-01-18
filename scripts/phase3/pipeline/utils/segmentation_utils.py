"""
Segmentation utilities for wrist camera.
"""
import numpy as np
from typing import Optional, List, Tuple, Set


def get_table_body_name(env) -> Optional[str]:
    """
    Detect the table body name from LIBERO environment.

    This function identifies the workspace table body (kitchen_table, living_room_table,
    study_table) which may have unnamed geoms that can't be detected by keyword matching.

    Args:
        env: LIBERO environment with MuJoCo simulation

    Returns:
        Table body name (e.g., "kitchen_table", "living_room_table", "study_table")
        or None if not found
    """
    # Method 1: Try to get from env attributes (LIBERO envs have workspace_name)
    # Access the base env if wrapped
    base_env = env
    while hasattr(base_env, 'env'):
        if hasattr(base_env, 'workspace_name'):
            return base_env.workspace_name
        base_env = base_env.env

    if hasattr(base_env, 'workspace_name'):
        return base_env.workspace_name

    # Method 2: Search for bodies containing 'table' in name
    sim = env.sim
    table_candidates = []
    workspace_tables = ['kitchen_table', 'living_room_table', 'study_table']

    for body_id in range(sim.model.nbody):
        body_name = sim.model.body_id2name(body_id)
        if body_name and 'table' in body_name.lower():
            # Prioritize workspace table names
            if any(ws in body_name for ws in workspace_tables):
                return body_name
            table_candidates.append(body_name)

    # Fallback: return first table candidate if any
    return table_candidates[0] if table_candidates else None


def get_table_geom_ids(sim, table_body_name: str) -> Set[int]:
    """
    Get all geom IDs belonging to the table body (including child bodies).

    This handles tables with unnamed collision geoms (like living_room_table)
    by collecting ALL geoms under the table body hierarchy.

    Args:
        sim: MuJoCo simulation object
        table_body_name: Table body name (e.g., "living_room_table")

    Returns:
        Set of geom IDs belonging to the table and its child bodies
    """
    table_geom_ids = set()

    if not table_body_name:
        return table_geom_ids

    # Find all bodies that are children of the table body
    # (e.g., "living_room_table", "living_room_table_col")
    prefix = table_body_name if not table_body_name.endswith('_') else table_body_name

    # Find all bodies starting with this prefix
    for body_id in range(sim.model.nbody):
        body_name = sim.model.body_id2name(body_id)
        if body_name and body_name.startswith(prefix):
            # Collect all geoms belonging to this body
            for geom_id in range(sim.model.ngeom):
                if sim.model.geom_bodyid[geom_id] == body_id:
                    table_geom_ids.add(geom_id)

    return table_geom_ids


def get_related_bodies(sim, target_object: str) -> List[str]:
    """
    Automatically find all related bodies for hierarchical objects.

    For hierarchical objects (stove, cabinet, etc.), finds all bodies that share
    the same prefix. This ensures all child bodies (burners, knobs, drawers, etc.)
    are included without hardcoding.

    Examples:
        "flat_stove_2_main" -> ["flat_stove_2_main", "flat_stove_2_burner",
                                "flat_stove_2_g4", "flat_stove_2_burner_plate"]
        "wooden_cabinet_1_cabinet_top" -> ["wooden_cabinet_1_main",
                                           "wooden_cabinet_1_cabinet_top", ...]

    Args:
        sim: MuJoCo simulation object
        target_object: Target object name (e.g., "flat_stove_2_main")

    Returns:
        List of related body names (including the target object itself)
    """
    # Extract prefix from target_object
    # "flat_stove_2_main" -> "flat_stove_2_"
    if "_main" in target_object:
        prefix = target_object.split("_main")[0] + "_"
    else:
        # For objects like "wooden_cabinet_1_cabinet_top" -> "wooden_cabinet_1_"
        # Extract everything before the last underscore
        parts = target_object.rsplit("_", 1)
        if len(parts) > 1:
            prefix = parts[0] + "_"
        else:
            # No underscore found, return object as-is
            return [target_object]

    # Find all bodies starting with this prefix
    related_bodies = []
    for body_id in range(sim.model.nbody):
        body_name = sim.model.body_id2name(body_id)
        if body_name and body_name.startswith(prefix):
            related_bodies.append(body_name)

    # Return found bodies, or original target if nothing found
    return related_bodies if related_bodies else [target_object]


def create_wrist_segmentation_mask(env, target_object: str, resolution: int = 256) -> np.ndarray:
    """
    Create segmentation mask for wrist camera showing only robot gripper + target object.

    Target object pixels are drawn AFTER robot pixels, so target object overwrites any
    overlapping robot pixels (e.g., black lines from gripper).

    This function saves and restores simulation state to prevent sim.render() from
    corrupting the simulation during trajectory execution.

    Args:
        env: LIBERO environment
        target_object: Target object name (e.g., "akita_black_bowl_1_main")
        resolution: Image resolution (default 256x256)

    Returns:
        Binary segmentation mask (resolution x resolution) as uint8 (0 or 255)
    """
    sim = env.sim
    camera_name = "robot0_eye_in_hand"

    # Save simulation state before rendering to prevent corruption
    qpos_before = sim.data.qpos.copy()
    qvel_before = sim.data.qvel.copy()
    ctrl_before = sim.data.ctrl.copy()

    # Render segmentation image
    seg_img = sim.render(width=resolution, height=resolution, camera_name=camera_name, segmentation=True)

    # Restore simulation state after rendering
    sim.data.qpos[:] = qpos_before
    sim.data.qvel[:] = qvel_before
    sim.data.ctrl[:] = ctrl_before
    sim.forward()

    # Collect robot gripper geom IDs (gripper, finger, hand, pad, link6, link7)
    # IMPORTANT: This includes ALL gripper geoms (visual + collision)
    #   - Visual geoms: Visible parts (hand_visual, finger_visual)
    #   - Collision geoms: Invisible physics geoms (hand_collision, finger_collision, pad_collision)
    #   - Robot arm end-effector links: robot0_link6, robot0_link7 (wrist links visible in wrist camera)
    #   - The link6/link7 collision geoms include invisible structures like the line between gripper
    #   - ALL collision geoms are INTENTIONALLY included to prevent random erasing of gripper structure
    robot_keywords = ['gripper', 'finger', 'hand', 'pad', 'robot0_link6', 'robot0_link7']
    robot_geom_ids = set()
    for geom_id in range(sim.model.ngeom):
        geom_name = sim.model.geom_id2name(geom_id)
        if geom_name and any(kw in geom_name.lower() for kw in robot_keywords):
            robot_geom_ids.add(geom_id)

    # Collect target object geom IDs
    target_geom_ids = set()

    # Automatically find all related bodies (handles hierarchical objects like stove, cabinet)
    bodies_to_check = get_related_bodies(sim, target_object)

    # Collect geoms from all relevant bodies
    for body_name in bodies_to_check:
        try:
            body_id = sim.model.body_name2id(body_name)
            for geom_id in range(sim.model.ngeom):
                if sim.model.geom_bodyid[geom_id] == body_id:
                    target_geom_ids.add(geom_id)
        except:
            pass  # Body not found, skip

    # Create mask: robot first, then target object (so target overwrites robot)
    mask = np.zeros((resolution, resolution), dtype=bool)

    # Add robot gripper pixels
    for geom_id in robot_geom_ids:
        mask |= (seg_img[:, :, 1] == geom_id)

    # Add target object pixels (overwrites robot if overlapping)
    for geom_id in target_geom_ids:
        mask |= (seg_img[:, :, 1] == geom_id)

    # Convert to uint8 (0 or 255)
    return (mask.astype(np.float32) * 255).astype(np.uint8)


def create_wrist_segmentation_mask_with_grasped(env, target_object: str, grasped_object_name: Optional[str], resolution: int = 256) -> np.ndarray:
    """
    Create segmentation mask for wrist camera showing robot gripper + target object + grasped object.
    
    For place skills, this includes the grasped object that is being held.

    Args:
        env: LIBERO environment
        target_object: Target object name (e.g., "wooden_cabinet_1_cabinet_bottom")
        grasped_object_name: Grasped object name (e.g., "akita_black_bowl_1_main") or None
        resolution: Image resolution (default 256x256)

    Returns:
        Binary segmentation mask (resolution x resolution) as uint8 (0 or 255)
    """
    sim = env.sim
    camera_name = "robot0_eye_in_hand"

    # Save simulation state before rendering to prevent corruption
    qpos_before = sim.data.qpos.copy()
    qvel_before = sim.data.qvel.copy()
    ctrl_before = sim.data.ctrl.copy()

    # Render segmentation image
    seg_img = sim.render(width=resolution, height=resolution, camera_name=camera_name, segmentation=True)

    # Restore simulation state after rendering
    sim.data.qpos[:] = qpos_before
    sim.data.qvel[:] = qvel_before
    sim.data.ctrl[:] = ctrl_before
    sim.forward()

    # Collect robot gripper geom IDs (gripper, finger, hand, pad, link6, link7)
    # IMPORTANT: This includes ALL gripper geoms (visual + collision)
    #   - Visual geoms: Visible parts (hand_visual, finger_visual)
    #   - Collision geoms: Invisible physics geoms (hand_collision, finger_collision, pad_collision)
    #   - Robot arm end-effector links: robot0_link6, robot0_link7 (wrist links visible in wrist camera)
    #   - The link6/link7 collision geoms include invisible structures like the line between gripper
    #   - ALL collision geoms are INTENTIONALLY included to prevent random erasing of gripper structure
    robot_keywords = ['gripper', 'finger', 'hand', 'pad', 'robot0_link6', 'robot0_link7']
    robot_geom_ids = set()
    for geom_id in range(sim.model.ngeom):
        geom_name = sim.model.geom_id2name(geom_id)
        if geom_name and any(kw in geom_name.lower() for kw in robot_keywords):
            robot_geom_ids.add(geom_id)

    # Collect target object geom IDs
    target_geom_ids = set()

    # Automatically find all related bodies (handles hierarchical objects like stove, cabinet)
    bodies_to_check = get_related_bodies(sim, target_object)

    for body_name in bodies_to_check:
        try:
            body_id = sim.model.body_name2id(body_name)
            for geom_id in range(sim.model.ngeom):
                if sim.model.geom_bodyid[geom_id] == body_id:
                    target_geom_ids.add(geom_id)
        except:
            pass

    # Collect grasped object geom IDs
    grasped_geom_ids = set()
    if grasped_object_name is not None:
        # Automatically find all related bodies for grasped object (handles hierarchical objects)
        grasped_bodies = get_related_bodies(sim, grasped_object_name)
        for body_name in grasped_bodies:
            try:
                body_id = sim.model.body_name2id(body_name)
                for geom_id in range(sim.model.ngeom):
                    if sim.model.geom_bodyid[geom_id] == body_id:
                        grasped_geom_ids.add(geom_id)
            except:
                pass  # Body not found, skip

    # Create mask: robot first, then grasped object, then target object (so target overwrites)
    mask = np.zeros((resolution, resolution), dtype=bool)

    # Add robot gripper pixels
    for geom_id in robot_geom_ids:
        mask |= (seg_img[:, :, 1] == geom_id)

    # Add grasped object pixels (overwrites robot if overlapping)
    for geom_id in grasped_geom_ids:
        mask |= (seg_img[:, :, 1] == geom_id)

    # Add target object pixels (overwrites robot and grasped if overlapping)
    for geom_id in target_geom_ids:
        mask |= (seg_img[:, :, 1] == geom_id)

    # Convert to uint8 (0 or 255)
    return (mask.astype(np.float32) * 255).astype(np.uint8)


def apply_distractor_masking(
    env,
    wrist_img: np.ndarray,
    target_object: str,
    grasped_object_name: Optional[str] = None,
    resolution: int = 256,
    unmask_important_objects: bool = True
) -> np.ndarray:
    """
    Mask out distractor objects in wrist camera view by blacking out their bounding boxes.

    Keeps visible: gripper + target object + grasped object + background/table
    Masks out: all other objects (distractor objects like other bowls, plates, etc.)

    Args:
        env: LIBERO environment
        wrist_img: Input wrist camera image (H, W, 3), dtype uint8
        target_object: Target object name (e.g., "akita_black_bowl_1_main")
        grasped_object_name: Grasped object name (e.g., "akita_black_bowl_2_main") or None
        resolution: Image resolution (default 256x256)
        unmask_important_objects: If True, restore important objects (gripper/target/grasped)
                                   that may have been accidentally masked by large distractor bboxes

    Returns:
        Masked image with distractor objects blacked out (H, W, 3), dtype uint8
    """

    sim = env.sim
    camera_name = "robot0_eye_in_hand"

    # Save simulation state before rendering
    qpos_before = sim.data.qpos.copy()
    qvel_before = sim.data.qvel.copy()
    ctrl_before = sim.data.ctrl.copy()

    # Render segmentation image
    seg_img = sim.render(width=resolution, height=resolution, camera_name=camera_name, segmentation=True)

    # Rotate segmentation 180° to match wrist image preprocessing (libero_utils.py: img[::-1, ::-1])
    # This ensures bounding box coordinates align with the rotated wrist image
    seg_img = seg_img[::-1, ::-1]

    # Restore simulation state
    sim.data.qpos[:] = qpos_before
    sim.data.qvel[:] = qvel_before
    sim.data.ctrl[:] = ctrl_before
    sim.forward()

    # Collect robot gripper geom IDs (gripper, finger, hand, pad, link6, link7)
    # IMPORTANT: This includes ALL gripper geoms (visual + collision)
    #   - Visual geoms: Visible parts (hand_visual, finger_visual)
    #   - Collision geoms: Invisible physics geoms (hand_collision, finger_collision, pad_collision)
    #   - Robot arm end-effector links: robot0_link6, robot0_link7 (wrist links visible in wrist camera)
    #   - The link6/link7 collision geoms include invisible structures like the line between gripper
    #   - ALL collision geoms are INTENTIONALLY included to prevent random erasing of gripper structure
    robot_keywords = ['gripper', 'finger', 'hand', 'pad', 'robot0_link6', 'robot0_link7']
    robot_geom_ids = set()
    for geom_id in range(sim.model.ngeom):
        geom_name = sim.model.geom_id2name(geom_id)
        if geom_name and any(kw in geom_name.lower() for kw in robot_keywords):
            robot_geom_ids.add(geom_id)

    # Collect target object geom IDs
    target_geom_ids = set()

    # Automatically find all related bodies (handles hierarchical objects like stove, cabinet)
    target_bodies = get_related_bodies(sim, target_object)

    for body_name in target_bodies:
        try:
            body_id = sim.model.body_name2id(body_name)
            for geom_id in range(sim.model.ngeom):
                if sim.model.geom_bodyid[geom_id] == body_id:
                    target_geom_ids.add(geom_id)
        except:
            pass

    # Collect grasped object geom IDs
    grasped_geom_ids = set()
    if grasped_object_name is not None:
        # Automatically find all related bodies for grasped object (handles hierarchical objects)
        grasped_bodies = get_related_bodies(sim, grasped_object_name)
        for body_name in grasped_bodies:
            try:
                body_id = sim.model.body_name2id(body_name)
                for geom_id in range(sim.model.ngeom):
                    if sim.model.geom_bodyid[geom_id] == body_id:
                        grasped_geom_ids.add(geom_id)
            except:
                pass

    # Collect static furniture/background geom IDs (table, cabinet bases, walls, etc.)
    # Method 1: Body-based table detection (robust, handles unnamed geoms in living_room_table)
    table_body_name = get_table_body_name(env)
    table_geom_ids = get_table_geom_ids(sim, table_body_name) if table_body_name else set()

    # Method 2: Keyword-based detection (handles named geoms in kitchen_table + other static objects)
    # Keep 'table' in keywords to handle kitchen_table (which has named geoms)
    # Body-based detection handles living_room_table (which has unnamed geoms)
    # Set union prevents duplicates between both methods
    static_keywords = ['floor', 'wall', 'table', 'room', 'counter', 'shelf']
    keyword_geom_ids = set()
    for geom_id in range(sim.model.ngeom):
        geom_name = sim.model.geom_id2name(geom_id)
        if geom_name and any(kw in geom_name.lower() for kw in static_keywords):
            keyword_geom_ids.add(geom_id)

    # Combine both methods (set union prevents duplicates)
    static_geom_ids = table_geom_ids | keyword_geom_ids


    # Identify all geom IDs to KEEP (don't mask)
    keep_geom_ids = robot_geom_ids | target_geom_ids | grasped_geom_ids | static_geom_ids

    # Find distractor objects: all other geoms that appear in the image
    distractor_geom_ids = set()
    unique_geoms = np.unique(seg_img[:, :, 1])
    for geom_id in unique_geoms:
        if geom_id > 0 and geom_id not in keep_geom_ids:  # geom_id=0 is background
            distractor_geom_ids.add(geom_id)
            geom_name = sim.model.geom_id2name(geom_id) if geom_id < sim.model.ngeom else "unknown"

    # For each distractor object, compute its bounding box and black it out
    masked_img = wrist_img.copy()
    num_masked = 0

    for geom_id in distractor_geom_ids:
        # Find all pixels belonging to this distractor geom
        distractor_mask = (seg_img[:, :, 1] == geom_id)

        if not np.any(distractor_mask):
            continue

        # Compute bounding box (min rectangle)
        rows, cols = np.where(distractor_mask)
        if len(rows) == 0:
            continue

        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()
        bbox_size = (y_max - y_min + 1) * (x_max - x_min + 1)

        # Black out the bounding box region
        masked_img[y_min:y_max+1, x_min:x_max+1] = 0
        num_masked += 1
        geom_name = sim.model.geom_id2name(geom_id) if geom_id < sim.model.ngeom else "unknown"


    # Restore important objects (gripper/target/grasped) that may have been accidentally masked
    # by large distractor bounding boxes
    if unmask_important_objects:
        # Create mask of important pixels (robot + target + grasped)
        important_mask = np.zeros((resolution, resolution), dtype=bool)

        # Add robot gripper pixels
        for geom_id in robot_geom_ids:
            important_mask |= (seg_img[:, :, 1] == geom_id)

        # Add target object pixels
        for geom_id in target_geom_ids:
            important_mask |= (seg_img[:, :, 1] == geom_id)

        # Add grasped object pixels
        for geom_id in grasped_geom_ids:
            important_mask |= (seg_img[:, :, 1] == geom_id)

        # Restore original pixels for important objects
        num_restored = np.sum(important_mask)
        if num_restored > 0:
            masked_img[important_mask] = wrist_img[important_mask]

    return masked_img
