# MPlib Grasped Object Collision Issue - Root Cause Analysis

## Problem Statement
MPlib planning shows grasped object bbox is correct in PLY visualization and scene pointcloud is correct, but during execution, the grasped object collides with scene objects.

## Code Flow Analysis

### 1. Setup Phase (`planner.py:296-332`)
- Creates `attached_obj` dict with `aabb` (half-extents × safety_margin) and `grasp_pose`
- Only when `skill_type == "place"` AND `grasped_object_name is not None`

### 2. Planning Phase (`planner_core.py:851-873`)
```python
plan_to_pose_collision_aware():
  Line 853: planner.clear_scene()           # Clears pointclouds ONLY
  Line 857: planner.update_scene(scene_pts)  # Adds scene
  Line 860: planner.attach_obj(aabb, grasp_pose)  # ← Attaches box to move_group link
  Line 863: planner.plan_to_pose()          # Starts planning
```

### 3. MPlib Attachment (`mplib_core.py:61-70`)
```python
attach_obj():
  Line 68: full_size = np.array(aabb) * 2.0      # Half-extents → Full size
  Line 69: self.planner.update_attached_box(full_size, pose=Pose(grasp_pose), ...)
```

### 4. Planning Strategies (`planning_strategies.py:150`)
```python
planner.planner.plan_pose(...)  # MPlib internal RRT planner
```

## Verification Points

✅ **Attached object IS set** - Line 860 calls `attach_obj()` before planning
✅ **Scene IS set** - Line 857 adds pointcloud before planning
✅ **No clearing during perturbations** - Attached object persists across all 50 perturbation attempts
✅ **Correct conversion** - Half-extents × 2 = Full size (MPlib requirement)

## Possible Root Causes

### 1. **Grasp Pose Computation Error** (MOST LIKELY)

**Location**: `planner.py:323-325`
```python
grasp_pose = compute_grasp_pose_relative_to_move_group(
    bbox["pose"], controller_pose, self.planner.mv_link_to_ctrl
)
```

**Problem**:
- `grasp_pose` must be relative to move_group link frame
- If transformation is wrong, attached box appears at **wrong location** during planning
- MPlib would check collisions at wrong position → allows invalid paths

**Check**:
- Verify `bbox["pose"]` is object's AABB center (not pointcloud center)
- Verify `mv_link_to_ctrl` transformation direction
- Line 324: Uses `mv_link_to_ctrl` (controller → move_group transform)

### 2. **Safety Margin Double-Application** (LIKELY)

**Location**: `planner_core.py:739` + `mplib_core.py:68`

```python
# planner_core.py:739
aabb_half_extents = (maxs - mins) / 2.0 * safety_margin  # Already inflated

# mplib_core.py:68
full_size = np.array(aabb) * 2.0  # Convert to full size
# Result: full_size = (maxs - mins) * safety_margin
```

**Problem**:
- If `safety_margin=1.5`, bbox is 50% larger than actual object
- But visualization shows **correct** bbox in PLY (before ×2 conversion)
- MPlib uses **full_size** which is ×2 of what you see in PLY
- If bbox in PLY looks correct, MPlib sees a bbox **2× that size**

**Discrepancy**:
```
PLY visualization: half_extents × 1.5 (safety_margin)
MPlib collision:   half_extents × 1.5 × 2.0 = 3.0× half_extents
```

### 3. **AABB Center vs Pointcloud Center Mismatch** (LIKELY)

**Location**: `planner_core.py:750-755`

```python
# Line 751: AABB center (accounting for asymmetry)
aabb_center_world = pts_center_world + aabb_center_offset_world

# Line 754-755: Pose is at AABB center
obj_pose[:3, :3] = body_rot
obj_pose[:3, 3] = aabb_center_world
```

**Problem**:
- `grasp_pose` is computed using `bbox["pose"]` which is AABB center
- But during execution, object is at MuJoCo body center (not AABB center)
- For asymmetric objects (bowls, pots), these can differ significantly
- Attached box would be **offset** from actual grasped object

### 4. **MPlib Not Using Attached Box** (UNLIKELY but POSSIBLE)

**Potential MPlib Internal Issues**:
- Attached box might be ignored during RRT collision checking
- `update_attached_box()` might not properly register with collision checker
- Perturbation might somehow invalidate attached box

**Evidence Against**: PLY files show correct bbox → attachment code works for visualization

### 5. **Collision Checking Timing Issue**

**Location**: Execution happens AFTER planning

**Problem**:
- Planning: Attached box checked at planned joint configurations
- Execution: Robot follows planned trajectory, but object moves with gripper
- If trajectory interpolation differs between planning and execution:
  - Planning: Smooth RRT path
  - Execution: Linear interpolation between waypoints
  - Collision can occur **between waypoints** even if waypoints are collision-free

## Recommended Investigation Steps

1. **Verify grasp_pose computation**:
   - Add logging in `planner.py:323-325` to print `grasp_pose` matrix
   - Check if it matches expected relative pose

2. **Check safety_margin effect**:
   - Print `aabb` (half-extents) and `full_size` in `mplib_core.py:68`
   - Compare with PLY visualization
   - If PLY bbox looks correct but collisions still happen, issue is likely in grasp_pose

3. **Verify MPlib attachment**:
   - Add logging in `mplib_core.py:69` to confirm `update_attached_box()` is called
   - Check MPlib's internal state to see if attached box is registered

4. **Test with simplified scenario**:
   - Use symmetric object (cube) instead of asymmetric (bowl)
   - This eliminates AABB center vs body center offset issue
   - If cube works but bowl doesn't → Issue #3 is the problem

5. **Check actual collision location**:
   - When collision happens, print robot joint configuration
   - Compute where attached box SHOULD be at that configuration
   - Compare with where grasped object ACTUALLY is

## Most Likely Root Cause

**Issue #1 (Grasp Pose) + Issue #3 (AABB vs Body Center)**:

For asymmetric objects like bowls, the AABB center used for `grasp_pose` doesn't match the MuJoCo body center where the object is actually attached. This causes MPlib to check collisions at the wrong location during planning.

**Evidence**:
- Bbox visualization is correct (shows data is valid)
- But collisions still happen (suggests position/orientation mismatch)
- Asymmetric objects are more prone to this issue

## How to Add Additional Camera for Better Coverage

### 1. Define Camera in Environment (if custom camera needed)

**File**: `externals/boss/libero/libero/envs/problems/libero_kitchen_tabletop_manipulation.py`

**Location**: In `_setup_camera()` method (around line 237-267)

```python
mujoco_arena.set_camera(
    camera_name="your_camera_name",  # e.g., "leftview", "rightview"
    pos=[x, y, z],                   # Position
    quat=[w, x, y, z]                # Orientation quaternion
)
```

### 2. Add Camera to Extraction List (2 locations)

**File**: `scripts/phase3/pipeline/motion_planning/mplib/planner.py`

**Location 1 - Line 335** (scene extraction for planning):
```python
camera_names = ["agentview"] if self.use_agentview_only else ["agentview", "birdview", "sideview", "YOUR_NEW_CAMERA"]
```

**Location 2 - Line 430** (scene rendering for visualization):
```python
camera_names = ["agentview"] if self.use_agentview_only else ["agentview", "birdview", "sideview", "YOUR_NEW_CAMERA"]
```

### 3. Pointcloud Combination (already automatic)

**File**: `scripts/phase3/pipeline/motion_planning/mplib/planner_core.py`

**Function**: `extract_scene_pointcloud()` (lines 185-348)

- **Lines 253-322**: Loops through all cameras in `camera_names`
- **Line 328**: Automatically concatenates pointclouds: `merged_masked = np.concatenate(all_points_masked, axis=0)`

**No changes needed** - combination is automatic once you add camera to the list.

## Summary

1. **If using existing MuJoCo camera**: Just add name to `planner.py` lines 335 & 430
2. **If adding new camera**: Define in `_setup_camera()` first, then add to `planner.py`
3. **Pointcloud merging**: Automatic in `extract_scene_pointcloud()`

Current cameras: `agentview` (front), `birdview` (top), `sideview` (side)
