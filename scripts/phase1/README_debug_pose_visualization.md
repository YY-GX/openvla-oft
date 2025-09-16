# Pose Visualization Debug Script

## Purpose

This script helps debug local pose calculation issues in the long-horizon pipeline by:

1. **Calculating local poses** for each skill in a long-horizon task
2. **Moving the robot** to those calculated poses using IK
3. **Saving visual comparisons** between calculated poses and reference atomic skill environments
4. **Identifying pose mismatches** that cause VLA skill failures

## Usage

```bash
# Debug Complete Kitchen Organization task
python scripts/phase1/debug_pose_visualization.py --task_name "Complete Kitchen Organization"

# Debug other long-horizon tasks  
python scripts/phase1/debug_pose_visualization.py --task_name "Switch Table Objects"
python scripts/phase1/debug_pose_visualization.py --task_name "Cooking Preparation Setup"
```

## Output

The script creates a timestamped folder in `images_videos/pose_debug/` containing:

### For each skill (e.g., 7 skills total):
- `skill_01_calculated_pose_pick_black_bowl_agentview.png` - Agent view of calculated pose
- `skill_01_calculated_pose_pick_black_bowl_wrist.png` - Wrist view of calculated pose  
- `skill_01_atomic_reference_pick_black_bowl_agentview.png` - Agent view of reference atomic skill
- `skill_01_atomic_reference_pick_black_bowl_wrist.png` - Wrist view of reference atomic skill

### Total images:
- **7 skills × 4 images per skill = 28 images** for comparison

## What to Look For

When comparing the images:

1. **Robot arm position** - Is the calculated pose similar to the atomic skill reference?
2. **End-effector location** - Is the robot reaching toward the correct object?
3. **Object proximity** - Is the robot close enough to the target object for manipulation?
4. **Scene layout differences** - Are objects in similar positions between calculated and reference scenes?

## Debugging Process

1. **Run the script** to generate pose visualization images
2. **Compare calculated vs reference** images side by side
3. **Identify pose mismatches** where calculated poses look wrong
4. **Investigate pose calculation logic** in `4_gt_pose_calculator.py`
5. **Adjust pose calculation parameters** as needed

## Common Issues to Check

- **Coordinate frame mismatches** between long-horizon and atomic environments
- **Object position differences** affecting pose calculations  
- **IK solver limitations** not reaching intended poses
- **Pose offset errors** in the GT pose calculation logic

This visual debugging approach should help identify why VLA skills are failing even when they work correctly in atomic environments.