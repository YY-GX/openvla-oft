#!/usr/bin/env python3
"""
Visualize initial states from focused mode init files.

This script loads an init file from the focused skills mapping, visualizes up to 10 
initial states, and saves rendered images to show the scene setup for pose extraction.
Useful for debugging and understanding the data quality.
"""

import os
import sys
import pickle
import numpy as np
import csv
import argparse
from typing import Dict
from pathlib import Path
from PIL import Image

# Add project paths
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')

# LIBERO imports
from libero.libero.envs import OffScreenRenderEnv

# Import get_object_pose function
import importlib.util
spec_detector = importlib.util.spec_from_file_location("contact_detector", "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase1/3_contact_object_detector.py")
contact_detector = importlib.util.module_from_spec(spec_detector)
spec_detector.loader.exec_module(contact_detector)
get_object_pose = contact_detector.get_object_pose


def load_skills_mapping() -> Dict[str, str]:
    """Load skills mapping from CSV file."""
    csv_file = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/prompts/for_ai/skills_objects_mapping_updated.csv"
    mapping = {}
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                init_files = row['init_files'].strip()
                object_name = row['object'].strip()
                
                # Parse multiple init files separated by ';'
                for init_file in init_files.split(';'):
                    init_file = init_file.strip()
                    if init_file:
                        mapping[init_file] = object_name
        
        return mapping
        
    except Exception as e:
        print(f"❌ Error loading CSV mapping: {e}")
        return {}


def get_ee_pose(env):
    """Get end-effector pose from environment."""
    data = env.sim.data
    model = env.sim.model
    
    # Find end-effector body
    possible_ee_names = ["gripper0_eef", "robot0_eef", "gripper_eef", "eef"]
    for ee_name in possible_ee_names:
        try:
            if ee_name in [model.body_id2name(i) for i in range(model.nbody)]:
                ee_body_id = model.body_name2id(ee_name)
                ee_pos = data.body_xpos[ee_body_id].copy()
                ee_quat = data.body_xquat[ee_body_id].copy()
                return ee_pos, ee_quat, ee_name
        except:
            continue
    
    return None, None, None


def setup_better_camera(env):
    """Setup a better camera position for visualization."""
    try:
        model = env.sim.model
        
        # Try different camera configurations for better overview
        camera_configs = [
            ("agentview", [1.5, 1.0, 1.8], [0.6, 0.3, 0.3, 0.6]),    # High overview
            ("frontview", [2.0, 0.0, 1.2], [0.7, 0.0, 0.0, 0.7]),     # Front view
            ("birdview", [0.0, 0.0, 3.0], [1.0, 0.0, 0.0, 0.0]),      # Top-down
        ]
        
        for cam_name, pos, quat in camera_configs:
            try:
                cam_id = model.camera_name2id(cam_name)
                model.cam_pos[cam_id] = pos
                model.cam_quat[cam_id] = quat
                print(f"      📷 Setup {cam_name} camera: pos={pos}, quat={quat}")
            except:
                continue
            
    except Exception as e:
        print(f"      ⚠️  Failed to setup cameras: {e}")


def render_scene(env, camera_name: str = "agentview", setup_cameras: bool = True):
    """Render scene from specified camera."""
    try:
        # Setup better camera positions once
        if setup_cameras:
            setup_better_camera(env)
        
        # Render image with larger resolution for better detail
        img = env.sim.render(width=640, height=480, camera_name=camera_name)
        
        # Convert to PIL Image (MuJoCo renders as RGB)
        if img is not None:
            img_pil = Image.fromarray(img)
            return img_pil
        else:
            return None
            
    except Exception as e:
        print(f"      ⚠️  Failed to render {camera_name}: {e}")
        return None


def visualize_init_file(init_file_path: str, target_object: str, output_dir: str, max_states: int = 10):
    """
    Visualize initial states from an init file.
    
    Args:
        init_file_path: Path to the .init file
        target_object: Target object name from CSV
        output_dir: Directory to save images
        max_states: Maximum number of states to visualize
    """
    skill_name = Path(init_file_path).stem
    print(f"\n🎬 Visualizing: {skill_name}")
    print(f"  🎯 Target object: {target_object}")
    
    try:
        # Load initial states
        with open(init_file_path, 'rb') as f:
            initial_states = pickle.load(f)
        
        if not initial_states:
            print(f"  ❌ No initial states found")
            return False
        
        num_states = min(len(initial_states), max_states)
        print(f"  📊 Visualizing {num_states} states (out of {len(initial_states)} total)")
        
        # Create output directory
        skill_output_dir = os.path.join(output_dir, skill_name)
        os.makedirs(skill_output_dir, exist_ok=True)
        
        # Load BDDL file
        bddl_file = f"/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss/libero/libero/bddl_files/atomic_skills/{skill_name}.bddl"
        
        if not os.path.exists(bddl_file):
            print(f"  ❌ BDDL file not found: {bddl_file}")
            return False
        
        # Process each state
        successful_renders = 0
        
        for state_idx in range(num_states):
            print(f"  🎬 Rendering state {state_idx + 1}/{num_states}...", end=' ')
            
            try:
                # Initialize environment
                env_args = {
                    'bddl_file_name': bddl_file,
                    'camera_heights': 512,
                    'camera_widths': 512
                }
                
                env = OffScreenRenderEnv(**env_args)
                env.reset()
                
                # FIXED: Skip first 9 dims (joint+gripper) and use only simulation state (84-dim)
                full_init_state = initial_states[state_idx]
                print(f"      🔧 Full init state shape: {full_init_state.shape}")
                
                sim_state_only = full_init_state[9:]

                # Set initial state with correct simulation state
                env.sim.set_state_from_flattened(sim_state_only)
                env.sim.forward()
                
                # Investigate all objects in scene
                model = env.sim.model
                data = env.sim.data
                
                print(f"\n      🔍 Investigating objects in scene:")
                scene_objects = []
                for i in range(model.nbody):
                    body_name = model.body_id2name(i)
                    if body_name and any(keyword in body_name.lower() for keyword in 
                                       ['bowl', 'plate', 'mug', 'bottle', 'pan', 'pot', 'stove', 'cabinet', 'drawer', 'microwave']):
                        pos = data.body_xpos[i]
                        scene_objects.append((body_name, pos))
                        print(f"        - {body_name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                
                # Get target object and EE poses
                obj_pos, obj_quat = get_object_pose(env, target_object)
                ee_pos, ee_quat, ee_name = get_ee_pose(env)
                
                # Report object detection results
                if obj_pos is not None:
                    print(f"      ✅ Target object '{target_object}' found at: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
                else:
                    print(f"      ❌ Target object '{target_object}' NOT FOUND!")
                    print(f"      💡 Available objects: {[obj[0] for obj in scene_objects[:5]]}")
                
                if ee_pos is not None:
                    print(f"      ✅ End-effector '{ee_name}' found at: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
                else:
                    print(f"      ❌ End-effector NOT FOUND!")
                
                # Calculate distance if both poses available
                distance = None
                if obj_pos is not None and ee_pos is not None:
                    distance = np.linalg.norm(ee_pos - obj_pos)
                    print(f"      📏 EE-Object distance: {distance:.3f}m")
                
                # Setup cameras once and try different views
                cameras_to_try = ["agentview", "birdview", "frontview"]
                rendered_any = False
                setup_done = False
                
                for i, camera in enumerate(cameras_to_try):
                    # Setup cameras only once
                    img = render_scene(env, camera, setup_cameras=(not setup_done))
                    setup_done = True
                    
                    if img is not None:
                        # Save image with metadata in filename
                        distance_str = f"_d{distance:.3f}m" if distance is not None else "_dNA"
                        found_str = "_found" if obj_pos is not None else "_missing"
                        img_filename = f"state_{state_idx:02d}_{camera}{distance_str}{found_str}.png"
                        img_path = os.path.join(skill_output_dir, img_filename)
                        img.save(img_path)
                        
                        if not rendered_any:
                            status = f"({distance:.3f}m)" if distance else "(no distance)"
                            obj_status = "✅" if obj_pos is not None else "❌"
                            print(f"✅ {camera} {status} {obj_status}")
                            rendered_any = True
                
                if not rendered_any:
                    print("❌ No cameras worked")
                else:
                    successful_renders += 1
                
                env.close()
                
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        print(f"  ✅ Successfully rendered {successful_renders}/{num_states} states")
        print(f"  💾 Images saved to: {skill_output_dir}")
        
        return successful_renders > 0
        
    except Exception as e:
        print(f"❌ Error visualizing {init_file_path}: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Visualize focused mode init files')
    parser.add_argument('--skill', type=str, 
                        help='Specific init filename to visualize (e.g., KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_pick.init)')
    parser.add_argument('--max-states', type=int, default=10,
                        help='Maximum number of states to visualize per init file (default: 10)')
    parser.add_argument('--output-dir', type=str, 
                        default='/mnt/arc/yygx/pkgs_baselines/openvla-oft/images_videos/focused_init_vis',
                        help='Output directory for images')
    args = parser.parse_args()
    
    print("🎬 Focused Init File Visualizer")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎯 Max states per file: {args.max_states}")
    
    # Load skills mapping
    skills_mapping = load_skills_mapping()
    if not skills_mapping:
        print("❌ No skills mapping loaded")
        return
    
    print(f"📋 Loaded mapping for {len(skills_mapping)} init files")
    
    # Input directory
    input_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_local_demos"
    
    if args.skill:
        # Visualize specific skill
        if args.skill not in skills_mapping:
            print(f"❌ Skill '{args.skill}' not found in mapping")
            print(f"Available skills: {list(skills_mapping.keys())[:5]}...")
            return
        
        # Find the init file
        init_file_path = None
        for root, _, files in os.walk(input_dir):
            if args.skill in files:
                init_file_path = os.path.join(root, args.skill)
                break
        
        if not init_file_path:
            print(f"❌ Init file not found: {args.skill}")
            return
        
        target_object = skills_mapping[args.skill]
        success = visualize_init_file(init_file_path, target_object, args.output_dir, args.max_states)
        
        if success:
            print("\n🎉 Visualization completed successfully!")
        else:
            print("\n❌ Visualization failed!")
            
    else:
        # Show available skills for selection
        print(f"\n📋 Available skills to visualize:")
        print("=" * 80)
        
        available_skills = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.init') and file in skills_mapping:
                    available_skills.append(file)
        
        # Group by skill type
        pick_skills = [s for s in available_skills if '_pick' in s]
        place_skills = [s for s in available_skills if '_place' in s]
        atomic_skills = [s for s in available_skills if '_pick' not in s and '_place' not in s]
        
        print(f"🤏 Pick skills ({len(pick_skills)}):")
        for skill in sorted(pick_skills)[:5]:
            obj = skills_mapping[skill]
            print(f"  {skill} → {obj}")
        if len(pick_skills) > 5:
            print(f"  ... and {len(pick_skills) - 5} more")
        
        print(f"\n📦 Place skills ({len(place_skills)}):")
        for skill in sorted(place_skills)[:5]:
            obj = skills_mapping[skill]
            print(f"  {skill} → {obj}")
        if len(place_skills) > 5:
            print(f"  ... and {len(place_skills) - 5} more")
        
        print(f"\n🔧 Atomic skills ({len(atomic_skills)}):")
        for skill in sorted(atomic_skills)[:5]:
            obj = skills_mapping[skill]
            print(f"  {skill} → {obj}")
        if len(atomic_skills) > 5:
            print(f"  ... and {len(atomic_skills) - 5} more")
        
        print(f"\nUsage examples:")
        print(f"  python {os.path.basename(__file__)} --skill KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_pick.init")
        print(f"  python {os.path.basename(__file__)} --skill KITCHEN_SCENE3_turn_on_the_stove.init --max-states 5")


if __name__ == "__main__":
    main()