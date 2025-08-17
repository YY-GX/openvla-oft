"""
Script to capture initial scene images for all atomic skills tasks.

This script loads each of the 76 tasks in the atomic_skills benchmark and 
captures initial scene images without requiring proper initial state files.
Uses random object placement as fallback when specific initial states are not available.

Usage:
    # Process all 76 tasks
    python capture_initial_images.py
    
    # Debug mode: only process first 10 tasks
    python capture_initial_images.py --debug
    
    # Only process SCENE7 tasks (useful for testing white bowl fixes)
    python capture_initial_images.py --scene SCENE7
    
    # Debug mode + scene filter
    python capture_initial_images.py --debug --scene SCENE7
"""

import os
import traceback
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image


def extract_language_from_task_name(task_name):
    """Extract language description from BDDL file name using the same logic as benchmark."""
    # Remove .bddl extension if present
    if task_name.endswith('.bddl'):
        task_name = task_name[:-5]
    
    # Extract language part after SCENE numbers
    if "SCENE10" in task_name:
        language = " ".join(task_name[task_name.find("SCENE") + 8:].split("_"))
    else:
        language = " ".join(task_name[task_name.find("SCENE") + 7:].split("_"))
    
    return language


def get_libero_image_simple(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def capture_scene_with_random_init(task, output_dir):
    """
    Capture a scene image using random initialization.
    
    Args:
        task: Task object from the benchmark
        output_dir: Directory to save the image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create environment directly without tensorflow dependencies
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env_args = {
            "bddl_file_name": task_bddl_file, 
            "camera_heights": 256, 
            "camera_widths": 256, 
            "camera_depths": True
        }
        env = OffScreenRenderEnv(**env_args)
        
        # Reset environment with random seed to get randomized initial positions
        env.seed(42)  # Use fixed seed for reproducibility
        obs = env.reset()
        
        # Capture initial scene image
        agent_img = get_libero_image_simple(obs)
        
        # Use language description as filename (cleaned up) as requested in prompt
        language = extract_language_from_task_name(task.name)
        filename = language.replace(" ", "_").replace(".", "").replace(",", "") + ".png"
        image_path = os.path.join(output_dir, filename)
        
        # Save image
        Image.fromarray(agent_img).save(image_path)
        
        print(f"[✔] Saved: {image_path}")
        
        # Clean up
        env.close()
        return True
        
    except Exception as e:
        print(f"[✗] Failed to capture {task.name}: {str(e)}")
        traceback.print_exc()
        return False


def capture_all_atomic_skills_images(debug_mode=False, scene_filter=None):
    """Main function to capture images for all 76 atomic skills tasks.
    
    Args:
        debug_mode (bool): If True, only process a subset of tasks for debugging
        scene_filter (str): If provided, only process tasks containing this scene (e.g., "SCENE7")
    """
    
    print("Loading atomic_skills benchmark...")
    
    # Load the atomic_skills benchmark
    try:
        atomic_skills_benchmark = benchmark.get_benchmark("atomic_skills")()
        print(f"[INFO] Loaded benchmark with {atomic_skills_benchmark.get_num_tasks()} tasks")
        print(f"[DEBUG] First few task names: {[t.name for t in atomic_skills_benchmark.tasks[:5]]}")
    except Exception as e:
        print(f"[ERROR] Failed to load atomic_skills benchmark: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Output directory
    output_dir = "imgs/atomic_skills_scene_images"
    if debug_mode:
        output_dir += "_debug"
    os.makedirs(output_dir, exist_ok=True)
    
    # Track success/failure counts and failed tasks
    successful_captures = 0
    failed_captures = 0
    failed_tasks = []
    
    # Filter tasks if debug mode or scene filter is enabled
    tasks_to_process = []
    for i in range(atomic_skills_benchmark.get_num_tasks()):
        task = atomic_skills_benchmark.get_task(i)
        
        # Apply scene filter if specified
        if scene_filter and scene_filter not in task.name:
            continue
            
        # Apply debug mode filter (only first 10 tasks)
        if debug_mode and i >= 10:
            continue
            
        tasks_to_process.append((i, task))
    
    print(f"[INFO] Processing {len(tasks_to_process)} tasks")
    if debug_mode:
        print(f"[DEBUG] Debug mode enabled - limited to first 10 tasks")
    if scene_filter:
        print(f"[DEBUG] Scene filter enabled - only processing tasks containing '{scene_filter}'")
    
    # Iterate through filtered tasks
    for idx, (i, task) in enumerate(tasks_to_process):
        print(f"\n[{idx+1}/{len(tasks_to_process)}] Processing: {task.name}")
        print(f"  Language: {task.language}")
        
        # Attempt to capture the scene
        if capture_scene_with_random_init(task, output_dir):
            successful_captures += 1
        else:
            failed_captures += 1
            failed_tasks.append({
                'index': i,
                'name': task.name,
                'language': task.language
            })
    
    # Summary
    print(f"\n" + "="*50)
    print(f"CAPTURE SUMMARY:")
    print(f"  Total tasks in benchmark: {atomic_skills_benchmark.get_num_tasks()}")
    print(f"  Tasks processed: {len(tasks_to_process)}")
    print(f"  Successful: {successful_captures}")
    print(f"  Failed: {failed_captures}")
    print(f"  Output directory: {output_dir}")
    
    # List failed tasks if any
    if failed_tasks:
        print(f"\nFAILED TASKS ({len(failed_tasks)}):")
        print("-" * 30)
        for failed_task in failed_tasks:
            print(f"  [{failed_task['index']}] {failed_task['name']}")
            print(f"      Language: {failed_task['language']}")
        print("-" * 30)
    else:
        print(f"\n🎉 All processed tasks completed successfully!")
    
    print(f"="*50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Capture initial scene images for atomic skills tasks")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (only process first 10 tasks)")
    parser.add_argument("--scene", type=str, help="Only process tasks containing specific scene (e.g., SCENE7)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("ATOMIC SKILLS SCENE IMAGE CAPTURE")
    print("="*60)
    
    if args.debug:
        print("[DEBUG] Debug mode enabled")
    if args.scene:
        print(f"[DEBUG] Scene filter: {args.scene}")
    
    capture_all_atomic_skills_images(debug_mode=args.debug, scene_filter=args.scene)