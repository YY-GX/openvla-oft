#!/usr/bin/env python3
"""
Sample 2 demos from each source for multi-BDDL skills and create side-by-side videos.
Video naming: {skill_name}_source_{id}_demo_{demo_idx}.mp4 (id 0 = source with most demos, id 1 = second most, etc.)

Both agent view and wrist camera view are rotated 180 degrees.
If a source has only 1 demo, only 1 video will be saved.
"""

import os
import h5py
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict

# Settings
atomic_demos_path = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_above_fewer/all"
output_video_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase3/pipeline/outputs/videos/demo_gen/sources_video_check"

# First 3 skills
SKILLS_TO_PROCESS = [
    "pick_black_bowl",
    "pick_frying_pan",
    "place_black_bowl_on_the_plate"
]

os.makedirs(output_video_dir, exist_ok=True)


def analyze_source_distribution(hdf5_path: str) -> dict:
    """
    Analyze source_hdf5 distribution in an HDF5 file.
    Returns: {source_base_name: list of (demo_idx, demo_key)}
    """
    source_to_demos = defaultdict(list)
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'data' not in f:
            return {}
        
        n_demos = len(f['data'])
        for i in range(n_demos):
            demo_key = f'demo_{i}'
            if demo_key not in f['data']:
                continue

            demo = f['data'][demo_key]
            if 'metadata' not in demo:
                continue

            meta = demo['metadata']
            source_hdf5 = None

            # Try to get source_hdf5 from attrs
            if 'source_hdf5' in meta.attrs:
                source_hdf5 = meta.attrs['source_hdf5']
                if isinstance(source_hdf5, bytes):
                    source_hdf5 = source_hdf5.decode('utf-8')

            if source_hdf5:
                # Extract base name from path
                base_name = os.path.basename(source_hdf5).replace('_demo.hdf5', '')
                source_to_demos[base_name].append((i, demo_key))

    return source_to_demos


def get_demo_images(hdf5_path: str, demo_key: str):
    """Extract agentview_rgb and eye_in_hand_rgb images from a demo."""
    with h5py.File(hdf5_path, 'r') as f:
        demo = f['data'][demo_key]
        
        if 'obs' not in demo:
            return None, None
        
        obs = demo['obs']
        
        # Get agentview images
        agentview_images = None
        if 'agentview_rgb' in obs:
            agentview_images = obs['agentview_rgb'][:]
        elif 'agentview_image' in obs:
            agentview_images = obs['agentview_image'][:]
        
        # Get wrist camera images
        wrist_images = None
        if 'eye_in_hand_rgb' in obs:
            wrist_images = obs['eye_in_hand_rgb'][:]
        elif 'robot0_eye_in_hand_image' in obs:
            wrist_images = obs['robot0_eye_in_hand_image'][:]
        elif 'wrist_image' in obs:
            wrist_images = obs['wrist_image'][:]
        
        return agentview_images, wrist_images


def rotate_frame_180(frame: np.ndarray) -> np.ndarray:
    """Rotate a frame 180 degrees using OpenCV."""
    return cv2.rotate(frame, cv2.ROTATE_180)


def create_side_by_side_video(agentview_images, wrist_images, output_path, fps=30):
    """Create a side-by-side video from agentview and wrist camera images (both rotated 180°)."""
    if agentview_images is None or len(agentview_images) == 0:
        print(f"  ⚠️  No agentview images")
        return False
    
    if wrist_images is None or len(wrist_images) == 0:
        print(f"  ⚠️  No wrist images, using agentview only")
        wrist_images = agentview_images.copy()
    
    # Ensure uint8 format
    if agentview_images.dtype != np.uint8:
        if agentview_images.max() <= 1.0:
            agentview_images = (agentview_images * 255).astype(np.uint8)
        else:
            agentview_images = agentview_images.astype(np.uint8)
    
    if wrist_images.dtype != np.uint8:
        if wrist_images.max() <= 1.0:
            wrist_images = (wrist_images * 255).astype(np.uint8)
        else:
            wrist_images = wrist_images.astype(np.uint8)
    
    # Get target dimensions from agentview
    target_h, target_w = agentview_images[0].shape[:2]
    target_size = (target_w, target_h)
    
    # Determine minimum frame count
    min_frames = min(len(agentview_images), len(wrist_images))
    agentview_images = agentview_images[:min_frames]
    wrist_images = wrist_images[:min_frames]
    
    # Combine frames side by side (both rotated 180°)
    combined_frames = []
    for agent_frame, wrist_frame in zip(agentview_images, wrist_images):
        # Rotate both frames 180 degrees
        agent_frame_rotated = rotate_frame_180(agent_frame)
        wrist_frame_rotated = rotate_frame_180(wrist_frame)
        
        # Resize wrist frame if needed
        if wrist_frame_rotated.shape[0:2] != (target_h, target_w):
            try:
                wrist_frame_rotated = np.array(Image.fromarray(wrist_frame_rotated).resize(target_size, Image.BILINEAR))
            except Exception:
                # Fallback: crop or pad
                h, w = wrist_frame_rotated.shape[:2]
                if h > target_h or w > target_w:
                    wrist_frame_rotated = wrist_frame_rotated[:target_h, :target_w]
                else:
                    # Pad with zeros
                    padded = np.zeros((target_h, target_w, 3), dtype=wrist_frame_rotated.dtype)
                    padded[:h, :w] = wrist_frame_rotated
                    wrist_frame_rotated = padded
        
        # Concatenate side by side
        combined_frame = np.concatenate([agent_frame_rotated, wrist_frame_rotated], axis=1)
        combined_frames.append(combined_frame)
    
    if len(combined_frames) == 0:
        print(f"  ⚠️  No frames to write")
        return False
    
    frames = np.stack(combined_frames)
    height, width = frames[0].shape[:2]
    
    # Try different codecs
    codecs = [
        ('avc1', '.mp4'),  # H.264
        ('mp4v', '.mp4'),  # MPEG-4
        ('XVID', '.avi'),  # Xvid
    ]
    
    for codec, ext in codecs:
        try:
            if not output_path.endswith(ext):
                test_output_path = output_path.rsplit('.', 1)[0] + ext
            else:
                test_output_path = output_path
            
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(test_output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                continue
            
            # Write frames
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                if frame.shape[-1] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                out.write(frame_bgr)
            
            out.release()
            
            # Verify file was created
            if os.path.exists(test_output_path) and os.path.getsize(test_output_path) > 0:
                return True
        except Exception as e:
            continue
    
    return False


def process_skill(skill_name: str):
    """Process one skill: sample 2 demos from each source (if available) and create videos."""
    hdf5_path = os.path.join(atomic_demos_path, f"{skill_name}.hdf5")
    
    if not os.path.exists(hdf5_path):
        print(f"❌ File not found: {hdf5_path}")
        return
    
    print(f"\n📊 Processing {skill_name}")
    print(f"   File: {hdf5_path}")
    
    # Analyze source distribution
    source_to_demos = analyze_source_distribution(hdf5_path)
    
    if not source_to_demos:
        print(f"   ⚠️  No source metadata found")
        return
    
    # Sort sources by count (descending) to assign IDs
    source_counts = {source: len(demos) for source, demos in source_to_demos.items()}
    sorted_sources = sorted(source_counts.items(), key=lambda x: -x[1])
    
    print(f"   Found {len(sorted_sources)} sources:")
    for idx, (source, count) in enumerate(sorted_sources):
        print(f"      Source {idx}: {source} ({count} demos)")
    
    # Sample up to 2 demos from each source and create videos
    for source_id, (source, count) in enumerate(sorted_sources):
        demos = source_to_demos[source]
        
        # Determine how many demos to sample (up to 2, or all if less than 2)
        num_demos_to_sample = min(2, len(demos))
        
        print(f"\n   Processing source {source_id}: {source}")
        print(f"      Sampling {num_demos_to_sample} demo(s) from {len(demos)} available")
        
        for demo_sample_idx in range(num_demos_to_sample):
            demo_idx, demo_key = demos[demo_sample_idx]
            
            print(f"      Demo {demo_sample_idx + 1}: {demo_key} (demo_{demo_idx})")
            
            # Extract images
            agentview_images, wrist_images = get_demo_images(hdf5_path, demo_key)
            
            if agentview_images is None:
                print(f"         ⚠️  Failed to extract images from {demo_key}")
                continue
            
            # Create video filename: {skill_name}_source_{source_id}_demo_{demo_idx}.mp4
            video_filename = f"{skill_name}_source_{source_id}_demo_{demo_idx}.mp4"
            video_path = os.path.join(output_video_dir, video_filename)
            
            # Create side-by-side video
            success = create_side_by_side_video(agentview_images, wrist_images, video_path)
            
            if success:
                print(f"         ✓ Saved video: {video_filename}")
            else:
                print(f"         ❌ Failed to create video: {video_filename}")


def main():
    print("=" * 80)
    print("Sampling Demo Videos from Multi-BDDL Skills")
    print("=" * 80)
    print(f"Output directory: {output_video_dir}")
    print()
    
    for skill_name in SKILLS_TO_PROCESS:
        process_skill(skill_name)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()

