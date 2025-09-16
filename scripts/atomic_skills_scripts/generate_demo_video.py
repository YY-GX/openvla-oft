#!/usr/bin/env python3
"""
Demo Video Generation Script

This script processes HDF5 demonstration files and creates video files
from the eye_in_hand_rgb image sequences for visual inspection of generated demonstrations.

Key Features:
- Generates one video per HDF5 file (extracts first demo only)
- Optional 3rd person view video generation (agentview_rgb)
- Preserves input folder structure in output directory
- Supports recursive subdirectory search
- Can process single files or entire directories

Video Types:
- Wrist camera view (eye_in_hand_rgb): filename.mp4
- 3rd person view (agentview_rgb): filename_3rd_view.mp4 (when --include_3rd_view is used)

Usage:
    python generate_demo_video.py <path_to_hdf5_file> [--output_dir <output_directory>]
    python generate_demo_video.py --demo_dir <directory_path> [--output_dir <output_directory>]
"""

import argparse
import os
import sys
import glob
import h5py
import numpy as np
import cv2
from pathlib import Path


def load_hdf5_file(file_path):
    """
    Load and validate HDF5 demonstration file.
    
    Args:
        file_path (str): Path to the HDF5 file
        
    Returns:
        h5py.File: Opened HDF5 file handle
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file structure is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")
    
    try:
        hdf5_file = h5py.File(file_path, 'r')
        
        # Validate file structure
        if 'data' not in hdf5_file:
            raise ValueError("Invalid HDF5 structure: 'data' group not found")
        
        return hdf5_file
    except Exception as e:
        raise ValueError(f"Failed to load HDF5 file: {e}")


def get_demo_groups(hdf5_file):
    """
    Extract all demo groups from the HDF5 file.
    
    Args:
        hdf5_file (h5py.File): Opened HDF5 file
        
    Returns:
        list: List of demo group names (e.g., ['demo_0', 'demo_1', ...])
    """
    data_group = hdf5_file['data']
    demo_groups = [key for key in data_group.keys() if key.startswith('demo_')]
    demo_groups.sort()  # Ensure consistent ordering
    return demo_groups


def extract_frames(demo_group, include_3rd_view=False):
    """
    Extract eye_in_hand_rgb frames from a demo group.
    
    Args:
        demo_group (h5py.Group): HDF5 demo group
        include_3rd_view (bool): Whether to also extract 3rd person view frames
        
    Returns:
        dict: Dictionary containing 'wrist' frames and optionally '3rd_view' frames
        
    Raises:
        ValueError: If required datasets are missing
    """
    if 'obs' not in demo_group:
        raise ValueError("Demo group missing 'obs' subgroup")
    
    obs_group = demo_group['obs']
    frames = {}
    
    # Extract wrist camera frames
    if 'eye_in_hand_rgb' not in obs_group:
        raise ValueError("Demo group missing 'obs/eye_in_hand_rgb' dataset")
    
    wrist_frames = obs_group['eye_in_hand_rgb'][:]
    
    # Validate frame shape
    if len(wrist_frames.shape) != 4:
        raise ValueError(f"Invalid wrist frame shape: {wrist_frames.shape}. Expected (T, H, W, 3)")
    
    # Ensure uint8 format for video encoding
    if wrist_frames.dtype != np.uint8:
        wrist_frames = wrist_frames.astype(np.uint8)
    
    frames['wrist'] = wrist_frames
    
    # Extract 3rd person view frames if requested
    if include_3rd_view:
        if 'agentview_rgb' not in obs_group:
            print("⚠️  Warning: 'obs/agentview_rgb' dataset not found, skipping 3rd person view")
        else:
            third_view_frames = obs_group['agentview_rgb'][:]
            
            # Validate frame shape
            if len(third_view_frames.shape) != 4:
                print(f"⚠️  Warning: Invalid 3rd view frame shape: {third_view_frames.shape}, skipping")
            else:
                # Ensure uint8 format for video encoding
                if third_view_frames.dtype != np.uint8:
                    third_view_frames = third_view_frames.astype(np.uint8)
                
                frames['3rd_view'] = third_view_frames
    
    return frames


def create_video(frames, output_path, fps=30):
    """
    Create MP4 video from image frames.
    
    Args:
        frames (numpy.ndarray): Array of shape (T, H, W, 3) containing RGB frames
        output_path (str): Path for output video file
        fps (int): Frames per second for output video
        
    Returns:
        bool: True if video creation was successful, False otherwise
    """
    try:
        num_frames, height, width, channels = frames.shape
        
        if channels != 3:
            raise ValueError(f"Expected 3 channels (RGB), got {channels}")
        
        # Define video codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")
        
        # Write frames to video
        for frame_idx in range(num_frames):
            frame = frames[frame_idx]
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        
        video_writer.release()
        
        print(f"✅ Video created successfully: {output_path}")
        print(f"   Frames: {num_frames}, Resolution: {width}x{height}, FPS: {fps}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating video {output_path}: {e}")
        return False


def generate_output_filename(input_path, demo_name, output_dir, video_type='wrist'):
    """
    Generate output video filename based on input file and demo name.
    
    Args:
        input_path (str): Path to input HDF5 file
        demo_name (str): Name of demo (e.g., 'demo_0')
        output_dir (str): Output directory path
        video_type (str): Type of video ('wrist' or '3rd_view')
        
    Returns:
        str: Full path to output video file
    """
    input_path_obj = Path(input_path)
    input_filename = input_path_obj.stem  # Remove .hdf5 extension
    
    # Extract the subdirectory structure relative to the demo_dir
    # This will preserve folder structure like closer_pick, farther_pick
    demo_dir = Path(output_dir).parent  # Get the parent of output_dir to find relative path
    try:
        # Find the relative path from demo_dir to the input file's directory
        input_dir = input_path_obj.parent
        relative_dir = input_dir.relative_to(demo_dir)
        # Create output subdirectory
        output_subdir = os.path.join(output_dir, str(relative_dir))
        os.makedirs(output_subdir, exist_ok=True)
    except ValueError:
        # If we can't determine relative path, use output_dir directly
        output_subdir = output_dir
    
    # Add demo name and suffix for 3rd person view
    if video_type == '3rd_view':
        output_filename = f"{input_filename}_{demo_name}_3rd_view.mp4"
    else:
        output_filename = f"{input_filename}_{demo_name}.mp4"
    
    return os.path.join(output_subdir, output_filename)


def process_hdf5_file(input_path, output_dir, fps=30, include_3rd_view=False, all_demos=False):
    """
    Process entire HDF5 file and create videos for all demos.
    
    Args:
        input_path (str): Path to input HDF5 file
        output_dir (str): Output directory for video files
        fps (int): Frames per second for output videos
        include_3rd_view (bool): Whether to generate 3rd person view videos
        
    Returns:
        tuple: (success_count, total_count) indicating processing results
    """
    print(f"🎬 Processing HDF5 file: {input_path}")
    
    # Load HDF5 file
    try:
        hdf5_file = load_hdf5_file(input_path)
    except Exception as e:
        print(f"❌ Failed to load HDF5 file: {e}")
        return 0, 0
    
    try:
        # Get all demo groups
        demo_groups = get_demo_groups(hdf5_file)
        
        if not demo_groups:
            print("⚠️  No demo groups found in HDF5 file")
            return 0, 0
        
        if all_demos:
            print(f"📊 Found {len(demo_groups)} demos, processing ALL demos")
            # Count expected videos: all demos * (1 wrist + 1 third view if requested)
            expected_videos = len(demo_groups) * (1 + (1 if include_3rd_view else 0))
        else:
            print(f"📊 Found {len(demo_groups)} demos, processing first demo only")
            # Count expected videos: 1 wrist + 1 third view (if requested and available)
            expected_videos = 1 + (1 if include_3rd_view else 0)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        success_count = 0
        
        # Determine which demos to process
        demos_to_process = demo_groups if all_demos else [demo_groups[0]]
        
        for demo_name in demos_to_process:
            print(f"\n🎥 Processing {demo_name}...")
            
            try:
                demo_group = hdf5_file['data'][demo_name]
                
                # Extract frames
                frames = extract_frames(demo_group, include_3rd_view)
                print(f"   Extracted {len(frames['wrist'])} wrist camera frames")
                if '3rd_view' in frames:
                    print(f"   Extracted {len(frames['3rd_view'])} 3rd person view frames")
                
                # Generate output path for wrist camera
                wrist_output_path = generate_output_filename(input_path, demo_name, output_dir, 'wrist')
                
                # Create wrist camera video
                if create_video(frames['wrist'], wrist_output_path, fps):
                    success_count += 1
                    
                # Generate 3rd person view video if requested and available
                if include_3rd_view and '3rd_view' in frames:
                    third_view_output_path = generate_output_filename(input_path, demo_name, output_dir, '3rd_view')
                    if create_video(frames['3rd_view'], third_view_output_path, fps):
                        success_count += 1
                        
            except Exception as e:
                print(f"❌ Error processing {demo_name}: {e}")
        
        return success_count, expected_videos
        
    finally:
        hdf5_file.close()


def main():
    """Main function to handle command-line arguments and execute processing."""
    parser = argparse.ArgumentParser(
        description="Generate video files from HDF5 demonstration data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process HDF5 file with default output directory
  python generate_demo_video.py /path/to/demo_file.hdf5
  
  # Process HDF5 file with custom output directory
  python generate_demo_video.py /path/to/demo_file.hdf5 --output_dir ./videos
  
  # Process HDF5 file with ALL demos (not just first)
  python generate_demo_video.py /path/to/demo_file.hdf5 --all_demos
  
  # Process HDF5 file with ALL demos and both views
  python generate_demo_video.py /path/to/demo_file.hdf5 --all_demos --include_3rd_view
  
  # Process all HDF5 files in closer_pick directory
  python generate_demo_video.py --demo_dir datasets/hdf5_datasets/atomic_local_demos/closer_pick
  
  # Process all HDF5 files in farther_pick directory
  python generate_demo_video.py --demo_dir datasets/hdf5_datasets/atomic_local_demos/farther_pick
  
  # Process all HDF5 files in closer_pick with both wrist and 3rd person views
  python generate_demo_video.py --demo_dir datasets/hdf5_datasets/atomic_local_demos/closer_pick --include_3rd_view
  
  # Process all HDF5 files in closer_pick with custom output and FPS
  python generate_demo_video.py --demo_dir datasets/hdf5_datasets/atomic_local_demos/closer_pick --output_dir ./closer_videos --fps 60 --include_3rd_view
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        nargs='?',
        default=None,
        help='Path to the input HDF5 demonstration file. If not provided, processes all .hdf5 files in atomic_local_demos directory'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./demo_videos',
        help='Output directory for generated video files (default: ./demo_videos)'
    )
    
    parser.add_argument(
        '--demo_dir',
        type=str,
        default='datasets/hdf5_datasets/atomic_local_demos',
        help='Directory containing HDF5 demo files (default: datasets/hdf5_datasets/atomic_local_demos)'
    )
    
    parser.add_argument(
        '--include_3rd_view',
        action='store_true',
        help='Generate both wrist camera view and 3rd person view videos (default: wrist camera only)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second for output videos (default: 30)'
    )
    
    parser.add_argument(
        '--all_demos',
        action='store_true',
        help='Generate videos for all demos in the HDF5 file (default: only first demo)'
    )
    
    args = parser.parse_args()
    
    # Determine input files
    if args.input_file:
        # Single file mode
        if not os.path.exists(args.input_file):
            print(f"❌ Error: Input file does not exist: {args.input_file}")
            sys.exit(1)
        
        if not args.input_file.endswith('.hdf5'):
            print(f"❌ Error: Input file must be an HDF5 file (.hdf5): {args.input_file}")
            sys.exit(1)
        
        input_files = [args.input_file]
    else:
        # Multiple files mode - find all HDF5 files in demo_dir and its subdirectories
        demo_dir = args.demo_dir
        if not os.path.exists(demo_dir):
            print(f"❌ Error: Demo directory does not exist: {demo_dir}")
            print("Please provide a specific HDF5 file path or ensure the demo directory exists")
            sys.exit(1)
        
        # Recursively search for HDF5 files in demo_dir and subdirectories
        input_files = []
        for root, dirs, files in os.walk(demo_dir):
            for file in files:
                if file.endswith('.hdf5'):
                    input_files.append(os.path.join(root, file))
        
        if not input_files:
            print(f"❌ Error: No HDF5 demo files found in: {demo_dir}")
            sys.exit(1)
        
        input_files.sort()  # Sort for consistent ordering
        print(f"🔍 Found {len(input_files)} HDF5 files to process")
    
    # Validate FPS
    if args.fps <= 0:
        print(f"❌ Error: FPS must be positive: {args.fps}")
        sys.exit(1)
    
    print("🎬 Demo Video Generator")
    print("=" * 50)
    if len(input_files) == 1:
        print(f"Input file: {input_files[0]}")
    else:
        print(f"Input files: {len(input_files)} HDF5 files")
    print(f"Output directory: {args.output_dir}")
    print(f"FPS: {args.fps}")
    print("=" * 50)
    
    # Process all files
    total_success_count = 0
    total_demo_count = 0
    
    for input_file in input_files:
        success_count, demo_count = process_hdf5_file(
            input_file, 
            args.output_dir, 
            args.fps,
            args.include_3rd_view,
            args.all_demos
        )
        total_success_count += success_count
        total_demo_count += demo_count
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Processing Summary:")
    print(f"   HDF5 files processed: {len(input_files)}")
    print(f"   Total demos: {total_demo_count}")
    print(f"   Successful videos: {total_success_count}")
    print(f"   Failed videos: {total_demo_count - total_success_count}")
    
    if total_success_count == total_demo_count:
        print("✅ All videos generated successfully!")
    elif total_success_count > 0:
        print("⚠️  Some videos failed to generate")
    else:
        print("❌ No videos were generated")
        sys.exit(1)


if __name__ == "__main__":
    main()