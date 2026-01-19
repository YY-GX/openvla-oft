#!/usr/bin/env python3
"""
Extract demos from each HDF5 file in atomic_above_fewer/all and save as video.

This script:
1. Reads all HDF5 files from datasets/hdf5_datasets/atomic_above_fewer/all
2. For each file, extracts 2 demos (preferring non-shifted if available)
3. Saves videos to scripts/phase3/pipeline/outputs/videos/gen_dataset_videos

Alternatively, can check a specific HDF5 file and extract 10 demos from it.


python scripts/phase3/pipeline/utils/extract_dataset_videos.py --hdf5-file /mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_above_fewer/v7/pick_red_mug_failure.hdf5
"""

import os
import h5py
import numpy as np
import imageio
import cv2
from PIL import Image
from typing import Dict, List, Optional
from tqdm import tqdm
import json
import argparse

# Directories
HDF5_DIR = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_above_fewer/all"
OUTPUT_VIDEO_DIR = "scripts/phase3/pipeline/outputs/videos/demo_gen/dataset_samples_all"

# HDF5_DIR = "datasets/hdf5_datasets/atomic_recovery_demos"
# OUTPUT_VIDEO_DIR = "scripts/phase3/pipeline/outputs/videos/demo_gen/recovery_demos"

# HDF5_DIR = "datasets/hdf5_datasets/atomic_above_fewer/place_update_seg/"
# OUTPUT_VIDEO_DIR = "scripts/phase3/pipeline/outputs/videos/demo_gen/dataset_samples_place_update_seg"

HDF5_DIR = "datasets/hdf5_datasets/atomic_above_fewer/all_downsampled_fixed_seg"
OUTPUT_VIDEO_DIR = "scripts/phase3/pipeline/outputs/videos/demo_gen/dataset_samples_all_downsampled_fixed_seg"

HDF5_DIR = "datasets/hdf5_datasets/atomic_above_fewer/v1"
OUTPUT_VIDEO_DIR = "scripts/phase3/pipeline/outputs/videos/demo_gen/failure_videos_examples"

HDF5_DIR = "datasets/hdf5_datasets/atomic_above_fewer/v7"
OUTPUT_VIDEO_DIR = "scripts/phase3/pipeline/outputs/videos/demo_gen/failure_videos_examples_v4"

# HDF5_DIR = "datasets/hdf5_datasets/atomic_above_fewer/all_downsampled"
# OUTPUT_VIDEO_DIR = "scripts/phase3/pipeline/outputs/videos/demo_gen/dataset_samples_all_downsampled"

# HDF5_DIR = "datasets/hdf5_datasets/atomic_above_27_skills/all_downsampled"
# OUTPUT_VIDEO_DIR = "scripts/phase3/pipeline/outputs/videos/demo_gen/dataset_samples_all_27skills"


# HDF5_DIR = "datasets/hdf5_datasets/atomic_above_27_skills/all_downsampled_fixed_seg"
# OUTPUT_VIDEO_DIR = "scripts/phase3/pipeline/outputs/videos/demo_gen/dataset_samples_all_27skills_fixed_seg"

def load_hdf5_demo(h5file: h5py.File, demo_key: str) -> Optional[Dict]:
    """
    Load a single demo from HDF5 file.
    
    Args:
        h5file: Open HDF5 file
        demo_key: Demo key (e.g., "demo_0")
        
    Returns:
        Demo dictionary or None if failed
    """
    try:
        demo_group = h5file['data'][demo_key]
        demo = {}
        
        # Load observations
        if 'obs' in demo_group:
            demo['obs'] = {}
            obs_group = demo_group['obs']
            for key in obs_group.keys():
                demo['obs'][key] = np.array(obs_group[key])
        
        # Load metadata if available
        if 'metadata' in demo_group:
            metadata = {}
            meta_group = demo_group['metadata']
            # Load attributes
            for key in meta_group.attrs.keys():
                try:
                    val = meta_group.attrs[key]
                    if isinstance(val, str) and val.startswith('{'):
                        metadata[key] = json.loads(val)
                    elif isinstance(val, (np.integer, np.floating)):
                        metadata[key] = val.item()
                    else:
                        metadata[key] = val
                except:
                    pass
            # Load scalar datasets
            for key in meta_group.keys():
                try:
                    val = meta_group[key]
                    if isinstance(val, h5py.Dataset):
                        if val.shape == ():
                            metadata[key] = val[()].item() if hasattr(val[()], 'item') else val[()]
                        elif len(val.shape) == 1 and val.shape[0] == 1:
                            metadata[key] = val[0].item() if hasattr(val[0], 'item') else val[0]
                except:
                    pass
            demo['metadata'] = metadata
        
        return demo
    except Exception as e:
        print(f"  ⚠️  Warning: Could not load demo {demo_key}: {e}")
        return None


def find_best_demo_key(h5file: h5py.File) -> Optional[str]:
    """
    Find the best demo key to extract (prefer non-shifted, otherwise first).
    Only checks metadata without loading full demo.
    
    Args:
        h5file: Open HDF5 file
        
    Returns:
        Demo key (e.g., "demo_0") or None
    """
    if 'data' not in h5file:
        return None
    
    data_group = h5file['data']
    demo_keys = sorted(data_group.keys())
    
    if not demo_keys:
        return None
    
    # Try to find a non-shifted demo first (check metadata only)
    for demo_key in demo_keys:
        try:
            demo_group = data_group[demo_key]
            if 'metadata' not in demo_group:
                continue
            
            meta_group = demo_group['metadata']
            shifted = False
            
            # Check attributes first
            if 'shifted' in meta_group.attrs:
                shifted = meta_group.attrs['shifted']
            elif 'iteration' in meta_group.attrs:
                iteration = meta_group.attrs['iteration']
                if isinstance(iteration, (np.integer, np.floating)):
                    iteration = iteration.item()
                shifted = (iteration > 0)
            else:
                # Check scalar datasets
                if 'shifted' in meta_group:
                    val = meta_group['shifted']
                    if isinstance(val, h5py.Dataset) and val.shape == ():
                        shifted = val[()].item() if hasattr(val[()], 'item') else val[()]
                elif 'iteration' in meta_group:
                    val = meta_group['iteration']
                    if isinstance(val, h5py.Dataset):
                        if val.shape == ():
                            iteration = val[()].item() if hasattr(val[()], 'item') else val[()]
                        elif len(val.shape) == 1 and val.shape[0] == 1:
                            iteration = val[0].item() if hasattr(val[0], 'item') else val[0]
                        else:
                            iteration = 0
                        shifted = (iteration > 0)
            
            if not shifted:
                return demo_key
        except:
            # If metadata check fails, continue to next demo
            continue
    
    # If no non-shifted demo found, return first one
    return demo_keys[0]


WRIST_VIEW_KEYS = [
    'robot0_eye_in_hand_image',
    'wrist_image',
    'wrist_rgb',
    'eye_in_hand_rgb',
    'robot0_eye_in_hand_rgb',
]


def _prepare_rgb_frames(frames: np.ndarray) -> np.ndarray:
    """Rotate frames by 180° and ensure uint8 format."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected frames with shape (T, H, W, 3)")
    
    if frames.dtype != np.uint8:
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)
    
    rotated_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        img_rotated = img.rotate(180)
        rotated_frames.append(np.array(img_rotated))
    return np.stack(rotated_frames)


def _get_wrist_frames(obs: Dict) -> Optional[np.ndarray]:
    """Fetch wrist camera frames from observation dict."""
    for key in WRIST_VIEW_KEYS:
        if key in obs:
            try:
                return np.asarray(obs[key])
            except Exception:
                continue
    return None


def _prepare_segmentation_frames(seg_frames: np.ndarray) -> np.ndarray:
    """Rotate segmentation frames by 180° and convert grayscale to RGB."""
    seg_frames = np.asarray(seg_frames)

    # Ensure uint8 format
    if seg_frames.dtype != np.uint8:
        if seg_frames.max() <= 1.0:
            seg_frames = (seg_frames * 255).astype(np.uint8)
        else:
            seg_frames = seg_frames.astype(np.uint8)

    rotated_frames = []
    for frame in seg_frames:
        # Rotate 180 degrees
        frame_rotated = cv2.rotate(frame, cv2.ROTATE_180)
        # Convert grayscale to RGB
        if len(frame_rotated.shape) == 2:
            frame_rgb = cv2.cvtColor(frame_rotated, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = frame_rotated
        rotated_frames.append(frame_rgb)
    return np.stack(rotated_frames)


def save_demo_video_with_labels(demo: Dict, skill_name: str, video_dir: str,
                                 demo_suffix: str = "", labels: List[Dict] = None):
    """
    Save video with label overlay (stage_name, reachout, reward, success_prob).

    Args:
        demo: Demo dict with 'obs' containing camera frames
        skill_name: Skill name for filename
        video_dir: Directory to save video
        demo_suffix: Optional suffix to add to filename (e.g., "_demo0")
        labels: List of label dicts with keys: t, stage_name, reachout, reward, success_prob
    """
    if 'obs' not in demo or 'agentview_rgb' not in demo['obs']:
        print(f"  Warning: No agentview_rgb in demo for {skill_name}")
        return

    agent_frames = demo['obs']['agentview_rgb']
    if agent_frames.shape[0] == 0:
        print(f"  Warning: Empty agentview frames for {skill_name}")
        return

    try:
        agent_frames = _prepare_rgb_frames(agent_frames)
    except Exception as exc:
        print(f"  Warning: Failed to prepare agentview frames for {skill_name}: {exc}")
        return

    wrist_frames_raw = _get_wrist_frames(demo['obs'])
    if wrist_frames_raw is None:
        print(f"  Warning: No wrist camera frames found for {skill_name}, duplicating agentview")
        wrist_frames = agent_frames.copy()
    else:
        try:
            wrist_frames = _prepare_rgb_frames(wrist_frames_raw)
        except Exception as exc:
            print(f"  Warning: Failed to prepare wrist frames for {skill_name}: {exc}")
            wrist_frames = agent_frames.copy()

    # Determine minimum frame count
    min_frames = min(agent_frames.shape[0], wrist_frames.shape[0])
    if min_frames == 0:
        print(f"  Warning: No usable frames for {skill_name}")
        return

    agent_frames = agent_frames[:min_frames]
    wrist_frames = wrist_frames[:min_frames]

    # Build label lookup by timestep
    label_lookup = {}
    if labels:
        for lbl in labels:
            label_lookup[lbl['t']] = lbl

    combined_frames = []
    target_hw = agent_frames.shape[1:3]
    target_size = (target_hw[1], target_hw[0])

    for t, (agent_frame, wrist_frame) in enumerate(zip(agent_frames, wrist_frames)):
        # Resize wrist frame if needed
        if wrist_frame.shape[0:2] != target_hw:
            try:
                wrist_frame = np.array(Image.fromarray(wrist_frame).resize(target_size, Image.BILINEAR))
            except Exception:
                wrist_frame = wrist_frame[:target_hw[0], :target_hw[1]]

        # Concatenate frames
        combined = np.concatenate([agent_frame, wrist_frame], axis=1)

        # Add label overlay if available
        if t in label_lookup:
            lbl = label_lookup[t]
            stage = lbl.get('stage_name', '?')
            reachout = lbl.get('reachout', 0)
            reward = lbl.get('reward', 0)
            success_prob = lbl.get('success_prob', 0)

            # Draw text overlay
            text_lines = [
                f"t={t} stage={stage}",
                f"reachout={reachout:.2f}",
                f"reward={reward:.2f}",
                f"success={success_prob:.2f}"
            ]

            # Use OpenCV to draw text
            y_offset = 20
            for line in text_lines:
                cv2.putText(combined, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(combined, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                y_offset += 18

        combined_frames.append(combined)

    frames = np.stack(combined_frames)

    # Create video filename
    video_filename = f"{skill_name}{demo_suffix}_labeled.mp4"
    video_path = os.path.join(video_dir, video_filename)

    # Save video using imageio
    try:
        imageio.mimwrite(video_path, frames, fps=30, quality=8)
        print(f"  Saved labeled video: {video_filename}")
    except Exception as e:
        print(f"  Failed to save video {video_filename}: {e}")


def save_demo_video(demo: Dict, skill_name: str, video_dir: str, demo_suffix: str = ""):
    """
    Save video combining agentview, wrist camera, and segmentation frames side-by-side.

    Args:
        demo: Demo dict with 'obs' containing camera frames
        skill_name: Skill name for filename
        video_dir: Directory to save video
        demo_suffix: Optional suffix to add to filename (e.g., "_demo0")
    """
    if 'obs' not in demo or 'agentview_rgb' not in demo['obs']:
        print(f"  ⚠️  Warning: No agentview_rgb in demo for {skill_name}")
        return

    agent_frames = demo['obs']['agentview_rgb']
    if agent_frames.shape[0] == 0:
        print(f"  ⚠️  Warning: Empty agentview frames for {skill_name}")
        return

    try:
        agent_frames = _prepare_rgb_frames(agent_frames)
    except Exception as exc:
        print(f"  ⚠️  Warning: Failed to prepare agentview frames for {skill_name}: {exc}")
        return

    wrist_frames_raw = _get_wrist_frames(demo['obs'])
    if wrist_frames_raw is None:
        print(f"  ⚠️  Warning: No wrist camera frames found for {skill_name}, duplicating agentview")
        wrist_frames = agent_frames.copy()
    else:
        try:
            wrist_frames = _prepare_rgb_frames(wrist_frames_raw)
        except Exception as exc:
            print(f"  ⚠️  Warning: Failed to prepare wrist frames for {skill_name}: {exc}")
            wrist_frames = agent_frames.copy()

    # Load segmentation frames if available
    seg_frames = None
    if 'eye_in_hand_segmentation' in demo['obs']:
        try:
            seg_frames = _prepare_segmentation_frames(demo['obs']['eye_in_hand_segmentation'])
        except Exception as exc:
            print(f"  ⚠️  Warning: Failed to prepare segmentation frames for {skill_name}: {exc}")
            seg_frames = None

    # Determine minimum frame count
    min_frames = min(agent_frames.shape[0], wrist_frames.shape[0])
    if seg_frames is not None:
        min_frames = min(min_frames, seg_frames.shape[0])

    if min_frames == 0:
        print(f"  ⚠️  Warning: No usable frames for {skill_name}")
        return

    agent_frames = agent_frames[:min_frames]
    wrist_frames = wrist_frames[:min_frames]
    if seg_frames is not None:
        seg_frames = seg_frames[:min_frames]

    combined_frames = []
    target_hw = agent_frames.shape[1:3]
    target_size = (target_hw[1], target_hw[0])  # PIL uses (width, height)

    if seg_frames is not None:
        # Concatenate 3 views: agentview | wrist_rgb | wrist_segmentation
        for agent_frame, wrist_frame, seg_frame in zip(agent_frames, wrist_frames, seg_frames):
            # Resize wrist frame if needed
            if wrist_frame.shape[0:2] != target_hw:
                try:
                    wrist_frame = np.array(Image.fromarray(wrist_frame).resize(target_size, Image.BILINEAR))
                except Exception:
                    wrist_frame = wrist_frame[:target_hw[0], :target_hw[1]]
            # Resize segmentation frame if needed
            if seg_frame.shape[0:2] != target_hw:
                try:
                    seg_frame = np.array(Image.fromarray(seg_frame).resize(target_size, Image.BILINEAR))
                except Exception:
                    seg_frame = seg_frame[:target_hw[0], :target_hw[1]]
            combined_frames.append(np.concatenate([agent_frame, wrist_frame, seg_frame], axis=1))
    else:
        # No segmentation, just concatenate 2 views: agentview | wrist_rgb
        for agent_frame, wrist_frame in zip(agent_frames, wrist_frames):
            if wrist_frame.shape[0:2] != target_hw:
                try:
                    wrist_frame = np.array(Image.fromarray(wrist_frame).resize(target_size, Image.BILINEAR))
                except Exception:
                    wrist_frame = wrist_frame[:target_hw[0], :target_hw[1]]
            combined_frames.append(np.concatenate([agent_frame, wrist_frame], axis=1))

    frames = np.stack(combined_frames)
    
    # Create video filename
    video_filename = f"{skill_name}{demo_suffix}.mp4"
    video_path = os.path.join(video_dir, video_filename)
    
    # Save video using imageio
    try:
        imageio.mimwrite(video_path, frames, fps=30, quality=8)
        print(f"  ✓ Saved video: {video_filename}")
    except Exception as e:
        print(f"  ⚠️  Failed to save video {video_filename}: {e}")


def get_all_demo_keys(h5file: h5py.File) -> List[str]:
    """
    Get all demo keys from an HDF5 file.
    
    Args:
        h5file: Open HDF5 file
        
    Returns:
        List of demo keys (e.g., ["demo_0", "demo_1", ...])
    """
    if 'data' not in h5file:
        return []
    
    data_group = h5file['data']
    demo_keys = sorted(data_group.keys())
    return demo_keys


def main():
    """Main function to extract videos from HDF5 files."""
    parser = argparse.ArgumentParser(description="Extract videos from HDF5 dataset files")
    parser.add_argument(
        '--check-hdf5',
        action='store_true',
        help='Enable HDF5 check mode: process only a specific HDF5 file and sample 10 demos'
    )
    parser.add_argument(
        '--hdf5-file',
        type=str,
        default=None,
        help='Path to HDF5 file to check (only used in check mode). If specified, processes all demos.'
    )
    parser.add_argument(
        '--failure-mode',
        action='store_true',
        default=False,
        help='If enabled, only extract videos from _failure.hdf5 files (default: False)'
    )
    parser.add_argument(
        '--labels-file',
        type=str,
        default=None,
        help='Path to timestep labels JSON file. Will extract videos with label overlays for specified demos.'
    )
    parser.add_argument(
        '--demos',
        type=str,
        default='demo_0,demo_1',
        help='Comma-separated list of demo IDs to extract when using --labels-file (default: demo_0,demo_1)'
    )

    args = parser.parse_args()
    
    print("=" * 80)
    print("Extract Dataset Videos")
    print("=" * 80)

    # Handle labels file mode (--labels-file)
    if args.labels_file is not None:
        print("Mode: Labels Visualization Mode")
        print(f"Labels file: {args.labels_file}")

        # Load labels (one JSON object per line, with trailing commas)
        if not os.path.exists(args.labels_file):
            print(f"  Error: Labels file does not exist: {args.labels_file}")
            return

        all_labels = []
        with open(args.labels_file, 'r') as f:
            for line in f:
                line = line.strip().rstrip(',')
                if line:
                    all_labels.append(json.loads(line))

        if not all_labels:
            print(f"  Error: Labels file is empty")
            return

        # Get hdf5_path from first label record
        hdf5_path = all_labels[0].get('hdf5_path')
        skill_name = all_labels[0].get('skill_name', 'unknown')
        if not hdf5_path or not os.path.exists(hdf5_path):
            print(f"  Error: HDF5 path not found or invalid: {hdf5_path}")
            return

        print(f"HDF5 file: {hdf5_path}")
        print(f"Skill: {skill_name}")

        # Parse demos to extract
        demo_ids = [d.strip() for d in args.demos.split(',')]
        print(f"Demos to extract: {demo_ids}")

        # Group labels by demo_id
        labels_by_demo = {}
        for lbl in all_labels:
            demo_id = lbl.get('demo_id')
            if demo_id not in labels_by_demo:
                labels_by_demo[demo_id] = []
            labels_by_demo[demo_id].append(lbl)

        # Output directory
        labels_basename = os.path.splitext(os.path.basename(args.labels_file))[0]
        output_video_dir = f"scripts/phase3/pipeline/outputs/videos/demo_gen/labels_check/{labels_basename}"
        os.makedirs(output_video_dir, exist_ok=True)
        print(f"Output directory: {output_video_dir}")
        print()

        # Process each requested demo
        success_count = 0
        with h5py.File(hdf5_path, 'r') as h5file:
            for demo_id in demo_ids:
                if demo_id not in labels_by_demo:
                    print(f"  Warning: No labels found for {demo_id}")
                    continue

                print(f"Processing {demo_id}...")
                demo = load_hdf5_demo(h5file, demo_id)
                if demo is None:
                    print(f"  Warning: Could not load {demo_id}")
                    continue

                demo_labels = labels_by_demo[demo_id]
                save_demo_video_with_labels(demo, skill_name, output_video_dir,
                                            f"_{demo_id}", demo_labels)
                success_count += 1

        print()
        print("=" * 80)
        print(f"Done! {success_count} videos saved to {output_video_dir}")
        print("=" * 80)
        return

    # Handle single HDF5 file mode (when --hdf5-file is specified or --check-hdf5 is set)
    if args.hdf5_file is not None or args.check_hdf5:
        # If --check-hdf5 is set but --hdf5-file is not provided, use default
        if args.check_hdf5 and args.hdf5_file is None:
            args.hdf5_file = 'datasets/hdf5_datasets/atomic_above_fewer_v1/all/place_black_bowl_on_the_plate.hdf5'
            process_all_demos = False
        else:
            # When --hdf5-file is explicitly provided, always process all demos
            process_all_demos = True
        
        print("Mode: Single HDF5 File Mode")
        print(f"HDF5 file: {args.hdf5_file}")
        if process_all_demos:
            print("Processing all demos (--hdf5-file was explicitly specified)")
        else:
            print("Sampling demos (using default file with --check-hdf5)")
        if args.failure_mode:
            print("Failure mode: ENABLED (only processing failure files)")
        
        # Check if failure mode is enabled and file doesn't match
        if args.failure_mode and '_failure.hdf5' not in args.hdf5_file:
            print(f"  ⚠️  Warning: Failure mode is enabled but file is not a failure file: {args.hdf5_file}")
            print(f"  Skipping...")
            return
        
        # Use different output directory for check mode
        hdf5_basename = os.path.splitext(os.path.basename(args.hdf5_file))[0]
        output_video_dir = f"scripts/phase3/pipeline/outputs/videos/demo_gen/hdf5_check/{hdf5_basename}"
        print(f"Output video directory: {output_video_dir}")
        print()
        
        # Create output directory
        os.makedirs(output_video_dir, exist_ok=True)
        
        # Check if file exists
        if not os.path.exists(args.hdf5_file):
            print(f"  ❌ Error: HDF5 file does not exist: {args.hdf5_file}")
            return
        
        # Process the specific HDF5 file
        skill_name = hdf5_basename
        success_count = 0
        failed_count = 0
        demo_keys = []
        num_samples = 0
        
        try:
            with h5py.File(args.hdf5_file, 'r') as h5file:
                # Get all demo keys
                demo_keys = get_all_demo_keys(h5file)
                
                if not demo_keys:
                    print(f"  ❌ Error: No demos found in {args.hdf5_file}")
                    return
                
                print(f"Found {len(demo_keys)} demos in file")
                
                if process_all_demos:
                    # Process all demos
                    print(f"Processing all {len(demo_keys)} demos...")
                    print()
                    demos_to_process = demo_keys
                else:
                    # Sample 4 demos (or all if less than 4)
                    num_samples = min(4, len(demo_keys))
                    demos_to_process = np.random.choice(demo_keys, size=num_samples, replace=False).tolist()
                    print(f"Sampling {num_samples} demos: {demos_to_process}")
                    print()
                
                # Process each demo
                for demo_key in tqdm(demos_to_process, desc="Processing demos", unit="demo"):
                    # Load the demo
                    demo = load_hdf5_demo(h5file, demo_key)
                    
                    if demo is None:
                        print(f"  ⚠️  Warning: Could not load {demo_key}")
                        failed_count += 1
                        continue
                    
                    # Save video with demo suffix
                    demo_suffix = f"_{demo_key}"
                    save_demo_video(demo, skill_name, output_video_dir, demo_suffix)
                    success_count += 1
                    
        except Exception as e:
            print(f"  ❌ Error processing {args.hdf5_file}: {e}")
            return
        
        print()
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"HDF5 file: {args.hdf5_file}")
        print(f"Total demos in file: {len(demo_keys)}")
        if not process_all_demos:
            print(f"Demos sampled: {num_samples}")
        print(f"Successfully processed: {success_count}")
        print(f"Failed: {failed_count}")
        print(f"Videos saved to: {output_video_dir}")
        print("=" * 80)
        return
    
    # Normal mode: process all HDF5 files
    print("Mode: Normal Mode")
    if args.failure_mode:
        print("Failure mode: ENABLED (only processing failure files)")
    print(f"HDF5 directory: {HDF5_DIR}")
    print(f"Output video directory: {OUTPUT_VIDEO_DIR}")
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
    
    # Find all HDF5 files
    if not os.path.exists(HDF5_DIR):
        print(f"  ❌ Error: HDF5 directory does not exist: {HDF5_DIR}")
        return
    
    hdf5_files = [f for f in os.listdir(HDF5_DIR) if f.endswith('.hdf5')]
    
    # Filter to only failure files if failure_mode is enabled
    if args.failure_mode:
        hdf5_files = [f for f in hdf5_files if '_failure.hdf5' in f]
        print(f"Failure mode enabled: filtering to only _failure.hdf5 files")
    
    hdf5_files = sorted(hdf5_files)
    
    if not hdf5_files:
        print(f"  ⚠️  Warning: No HDF5 files found in {HDF5_DIR}")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files")
    print()
    
    # Process each HDF5 file
    success_count = 0
    failed_count = 0
    
    for hdf5_filename in tqdm(hdf5_files, desc="Processing HDF5 files", unit="file"):
        skill_name = os.path.splitext(hdf5_filename)[0]
        hdf5_path = os.path.join(HDF5_DIR, hdf5_filename)
        # if "place_black_bowl_on_the_plate.hdf5" not in hdf5_path:
        #     continue
        
        try:
            with h5py.File(hdf5_path, 'r') as h5file:
                # Get all demo keys
                demo_keys = get_all_demo_keys(h5file)
                
                if not demo_keys:
                    print(f"  ⚠️  Warning: No demos found in {hdf5_filename}")
                    failed_count += 1
                    continue
                
                # Sample 2 demos (or all if less than 2)
                num_samples = min(5, len(demo_keys))
                
                # Prefer non-shifted demos if available
                non_shifted_keys = []
                shifted_keys = []
                
                for demo_key in demo_keys:
                    try:
                        demo_group = h5file['data'][demo_key]
                        if 'metadata' not in demo_group:
                            shifted_keys.append(demo_key)
                            continue
                        
                        meta_group = demo_group['metadata']
                        shifted = False
                        
                        # Check attributes first
                        if 'shifted' in meta_group.attrs:
                            shifted = meta_group.attrs['shifted']
                        elif 'iteration' in meta_group.attrs:
                            iteration = meta_group.attrs['iteration']
                            if isinstance(iteration, (np.integer, np.floating)):
                                iteration = iteration.item()
                            shifted = (iteration > 0)
                        else:
                            # Check scalar datasets
                            if 'shifted' in meta_group:
                                val = meta_group['shifted']
                                if isinstance(val, h5py.Dataset) and val.shape == ():
                                    shifted = val[()].item() if hasattr(val[()], 'item') else val[()]
                            elif 'iteration' in meta_group:
                                val = meta_group['iteration']
                                if isinstance(val, h5py.Dataset):
                                    if val.shape == ():
                                        iteration = val[()].item() if hasattr(val[()], 'item') else val[()]
                                    elif len(val.shape) == 1 and val.shape[0] == 1:
                                        iteration = val[0].item() if hasattr(val[0], 'item') else val[0]
                                    else:
                                        iteration = 0
                                    shifted = (iteration > 0)
                        
                        if not shifted:
                            non_shifted_keys.append(demo_key)
                        else:
                            shifted_keys.append(demo_key)
                    except:
                        shifted_keys.append(demo_key)
                
                # Prioritize non-shifted demos, then fill with shifted if needed
                # Uniformly sample from each category
                selected_keys = []
                if non_shifted_keys:
                    num_non_shifted = min(num_samples, len(non_shifted_keys))
                    selected_keys.extend(np.random.choice(non_shifted_keys, size=num_non_shifted, replace=False).tolist())
                if len(selected_keys) < num_samples and shifted_keys:
                    remaining = num_samples - len(selected_keys)
                    num_shifted = min(remaining, len(shifted_keys))
                    selected_keys.extend(np.random.choice(shifted_keys, size=num_shifted, replace=False).tolist())
                
                # Process each selected demo
                skill_success = 0
                for demo_key in selected_keys:
                    # Load the demo
                    demo = load_hdf5_demo(h5file, demo_key)
                    
                    if demo is None:
                        print(f"  ⚠️  Warning: Could not load {demo_key} from {hdf5_filename}")
                        continue
                    
                    # Extract the number from demo_key (e.g., "demo_2" -> "2", "demo_32" -> "32")
                    demo_num = demo_key.split('_')[-1] if '_' in demo_key else demo_key.replace('demo', '')
                    demo_suffix = f"_{demo_num}"
                    save_demo_video(demo, skill_name, OUTPUT_VIDEO_DIR, demo_suffix)
                    skill_success += 1
                
                if skill_success > 0:
                    success_count += skill_success
                else:
                    failed_count += 1
                
        except Exception as e:
            print(f"  ⚠️  Error processing {hdf5_filename}: {e}")
            failed_count += 1
            continue
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total HDF5 files: {len(hdf5_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Videos saved to: {OUTPUT_VIDEO_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()

