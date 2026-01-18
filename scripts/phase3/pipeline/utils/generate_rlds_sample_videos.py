"""
Generate sample videos from RLDS dataset to visualize wrist camera images.

This script loads an RLDS dataset and generates one video per skill (language instruction),
showing the wrist camera view across all timesteps of one episode.

Usage:
    python scripts/phase3/pipeline/utils/generate_rlds_sample_videos.py \
        --dataset_name libero_above_atomic_libero_long_all17 \
        --output_dir scripts/phase3/pipeline/outputs/videos/demo_gen/seg_ds_videos_libero_long_all17
"""

import argparse
import os
from collections import defaultdict

import cv2
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image


def generate_videos(dataset_name: str, data_dir: str, output_dir: str, fps: int = 10):
    """
    Generate one video per skill from an RLDS dataset.

    Args:
        dataset_name: Name of the RLDS dataset (e.g., 'libero_above_atomic_long_id10')
        data_dir: Directory containing RLDS datasets
        output_dir: Directory to save output videos
        fps: Frames per second for output videos
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    print(f'Loading dataset: {dataset_name}')
    ds = tfds.load(dataset_name, data_dir=data_dir, split='train')

    # Group episodes by skill (language instruction)
    skill_episodes = defaultdict(list)
    print('Grouping episodes by skill...')

    for episode_idx, episode in enumerate(ds):
        # Get language instruction from first step
        steps_list = list(episode['steps'])
        if len(steps_list) > 0:
            language = steps_list[0]['language_instruction'].numpy().decode('utf-8')
            skill_episodes[language].append(episode)

        if episode_idx % 100 == 0:
            print(f'  Processed {episode_idx} episodes, found {len(skill_episodes)} unique skills')

    print(f'\nFound {len(skill_episodes)} unique skills')
    print(f'Generating 1 video per skill...\n')

    # Generate one video per skill
    for skill_idx, (skill_name, episodes) in enumerate(sorted(skill_episodes.items()), 1):
        print(f'[{skill_idx}/{len(skill_episodes)}] Skill: {skill_name}')

        # Use first episode for this skill
        episode = episodes[0]

        # Collect wrist images from all steps
        frames = []
        for step in episode['steps']:
            wrist_image = step['observation']['wrist_image'].numpy()
            frames.append(wrist_image)

        print(f'  Collected {len(frames)} frames')

        # Create video
        if len(frames) > 0:
            # Sanitize skill name for filename
            safe_skill_name = skill_name.replace(' ', '_').replace('/', '_')
            video_path = os.path.join(output_dir, f'{safe_skill_name}.mp4')

            # Video writer setup
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            # Write frames
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()
            print(f'  ✓ Saved: {video_path}')
        else:
            print(f'  ⚠️  No frames found for this skill')

    print(f'\nDone! Generated {len(skill_episodes)} videos in {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='Generate sample videos from RLDS dataset')
    parser.add_argument('--dataset_name', type=str, default='libero_above_atomic_long_id10',
                       help='Name of the RLDS dataset')
    parser.add_argument('--data_dir', type=str, default='datasets/rlds_datasets',
                       help='Directory containing RLDS datasets')
    parser.add_argument('--output_dir', type=str,
                       default='scripts/phase3/pipeline/outputs/videos/demo_gen/seg_ds_videos',
                       help='Directory to save output videos')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for output videos')

    args = parser.parse_args()

    generate_videos(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        fps=args.fps
    )


if __name__ == '__main__':
    main()
