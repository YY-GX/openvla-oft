"""
Generate sample videos from RLDS dataset to visualize wrist camera images.

This script loads an RLDS dataset and generates multiple videos per skill (language instruction),
showing the wrist camera view across all timesteps of randomly sampled episodes.

Usage:

python scripts/phase3/pipeline/utils/generate_rlds_sample_videos.py \
    --dataset_name libero_above_atomic_libero_all \
    --output_dir scripts/phase3/pipeline/outputs/videos/demo_gen/seg_ds_videos_libero_above_atomic_libero_all \
    --num_samples 3

python scripts/phase3/pipeline/utils/generate_rlds_sample_videos.py \
    --dataset_name libero_above_atomic_libero_complex_tasks \
    --output_dir scripts/phase3/pipeline/outputs/videos/demo_gen/seg_ds_videos_libero_above_atomic_libero_complex_tasks \
    --num_samples 3

python scripts/phase3/pipeline/utils/generate_rlds_sample_videos.py \
    --dataset_name libero_above_atomic_libero_long_all17 \
    --output_dir scripts/phase3/pipeline/outputs/videos/demo_gen/seg_ds_videos_libero_above_atomic_libero_long_all17 \
    --num_samples 3

"""

import argparse
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image


def generate_videos(dataset_name: str, data_dir: str, output_dir: str, fps: int = 10, num_samples: int = 3):
    """
    Generate multiple videos per skill from an RLDS dataset.

    Args:
        dataset_name: Name of the RLDS dataset (e.g., 'libero_above_atomic_long_id10')
        data_dir: Directory containing RLDS datasets
        output_dir: Directory to save output videos
        fps: Frames per second for output videos
        num_samples: Number of randomly sampled episodes per skill to generate videos for
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
    print(f'Generating {num_samples} videos per skill...\n')

    total_videos = 0
    # Generate multiple videos per skill
    for skill_idx, (skill_name, episodes) in enumerate(sorted(skill_episodes.items()), 1):
        print(f'[{skill_idx}/{len(skill_episodes)}] Skill: {skill_name}')

        # Randomly sample episodes for this skill
        num_available = len(episodes)
        num_to_sample = min(num_samples, num_available)
        sampled_episodes = random.sample(episodes, num_to_sample)

        print(f'  Sampling {num_to_sample} episode(s) from {num_available} available')

        # Generate video for each sampled episode
        for demo_idx, episode in enumerate(sampled_episodes):
            # Collect wrist images from all steps
            frames = []
            for step in episode['steps']:
                wrist_image = step['observation']['wrist_image'].numpy()
                frames.append(wrist_image)

            print(f'  Demo {demo_idx + 1}/{num_to_sample}: Collected {len(frames)} frames')

            # Create video
            if len(frames) > 0:
                # Sanitize skill name for filename
                safe_skill_name = skill_name.replace(' ', '_').replace('/', '_')
                video_path = os.path.join(output_dir, f'{safe_skill_name}_demo{demo_idx + 1}.mp4')

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
                print(f'    ✓ Saved: {video_path}')
                total_videos += 1
            else:
                print(f'    ⚠️  No frames found for this demo')

    print(f'\nDone! Generated {total_videos} videos in {output_dir}')


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
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Number of randomly sampled episodes per skill to generate videos for (default: 3)')

    args = parser.parse_args()

    generate_videos(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        num_samples=args.num_samples
    )


if __name__ == '__main__':
    main()
