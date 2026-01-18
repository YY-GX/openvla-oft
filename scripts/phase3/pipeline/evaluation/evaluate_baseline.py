#!/usr/bin/env python3
"""
Baseline VLA Evaluation - Pure VLA Rollout Without Motion Planning

This script evaluates VLA models with optional object pose information in a pure
end-to-end rollout fashion (no motion planning, no above-pose waypoints).

Observation space options:
- With object pose (default): agentview + wrist + EE state (8D) + object pose (6D) = 14D
- Without object pose: agentview + wrist + EE state (8D)

Usage:
CUDA_VISIBLE_DEVICES=0 python scripts/phase3/pipeline/evaluation/evaluate_baseline.py \
    --vla_checkpoint runs/libero_oft_obj_long_id10/1.0.0/... \
    --task_name 'Cooking Preparation Setup V1' \
    --include_object_pose True
"""

import os
import sys
import time
import json
import argparse
import warnings
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from typing import Optional

# Suppress warnings
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft')
sys.path.append('/mnt/arc/yygx/pkgs_baselines/openvla-oft/externals/boss')

from scipy.spatial.transform import Rotation as R

# Import pose calculation functions
from scripts.phase3.pipeline.utils.local_pose_tools import get_object_pose


def load_module(path: str, name: str):
    """Load script module dynamically."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BaselinePipeline:
    def __init__(self, vla_checkpoint: str, task_name: str, wrist_only: bool = False,
                 visualize_poses: bool = False, ee_offset: float = 0.0,
                 continue_on_failure: bool = False, max_vla_retries: int = 3,
                 vla_horizon: int = 200, verbose_vla_failure: bool = False,
                 benchmark_name: str = "long_horizon_tasks_v1",
                 apply_background_erasing: bool = False, apply_distractor_masking: bool = False,
                 include_object_pose: bool = True):

        self.task_name = task_name
        self.wrist_only = wrist_only
        self.visualize_poses = visualize_poses
        self.ee_offset = ee_offset
        self.continue_on_failure = continue_on_failure
        self.max_vla_retries = max_vla_retries
        self.vla_horizon = vla_horizon
        self.verbose_vla_failure = verbose_vla_failure
        self.benchmark_name = benchmark_name
        self.apply_background_erasing = apply_background_erasing
        self.apply_distractor_masking = apply_distractor_masking
        self.include_object_pose = include_object_pose

        self.env = None
        self.vla = None
        self.skill_sequence = []
        self.current_masked_wrist_img = None  # Store masked wrist image for video recording

        # Video/image recording setup
        self.video_writer = None
        self.video_filename = None
        self.images_dir = None
        self.video_dir = None
        self.trial_num = 1
        self._setup_recording()

        # Load components
        script_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase3/pipeline"
        self.task_planner = load_module(f"{script_dir}/utils/task_planner.py", "task_planner")
        self.pose_calculator_module = load_module(f"{script_dir}/utils/pose_calculator.py", "pose_calculator")
        self.skill_checker = load_module(f"{script_dir}/utils/skill_checker.py", "skill_checker")

        # Load task config for target object lookup
        task_config_path = f"{script_dir}/config/tasks_and_skills.json"
        with open(task_config_path, 'r') as f:
            self.task_config = json.load(f)
        self.skill_mappings = self.task_config.get("skill_mappings", {})

        # Load skill config for BDDL filename lookup
        skill_config_path = f"{script_dir}/config/skill_config.json"
        with open(skill_config_path, 'r') as f:
            self.skill_config = json.load(f)

        if not visualize_poses:
            from dataclasses import dataclass
            from typing import Union
            from experiments.robot.openvla_utils import get_processor
            from experiments.robot.robot_utils import get_model

            @dataclass
            class VLAConfig:
                model_family: str = "openvla"
                pretrained_checkpoint: Union[str, Path] = ""
                use_l1_regression: bool = True
                use_diffusion: bool = False
                num_diffusion_steps: int = 50
                use_film: bool = False
                num_images_in_input: int = 2
                use_proprio: bool = True
                center_crop: bool = True
                num_open_loop_steps: int = 8
                unnorm_key: Union[str, Path] = ""
                load_in_8bit: bool = False
                load_in_4bit: bool = False
                task_suite_name: str = "atomic_skills"
                wrist_only: bool = False
                agent_only: bool = False
                pro_only: bool = False
                is_oss: bool = False
                is_depth: bool = False

            # Use default checkpoint if none provided
            if vla_checkpoint is None:
                vla_checkpoint = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/libero_oft_obj_long_id10/1.0.0/openvla-7b+libero_oft_obj_long_id10+b16+lr-0.0005+lora-r32+dropout-0.0--baseline_obj_id_10--8_acts_chunk--continuous_acts--L1_regression--wrist_img--14d_state_with_target_obj_pose--100000_chkpt"

            cfg = VLAConfig()
            cfg.pretrained_checkpoint = vla_checkpoint
            cfg.num_images_in_input = 2  # Always use 2 images (agentview + wrist)
            cfg.wrist_only = False  # Use both cameras

            self.vla = get_model(cfg)
            self.processor = get_processor(cfg)

            # Load action head if needed
            from experiments.robot.openvla_utils import get_action_head, get_proprio_projector, get_noisy_action_projector
            self.action_head = None
            self.proprio_projector = None
            self.noisy_action_projector = None

            if cfg.use_l1_regression or cfg.use_diffusion:
                self.action_head = get_action_head(cfg, self.vla.llm_dim)

            if cfg.use_proprio:
                self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, proprio_dim=14 if include_object_pose else 8)

            if cfg.use_diffusion:
                self.noisy_action_projector = get_noisy_action_projector(cfg, self.vla.llm_dim)

            # Auto-detect unnorm_key from checkpoint path
            checkpoint_filename = Path(vla_checkpoint).name
            cfg.unnorm_key = None

            if "+" in checkpoint_filename:
                parts = checkpoint_filename.split("+")
                if len(parts) >= 2:
                    cfg.unnorm_key = parts[1]

            if not cfg.unnorm_key:
                checkpoint_parts = Path(vla_checkpoint).parts
                if "runs" in checkpoint_parts:
                    runs_idx = checkpoint_parts.index("runs")
                    if runs_idx + 1 < len(checkpoint_parts):
                        cfg.unnorm_key = checkpoint_parts[runs_idx + 1]
                    else:
                        cfg.unnorm_key = "bridge_orig"
                else:
                    cfg.unnorm_key = "bridge_orig"

            if hasattr(self.vla, 'norm_stats') and self.vla.norm_stats:
                if cfg.unnorm_key not in self.vla.norm_stats:
                    base_key = cfg.unnorm_key.split(":")[0] if ":" in cfg.unnorm_key else cfg.unnorm_key
                    matching_keys = [k for k in self.vla.norm_stats.keys() if k.startswith(base_key)]
                    if matching_keys:
                        cfg.unnorm_key = matching_keys[0]
                        print(f"🔧 Matched unnorm_key: {cfg.unnorm_key}")
                    else:
                        print(f"⚠️  Warning: unnorm_key '{cfg.unnorm_key}' not found, using 'bridge_orig'")
                        cfg.unnorm_key = "bridge_orig"

            self.cfg = cfg
            print(f"🎯 VLA Model loaded: {vla_checkpoint}")
            print(f"📊 Unnorm key: {cfg.unnorm_key}")
            print(f"📊 Observation space: {'14D (with object pose)' if include_object_pose else '8D (EE state only)'}")

        # Initialize metadata cache
        self._skill_metadata_cache = {}
        self._skill_metadata_lower = {}
        for skill_lang, meta in self.skill_mappings.items():
            self._skill_metadata_cache[skill_lang] = meta
            self._skill_metadata_lower[skill_lang.lower()] = meta

    def _setup_recording(self):
        """Setup video and image recording directories."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_clean = self.task_name.replace(" ", "_").lower()

        # Video directory
        video_base = Path("/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase3/pipeline/outputs/videos/baseline_eval")
        self.video_dir = video_base / f"{task_clean}_{timestamp}"
        self.video_dir.mkdir(parents=True, exist_ok=True)

        # Images directory
        images_base = Path("/mnt/arc/yygx/pkgs_baselines/openvla-oft/images_videos/phase2/long_horizon_tasks_recordings/images")
        self.images_dir = images_base / f"{task_clean}_{timestamp}"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        print(f"📹 Videos will be saved to: {self.video_dir}")
        print(f"📸 Images will be saved to: {self.images_dir}")

    def _start_video(self, trial_name: str):
        """Start video recording."""
        if self.video_writer is not None:
            self._stop_video()

        self.video_filename = self.video_dir / f"{trial_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(str(self.video_filename), fourcc, 30.0, (256, 256))

    def _record_frame(self, obs):
        """Record a single frame to video."""
        if self.video_writer is None:
            return

        # Use masked wrist image if available (from VLA execution)
        if self.current_masked_wrist_img is not None:
            frame = self.current_masked_wrist_img
            self.current_masked_wrist_img = None  # Reset after use
        else:
            # Get wrist image from observation
            frame = self.get_libero_wrist_image(obs)

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame_bgr)

    def _stop_video(self):
        """Stop video recording."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print(f"  💾 Video saved: {self.video_filename.name}")

    def get_libero_wrist_image(self, obs):
        """Get wrist camera image from LIBERO observation."""
        return obs.get('robot0_eye_in_hand_image', obs.get('eye_in_hand_image'))

    def execute(self):
        """Main execution loop."""
        # Setup environment
        init_states, init_info, env, task_lang, skill_sequence = self.task_planner.setup_task_and_env(
            self.task_name, self.benchmark_name
        )
        self.env = env
        self.skill_sequence = skill_sequence

        print(f"\n🎯 Task: {task_lang}")
        print(f"📋 Skills: {', '.join(skill_sequence)}")

        # Reset environment
        obs = env.reset()
        env.reset_to(init_states)
        self._record_frame(obs)

        # Start video recording
        trial_name = f"trial_{self.trial_num}"
        self._start_video(trial_name)
        self.trial_num += 1

        # Execute skill sequence
        overall_success = True
        grasped_object = None  # Track grasped object for place skills

        for skill_idx, skill in enumerate(skill_sequence):
            print(f"\n{'='*60}")
            print(f"🎯 Skill {skill_idx + 1}/{len(skill_sequence)}: {skill}")
            print(f"{'='*60}")

            # Get skill metadata
            skill_meta = self._skill_metadata_cache.get(skill)
            if skill_meta is None:
                skill_meta = self._skill_metadata_lower.get(skill.lower())

            if not skill_meta:
                print(f"  ❌ Skill metadata not found for '{skill}'")
                overall_success = False
                if not self.continue_on_failure:
                    break
                continue

            # Get target object and VLA language
            target_object = skill_meta.get("target_object")
            vla_language = skill_meta.get("language", skill)

            print(f"  🎯 Target object: {target_object}")
            print(f"  💬 VLA language: {vla_language}")

            # Update grasped object tracking
            if "pick" in skill.lower():
                grasped_object = target_object
            elif "place" in skill.lower() or "open" in skill.lower() or "close" in skill.lower() or "turn" in skill.lower():
                # For place skills, target_object is where to place
                # Keep grasped_object for distractor masking
                pass

            # Execute VLA skill with retries
            skill_success = False
            for attempt in range(self.max_vla_retries):
                if attempt > 0:
                    print(f"\n  🔄 Retry {attempt}/{self.max_vla_retries - 1}")

                # Execute VLA
                skill_success, overall_task_done = self._execute_vla(
                    skill, vla_language, obs, target_object, grasped_object
                )

                if skill_success:
                    print(f"  ✅ Skill succeeded on attempt {attempt + 1}")
                    break
                else:
                    print(f"  ❌ Skill failed on attempt {attempt + 1}")

            if not skill_success:
                print(f"  ❌ Skill failed after {self.max_vla_retries} attempts")
                overall_success = False
                if not self.continue_on_failure:
                    break

            # Clear grasped object after successful place
            if skill_success and "place" in skill.lower():
                grasped_object = None

        # Stop video recording
        self._stop_video()

        # Final result
        print(f"\n{'='*60}")
        print(f"{'✅ TASK SUCCESS' if overall_success else '❌ TASK FAILED'}")
        print(f"{'='*60}\n")

        return overall_success

    def _execute_vla(self, skill: str, language: str, obs, target_object: str, grasped_object_name: Optional[str] = None) -> tuple:
        """
        Execute VLA skill.

        Args:
            skill: Skill name with numbers (e.g., "pick black bowl 2") - for success checking
            language: Clean VLA language (e.g., "pick black bowl") - for VLA execution
            obs: Current observation
            target_object: Target object name
            grasped_object_name: Grasped object name (for place skills) or None

        Returns:
            (skill_success, overall_task_done)
        """
        from collections import deque
        from experiments.robot.robot_utils import get_action, normalize_gripper_action, invert_gripper_action, get_image_resize_size
        from experiments.robot.libero.libero_utils import quat2axisangle
        from experiments.robot.openvla_utils import resize_image_for_policy
        from scripts.phase3.pipeline.utils.segmentation_utils import create_wrist_segmentation_mask

        # Get fresh observation
        obs = self.env.env._get_observations()
        self._record_frame(obs)

        action_queue = deque(maxlen=self.cfg.num_open_loop_steps)
        resize_size = get_image_resize_size(self.cfg)
        overall_task_done = False

        for t in range(self.vla_horizon):
            if len(action_queue) == 0:
                # Get images
                wrist_img = self.get_libero_wrist_image(obs)
                agentview_img = obs.get('agentview_image', obs.get('robot0_agentview_image'))

                # Apply background erasing if enabled
                if self.apply_background_erasing and target_object:
                    try:
                        seg_mask = create_wrist_segmentation_mask(self.env, target_object)
                        from scripts.phase3.pipeline.utils.random_erasing_mask import apply_random_erasing_to_mask
                        erased_mask = apply_random_erasing_to_mask(seg_mask, erasing_ratio=0.25, num_rectangles=5)
                        mask_3ch = (erased_mask[:, :, None] / 255.0).astype(np.float32)
                        wrist_img = (wrist_img * mask_3ch).astype(np.uint8)
                    except Exception as e:
                        pass

                # Apply distractor masking if enabled
                if self.apply_distractor_masking and target_object:
                    try:
                        from scripts.phase3.pipeline.utils.segmentation_utils import apply_distractor_masking
                        wrist_img = apply_distractor_masking(
                            self.env,
                            wrist_img,
                            target_object,
                            grasped_object_name=grasped_object_name,
                            resolution=256
                        )
                    except Exception as e:
                        pass

                # Store masked wrist image for video recording
                if self.apply_background_erasing or self.apply_distractor_masking:
                    self.current_masked_wrist_img = wrist_img.copy()

                # Resize images
                wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
                agentview_img_resized = resize_image_for_policy(agentview_img, resize_size)

                # Construct proprioception state
                ee_pos = obs["robot0_eef_pos"]
                ee_quat_xyzw = obs["robot0_eef_quat"]
                gripper_state = obs["robot0_gripper_qpos"]

                # Convert EE quaternion to axis-angle
                ee_rot = R.from_quat(ee_quat_xyzw)
                ee_ori_axisangle = ee_rot.as_rotvec()

                if self.include_object_pose:
                    # 14D state: EE pose (6D) + gripper (2D) + object pose (6D)
                    obj_pos, obj_quat_wxyz = get_object_pose(self.env, target_object)

                    # Convert object quaternion to axis-angle
                    obj_quat_xyzw = np.array([obj_quat_wxyz[1], obj_quat_wxyz[2], obj_quat_wxyz[3], obj_quat_wxyz[0]])
                    obj_rot = R.from_quat(obj_quat_xyzw)
                    obj_ori_axisangle = obj_rot.as_rotvec()

                    state = np.concatenate((
                        ee_pos,              # 3D
                        ee_ori_axisangle,    # 3D
                        gripper_state,       # 2D
                        obj_pos,             # 3D (absolute object position)
                        obj_ori_axisangle,   # 3D (absolute object orientation)
                    ))
                else:
                    # 8D state: EE pose (6D) + gripper (2D)
                    state = np.concatenate((
                        ee_pos,              # 3D
                        ee_ori_axisangle,    # 3D
                        gripper_state,       # 2D
                    ))

                # Create observation dict
                observation = {
                    "full_image": agentview_img_resized,  # Main camera (3rd person view)
                    "wrist_image": wrist_img_resized,     # Wrist camera
                    "state": state
                }

                # Get VLA action
                action = get_action(
                    self.cfg,
                    self.vla,
                    self.processor,
                    observation,
                    language,
                    self.action_head,
                    self.proprio_projector,
                    self.noisy_action_projector
                )

                # Normalize and invert gripper action
                action = normalize_gripper_action(action, binarize=True)
                action = invert_gripper_action(action)

                # Add to action queue
                for _ in range(self.cfg.num_open_loop_steps):
                    action_queue.append(action)

            # Execute action
            action = action_queue.popleft()
            obs, reward, done, info = self.env.step(action)
            self._record_frame(obs)

            # Check for termination
            gripper_closed = obs["robot0_gripper_qpos"][0] < 0.01
            if gripper_closed or t >= self.vla_horizon - 1:
                break

        # Check skill success
        skill_success = self.skill_checker.check_skill_success_by_language(
            self.env, skill, debug=False
        )

        return skill_success, overall_task_done


def main():
    parser = argparse.ArgumentParser(description='Baseline VLA Evaluation')
    parser.add_argument('--vla_checkpoint', type=str, default=None,
                       help='Path to VLA checkpoint')
    parser.add_argument('--task_name', type=str, default='Cooking Preparation Setup V1',
                       help='Task name from LIBERO benchmark')
    parser.add_argument('--wrist_only', action='store_true', default=False,
                       help='Use wrist camera only (not recommended for baseline)')
    parser.add_argument('--visualize_poses', action='store_true', default=False,
                       help='Visualize poses only, do not execute')
    parser.add_argument('--ee_offset', type=float, default=0.0,
                       help='EE backward offset in meters (not used in baseline)')
    parser.add_argument('--continue_on_failure', action='store_true', default=False,
                       help='Continue executing remaining skills after failure')
    parser.add_argument('--max_vla_retries', type=int, default=3,
                       help='Maximum number of retry attempts for VLA execution')
    parser.add_argument('--vla_horizon', type=int, default=200,
                       help='Maximum timesteps for VLA skill execution')
    parser.add_argument('--verbose_vla_failure', action='store_true', default=False,
                       help='Print verbose VLA failure information')
    parser.add_argument('--benchmark', type=str, default='long_horizon_tasks_v1',
                       help='LIBERO benchmark name')
    parser.add_argument('--apply_background_erasing', action='store_true', default=False,
                       help='Apply random erasing to wrist camera background')
    parser.add_argument('--apply_distractor_masking', action='store_true', default=False,
                       help='Mask out distractor objects in wrist camera')
    parser.add_argument('--include_object_pose', type=lambda x: x.lower() == 'true', default=True,
                       help='Include target object pose in proprioception (default: True, 14D state; False: 8D state)')

    args = parser.parse_args()

    # Create pipeline
    pipeline = BaselinePipeline(
        vla_checkpoint=args.vla_checkpoint,
        task_name=args.task_name,
        wrist_only=args.wrist_only,
        visualize_poses=args.visualize_poses,
        ee_offset=args.ee_offset,
        continue_on_failure=args.continue_on_failure,
        max_vla_retries=args.max_vla_retries,
        vla_horizon=args.vla_horizon,
        verbose_vla_failure=args.verbose_vla_failure,
        benchmark_name=args.benchmark,
        apply_background_erasing=args.apply_background_erasing,
        apply_distractor_masking=args.apply_distractor_masking,
        include_object_pose=args.include_object_pose,
    )

    # Execute task
    success = pipeline.execute()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
