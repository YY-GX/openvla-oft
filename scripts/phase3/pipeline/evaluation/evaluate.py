#!/usr/bin/env python3
"""
Streamlined Long-Horizon Pipeline - Minimal logging, maximum clarity.

Executes long-horizon tasks using VLA with atomic skill decomposition.
"""

"""
CUDA_VISIBLE_DEVICES=7 python scripts/phase3/pipeline/evaluation/evaluate.py --motion_planner_type mplib
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

from scripts.phase3.pipeline.motion_planning.standard_planner import MotionPlanner as StandardMotionPlanner
from scripts.phase3.pipeline.motion_planning.mplib.planner import MPlibMotionPlanner

from scipy.spatial.transform import Rotation as R


def load_module(path: str, name: str):
    """Load script module dynamically."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def move_ee_up(env, motion_planner, next_skill_type: str = 'other', grasped_object: str = None) -> bool:
    """Move end-effector up 3cm in Z-axis using motion planner."""
    try:
        # Get current observation
        obs = env.env._get_observations()
        current_ee_pos = obs['robot0_eef_pos'].copy()
        current_ee_quat = obs['robot0_eef_quat'].copy()

        # Move up 3cm in Z-axis
        target_ee_pos = current_ee_pos.copy()
        target_ee_pos[2] += 0.050  # 3cm up
        # Convert from [x, y, z, w] (xyzw - robosuite format) to [w, x, y, z] (wxyz - for move_to_pose)
        target_ee_quat = np.array([current_ee_quat[3], current_ee_quat[0], current_ee_quat[1], current_ee_quat[2]])

        print(f"   📈 Moving EE up 3cm: {current_ee_pos[2]:.3f} → {target_ee_pos[2]:.3f}")

        move_kwargs = {
            'position_threshold': 0.02,  # 1cm threshold
            'orientation_threshold': 0.348,  # 20 degrees
            'skill_type': next_skill_type
        }

        if isinstance(motion_planner, MPlibMotionPlanner):
            # Pass grasped object if provided (e.g., after pick)
            move_kwargs['grasped_object_name'] = grasped_object
        else:
            move_kwargs['num_steps'] = 200  # Limit steps for quick upward movement

        # Use motion planner to move up with limited steps
        move_success, [observations, _, _, _] = motion_planner.move_to_pose(
            target_ee_pos, target_ee_quat,
            **move_kwargs
        )

        # Record all frames if pipeline has recording capability
        if hasattr(env, 'pipeline') and hasattr(env.pipeline, '_record_frame'):
            for obs in observations:
                env.pipeline._record_frame(obs)

        if move_success:
            print(f"   ✅ Successfully moved EE up 3cm")
        else:
            print(f"   ❌ Failed to move EE up 3cm")

        return move_success

    except Exception as e:
        print(f"   ❌ Error moving EE up: {e}")
        return False


def post_actions(skill: str, env, motion_planner, skill_succeeded: bool = False, next_skill: str = None, grasped_object: str = None) -> bool:
    """Execute gripper actions and move EE up."""
    skill_lower = skill.lower()
    if 'pick' in skill_lower:
        gripper_action = 1.0  # Close gripper
    elif 'place' in skill_lower:
        gripper_action = -1.0  # Open gripper
    else:
        gripper_action = 0.0  # No gripper action

    action = np.array([0, 0, 0, 0, 0, 0, gripper_action])

    print(f"   🤏 Executing post-actions: {'close' if gripper_action > 0 else 'open' if gripper_action < 0 else 'no'} gripper")
    for _ in range(5):
        obs, _, _, _ = env.step(action)
        # Record frames during post-actions if pipeline has recording capability
        if hasattr(env, 'pipeline') and hasattr(env.pipeline, '_record_frame'):
            env.pipeline._record_frame(obs)

    # Move EE up 3cm with next skill type
    if next_skill:
        next_skill_lower = next_skill.lower()
        if 'pick' in next_skill_lower:
            next_skill_type = 'pick'
        elif 'place' in next_skill_lower:
            next_skill_type = 'place'
        else:
            next_skill_type = 'other'

        # Pass grasped object if we just completed a pick
        held_object = grasped_object if 'pick' in skill_lower else None
        move_ee_up(env, motion_planner, next_skill_type, grasped_object=held_object)

    return True


class LongHorizonPipeline:
    def __init__(self, vla_checkpoint: str, task_name: str, wrist_only: bool = False,
                 visualize_poses: bool = False, ee_offset: float = 0.0,
                 motion_planner_method: str = "cartesian_linear", continue_on_failure: bool = False,
                 mp_steps: int = 400, mp_pos_gain: float = 5.0, mp_ori_gain: float = 5.0,
                 mp_pos_threshold: float = 0.02, mp_ori_threshold: float = 0.524,
                 mp_pos_success_threshold: float = 0.025, mp_ori_success_threshold: float = 0.524,
                 motion_planner_type: str = "standard", collision_aware: bool = True,
                 mp_velocity_factor: float = 0.9, obj_ee_pairs_dir: str = None,
                 max_vla_retries: int = 3,
                mplib_joint_vel_limits: float = 2.0, mplib_joint_acc_limits: float = 4.0,
                mplib_waypoint_target_ratio: int = 50, mplib_planning_time: float = 1.0,
                 mplib_safety_margin: float = 1.1, mplib_erode_kernel_size: int = 3,
                 mplib_outlier_std_ratio: float = 0.0, vla_horizon: int = 200,
                 verbose_vla_failure: bool = False):

        self.task_name = task_name
        self.wrist_only = wrist_only
        self.visualize_poses = visualize_poses
        self.ee_offset = ee_offset
        self.motion_planner_method = motion_planner_method
        self.motion_planner_type = motion_planner_type
        self.collision_aware = collision_aware
        self.continue_on_failure = continue_on_failure
        self.mp_steps = mp_steps
        self.mp_pos_gain = mp_pos_gain
        self.mp_ori_gain = mp_ori_gain
        self.mp_pos_threshold = mp_pos_threshold
        self.mp_ori_threshold = mp_ori_threshold
        self.mp_pos_success_threshold = mp_pos_success_threshold
        self.mp_ori_success_threshold = mp_ori_success_threshold
        self.mp_velocity_factor = mp_velocity_factor

        # MPlib-specific parameters
        self.mplib_joint_vel_limits = mplib_joint_vel_limits
        self.mplib_joint_acc_limits = mplib_joint_acc_limits
        self.mplib_waypoint_target_ratio = mplib_waypoint_target_ratio
        self.mplib_planning_time = mplib_planning_time
        self.mplib_safety_margin = mplib_safety_margin
        self.mplib_erode_kernel_size = mplib_erode_kernel_size
        self.mplib_outlier_std_ratio = mplib_outlier_std_ratio
        self.obj_ee_pairs_dir = obj_ee_pairs_dir
        self.max_vla_retries = max_vla_retries
        self.vla_horizon = vla_horizon
        self.verbose_vla_failure = verbose_vla_failure

        self.env = None
        self.motion_planner = None
        self.vla = None
        self.skill_sequence = []

        # Video/image recording setup
        self.video_writer = None
        self.video_filename = None
        self.images_dir = None
        self._setup_recording()

        # Load components
        script_dir = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase3/pipeline"
        self.task_planner = load_module(f"{script_dir}/utils/task_planner.py", "task_planner")
        self.pose_calculator_module = load_module(f"{script_dir}/utils/pose_calculator.py", "pose_calculator")
        self.skill_checker = load_module(f"{script_dir}/utils/skill_checker.py", "skill_checker")

        if not visualize_poses:
            from dataclasses import dataclass
            from typing import Union
            from pathlib import Path
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
                vla_checkpoint = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/libero_atomic_skills_augmented_closer/1.0.0/openvla-7b+libero_atomic_skills_augmented_closer+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_closer--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt"

            cfg = VLAConfig()
            cfg.pretrained_checkpoint = vla_checkpoint
            cfg.num_images_in_input = 1 if wrist_only else 2 
            cfg.wrist_only = wrist_only

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
                self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, proprio_dim=8)

            if cfg.use_diffusion:
                self.noisy_action_projector = get_noisy_action_projector(cfg, self.vla.llm_dim)

            # Automatically determine unnorm_key from checkpoint path
            if "libero_atomic_skills_augmented_closer_original" in vla_checkpoint:
                cfg.unnorm_key = "libero_atomic_skills_augmented_closer_original"
            elif "libero_atomic_skills_augmented_closer" in vla_checkpoint:
                cfg.unnorm_key = "libero_atomic_skills_augmented_closer"
            elif "libero_atomic_skills_augmented_farther" in vla_checkpoint:
                cfg.unnorm_key = "libero_atomic_skills_augmented_farther"
            elif "libero_atomic_skills_augmented_long_id1" in vla_checkpoint:
                cfg.unnorm_key = "libero_atomic_skills_augmented_long_id1"
            elif "libero_atomic_skills_augmented_long_id2" in vla_checkpoint:
                cfg.unnorm_key = "libero_atomic_skills_augmented_long_id2"
            elif "libero_atomic_skills_augmented_long_id3" in vla_checkpoint:                
                cfg.unnorm_key = "libero_atomic_skills_augmented_long_id3"
            else:
                cfg.unnorm_key = "libero_atomic_skills"

            # Import VLA execution utilities
            from experiments.robot.libero.libero_utils import get_libero_wrist_image
            self.get_libero_wrist_image = get_libero_wrist_image
            self.cfg = cfg

    def _setup_recording(self):
        """Setup video and image recording directories."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_clean = self.task_name.replace(" ", "_").lower()

        if self.visualize_poses:
            # Pose visualization mode - only save images
            pose_dir = Path("/mnt/arc/yygx/pkgs_baselines/openvla-oft/images_videos/phase2/long_horizon_tasks_recordings/pose_visualization")
            self.images_dir = pose_dir / f"{task_clean}_{timestamp}"
            self.images_dir.mkdir(parents=True, exist_ok=True)
            print(f"📷 Pose images will be saved to: {self.images_dir}")
        else:
            # Normal mode - video + images
            video_dir = Path("/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase3/pipeline/outputs/videos/long_eval")
            video_dir.mkdir(parents=True, exist_ok=True)
            self.video_filename = video_dir / f"{task_clean}_{timestamp}.mp4"

            images_base = Path("/mnt/arc/yygx/pkgs_baselines/openvla-oft/images_videos/phase2/long_horizon_tasks_recordings/images")
            self.images_dir = images_base / f"{task_clean}_{timestamp}"
            self.images_dir.mkdir(parents=True, exist_ok=True)

    def _start_video(self, obs):
        """Start video recording."""
        if self.visualize_poses or obs is None or not isinstance(obs, dict) or 'agentview_image' not in obs:
            return
        h, w = obs['agentview_image'].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(str(self.video_filename), fourcc, 20, (w, h))

    def _record_frame(self, obs):
        """Record frame to video."""
        # Guard against non-dict observations (e.g., strings)
        if self.video_writer is None or obs is None or not isinstance(obs, dict) or 'agentview_image' not in obs:
            return
        frame = obs['agentview_image']
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        # Rotate 180 degrees
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def _save_mp_images(self, obs, skill_idx: int, skill_name: str):
        """Save images after motion planner completes."""
        if self.images_dir is None or obs is None or not isinstance(obs, dict):
            return

        prefix = f"skill_{skill_idx:02d}_{skill_name.replace(' ', '_')}"

        # Save agent view
        if 'agentview_image' in obs:
            img = obs['agentview_image']
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            cv2.imwrite(str(self.images_dir / f"{prefix}_agent.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Save wrist view
        if 'robot0_eye_in_hand_image' in obs:
            img = obs['robot0_eye_in_hand_image']
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            cv2.imwrite(str(self.images_dir / f"{prefix}_wrist.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def _stop_video(self):
        """Stop video recording."""
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"📹 Video saved: {self.video_filename}")

    def _print_recovery_summary(self, recovery_stats: dict, success_count: int, total_skills: int):
        """Print detailed recovery statistics."""
        print(f"\n{'='*80}")
        print(f"📊 RECOVERY STATISTICS SUMMARY")
        print(f"{'='*80}")

        # Overall progress
        print(f"\n🎯 Overall Progress: {success_count}/{total_skills} skills completed")

        # Final failure cause
        if recovery_stats['final_failure']:
            ff = recovery_stats['final_failure']
            print(f"\n❌ Final Failure:")
            print(f"   Skill: '{ff['skill']}' (#{ff['skill_idx']}, type: {ff['skill_type']})")
            print(f"   Failed Stage: {ff['stage'].upper()}")
            print(f"   Recovery Attempts:")
            for stage, count in ff['attempted_recoveries'].items():
                if count > 0:
                    stage_name = {'pose': 'Pose Calc', 'mp': 'Motion Planning', 'vla': 'VLA Execution', 'pick_recovery': 'Pick Recovery'}[stage]
                    print(f"      - {stage_name}: {count} attempts")

        # Per-skill recovery breakdown
        print(f"\n📋 Per-Skill Recovery Breakdown:")
        print(f"{'─'*80}")

        for skill_stat in recovery_stats['skills']:
            status = '✅' if skill_stat['success'] else '❌'
            print(f"{status} Skill #{skill_stat['skill_idx']}: {skill_stat['skill_name']} ({skill_stat['skill_type']})")

            # Show recovery attempts if any
            had_retries = False
            if skill_stat['pose_retries'] > 0:
                print(f"      🔄 Pose Calc retries: {skill_stat['pose_retries']}")
                had_retries = True
            if skill_stat['mp_retries'] > 0:
                print(f"      🔄 Motion Planning retries: {skill_stat['mp_retries']}")
                had_retries = True
            if skill_stat['vla_retries'] > 0:
                print(f"      🔄 VLA Execution retries: {skill_stat['vla_retries']}")
                had_retries = True
            if skill_stat['pick_recovery_attempts'] > 0:
                recovery_status = '✅' if skill_stat['pick_recovery_success'] else '❌'
                print(f"      🔄 Pick Recovery attempts: {skill_stat['pick_recovery_attempts']} {recovery_status}")
                had_retries = True

            if not had_retries and skill_stat['success']:
                print(f"      ✨ Success on first attempt!")

        # Aggregate statistics
        print(f"\n📈 Aggregate Statistics:")
        total_pose_retries = sum(s['pose_retries'] for s in recovery_stats['skills'])
        total_mp_retries = sum(s['mp_retries'] for s in recovery_stats['skills'])
        total_vla_retries = sum(s['vla_retries'] for s in recovery_stats['skills'])
        total_pick_recovery = sum(s['pick_recovery_attempts'] for s in recovery_stats['skills'])
        successful_pick_recovery = sum(1 for s in recovery_stats['skills'] if s['pick_recovery_success'])

        print(f"   Total Pose Calc retries: {total_pose_retries}")
        print(f"   Total Motion Planning retries: {total_mp_retries}")
        print(f"   Total VLA Execution retries: {total_vla_retries}")
        if total_pick_recovery > 0:
            print(f"   Total Pick Recovery attempts: {total_pick_recovery} ({successful_pick_recovery} successful)")

        # Skills requiring recovery
        skills_with_recovery = sum(1 for s in recovery_stats['skills']
                                   if s['pose_retries'] > 0 or s['mp_retries'] > 0 or
                                      s['vla_retries'] > 0 or s['pick_recovery_attempts'] > 0)
        print(f"   Skills requiring recovery: {skills_with_recovery}/{total_skills}")

        print(f"{'='*80}\n")

    def _execute_skill_once(self, skill: str, skill_idx: int, skill_type: str) -> tuple:
        """
        Execute a single skill attempt: pose calc → MP → VLA.
        No retry logic - just one attempt.

        Args:
            skill: Skill name
            skill_idx: Skill index for logging/saving
            skill_type: 'pick', 'place', or 'other'

        Returns:
            (success, task_done, vla_language, failure_stage, target_object)
            - success: True if all 3 stages passed
            - task_done: True if overall task completed
            - vla_language: Language instruction for VLA
            - failure_stage: 'pose', 'mp', 'vla', or None if success
            - target_object: MuJoCo object name (e.g., "akita_black_bowl_1_main")
        """


        """
        s
        1. 🎯 Target EE:  pos=[-0.0976, -0.3535, 1.1772], axis=[-176.9°, 3.6°, -28.8°]
        2.  🎯 Target EE:  pos=[-0.0637, -0.3519, 1.1772], axis=[-176.9°, 3.6°, -28.8°]
        plate: 🎯 Target EE:  pos=[-0.1442, 0.2257, 0.9321], axis=[-168.8°, 44.8°, -5.3°]

        mplib
        1. 🎯 Target EE:  pos=[0.0011, -0.4092, 1.2298], axis=[-15.0°, 5.5°, -2.6°]
        2. pos=[-0.0420, -0.3472, 1.1772], axis=[-176.9°, 3.6°, -28.8°]

        plate: 🎯 Target EE:  pos=[-0.1793, 0.1952, 0.9402], axis=[-150.9°, 66.5°, -23.1°]
        """
        # Stage 1: Pose calculation
        pose_result = self.pose_calculator_module.calculate_gt_local_pose(
            skill, self.env, pose_pairs_dir=self.obj_ee_pairs_dir)

        if pose_result is None:
            print(f"   1️⃣ ❌ Pose calculation failed")
            return False, False, None, 'pose', None

        target_pos, target_quat, vla_language, target_object = pose_result
        # from scipy.spatial.transform import Rotation as R
        # print(np.degrees(R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]]).as_rotvec()))
        # print(np.degrees(R.from_quat([target_quat[0], target_quat[1], target_quat[2], target_quat[3]]).as_rotvec()))
        # exit(0)

        # Apply EE offset if needed
        if self.ee_offset > 0:
            from scipy.spatial.transform import Rotation as R
            # FIX: target_quat is [w,x,y,z] from pose_calculator, convert to [x,y,z,w] for scipy
            ee_quat_xyzw = [target_quat[1], target_quat[2], target_quat[3], target_quat[0]]
            ee_rot = R.from_quat(ee_quat_xyzw)
            target_pos = target_pos - ee_rot.apply([0, 0, 1]) * self.ee_offset

        # Stage 2: Motion planning
        grasped_obj = self._get_grasped_object(skill_idx) if skill_type == 'place' else None

        if isinstance(self.motion_planner, MPlibMotionPlanner):
            # target_quat = [target_quat[3], target_quat[0], target_quat[1], target_quat[2]]
            move_success, [observations, _, _, _] = self.motion_planner.move_to_pose(
                target_pos, target_quat,
                position_threshold=self.mp_pos_threshold,
                orientation_threshold=self.mp_ori_threshold,
                skill_type=skill_type,
                grasped_object_name=grasped_obj)
        else:
            move_success, observations = self.motion_planner.move_to_pose(
                target_pos, target_quat,
                position_threshold=self.mp_pos_threshold,
                orientation_threshold=self.mp_ori_threshold,
                skill_type=skill_type)

        # Normalize observations to a list and record frames
        obs_list = observations if isinstance(observations, list) else ([observations] if observations is not None else [])
        for obs in obs_list:
            self._record_frame(obs)


        if obs_list:
            self._save_mp_images(obs_list[-1], skill_idx, skill)

        obs = self.env.env._get_observations()

        if not move_success:
            print(f"   1️⃣ ❌ Motion planning failed")
            return False, False, vla_language, 'mp', target_object

        print(f"   1️⃣ ✅ Motion planning success")

        # Visualization mode: skip VLA
        if self.visualize_poses:
            self._save_mp_images(obs, skill_idx, skill)
            return True, False, vla_language, None, target_object

        # Stage 3: VLA execution
        # # yy: debug
        # from scripts.phase3.pipeline.motion_planning.mplib.action_utils import get_controller_robot_pose
        # current_pos, current_rot_mat = get_controller_robot_pose(self.env, "right")
        # print("================================================")
        # from scipy.spatial.transform import Rotation as R
        # print('control coordinate system')
        # print(f"   📍 Current: pos={current_pos}")
        # print(f"   📍 Current: quat={R.from_matrix(current_rot_mat).as_quat()}") # -> xyzw (scipy format)
        # print(f"   📍 Current: axis={np.degrees(R.from_matrix(current_rot_mat).as_rotvec())}°")
        # position = obs['robot0_eef_pos'].copy()
        # quaternion = obs['robot0_eef_quat'].copy()  # [w,x,y,z] from LIBERO
        # print('libero coordinate system')
        # print(f"   📍 VLA: pos={position}")
        # print(f"   📍 VLA: quat={quaternion}")  # [w,x,y,z]
        # # Convert [w,x,y,z] to [x,y,z,w] for scipy
        # quat_xyzw = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
        # print(f"   📍 VLA: axis={np.degrees(R.from_quat(quat_xyzw).as_rotvec())}°")
        # exit(0)
        

        vla_success, task_done = self._execute_vla(skill, vla_language, obs)

        if not vla_success:
            print(f"   2️⃣ ❌ VLA execution failed")
            return False, task_done, vla_language, 'vla', target_object

        print(f"   2️⃣ ✅ VLA execution success")
        return True, task_done, vla_language, None, target_object

    def _get_grasped_object(self, current_skill_idx: int) -> Optional[str]:
        """Return mujoco name for previously picked object if available."""
        if not self.skill_sequence or current_skill_idx <= 1:
            return None

        if getattr(self, "_skill_metadata_cache", None) is None:
            config_path = Path("/mnt/arc/yygx/pkgs_baselines/openvla-oft/scripts/phase3/pipeline/config/tasks_and_skills.json")
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return None

            skill_mappings = config.get("skill_mappings", {})
            self._skill_metadata_cache = skill_mappings
            self._skill_metadata_lower = {name.lower(): meta for name, meta in skill_mappings.items()}

        for idx in range(current_skill_idx - 2, -1, -1):
            prev_skill = self.skill_sequence[idx]
            if "pick" not in prev_skill.lower():
                continue

            skill_meta = self._skill_metadata_cache.get(prev_skill)
            if skill_meta is None:
                skill_meta = self._skill_metadata_lower.get(prev_skill.lower())

            if not skill_meta:
                continue

            target_object = skill_meta.get("target_object")
            if target_object:
                return target_object

        return None

    def _execute_vla(self, skill: str, language: str, obs) -> tuple:
        """
        Execute VLA skill.

        Args:
            skill: Skill name with numbers (e.g., "pick black bowl 2") - for success checking
            language: Clean VLA language (e.g., "pick black bowl") - for VLA execution
            obs: Current observation

        Returns:
            (skill_success, overall_task_done)
        """
        from collections import deque
        from experiments.robot.robot_utils import get_action, normalize_gripper_action, invert_gripper_action, get_image_resize_size
        from experiments.robot.libero.libero_utils import quat2axisangle
        from experiments.robot.openvla_utils import resize_image_for_policy

        # Get fresh observation
        obs = self.env.env._get_observations()
        self._record_frame(obs)

        action_queue = deque(maxlen=self.cfg.num_open_loop_steps)
        resize_size = get_image_resize_size(self.cfg)
        overall_task_done = False

        for t in range(self.vla_horizon):
            if len(action_queue) == 0:
                wrist_img = self.get_libero_wrist_image(obs)
                wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
                # This is the correct one - use axis-angle representation -> axis for datasets/hdf5_datasets/libero_90_no_noops
                # observation = {
                #     "full_image": wrist_img_resized,
                #     "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
                # }

                # This is wrong - use euler representation, but in generated datasets, the euler representation is used for the state
                # obs["robot0_eef_quat"] is already [x,y,z,w] (xyzw - robosuite format), use directly
                quat_xyzw = obs["robot0_eef_quat"]
                observation = {
                    "full_image": wrist_img_resized,
                    "state": np.concatenate((obs["robot0_eef_pos"], R.from_quat(quat_xyzw).as_euler('xyz'), obs["robot0_gripper_qpos"]))
                }
                
                # Use clean language for VLA execution
                actions = get_action(
                    self.cfg,
                    self.vla,
                    observation,
                    language,  # ← Use clean language here
                    processor=self.processor,
                    action_head=self.action_head,
                    proprio_projector=self.proprio_projector,
                    noisy_action_projector=self.noisy_action_projector,
                    use_film=self.cfg.use_film
                )
                action_queue.extend(actions)

            action = action_queue.popleft()
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)

            obs, _, done, _ = self.env.step(action.tolist())
            self._record_frame(obs)

            if done:
                overall_task_done = True

            # Check if skill is complete using BDDL predicates
            # Use skill (with numbers) for success checking
            if self.skill_checker.check_skill_success_by_language(self.env, skill, debug=False):
                print(f"   ✅ VLA skill completed in {t+1} steps (goal achieved)")
                return True, overall_task_done

        print(f"   ❌ VLA skill failed after {self.vla_horizon} steps (goal not achieved)")
        
        # Print detailed diagnostic analysis if verbose flag is enabled
        if self.verbose_vla_failure:
            print(f"\n   📊 VLA Failure Analysis:")
            self.skill_checker.check_skill_success_by_language(self.env, skill, debug=True)
        
        return False, overall_task_done

    def _create_env(self) -> bool:
        """Create LIBERO environment."""
        try:
            from libero.libero import benchmark
            from libero.libero.envs import OffScreenRenderEnv

            bm = benchmark.get_benchmark_dict()["long_horizon_tasks_v0"]()
            task_names = bm.get_task_names()
            # Task names have format "LONG_HORIZON_task_name", so match against the suffix
            search_name = self.task_name.lower().replace(' ', '_')
            task_id = next((i for i, t in enumerate(task_names) if t.lower().endswith(search_name)), None)

            if task_id is None:
                print(f"Task '{self.task_name}' not found. Available: {bm.get_task_names()}")
                return False

            task = bm.get_task(task_id)

            from libero.libero import get_libero_path
            bddl_path = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)

            env_args = {
                "bddl_file_name": bddl_path,
                "camera_heights": 256, "camera_widths": 256,
                "has_renderer": False, "has_offscreen_renderer": True,
                "ignore_done": True, "use_camera_obs": True, "control_freq": 20,
                "camera_names": ["agentview", "robot0_eye_in_hand"]
            }

            env = OffScreenRenderEnv(**env_args)
            env.reset()

            # Set pipeline reference for motion planner to access recording functions
            env.pipeline = self

            self.env = env
            # Initialize motion planner based on type
            if self.motion_planner_type == "standard":
                self.motion_planner = StandardMotionPlanner(
                    env, method=self.motion_planner_method, num_steps=self.mp_steps,
                    pos_gain=self.mp_pos_gain, ori_gain=self.mp_ori_gain,
                    pos_success_threshold=self.mp_pos_success_threshold,
                    ori_success_threshold=self.mp_ori_success_threshold
                )
                print(f"🔧 Using Standard Motion Planner")
            elif self.motion_planner_type == "mplib":
                self.motion_planner = MPlibMotionPlanner(
                    env=self.env,
                    collision_aware=self.collision_aware,
                    use_agentview_only=True,
                    velocity_factor=self.mp_velocity_factor,
                    verbose=True,
                    safety_margin=self.mplib_safety_margin,
                    erode_kernel_size=self.mplib_erode_kernel_size,
                    outlier_std_ratio=self.mplib_outlier_std_ratio
                )
                print(f"🔧 Using MPlib Motion Planner (RRT-based path planning)")
            else:
                raise ValueError(f"Unknown motion planner type: {self.motion_planner_type}. Use 'standard' or 'mplib'.")

            return True
        except Exception as e:
            print(f"Failed to create env: {e}")
            import traceback
            traceback.print_exc()
            return False

    def execute(self) -> bool:
        """Execute pipeline.

        Returns:
            bool: True if overall task succeeded

        Note: self.success_count tracks number of successfully completed skills (for eval purposes)
        """
        # Initialize tracking attributes early (for eval script)
        self.success_count = 0
        self.total_skills = 0
        self.recovery_stats = None  # Will store detailed skill-level statistics

        # Reset video recording for each trial (important for multi-trial eval)
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self._setup_recording()

        if not self._create_env():
            print("❌ Failed to create environment")
            return False

        print("✅ Environment created")

        # Stabilize and start video
        for i in range(5):
            obs, _, _, _ = self.env.step(np.array([0, 0, 0, 0, 0, 0, -1]))
            if i == 0:
                self._start_video(obs)
            self._record_frame(obs)

        # Get skills
        skills = self.task_planner.plan_task_sequence(self.task_name)
        if not skills:
            return False

        # Update total_skills now that we have the skill list
        self.skill_sequence = skills
        self.total_skills = len(skills)

        overall_task_done = False
        previous_skill = None  # Track for place recovery
        previous_skill_idx = None

        # Recovery statistics tracking
        recovery_stats = {
            'skills': [],  # Per-skill recovery info
            'final_failure': None  # What caused the final failure
        }

        for i, skill in enumerate(skills, 1):
            print(f"\n0️⃣ Skill {i}/{len(skills)}: {skill}")

            # Initialize skill recovery stats
            skill_stats = {
                'skill_name': skill,
                'skill_idx': i,
                'skill_type': None,
                'success': False,
                'pose_retries': 0,
                'mp_retries': 0,
                'vla_retries': 0,
                'pick_recovery_attempts': 0,
                'pick_recovery_success': False
            }

            # Determine skill type
            skill_lower = skill.lower()
            if 'pick' in skill_lower:
                skill_type = 'pick'
            elif 'place' in skill_lower:
                skill_type = 'place'
            elif 'open' in skill_lower or 'close' in skill_lower or 'turn' in skill_lower:
                skill_type = 'other'
            else:
                skill_type = 'pick'  # Default

            skill_stats['skill_type'] = skill_type

            # Place skills: double retries with pick recovery on VLA failure
            if skill_type == 'place':
                max_retries = self.max_vla_retries * 1
                vla_success = False
                vla_language = None
                retry_count = 0

                while retry_count < max_retries and not vla_success:
                    if retry_count > 0:
                        print(f"   🔄 Place retry {retry_count + 1}/{max_retries}")

                    # Execute place skill
                    success, task_done, vla_lang, failure_stage, target_obj = self._execute_skill_once(
                        skill, i, skill_type)

                    if task_done:
                        overall_task_done = True

                    if success:
                        vla_success = True
                        vla_language = vla_lang
                        print(f"   ✅ Place succeeded on attempt {retry_count + 1}/{max_retries}")
                        break

                    # Track failure stage
                    if failure_stage == 'pose':
                        skill_stats['pose_retries'] += 1
                    elif failure_stage == 'mp':
                        skill_stats['mp_retries'] += 1
                    elif failure_stage == 'vla':
                        skill_stats['vla_retries'] += 1

                    # Handle failure
                    if failure_stage == 'vla' and previous_skill and 'pick' in previous_skill.lower():
                        # VLA failure: recover by re-executing pick
                        print(f"   ⚠️  VLA failed - object may be dropped")
                        print(f"   🔄 Recovering via pick: {previous_skill}")
                        skill_stats['pick_recovery_attempts'] += 1

                        # Open gripper first
                        for _ in range(5):
                            obs, _, _, _ = self.env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
                            self._record_frame(obs)

                        # Re-execute pick with retry loop
                        pick_success = False
                        pick_retry_count = 0
                        while pick_retry_count < self.max_vla_retries and not pick_success:
                            if pick_retry_count > 0:
                                print(f"   🔄 Pick recovery retry {pick_retry_count + 1}/{self.max_vla_retries}")

                            pick_success, _, pick_lang, pick_failure_stage, pick_target_obj = self._execute_skill_once(
                                previous_skill, previous_skill_idx, 'pick')

                            if pick_success:
                                # # Pick post-actions (close gripper + move up)
                                post_actions(pick_lang, self.env, self.motion_planner,
                                           skill_succeeded=True, next_skill=None, grasped_object=pick_target_obj)
                                print(f"   ✅ Pick recovery complete on attempt {pick_retry_count + 1}/{self.max_vla_retries}")
                                skill_stats['pick_recovery_success'] = True
                                break
                            else:
                                pick_retry_count += 1

                        if not pick_success:
                            print(f"   ❌ Pick recovery failed after {self.max_vla_retries} attempts")
                    elif failure_stage in ['pose', 'mp']:
                        # Pose/MP failure: just retry, no pick recovery needed
                        print(f"   ⚠️  {failure_stage.upper()} failed - retrying")

                    retry_count += 1

            else:
                # Non-place skills: regular retry loop
                vla_success = False
                vla_language = None
                retry_count = 0

                while retry_count < self.max_vla_retries and not vla_success:
                    if retry_count > 0:
                        print(f"   🔄 Retry {retry_count + 1}/{self.max_vla_retries}")

                    # Execute skill
                    success, task_done, vla_lang, failure_stage, target_obj = self._execute_skill_once(
                        skill, i, skill_type)

                    if task_done:
                        overall_task_done = True

                    if success:
                        vla_success = True
                        vla_language = vla_lang
                        print(f"   ✅ Succeeded on attempt {retry_count + 1}/{self.max_vla_retries}")
                    else:
                        # Track failure stage
                        if failure_stage == 'pose':
                            skill_stats['pose_retries'] += 1
                        elif failure_stage == 'mp':
                            skill_stats['mp_retries'] += 1
                        elif failure_stage == 'vla':
                            skill_stats['vla_retries'] += 1

                        print(f"   ⚠️  {failure_stage.upper()} failed on attempt {retry_count + 1}/{self.max_vla_retries}")
                        retry_count += 1

            # Post-actions ONLY if skill succeeded
            if vla_success and vla_language:
                next_skill = skills[i] if i < len(skills) else None
                # For place skills, get grasped_object from previous pick skill
                grasped_obj_for_post = self._get_grasped_object(i) if skill_type == 'place' else target_obj
                post_actions(vla_language, self.env, self.motion_planner,
                           skill_succeeded=True, next_skill=next_skill, grasped_object=grasped_obj_for_post)
                self.success_count += 1
                skill_stats['success'] = True
                # Track for place recovery
                previous_skill = skill
                previous_skill_idx = i
            else:
                max_r = self.max_vla_retries * 2 if skill_type == 'place' else self.max_vla_retries
                print(f"   ⚠️  Skipping post_actions (failed after {max_r} attempts)")

                # Record final failure cause
                if skill_stats['vla_retries'] > 0:
                    final_stage = 'vla'
                elif skill_stats['mp_retries'] > 0:
                    final_stage = 'mp'
                elif skill_stats['pose_retries'] > 0:
                    final_stage = 'pose'
                else:
                    final_stage = 'unknown'

                recovery_stats['final_failure'] = {
                    'skill': skill,
                    'skill_idx': i,
                    'skill_type': skill_type,
                    'stage': final_stage,
                    'attempted_recoveries': {
                        'pose': skill_stats['pose_retries'],
                        'mp': skill_stats['mp_retries'],
                        'vla': skill_stats['vla_retries'],
                        'pick_recovery': skill_stats['pick_recovery_attempts']
                    }
                }

                # Save skill stats before exiting
                recovery_stats['skills'].append(skill_stats)

                if not self.continue_on_failure:
                    self._print_recovery_summary(recovery_stats, self.success_count, self.total_skills)
                    return False

            # Save skill stats
            recovery_stats['skills'].append(skill_stats)

        print(f"\n{'='*60}")
        print(f"3️⃣ Overall Long-Horizon Task: {'✅ SUCCESS' if overall_task_done else '❌ FAILED'}")
        print(f"{'='*60}")

        # Print recovery statistics summary
        self._print_recovery_summary(recovery_stats, self.success_count, self.total_skills)

        self._stop_video()

        if self.env:
            self.env.close()

        # Store recovery stats for eval script access
        self.recovery_stats = recovery_stats

        return overall_task_done


def main():
    parser = argparse.ArgumentParser(description='Execute long horizon pipeline')
    parser.add_argument('--task_name', type=str, default='Complete Kitchen Organization',
                       choices=['Cooking Preparation Setup', 'Complete Kitchen Organization',
                               'Switch Table Objects', 'Pick White Bowl', 'Pick Black Bowl',
                               'Put The Black Bowl In The Bottom Drawer Of The Cabinet And Close It',
                               'Turn On The Stove And Put The Moka Pot On It'])
    parser.add_argument('--vla_checkpoint', type=str,
                       default="/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/libero_atomic_skills_augmented_closer/1.0.0/openvla-7b+libero_atomic_skills_augmented_closer+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_closer--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt")
    parser.add_argument('--wrist_only', action='store_true', default=True)
    parser.add_argument('--visualize_poses', action='store_true')
    parser.add_argument('--ee_backward_offset', type=float, default=0.0)
    parser.add_argument('--motion_planner_method', type=str, default='cartesian_linear',
                       choices=['ik_setjoint', 'cartesian_linear'])
    parser.add_argument('--motion_planner_type', type=str, default='standard', choices=['standard', 'mplib'],
                       help='Motion planner type: standard (default) or mplib (RRT-based path planning)')
    parser.add_argument('--continue_on_failure', action='store_true', default=False)
    parser.add_argument('--motion_planner_steps', type=int, default=400)
    parser.add_argument('--motion_planner_pos_gain', type=float, default=5.0)
    parser.add_argument('--motion_planner_ori_gain', type=float, default=5.0)
    parser.add_argument('--motion_planner_pos_threshold', type=float, default=0.02,
                       help='Position threshold for early convergence during motion planning (default: 0.02m = 2cm)')
    parser.add_argument('--motion_planner_ori_threshold', type=float, default=0.524,
                       help='Orientation threshold for early convergence during motion planning (default: 0.524 rad ≈ 30°)')
    parser.add_argument('--motion_planner_pos_success_threshold', type=float, default=0.025,
                       help='Looser position threshold for final success check after all steps (default: 0.025m = 2.5cm)')
    parser.add_argument('--motion_planner_ori_success_threshold', type=float, default=0.524,
                       help='Looser orientation threshold for final success check after all steps (default: 0.524 rad ≈ 30°)')
    parser.add_argument('--motion_planner_velocity_factor', type=float, default=0.9,
                       help='Velocity factor to scale controller limits (0.0-1.0, default: 0.9)')

    # MPlib-specific parameters
    parser.add_argument('--mplib_collision_aware', type=lambda x: x.lower() == 'true', default=True,
                       help='Enable MPlib collision avoidance planning (default: True)')
    parser.add_argument('--mplib_joint_vel_limits', type=float, default=2.0,
                       help='MPlib joint velocity limits in rad/s (lower = smoother/more accurate, default: 2.0)')
    parser.add_argument('--mplib_joint_acc_limits', type=float, default=4.0,
                       help='MPlib joint acceleration limits in rad/s^2 (lower = smoother, default: 4.0)')
    parser.add_argument('--mplib_waypoint_target_ratio', type=int, default=50,
                       help='MPlib target number of waypoints (higher = more accurate but slower, default: 50)')
    parser.add_argument('--mplib_planning_time', type=float, default=1.0,
                       help='MPlib planning time budget in seconds (higher = better paths, default: 1.0)')
    parser.add_argument('--mplib_safety_margin', type=float, default=1.1,
                       help='MPlib AABB safety margin multiplier (default: 1.1 = 10% margin)')
    parser.add_argument('--mplib_erode_kernel_size', type=int, default=3,
                       help='MPlib mask erosion kernel size for boundary pixel removal (default: 3)')
    parser.add_argument('--mplib_outlier_std_ratio', type=float, default=0.0,
                       help='MPlib 3D outlier removal std ratio (default: 0.0 = disabled, try 2.0 for aggressive filtering)')

    parser.add_argument('--obj_ee_pairs_dir', type=str,
                       default="datasets/hdf5_datasets/atomic_local_demos_augmented_pick_10_place_15_atomic_15/ee_target_obj_pairs/simple_extraction_obs_pose_v1",
                       help='Directory containing object-EE pose pairs (default: datasets/hdf5_datasets/atomic_local_demos_augmented_pick_10_place_15_atomic_15/ee_target_obj_pairs/simple_extraction_obs_pose)')
    parser.add_argument('--max_vla_retries', type=int, default=6,
                       help='Maximum number of retry attempts for VLA execution (default: 6)')
    parser.add_argument('--vla_horizon', type=int, default=200,
                       help='Maximum steps per skill VLA execution before considered failure (default: 200)')
    parser.add_argument('--verbose_vla_failure', action='store_true', default=False,
                       help='Print detailed diagnostic analysis when VLA fails (default: False)')

    args = parser.parse_args()

    pipeline = LongHorizonPipeline(
        vla_checkpoint=args.vla_checkpoint,
        task_name=args.task_name,
        wrist_only=args.wrist_only,
        visualize_poses=args.visualize_poses,
        ee_offset=args.ee_backward_offset,
        motion_planner_method=args.motion_planner_method,
        motion_planner_type=args.motion_planner_type,
        continue_on_failure=args.continue_on_failure,
        mp_steps=args.motion_planner_steps,
        mp_pos_gain=args.motion_planner_pos_gain,
        mp_ori_gain=args.motion_planner_ori_gain,
        mp_pos_threshold=args.motion_planner_pos_threshold,
        mp_ori_threshold=args.motion_planner_ori_threshold,
        mp_pos_success_threshold=args.motion_planner_pos_success_threshold,
        mp_ori_success_threshold=args.motion_planner_ori_success_threshold,
        mp_velocity_factor=args.motion_planner_velocity_factor,
        obj_ee_pairs_dir=args.obj_ee_pairs_dir,
        max_vla_retries=args.max_vla_retries,
        collision_aware=args.mplib_collision_aware,
        mplib_joint_vel_limits=args.mplib_joint_vel_limits,
        mplib_joint_acc_limits=args.mplib_joint_acc_limits,
        mplib_waypoint_target_ratio=args.mplib_waypoint_target_ratio,
        mplib_planning_time=args.mplib_planning_time,
        mplib_safety_margin=args.mplib_safety_margin,
        mplib_erode_kernel_size=args.mplib_erode_kernel_size,
        mplib_outlier_std_ratio=args.mplib_outlier_std_ratio,
        vla_horizon=args.vla_horizon,
        verbose_vla_failure=args.verbose_vla_failure
    )

    start = time.time()
    success = pipeline.execute()
    elapsed = time.time() - start

    print(f"\n⏱️  Total: {elapsed:.1f}s")
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())