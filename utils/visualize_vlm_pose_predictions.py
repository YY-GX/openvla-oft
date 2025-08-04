import os
import json
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import argparse
import random

# VLM and dataset imports (EXACTLY as in evaluate_pose_vlm_skill_aware.py)
from prismatic.vla.datasets.pose_dataset import create_pose_dataset
from prismatic.util.data_utils import PaddedCollatorForPosePrediction
from prismatic.models.vlms.pose_vlm import PoseVLM
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from transformers import AutoModelForVision2Seq

# Import the config class to fix torch.load bug
import sys
sys.path.append('vla-scripts')
from finetune_pose import PoseFinetuneConfig

# IK tools
from utils.tracik_tools import solve_ik, pose6d_to_matrix
from tracikpy import TracIKSolver

# LIBERO env creation (as in run_libero_eval.py)
from libero.libero import benchmark
import cv2
import pandas as pd
from experiments.robot.libero.libero_utils import get_libero_env

def save_env_image(env, save_path):
    # Try to use the public get_observation method if available
    if hasattr(env, 'get_observation'):
        obs = env.get_observation()
    else:
        obs = env.env._get_observations()
    img = obs["agentview_image"][::-1, ::-1]  # Rotate 180 degrees as in get_libero_image

    # Debug: print min, max, dtype, shape
    print(f"[DEBUG] Saving image: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")

    # If grayscale, convert to 3-channel
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    # Ensure image is uint8 for OpenCV
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    import cv2
    cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize VLM-predicted poses using IK and LIBERO env.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to VLM checkpoint')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of dataset')
    parser.add_argument('--split', type=str, default='val', help='Dataset split (train/val/test)')
    parser.add_argument('--save_dir', type=str, default='runs/eval_results/skill_pose_diversity', help='Directory to save images')
    parser.add_argument('--splits_folder', type=str, default='smaller_splits', help='Folder name for annotation splits (default: smaller_splits)')
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    data_root = args.data_root
    split = args.split
    save_dir = Path(args.save_dir)
    splits_folder = args.splits_folder
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load VLM model (EXACTLY as in evaluate_pose_vlm_skill_aware.py)
    print("Loading VLM model...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", None)
    if config is None:
        class SimpleConfig:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        config = SimpleConfig(
            pose_head_type="gmm",
            gmm_num_components=3,
            pose_dim=6,
            num_pose_tokens=6,
            num_images_in_input=1,
            max_length=512,
        )
    processor = PrismaticProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    base_vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    class OpenVLAWrapper(torch.nn.Module):
        def __init__(self, language_model, processor):
            super().__init__()
            self.llm = language_model
            self.tokenizer = processor.tokenizer
            self.config = language_model.config
            self.embed_dim = language_model.config.hidden_size
        def forward(self, **kwargs):
            return self.llm(**kwargs)
        def get_tokenizer(self):
            return self.tokenizer
    llm_backbone = OpenVLAWrapper(base_vla.language_model, processor)
    model = PoseVLM(
        model_id=f"pose_vlm_{config.pose_head_type}",
        vision_backbone=base_vla.vision_backbone,
        llm_backbone=llm_backbone,
        pose_head_type=config.pose_head_type,
        pose_dim=config.pose_dim,
        num_pose_tokens=config.num_pose_tokens,
        gmm_num_components=config.gmm_num_components,
        enable_mixed_precision_training=True,
    )
    state_dict = checkpoint["model"]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[7:]
        else:
            new_key = key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("Model loaded.")

    # 2. Load dataset and sample indices
    print("Loading dataset and sample indices...")
    val_dataset = create_pose_dataset(
        data_root=data_root,
        split=split,
        num_images_in_input=config.num_images_in_input,
        use_image_augmentation=False,
        tokenizer_name=processor.tokenizer.name_or_path,
        splits_folder=splits_folder,
        use_pose_normalization=True,  # Enable pose normalization for consistency
    )
    # 3. Load annotation for mapping to skill/task
    annotation_path = os.path.join(data_root, f"{split}_annotation.csv")
    if not os.path.exists(annotation_path):
        annotation_path = os.path.join(data_root, splits_folder, f"{split}_annotation.csv")
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found in {data_root} or {data_root}/{splits_folder}")
    annotation_df = pd.read_csv(annotation_path)
    poses = np.load(os.path.join(data_root, "local_poses.npy"))

    # Define the manual offset to apply to all poses (GT and pred)
    pose_offset = np.array([0.667, 0.0, -0.81, 0.0, 0.0, 0.0])

    # 4. Prepare LIBERO benchmark
    benchmark_dict = benchmark.get_benchmark_dict()

    # 5. Process each skill: predict pose, run IK, set joints, render
    print("Processing each skill to get one predicted pose...")
    
    # Get all unique skills
    unique_skills = annotation_df['task_name'].unique()
    print(f"Found {len(unique_skills)} unique skills:")
    for i, skill in enumerate(unique_skills):
        print(f"  {i+1}. {skill}")
    
    successful_skills = []
    failed_skills = []
    
    for skill_idx, skill_name in enumerate(unique_skills):
        print(f"\n{'='*20} Processing Skill {skill_idx+1}/{len(unique_skills)}: {skill_name} {'='*20}")
        
        # Create subfolder for this skill
        skill_name_for_file = skill_name.replace(" ", "_")
        skill_save_dir = save_dir / skill_name_for_file
        skill_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all samples for this skill
        skill_indices = [idx for idx, row in annotation_df.iterrows() if row['task_name'] == skill_name]
        if not skill_indices:
            print(f"[Warning] No samples found for skill {skill_name}")
            failed_skills.append(skill_name)
            continue
        
        # Try up to 10 different samples for this skill
        success = False
        for attempt in range(min(10, len(skill_indices))):
            idx = skill_indices[attempt]
            print(f"  Attempt {attempt + 1}/10: using sample idx={idx}")
            
            sample = val_dataset[idx]
            try:
                ann_row = annotation_df.iloc[idx]
            except Exception as e:
                print(f"    [Warning] Could not get annotation row for idx {idx}: {e}")
                continue
            
            # Prepare input batch for VLM
            batch = {}
            for k in ["pixel_values", "input_ids", "attention_mask"]:
                v = sample[k]
                if isinstance(v, torch.Tensor):
                    batch[k] = v.unsqueeze(0).to(device)
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.from_numpy(v).unsqueeze(0).to(device)
                elif np.isscalar(v):
                    batch[k] = torch.tensor([[v]], device=device)
                else:
                    batch[k] = torch.tensor(v, device=device).unsqueeze(0)

            # Predict pose
            with torch.no_grad():
                predictions = model.predict_pose(
                    images=batch["pixel_values"],
                    text=batch["input_ids"],
                    text_attention_mask=batch["attention_mask"],
                    num_samples=1,
                )
                if config.pose_head_type == "gmm":
                    sampled_poses = predictions['sampled_poses'].squeeze(1)
                    weights = predictions['weights']
                    first_token_weights = weights[:, 0, :]
                    first_token_poses = sampled_poses[:, 0, :, :]
                    best_component_indices = torch.argmax(first_token_weights, dim=-1)
                    pred_pose_tensor = first_token_poses[0, best_component_indices[0]]
                else:
                    pred_pose_tensor = predictions['predicted_poses'][0]
                # Fix: convert bfloat16 to float32 before .cpu().numpy()
                if pred_pose_tensor.dtype == torch.bfloat16:
                    pred_pose = pred_pose_tensor.to(torch.float32).cpu().numpy()
                else:
                    pred_pose = pred_pose_tensor.cpu().numpy()
            
            # Apply offset to predicted pose
            pred_pose_offset = pred_pose + pose_offset
            
            # Convert to 4x4 matrix
            pred_pose_mat = pose6d_to_matrix(pred_pose_offset)
            
            # Run IK
            pred_joints = solve_ik(pred_pose_mat)
            
            if pred_joints is not None:
                print(f"    ✓ IK succeeded on attempt {attempt + 1}")
                
                # Create env for this skill/task
                task_suite = benchmark_dict["libero_90"]()
                # Find the task id by name
                task_id = None
                for i in range(task_suite.n_tasks):
                    if task_suite.get_task(i).name == skill_name:
                        task_id = i
                        break
                
                if task_id is None:
                    print(f"    [Warning] Task {skill_name} not found in libero_90 suite.")
                    continue
                
                task = task_suite.get_task(task_id)
                env, _ = get_libero_env(task, model_family="openvla", resolution=256)
                
                # Set pred joints, render, save
                env.env.robots[0].set_robot_joint_positions(pred_joints)
                env.env.sim.forward()  # Ensure simulation state is updated
                if hasattr(env.env, '_update_observables'):
                    env.env._update_observables(force=True)
                
                # Save image
                demo_num = ann_row["source_demo_idx"] if "source_demo_idx" in ann_row else str(idx)
                pred_img_name = f"pred_{skill_name_for_file}_demo{demo_num}_attempt{attempt+1}.png"
                save_env_image(env, skill_save_dir / pred_img_name)
                
                env.close()
                successful_skills.append(skill_name)
                success = True
                break
            else:
                print(f"    ✗ IK failed on attempt {attempt + 1}")
        
        if not success:
            print(f"  ✗ Failed to get IK solution for skill {skill_name} after 10 attempts")
            failed_skills.append(skill_name)
    
    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total skills: {len(unique_skills)}")
    print(f"Successful: {len(successful_skills)}")
    print(f"Failed: {len(failed_skills)}")
    
    if successful_skills:
        print("\nSuccessful skills:")
        for skill in successful_skills:
            print(f"  ✓ {skill}")
    
    if failed_skills:
        print("\nFailed skills:")
        for skill in failed_skills:
            print(f"  ✗ {skill}")
    # Save stats
    stats = {
        "total_skills": len(unique_skills),
        "successful_skills": len(successful_skills),
        "failed_skills": len(failed_skills),
        "successful_skill_names": successful_skills,
        "failed_skill_names": failed_skills,
    }
    stats_path = save_dir / f"{split}_vlm_prediction_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"VLM prediction visualization complete. Images saved to {save_dir}")
    print(f"Stats saved to {stats_path}") 