#!/usr/bin/env python3
"""
train_simple_resnet_pose.py

Simple ResNet-based pose prediction for debugging.
Takes images + one-hot skill encoding → 6D pose output.
"""

import os
import argparse
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import wandb

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class SimplePoseDataset(Dataset):
    """Simple dataset for ResNet pose prediction with one-hot skill encoding."""
    
    def __init__(self, data_root, split, skill_vocab, image_size=224):
        self.data_root = data_root
        self.split = split
        self.skill_vocab = skill_vocab
        self.image_size = image_size
        
        # Load annotation
        annotation_path = os.path.join(data_root, f"{split}_annotation.csv")
        self.annotation_df = pd.read_csv(annotation_path)
        
        # Load poses (in parent directory)
        poses_path = os.path.join(os.path.dirname(data_root), "local_poses.npy")
        self.poses = np.load(poses_path)
        
        # Set up image directory (in parent directory)
        self.image_dir = os.path.join(os.path.dirname(data_root), "3rd_imgs")
        
        # Set up transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded {len(self.annotation_df)} samples for {split} split")
        print(f"Skill vocabulary size: {len(skill_vocab)}")

        # Build skill to pose indices mapping (for skill-aware metric)
        self.skill_to_pose_indices = {}
        for _, row in self.annotation_df.iterrows():
            skill = row['language_description']
            idx = row['ee_pose_idx']
            if skill not in self.skill_to_pose_indices:
                self.skill_to_pose_indices[skill] = set()
            self.skill_to_pose_indices[skill].add(idx)
        # Convert sets to sorted lists for consistency
        for k in self.skill_to_pose_indices:
            self.skill_to_pose_indices[k] = sorted(list(self.skill_to_pose_indices[k]))
    
    def __len__(self):
        return len(self.annotation_df)
    
    def __getitem__(self, idx):
        sample = self.annotation_df.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, f"{sample['overview_image_idx']:05d}.jpg")
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Get skill encoding
        skill = sample['language_description']
        skill_idx = self.skill_vocab[skill]
        skill_encoding = torch.zeros(len(self.skill_vocab))
        skill_encoding[skill_idx] = 1.0
        
        # Get pose target
        pose_target = torch.tensor(self.poses[sample['ee_pose_idx']], dtype=torch.float32)

        # Get all valid poses for this skill (for skill-aware metric)
        valid_pose_indices = self.skill_to_pose_indices[skill]
        valid_poses = self.poses[valid_pose_indices]
        valid_poses = torch.tensor(valid_poses, dtype=torch.float32)  # (N, 6)
        
        return {
            'image': image,
            'skill_encoding': skill_encoding,
            'pose_target': pose_target,
            'skill': skill,
            'valid_poses': valid_poses,  # (N, 6)
        }


class SimpleResNetPose(nn.Module):
    """Simple ResNet-based pose prediction model."""
    
    def __init__(self, num_skills, pose_dim=6, image_size=224):
        super().__init__()
        
        # Image encoder (ResNet-18, freeze early layers)
        resnet = models.resnet18(pretrained=True)
        # Remove the final classification layer
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Unfreeze ALL layers for maximum overfitting
        for param in self.image_encoder.parameters():
            param.requires_grad = True
        
        # Skill encoder
        self.skill_encoder = nn.Linear(num_skills, 64)
        
        # Fusion and output - massive network for extreme overfitting
        image_features = 512  # ResNet-18 output
        skill_features = 64
        
        self.fusion = nn.Sequential(
            nn.Linear(image_features + skill_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, pose_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, image, skill_encoding):
        # Encode image
        image_features = self.image_encoder(image).squeeze(-1).squeeze(-1)  # (B, 512)
        
        # Encode skill
        skill_features = self.skill_encoder(skill_encoding)  # (B, 64)
        
        # Concatenate and predict pose
        combined = torch.cat([image_features, skill_features], dim=1)  # (B, 576)
        pose = self.fusion(combined)  # (B, 6)
        
        return pose


def create_skill_vocab(data_root):
    """Create skill vocabulary from all splits."""
    splits = ['train', 'val', 'test']
    all_skills = set()
    
    for split in splits:
        annotation_path = os.path.join(data_root, f"{split}_annotation.csv")
        if os.path.exists(annotation_path):
            df = pd.read_csv(annotation_path)
            all_skills.update(df['language_description'].unique())
    
    skill_vocab = {skill: idx for idx, skill in enumerate(sorted(all_skills))}
    return skill_vocab


def compute_skill_aware_min_distance(pred_pose, valid_poses):
    """Compute minimum L2 distance between pred_pose and all valid_poses (N, 6)."""
    # pred_pose: (6,), valid_poses: (N, 6)
    dists = torch.norm(pred_pose.unsqueeze(0) - valid_poses, dim=1)  # (N,)
    return dists.min().item()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    skill_aware_errors = []
    
    for batch in tqdm(dataloader, desc="Training"):
        images = batch['image'].to(device)
        skill_encodings = batch['skill_encoding'].to(device)
        pose_targets = batch['pose_target'].to(device)
        valid_poses_batch = batch['valid_poses']  # list of tensors (batch_size, N, 6)
        
        # Forward pass
        pose_pred = model(images, skill_encodings)  # (B, 6)
        loss = criterion(pose_pred, pose_targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1

        # Skill-aware min distance metric (L2)
        for i in range(pose_pred.shape[0]):
            pred = pose_pred[i].detach().cpu()
            valid_poses = valid_poses_batch[i].cpu()
            skill_aware_errors.append(compute_skill_aware_min_distance(pred, valid_poses))
    
    mean_skill_aware_error = np.mean(skill_aware_errors)
    return total_loss / num_batches, mean_skill_aware_error


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    skill_aware_errors = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['image'].to(device)
            skill_encodings = batch['skill_encoding'].to(device)
            pose_targets = batch['pose_target'].to(device)
            valid_poses_batch = batch['valid_poses']
            
            # Forward pass
            pose_pred = model(images, skill_encodings)
            loss = criterion(pose_pred, pose_targets)
            
            total_loss += loss.item()
            num_batches += 1

            # Skill-aware min distance metric (L2)
            for i in range(pose_pred.shape[0]):
                pred = pose_pred[i].cpu()
                valid_poses = valid_poses_batch[i].cpu()
                skill_aware_errors.append(compute_skill_aware_min_distance(pred, valid_poses))
    mean_skill_aware_error = np.mean(skill_aware_errors)
    return total_loss / num_batches, mean_skill_aware_error


def main():
    parser = argparse.ArgumentParser(description="Train simple ResNet pose prediction")
    parser.add_argument("--data_root", type=str, default="datasets/local_pairs_datasets/debug_splits")
    parser.add_argument("--batch_size", type=int, default=8)  # Smaller batch for better gradient updates
    parser.add_argument("--learning_rate", type=float, default=5e-4)  # Higher learning rate
    parser.add_argument("--num_epochs", type=int, default=200)  # More epochs
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb_project", type=str, default="simple-resnet-pose")
    parser.add_argument("--wandb_entity", type=str, default="yygx")
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args)
    )
    
    # Create skill vocabulary
    skill_vocab = create_skill_vocab(args.data_root)
    print(f"Created skill vocabulary with {len(skill_vocab)} skills")
    
    # Create datasets
    train_dataset = SimplePoseDataset(args.data_root, "train", skill_vocab)
    val_dataset = SimplePoseDataset(args.data_root, "val", skill_vocab)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: {
        'image': torch.stack([item['image'] for item in x]),
        'skill_encoding': torch.stack([item['skill_encoding'] for item in x]),
        'pose_target': torch.stack([item['pose_target'] for item in x]),
        'skill': [item['skill'] for item in x],
        'valid_poses': [item['valid_poses'] for item in x],
    })
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=lambda x: {
        'image': torch.stack([item['image'] for item in x]),
        'skill_encoding': torch.stack([item['skill_encoding'] for item in x]),
        'pose_target': torch.stack([item['pose_target'] for item in x]),
        'skill': [item['skill'] for item in x],
        'valid_poses': [item['valid_poses'] for item in x],
    })
    
    # Create model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = SimpleResNetPose(num_skills=len(skill_vocab)).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training - aggressive overfitting settings
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)  # No weight decay
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)  # Commented out for overfitting
    
    # Training loop
    best_val_loss = float('inf')
    best_val_skill_aware = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss, train_skill_aware = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_skill_aware = validate(model, val_loader, criterion, device)
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_skill_aware_min_l2': train_skill_aware,
            'val_skill_aware_min_l2': val_skill_aware,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print(f"Train Skill-Aware Min L2: {train_skill_aware:.6f}, Val Skill-Aware Min L2: {val_skill_aware:.6f}")
        
        # Save best model (by skill-aware metric)
        if val_skill_aware < best_val_skill_aware:
            best_val_skill_aware = val_skill_aware
            torch.save(model.state_dict(), 'best_simple_resnet_pose.pt')
            print(f"Saved best model with val skill-aware min L2: {val_skill_aware:.6f}")
    
    print(f"Training completed! Best val skill-aware min L2: {best_val_skill_aware:.6f}")


if __name__ == "__main__":
    main() 