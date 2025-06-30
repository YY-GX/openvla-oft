"""
pose_heads.py

Implementations of pose prediction heads, following original action head patterns.
Supports GMM (Gaussian Mixture Model) for multi-modal pose distributions.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from prismatic.util import ensure_bfloat16

from prismatic.models.action_heads import MLPResNet


class GMMPoseHead(nn.Module):
    """
    Gaussian Mixture Model pose head for predicting multi-modal pose distributions.
    Tokenwise: applies a shared MLP to each pose token's hidden state.
    """
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 4096,
        pose_dim: int = 6,  # 3D position + 3D orientation
        num_components: int = 3,
        min_std: float = 0.01,
        max_std: float = 1.0,
    ):
        """
        Initialize GMM pose head.
        
        Args:
            input_dim: Input dimension from LLM hidden states
            hidden_dim: Hidden dimension for MLP
            pose_dim: Dimension of pose (6 for 3D pos + 3D ori)
            num_components: Number of GMM components
            min_std: Minimum standard deviation for numerical stability
            max_std: Maximum standard deviation
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pose_dim = pose_dim
        self.num_components = num_components
        self.min_std = min_std
        self.max_std = max_std
        
        # Calculate output dimensions for GMM parameters
        # For each component: means (pose_dim) + diagonal covariances (pose_dim) + weight (1)
        self.gmm_output_dim = num_components * (pose_dim + pose_dim + 1)
        
        # Shared MLP for all pose tokens
        self.mlp = MLPResNet(
            num_blocks=2,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.gmm_output_dim
        )
    
    def forward(self, pose_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pose_hidden_states: (batch_size, num_pose_tokens, hidden_dim)
        Returns:
            means: (batch_size, num_pose_tokens, num_components, pose_dim)
            covariances: (batch_size, num_pose_tokens, num_components, pose_dim, pose_dim)
            weights: (batch_size, num_pose_tokens, num_components)
        """
        # Ensure input is in bfloat16
        pose_hidden_states = ensure_bfloat16(pose_hidden_states)
        
        batch_size, num_pose_tokens, hidden_dim = pose_hidden_states.shape
        # Apply shared MLP to each token
        gmm_params = self.mlp(pose_hidden_states)  # (batch, num_pose_tokens, gmm_output_dim)
        # Split GMM params for each token
        gmm_params = gmm_params.view(batch_size, num_pose_tokens, self.num_components, -1)
        means = ensure_bfloat16(gmm_params[..., :self.pose_dim])  # (batch, num_pose_tokens, num_components, pose_dim)
        cov_params = ensure_bfloat16(gmm_params[..., self.pose_dim:2*self.pose_dim])  # (batch, num_pose_tokens, num_components, pose_dim)
        # Diagonal covariance matrices
        covariances = torch.zeros(batch_size, num_pose_tokens, self.num_components, self.pose_dim, self.pose_dim, device=gmm_params.device, dtype=torch.bfloat16)
        diag_indices = torch.arange(self.pose_dim)
        covariances[..., diag_indices, diag_indices] = torch.exp(cov_params)
        weights = ensure_bfloat16(gmm_params[..., -1])  # (batch, num_pose_tokens, num_components)
        weights = torch.softmax(weights, dim=-1)
        return means, covariances, weights
    
    def predict_pose_distribution(self, pose_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict pose distribution (alias for forward).
        
        Args:
            pose_hidden_states: Hidden states from LLM for pose tokens
        
        Returns:
            Tuple of (means, covariances, weights)
        """
        return self.forward(pose_hidden_states)
    
    def sample_poses(self, pose_hidden_states: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        Sample poses from the predicted GMM distribution.
        
        Args:
            pose_hidden_states: Hidden states from LLM for pose tokens
            num_samples: Number of samples to generate per batch item
        
        Returns:
            Sampled poses: (batch_size, num_samples, pose_dim)
        """
        means, covariances, weights = self.forward(pose_hidden_states)
        batch_size = means.shape[0]
        
        # Sample from GMM using reparameterization trick
        samples = []
        for b in range(batch_size):
            # Sample component indices based on weights
            component_indices = torch.multinomial(weights[b], num_samples, replacement=True)
            
            # Sample from selected components
            batch_samples = []
            for i in range(num_samples):
                comp_idx = component_indices[i]
                mean = means[b, comp_idx]  # (pose_dim,)
                cov = covariances[b, comp_idx]  # (pose_dim, pose_dim)
                
                # Sample from multivariate normal
                sample = torch.distributions.MultivariateNormal(mean, cov).sample()
                batch_samples.append(sample)
            
            batch_samples = torch.stack(batch_samples)  # (num_samples, pose_dim)
            samples.append(batch_samples)
        
        return torch.stack(samples)  # (batch_size, num_samples, pose_dim)
    
    def compute_loss(self, pose_hidden_states: torch.Tensor, target_poses: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood loss for GMM.
        
        Args:
            pose_hidden_states: Hidden states from LLM for pose tokens
            target_poses: Target poses: (batch_size, pose_dim)
        
        Returns:
            Loss value
        """
        print("          - [GMMPoseHead.compute_loss] Starting...")
        
        # Ensure all tensors are in bfloat16 to prevent dtype mismatches
        print("          - [GMMPoseHead.compute_loss] Ensuring bfloat16 dtype...")
        pose_hidden_states = ensure_bfloat16(pose_hidden_states)
        target_poses = ensure_bfloat16(target_poses)
        print("          - [GMMPoseHead.compute_loss] Dtype conversion completed")
        
        print("          - [GMMPoseHead.compute_loss] Calling forward to get GMM parameters...")
        means, covariances, weights = self.forward(pose_hidden_states)
        print("          - [GMMPoseHead.compute_loss] GMM parameters obtained")
        
        batch_size = means.shape[0]
        print(f"          - [GMMPoseHead.compute_loss] Batch size: {batch_size}, num_components: {self.num_components}")
        
        # Compute log-likelihood for each component
        print("          - [GMMPoseHead.compute_loss] Computing log-likelihood for each component...")
        log_probs = []
        for b in range(batch_size):
            print(f"            - [GMMPoseHead.compute_loss] Processing batch item {b}...")
            batch_log_probs = []
            for c in range(self.num_components):
                print(f"              - [GMMPoseHead.compute_loss] Processing component {c}...")
                mean = ensure_bfloat16(means[b, c])  # (pose_dim,)
                cov = ensure_bfloat16(covariances[b, c])  # (pose_dim, pose_dim)
                target = ensure_bfloat16(target_poses[b])  # (pose_dim,)
                
                # Compute log probability
                print(f"                - [GMMPoseHead.compute_loss] Creating MultivariateNormal distribution...")
                dist = torch.distributions.MultivariateNormal(mean, cov)
                print(f"                - [GMMPoseHead.compute_loss] Computing log probability...")
                log_prob = dist.log_prob(target)
                print(f"                - [GMMPoseHead.compute_loss] Log probability computed: {log_prob.item()}")
                batch_log_probs.append(log_prob)
            
            batch_log_probs = torch.stack(batch_log_probs)  # (num_components,)
            log_probs.append(batch_log_probs)
            print(f"            - [GMMPoseHead.compute_loss] Batch item {b} completed")
        
        log_probs = torch.stack(log_probs)  # (batch_size, num_components)
        print(f"          - [GMMPoseHead.compute_loss] Log probabilities shape: {log_probs.shape}")
        
        # Compute weighted log-likelihood
        print("          - [GMMPoseHead.compute_loss] Computing weighted log-likelihood...")
        weighted_log_probs = log_probs + torch.log(ensure_bfloat16(weights) + 1e-8)
        
        # Use logsumexp for numerical stability
        print("          - [GMMPoseHead.compute_loss] Computing logsumexp...")
        total_log_prob = torch.logsumexp(weighted_log_probs, dim=-1)
        
        # Return negative log-likelihood
        print("          - [GMMPoseHead.compute_loss] Computing final loss...")
        loss = -torch.mean(total_log_prob)
        print(f"          - [GMMPoseHead.compute_loss] Final loss: {loss.item()}")
        
        return loss


class SimplePoseHead(nn.Module):
    """
    Simple pose head for single pose prediction (non-GMM).
    
    Useful for comparison or when multi-modality is not needed.
    """
    
    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 4096,
        pose_dim: int = 6,
    ):
        """
        Initialize simple pose head.
        
        Args:
            input_dim: Input dimension from LLM hidden states
            hidden_dim: Hidden dimension for MLP
            pose_dim: Dimension of pose (6 for 3D pos + 3D ori)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pose_dim = pose_dim
        
        # MLP to predict single pose (following L1RegressionActionHead pattern)
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * pose_dim,
            hidden_dim=hidden_dim,
            output_dim=pose_dim
        )
    
    def forward(self, pose_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict single pose.
        
        Args:
            pose_hidden_states: Hidden states from LLM for pose tokens
                               Shape: (batch_size, pose_dim, hidden_dim)
        
        Returns:
            Predicted poses: (batch_size, pose_dim)
        """
        # Ensure input is in bfloat16
        pose_hidden_states = ensure_bfloat16(pose_hidden_states)
        
        batch_size = pose_hidden_states.shape[0]
        
        # Reshape to match original pattern
        rearranged_states = pose_hidden_states.reshape(batch_size, -1)
        
        # Predict pose
        pose = self.model(rearranged_states)
        
        return ensure_bfloat16(pose)
    
    def predict_pose(self, pose_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict pose (alias for forward).
        
        Args:
            pose_hidden_states: Hidden states from LLM for pose tokens
        
        Returns:
            Predicted poses: (batch_size, pose_dim)
        """
        return self.forward(pose_hidden_states)
    
    def compute_loss(self, pose_hidden_states: torch.Tensor, target_poses: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 loss for pose prediction.
        
        Args:
            pose_hidden_states: Hidden states from LLM for pose tokens
            target_poses: Target poses: (batch_size, pose_dim)
        
        Returns:
            Loss value
        """
        # Ensure all tensors are in bfloat16 to prevent dtype mismatches
        pose_hidden_states = ensure_bfloat16(pose_hidden_states)
        target_poses = ensure_bfloat16(target_poses)
        
        predicted_poses = self.forward(pose_hidden_states)
        return F.l1_loss(predicted_poses, target_poses) 