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

from prismatic.models.action_heads import MLPResNet


class GMMPoseHead(nn.Module):
    """
    Gaussian Mixture Model pose head for predicting multi-modal pose distributions.
    
    Follows original L1RegressionActionHead patterns but outputs GMM parameters
    instead of single pose predictions.
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
        
        # MLP to predict GMM parameters (following L1RegressionActionHead pattern)
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * pose_dim,  # Same as original pattern
            hidden_dim=hidden_dim,
            output_dim=self.gmm_output_dim
        )
    
    def forward(self, pose_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to predict GMM parameters.
        
        Args:
            pose_hidden_states: Hidden states from LLM for pose tokens
                               Shape: (batch_size, pose_dim, hidden_dim)
        
        Returns:
            Tuple of (means, covariances, weights):
                - means: (batch_size, num_components, pose_dim)
                - covariances: (batch_size, num_components, pose_dim, pose_dim)
                - weights: (batch_size, num_components)
        """
        batch_size = pose_hidden_states.shape[0]
        
        # Reshape to match original pattern (following L1RegressionActionHead)
        rearranged_states = pose_hidden_states.reshape(batch_size, -1)
        
        # Predict GMM parameters
        gmm_params = self.model(rearranged_states)  # (batch_size, gmm_output_dim)
        
        # Split parameters into means, covariances, and weights
        means, covariances, weights = self._split_gmm_params(gmm_params, batch_size)
        
        return means, covariances, weights
    
    def _split_gmm_params(self, gmm_params: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split GMM parameters into means, covariances, and weights.
        
        Args:
            gmm_params: Raw GMM parameters from MLP
            batch_size: Batch size
        
        Returns:
            Tuple of (means, covariances, weights)
        """
        # Calculate split indices
        means_dim = self.num_components * self.pose_dim
        covs_dim = self.num_components * self.pose_dim
        
        # Split parameters
        means_flat = gmm_params[:, :means_dim]  # (batch_size, num_components * pose_dim)
        covs_flat = gmm_params[:, means_dim:means_dim + covs_dim]  # (batch_size, num_components * pose_dim)
        weights_flat = gmm_params[:, means_dim + covs_dim:]  # (batch_size, num_components)
        
        # Reshape means
        means = means_flat.reshape(batch_size, self.num_components, self.pose_dim)
        
        # Reshape and process covariances (ensure positive definite)
        covs_flat = covs_flat.reshape(batch_size, self.num_components, self.pose_dim)
        # Apply softplus and clamp for numerical stability
        covs_flat = torch.clamp(F.softplus(covs_flat), min=self.min_std, max=self.max_std)
        # Create diagonal covariance matrices
        covariances = torch.diag_embed(covs_flat)  # (batch_size, num_components, pose_dim, pose_dim)
        
        # Process weights (ensure they sum to 1)
        weights = F.softmax(weights_flat, dim=-1)  # (batch_size, num_components)
        
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
        means, covariances, weights = self.forward(pose_hidden_states)
        batch_size = means.shape[0]
        
        # Compute log-likelihood for each component
        log_probs = []
        for b in range(batch_size):
            batch_log_probs = []
            for c in range(self.num_components):
                mean = means[b, c]  # (pose_dim,)
                cov = covariances[b, c]  # (pose_dim, pose_dim)
                target = target_poses[b]  # (pose_dim,)
                
                # Compute log probability
                dist = torch.distributions.MultivariateNormal(mean, cov)
                log_prob = dist.log_prob(target)
                batch_log_probs.append(log_prob)
            
            batch_log_probs = torch.stack(batch_log_probs)  # (num_components,)
            log_probs.append(batch_log_probs)
        
        log_probs = torch.stack(log_probs)  # (batch_size, num_components)
        
        # Compute weighted log-likelihood
        weighted_log_probs = log_probs + torch.log(weights + 1e-8)
        
        # Use logsumexp for numerical stability
        total_log_prob = torch.logsumexp(weighted_log_probs, dim=-1)
        
        # Return negative log-likelihood
        return -torch.mean(total_log_prob)


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
        batch_size = pose_hidden_states.shape[0]
        
        # Reshape to match original pattern
        rearranged_states = pose_hidden_states.reshape(batch_size, -1)
        
        # Predict pose
        pose = self.model(rearranged_states)
        
        return pose
    
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
        predicted_poses = self.forward(pose_hidden_states)
        return F.l1_loss(predicted_poses, target_poses) 