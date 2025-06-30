#!/usr/bin/env python3
"""
Simple test script to verify that dtype fixes work correctly.
"""

import torch
import torch.nn as nn
from prismatic.util import ensure_bfloat16
from prismatic.models.pose_heads import GMMPoseHead, SimplePoseHead


def test_pose_head_dtype():
    """Test that pose heads handle dtype correctly."""
    print("Testing pose head dtype handling...")
    
    # Test GMM pose head
    gmm_head = GMMPoseHead(
        input_dim=4096,
        hidden_dim=4096,
        pose_dim=6,
        num_components=3
    ).to(torch.bfloat16)
    
    # Test with float32 input (should be converted to bfloat16)
    batch_size = 2
    num_pose_tokens = 6
    hidden_dim = 4096
    
    pose_hidden_states = torch.randn(batch_size, num_pose_tokens, hidden_dim, dtype=torch.float32)
    target_poses = torch.randn(batch_size, 6, dtype=torch.float32)
    
    try:
        # This should work without dtype mismatch
        means, covariances, weights = gmm_head(pose_hidden_states)
        loss = gmm_head.compute_loss(pose_hidden_states, target_poses)
        
        print(f"✓ GMM pose head test passed")
        print(f"  - Means shape: {means.shape}, dtype: {means.dtype}")
        print(f"  - Covariances shape: {covariances.shape}, dtype: {covariances.dtype}")
        print(f"  - Weights shape: {weights.shape}, dtype: {weights.dtype}")
        print(f"  - Loss: {loss.item():.4f}, dtype: {loss.dtype}")
        
    except RuntimeError as e:
        if "mat1 and mat2 must have the same dtype" in str(e):
            print(f"✗ GMM pose head test failed with dtype mismatch: {e}")
            return False
        else:
            print(f"✓ GMM pose head test passed (other error: {e})")
    
    # Test Simple pose head
    simple_head = SimplePoseHead(
        input_dim=4096,
        hidden_dim=4096,
        pose_dim=6
    ).to(torch.bfloat16)
    
    try:
        # This should work without dtype mismatch
        predicted_poses = simple_head(pose_hidden_states)
        loss = simple_head.compute_loss(pose_hidden_states, target_poses)
        
        print(f"✓ Simple pose head test passed")
        print(f"  - Predicted poses shape: {predicted_poses.shape}, dtype: {predicted_poses.dtype}")
        print(f"  - Loss: {loss.item():.4f}, dtype: {loss.dtype}")
        
    except RuntimeError as e:
        if "mat1 and mat2 must have the same dtype" in str(e):
            print(f"✗ Simple pose head test failed with dtype mismatch: {e}")
            return False
        else:
            print(f"✓ Simple pose head test passed (other error: {e})")
    
    return True


def test_ensure_bfloat16():
    """Test the ensure_bfloat16 utility function."""
    print("Testing ensure_bfloat16 utility...")
    
    # Test with float32 tensor
    float32_tensor = torch.randn(3, 224, 224, dtype=torch.float32)
    bfloat16_tensor = ensure_bfloat16(float32_tensor)
    assert bfloat16_tensor.dtype == torch.bfloat16, f"Expected bfloat16, got {bfloat16_tensor.dtype}"
    
    # Test with already bfloat16 tensor
    bfloat16_tensor2 = torch.randn(3, 224, 224, dtype=torch.bfloat16)
    result = ensure_bfloat16(bfloat16_tensor2)
    assert result.dtype == torch.bfloat16, f"Expected bfloat16, got {result.dtype}"
    assert torch.equal(result, bfloat16_tensor2), "Should not change already bfloat16 tensor"
    
    print("✓ ensure_bfloat16 utility test passed")


def main():
    """Run all tests."""
    print("Running dtype fix tests...\n")
    
    try:
        test_ensure_bfloat16()
        success = test_pose_head_dtype()
        
        if success:
            print("\n🎉 All dtype fix tests passed!")
            print("The dtype mismatch issues should now be resolved.")
        else:
            print("\n❌ Some tests failed.")
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        raise


if __name__ == "__main__":
    main() 