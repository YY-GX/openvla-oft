#!/usr/bin/env python3
"""
Unit tests for Phase 3: Training Infrastructure

This script tests the training infrastructure components:
- Pose training script configuration
- Training shell script
- Partial loading functionality
- Integration with existing components

Usage:
    python test_phase3.py --mode self_contained  # Test only Phase 3 components
    python test_phase3.py --mode full            # Test full integration
"""

import argparse
import os
import sys
import tempfile
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_pose_finetune_config():
    """Test PoseFinetuneConfig dataclass"""
    print("=== Testing PoseFinetuneConfig ===")
    
    try:
        # Import using exec to handle the hyphen in directory name
        vla_scripts_path = project_root / "vla-scripts"
        sys.path.insert(0, str(vla_scripts_path))
        
        # Use importlib to handle the module with hyphen
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "finetune_pose", 
            vla_scripts_path / "finetune_pose.py"
        )
        finetune_pose_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(finetune_pose_module)
        
        PoseFinetuneConfig = finetune_pose_module.PoseFinetuneConfig
        
        # Test default configuration
        config = PoseFinetuneConfig()
        assert config.vla_path == "openvla/openvla-7b"
        assert config.pose_head_type == "gmm"
        assert config.gmm_num_components == 3
        assert config.pose_dim == 6
        assert config.num_pose_tokens == 6
        assert config.use_proprio == False
        assert config.pose_aug == True
        
        # Test custom configuration
        custom_config = PoseFinetuneConfig(
            pose_head_type="simple",
            gmm_num_components=5,
            pose_dim=7,
            use_proprio=True,
            pose_aug=False
        )
        assert custom_config.pose_head_type == "simple"
        assert custom_config.gmm_num_components == 5
        assert custom_config.pose_dim == 7
        assert custom_config.use_proprio == True
        assert custom_config.pose_aug == False
        
        print("✓ PoseFinetuneConfig tests passed")
        return True
        
    except Exception as e:
        print(f"✗ PoseFinetuneConfig tests failed: {e}")
        return False


def test_training_shell_script():
    """Test training shell script structure"""
    print("=== Testing Training Shell Script ===")
    
    try:
        shell_script_path = project_root / "shells" / "trainings" / "train_pose_vlm.sh"
        assert shell_script_path.exists(), f"Shell script not found at {shell_script_path}"
        
        # Read and check script content
        with open(shell_script_path, 'r') as f:
            content = f.read()
        
        # Check for required components
        assert "finetune_pose.py" in content, "Script should call finetune_pose.py"
        assert "--pose_head_type gmm" in content, "Script should specify pose head type"
        assert "--gmm_num_components 3" in content, "Script should specify GMM components"
        assert "--pose_dim 6" in content, "Script should specify pose dimension"
        assert "--use_proprio False" in content, "Script should disable proprioception"
        assert "--pose_aug True" in content, "Script should enable pose augmentation"
        
        print("✓ Training shell script tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Training shell script tests failed: {e}")
        return False


def test_partial_loading_self_contained():
    """Test partial loading functionality (self-contained mode)"""
    print("=== Testing Partial Loading (Self-contained) ===")
    
    try:
        # Import the module correctly
        import prismatic.models.load as load_module
        
        # Mock the load_vla function to avoid loading actual models
        with patch('prismatic.models.load.load_vla') as mock_load_vla:
            # Create a mock VLA model
            mock_vla = Mock()
            mock_vla.vision_backbone = Mock()
            mock_vla.llm_backbone = Mock()
            mock_vla.tokenizer = Mock()
            mock_vla.config.hidden_size = 4096
            mock_vla.parameters.return_value = [Mock(requires_grad=True)]
            
            mock_load_vla.return_value = mock_vla
            
            # Test loading with GMM pose head
            pose_vlm = load_module.load_pose_vla(
                vla_path="test_path",
                pose_head_type="gmm",
                gmm_num_components=3,
                pose_dim=6
            )
            
            assert pose_vlm is not None
            assert hasattr(pose_vlm, 'pose_head')
            assert pose_vlm.pose_head_type == "gmm"
            assert pose_vlm.use_proprio == False
            
            # Test loading with simple pose head
            pose_vlm_simple = load_module.load_pose_vla(
                vla_path="test_path",
                pose_head_type="simple",
                pose_dim=6
            )
            
            assert pose_vlm_simple is not None
            assert hasattr(pose_vlm_simple, 'pose_head')
            assert pose_vlm_simple.pose_head_type == "simple"
            
            print("✓ Partial loading (self-contained) tests passed")
            return True
            
    except Exception as e:
        print(f"✗ Partial loading (self-contained) tests failed: {e}")
        return False


def test_partial_loading_full():
    """Test partial loading functionality (full mode)"""
    print("=== Testing Partial Loading (Full) ===")
    
    try:
        import prismatic.models.load as load_module
        from prismatic.models.pose_heads import GMMPoseHead, SimplePoseHead
        
        # Create temporary directory for mock checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock checkpoint structure
            checkpoint_dir = temp_path / "mock_vla_checkpoint"
            checkpoint_dir.mkdir()
            
            # Mock the load_vla function
            with patch('prismatic.models.load.load_vla') as mock_load_vla:
                # Create a more realistic mock VLA model
                mock_vla = Mock()
                mock_vla.vision_backbone = Mock()
                mock_vla.llm_backbone = Mock()
                mock_vla.tokenizer = Mock()
                mock_vla.config.hidden_size = 4096
                
                # Mock parameters
                mock_param = Mock()
                mock_param.requires_grad = True
                mock_vla.parameters.return_value = [mock_param]
                
                mock_load_vla.return_value = mock_vla
                
                # Test loading for training with checkpoint
                pose_vlm = load_module.load_pose_vla(
                    vla_path=str(checkpoint_dir),
                    pose_head_type="gmm",
                    gmm_num_components=3,
                    pose_dim=6,
                    load_for_training=True
                )
                
                assert pose_vlm is not None
                assert isinstance(pose_vlm.pose_head, GMMPoseHead)
                assert pose_vlm.pose_head.num_components == 3
                
                # Test forward pass with mock data
                batch_size = 2
                hidden_dim = 4096
                pose_dim = 6
                
                mock_hidden_states = torch.randn(batch_size, hidden_dim, pose_dim)
                mock_target_poses = torch.randn(batch_size, pose_dim)
                
                # Test GMM pose head
                if pose_vlm.pose_head_type == "gmm":
                    means, covariances, weights = pose_vlm.pose_head(mock_hidden_states)
                    assert means.shape == (batch_size, 3, pose_dim)  # 3 components
                    assert covariances.shape == (batch_size, 3, pose_dim, pose_dim)
                    assert weights.shape == (batch_size, 3)
                    
                    loss = pose_vlm.pose_head.compute_loss(mock_hidden_states, mock_target_poses)
                    assert isinstance(loss, torch.Tensor)
                    assert loss.requires_grad
                
                print("✓ Partial loading (full) tests passed")
                return True
                
    except Exception as e:
        print(f"✗ Partial loading (full) tests failed: {e}")
        return False


def test_training_script_integration():
    """Test training script integration with existing components"""
    print("=== Testing Training Script Integration ===")
    
    try:
        # Test that training script can be imported
        vla_scripts_path = project_root / "vla-scripts"
        sys.path.insert(0, str(vla_scripts_path))
        
        # Use importlib to handle the module with hyphen
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "finetune_pose", 
            vla_scripts_path / "finetune_pose.py"
        )
        finetune_pose_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(finetune_pose_module)
        
        PoseFinetuneConfig = finetune_pose_module.PoseFinetuneConfig
        get_run_id = finetune_pose_module.get_run_id
        remove_ddp_in_checkpoint = finetune_pose_module.remove_ddp_in_checkpoint
        wrap_ddp = finetune_pose_module.wrap_ddp
        count_parameters = finetune_pose_module.count_parameters
        
        # Test get_run_id function
        config = PoseFinetuneConfig(
            vla_path="test/vla",
            dataset_name="test_dataset",
            pose_head_type="gmm",
            gmm_num_components=3
        )
        
        run_id = get_run_id(config)
        assert "test/vla" in run_id
        assert "test_dataset" in run_id
        assert "pose_gmm" in run_id
        assert "c3" in run_id
        
        # Test remove_ddp_in_checkpoint
        mock_state_dict = {
            "module.layer1.weight": torch.randn(10, 10),
            "layer2.weight": torch.randn(10, 10)
        }
        cleaned_dict = remove_ddp_in_checkpoint(mock_state_dict)
        assert "layer1.weight" in cleaned_dict
        assert "layer2.weight" in cleaned_dict
        assert "module.layer1.weight" not in cleaned_dict
        
        # Test count_parameters with mock module
        mock_module = Mock()
        mock_param1 = Mock()
        mock_param1.numel.return_value = 100
        mock_param1.requires_grad = True
        mock_param2 = Mock()
        mock_param2.numel.return_value = 50
        mock_param2.requires_grad = False
        mock_module.parameters.return_value = [mock_param1, mock_param2]
        
        # This should not raise an exception
        count_parameters(mock_module, "test_module")
        
        print("✓ Training script integration tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Training script integration tests failed: {e}")
        return False


def test_forward_pass_logic():
    """Test forward pass logic from training script"""
    print("=== Testing Forward Pass Logic ===")
    
    try:
        vla_scripts_path = project_root / "vla-scripts"
        sys.path.insert(0, str(vla_scripts_path))
        
        # Use importlib to handle the module with hyphen
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "finetune_pose", 
            vla_scripts_path / "finetune_pose.py"
        )
        finetune_pose_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(finetune_pose_module)
        
        run_forward_pass = finetune_pose_module.run_forward_pass
        
        # Create mock components
        mock_vla = Mock()
        mock_pose_head = Mock()
        mock_batch = {
            "images": torch.randn(2, 3, 224, 224),
            "text": torch.randint(0, 1000, (2, 10)),
            "text_attention_mask": torch.ones(2, 10),
            "poses": torch.randn(2, 6)
        }
        
        # Mock VLA outputs
        mock_hidden_states = torch.randn(2, 20, 4096)  # batch_size, seq_len, hidden_dim
        mock_vla_outputs = Mock()
        mock_vla_outputs.hidden_states = [torch.randn(2, 20, 4096)]  # List of hidden states
        mock_vla.return_value = mock_vla_outputs
        
        # Mock pose head
        mock_pose_head.return_value = (torch.randn(2, 3, 6), torch.randn(2, 3, 6, 6), torch.randn(2, 3))
        mock_pose_head.compute_loss.return_value = torch.tensor(0.5, requires_grad=True)
        
        # Mock config
        mock_config = Mock()
        mock_config.pose_head_type = "gmm"
        mock_config.num_pose_tokens = 6
        mock_config.pose_dim = 6
        
        # Test forward pass
        loss, metrics = run_forward_pass(
            vla=mock_vla,
            pose_head=mock_pose_head,
            batch=mock_batch,
            device_id=0,
            cfg=mock_config,
            use_film=False,
            num_patches=196,
            pose_augmentation=None
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "l1_error" in metrics
        
        print("✓ Forward pass logic tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Forward pass logic tests failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Phase 3: Training Infrastructure")
    parser.add_argument("--mode", choices=["self_contained", "full"], default="self_contained",
                       help="Test mode: self_contained (Phase 3 only) or full (with integration)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PHASE 3 TESTING")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Phase 3 tests in {args.mode} mode on {device}")
    
    # Test results
    results = {}
    
    # Phase 3 specific tests
    print("\n" + "=" * 60)
    results["pose_finetune_config"] = test_pose_finetune_config()
    results["training_shell_script"] = test_training_shell_script()
    results["partial_loading_self_contained"] = test_partial_loading_self_contained()
    
    if args.mode == "full":
        print("\n" + "=" * 60)
        results["partial_loading_full"] = test_partial_loading_full()
        results["training_script_integration"] = test_training_script_integration()
        results["forward_pass_logic"] = test_forward_pass_logic()
    
    # Summary
    print("\n" + "=" * 60)
    print("PHASE 3 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 3 tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 