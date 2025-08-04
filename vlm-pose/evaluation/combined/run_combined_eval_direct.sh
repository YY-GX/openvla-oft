#!/bin/bash
#SBATCH --job-name=combined_eval
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=combined_eval_%j.out
#SBATCH --error=combined_eval_%j.err

# Set up environment variables
export PATH="/mnt/arc/yygx/anaconda3/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/arc/yygx/anaconda3/lib:$LD_LIBRARY_PATH"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Navigate to the combined_eval directory
cd /mnt/arc/yygx/pkgs_baselines/openvla-oft/vla-evaluation/combined_eval

# Install missing library if needed
/mnt/arc/yygx/anaconda3/bin/conda install -c conda-forge orocos-kdl -y

# Test that the environment is working using full path
/mnt/arc/yygx/.conda/envs/openvla-oft/bin/python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Run the test directly using full path
/mnt/arc/yygx/.conda/envs/openvla-oft/bin/python test_combined_pipeline.py \
    --pose_vlm_checkpoint ../../runs/pose_vlm/1.0.0/openvla-7b+pose_dataset+b8+lr-1e-05+pose_simple+lora-r32+dropout-0.0--image_aug--pose_aug--simple_pose_head_pose_aug--smaller_splits/vla--20000_checkpoint.pt \
    --openvla_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/view_wrist/1.0.3/openvla-7b+libero44_local+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt \
    --task_id 16 \
    --num_episodes 1 \
    --max_ik_attempts 1

echo "Job completed!" 