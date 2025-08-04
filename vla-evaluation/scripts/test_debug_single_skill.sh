#!/bin/bash

# Debug single skill test script
# This script runs the combined evaluation in debug mode for a single skill

set -e

echo "=========================================="
echo "DEBUG SINGLE SKILL EVALUATION"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Change to the combined_eval directory
cd vla-evaluation/combined_eval

echo "Running debug single skill evaluation..."
echo "Skill: KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet"
echo "Episodes: 5"
echo "IK Attempts: 5"
echo ""

# Run the debug test
python test_debug_single_skill.py

echo ""
echo "=========================================="
echo "DEBUG EVALUATION COMPLETE"
echo "=========================================="
echo "Check debug images in: runs/eval_results/combined_eval/debug_single_skill"
echo "Check logs for detailed results" 