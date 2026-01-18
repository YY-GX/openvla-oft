#!/bin/bash

echo "🔄 Restarting parallel augmentation with proper conda environment"
echo "============================================================"

# Kill existing sessions if they exist
echo "🗑️  Killing existing tmux sessions..."
tmux kill-session -t phase2_v1 2>/dev/null || echo "   phase2_v1 not found"
tmux kill-session -t phase2_v2 2>/dev/null || echo "   phase2_v2 not found"
tmux kill-session -t phase2_v3 2>/dev/null || echo "   phase2_v3 not found"
tmux kill-session -t phase2_v4 2>/dev/null || echo "   phase2_v4 not found"

echo ""
echo "🚀 Re-launching with updated script..."
./shells/phase2/parallel_augmentation.sh