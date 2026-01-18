#!/bin/bash
# Launch 5 parallel processes for demo generation
# Each process processes all BDDL files with different random seeds and output folders

# ============================================================================
# CONFIGURATION - Edit these parameters as needed
# ============================================================================
SCRIPT_DIR="/mnt/arc/yygx/pkgs_baselines/openvla-oft"
PYTHON_SCRIPT="scripts/phase3/pipeline/data_generation/generate_above_augmented_demos.py"
BASE_OUTPUT_DIR="/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/atomic_above"
NUM_AUGMENTATIONS=5
NUM_PROCESSES=5

# Random seeds for each process (one per process)
SEEDS=(42 123 456 789 1011)

# Output folder suffixes (one per process)
OUTPUT_SUFFIXES=(v1 v2 v3 v4 v5)
# ============================================================================

# Validate configuration
if [ ${#SEEDS[@]} -ne ${NUM_PROCESSES} ]; then
    echo "Error: Number of seeds (${#SEEDS[@]}) must match NUM_PROCESSES (${NUM_PROCESSES})"
    exit 1
fi

if [ ${#OUTPUT_SUFFIXES[@]} -ne ${NUM_PROCESSES} ]; then
    echo "Error: Number of output suffixes (${#OUTPUT_SUFFIXES[@]}) must match NUM_PROCESSES (${NUM_PROCESSES})"
    exit 1
fi

cd "${SCRIPT_DIR}"

# Launch processes in tmux sessions
# Note: Using 1-indexed loop to work with both bash and zsh array indexing
for i in $(seq 1 ${NUM_PROCESSES}); do
    ARRAY_IDX=$((i - 1))  # Convert to 0-indexed for array access
    SEED=${SEEDS[$ARRAY_IDX]}
    OUTPUT_SUFFIX=${OUTPUT_SUFFIXES[$ARRAY_IDX]}
    OUTPUT_PATH="${BASE_OUTPUT_DIR}/${OUTPUT_SUFFIX}"
    SESSION_NAME="gen_${OUTPUT_SUFFIX}"
    
    # Enable video saving only for v1 (first process)
    VIDEO_FLAG=""
    if [ "${OUTPUT_SUFFIX}" = "v1" ]; then
        VIDEO_FLAG="--save_videos_normal"
    fi
    
    # Kill existing session if it exists
    tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
    
    # Create tmux session (detached)
    tmux new-session -d -s "${SESSION_NAME}"
    
    # Send commands to tmux session (wait for conda activation to complete)
    tmux send-keys -t "${SESSION_NAME}" "cd ${SCRIPT_DIR}" C-m
    tmux send-keys -t "${SESSION_NAME}" "conda activate openvla-oft" C-m
    sleep 1  # Wait for conda activation to complete
    tmux send-keys -t "${SESSION_NAME}" "python3 ${PYTHON_SCRIPT} --num_augmentations ${NUM_AUGMENTATIONS} --random_seed ${SEED} --output_path \"${OUTPUT_PATH}\" --resume ${VIDEO_FLAG}" C-m
    
    echo "Created tmux session: ${SESSION_NAME} (seed=${SEED}${VIDEO_FLAG:+ , videos enabled})"
done

echo ""
echo "All tmux sessions created!"
echo "Attach to a session with: tmux attach -t gen_v1"
echo "List all sessions with: tmux ls"
echo "Kill a session with: tmux kill-session -t gen_v1"
echo ""
echo "Sessions: gen_v1, gen_v2, gen_v3, gen_v4, gen_v5"

