#!/bin/bash

# Parallel Phase 2 Augmentation Script
# Launches 5 instances of 1_generate_augmented_demos.py with different seeds and output directories
# Uses offset parameters to automatically create structured output directories

echo "🚀 Starting parallel Phase 2 augmentation with 5 instances"
echo "============================================================"

# Base directories
BASE_DIR="/mnt/arc/yygx/pkgs_baselines/openvla-oft"
SCRIPT_PATH="scripts/phase2/1_generate_augmented_demos.py"
BASE_OUTPUT_DIR="datasets/hdf5_datasets/atomic_local_demos_augmented_pick_10_place_15_atomic_15"

# Offset parameters
PICK_OFFSET=10
PLACE_OFFSET=15
ATOMIC_OFFSET=15

# Create output directories
echo "📁 Creating output directories..."
mkdir -p "${BASE_DIR}/${BASE_OUTPUT_DIR}/v1"
mkdir -p "${BASE_DIR}/${BASE_OUTPUT_DIR}/v2"
mkdir -p "${BASE_DIR}/${BASE_OUTPUT_DIR}/v3"
mkdir -p "${BASE_DIR}/${BASE_OUTPUT_DIR}/v4"
mkdir -p "${BASE_DIR}/${BASE_OUTPUT_DIR}/v5"

# Configuration for each instance
declare -a GPU_IDS=(2 3 4 5 6)
declare -a SEEDS=(12345 67890 11111 99999 55555)
declare -a VERSIONS=(v1 v2 v3 v4 v5)

echo "📋 Configuration:"
echo "   Offsets: pick=${PICK_OFFSET}, place=${PLACE_OFFSET}, atomic=${ATOMIC_OFFSET}"
for i in {0..4}; do
    echo "   Instance $((i+1)): GPU ${GPU_IDS[i]}, Seed ${SEEDS[i]}, Output ${VERSIONS[i]}"
done

# Launch each instance in tmux session
cd "$BASE_DIR"

for i in {0..4}; do
    GPU_ID=${GPU_IDS[i]}
    SEED=${SEEDS[i]}
    VERSION=${VERSIONS[i]}

    # Debug: Check if variables are properly set
    if [[ -z "$GPU_ID" || -z "$SEED" || -z "$VERSION" ]]; then
        echo "❌ Error: Variables not properly set for iteration $i"
        echo "   GPU_ID='$GPU_ID', SEED='$SEED', VERSION='$VERSION'"
        continue
    fi

    SESSION_NAME="new_phase2_${VERSION}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${VERSION}"
    CHECKPOINT_FILE="${BASE_OUTPUT_DIR}/checkpoint_phase2_${VERSION}.pkl"

    echo ""
    echo "🚀 Launching instance $((i+1))/5: ${VERSION}"
    echo "   GPU: ${GPU_ID}, Seed: ${SEED}"
    echo "   Output: ${OUTPUT_DIR}"
    echo "   Checkpoint: ${CHECKPOINT_FILE}"
    echo "   Offsets: pick=${PICK_OFFSET}, place=${PLACE_OFFSET}, atomic=${ATOMIC_OFFSET}"

    # Create tmux session and run the script
    if tmux new-session -d -s "$SESSION_NAME" -c "$BASE_DIR"; then
        echo "   ✅ Tmux session created: $SESSION_NAME"
    else
        echo "   ❌ Failed to create tmux session: $SESSION_NAME"
        continue
    fi

    # Activate conda environment first, then run the script
    tmux send-keys -t "$SESSION_NAME" "conda activate openvla-oft" Enter
    sleep 3  # Wait for conda activation
    tmux send-keys -t "$SESSION_NAME" "CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --pick_offset ${PICK_OFFSET} \
        --place_offset ${PLACE_OFFSET} \
        --atomic_offset ${ATOMIC_OFFSET} \
        --random_seed ${SEED} \
        --checkpoint_file ${CHECKPOINT_FILE}" Enter

    echo "   ✅ Started in tmux session: ${SESSION_NAME}"

    # Small delay between launches
    sleep 2
done

echo ""
echo "🎯 All 5 instances launched successfully!"
echo "============================================================"
echo ""
echo "📊 Monitor progress with:"
echo "   tmux list-sessions"
echo "   tmux attach -t new_phase2_v1  # (or v2, v3, v4, v5)"
echo ""
echo "🔍 Check GPU usage with:"
echo "   nvidia-smi"
echo ""
echo "📁 Output locations (with offset structure):"
echo "   ${BASE_OUTPUT_DIR}/v1/"
echo "   ${BASE_OUTPUT_DIR}/v2/"
echo "   ${BASE_OUTPUT_DIR}/v3/"
echo "   ${BASE_OUTPUT_DIR}/v4/"
echo "   ${BASE_OUTPUT_DIR}/v5/"
echo ""
echo "💾 Checkpoint files:"
echo "   ${BASE_OUTPUT_DIR}/checkpoint_phase2_v1.pkl"
echo "   ${BASE_OUTPUT_DIR}/checkpoint_phase2_v2.pkl"
echo "   ${BASE_OUTPUT_DIR}/checkpoint_phase2_v3.pkl"
echo "   ${BASE_OUTPUT_DIR}/checkpoint_phase2_v4.pkl"
echo "   ${BASE_OUTPUT_DIR}/checkpoint_phase2_v5.pkl"
echo ""
echo "⚙️  Offset Configuration:"
echo "   Pick offset: ${PICK_OFFSET} steps"
echo "   Place offset: ${PLACE_OFFSET} steps"
echo "   Atomic offset: ${ATOMIC_OFFSET} steps"
echo ""
echo "⚡ Estimated time: ~32 hours per instance"
echo "🔄 Use --resume flag to restart from checkpoints if needed"