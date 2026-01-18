#!/bin/zsh

# Parallel Phase 2 Augmentation Script (Reaching Approach)
# Launches 5 instances of 0_generate_augmented_demos_reaching.py with different seeds and output directories

echo "🚀 Starting parallel Phase 2 augmentation (reaching) with 5 instances"
echo "============================================================"

# Base directories
BASE_DIR="/mnt/arc/yygx/pkgs_baselines/openvla-oft"
SCRIPT_PATH="scripts/phase2/0_generate_augmented_demos_reaching.py"
BASE_OUTPUT_DIR="datasets/hdf5_datasets/atomic_local_demos_augmented_reaching"

# MP threshold parameters (matching script defaults)
POSITION_THRESHOLD=0.02  # 2cm - tight threshold for MP3 final correction
ORIENTATION_THRESHOLD_DEG=15  # 15° - tight threshold for MP3
FAR_POSITION_THRESHOLD=0.05  # 5cm - loose threshold for MP1 far-away pose
FAR_ORIENTATION_THRESHOLD_DEG=30  # 30° - loose threshold for MP1
RETURN_POSITION_THRESHOLD=0.05  # 5cm - medium threshold for MP2 return
RETURN_ORIENTATION_THRESHOLD_DEG=60  # 60° - medium threshold for MP2
# Note: MP1+MP2 retries hardcoded to 20 in script (with distance shrinking after 5 attempts)
# Note: MP3 correction attempts hardcoded to 5 in script
NUM_AUGMENTATION_ITERATIONS=1

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
echo "   Each instance processes: ALL skills (atomic + pick + place) with ALL demos"
echo "   Position threshold (MP3): ${POSITION_THRESHOLD}m, Orientation: ${ORIENTATION_THRESHOLD_DEG}°"
echo "   Far position threshold (MP1): ${FAR_POSITION_THRESHOLD}m, Orientation: ${FAR_ORIENTATION_THRESHOLD_DEG}°"
echo "   Return position threshold (MP2): ${RETURN_POSITION_THRESHOLD}m, Orientation: ${RETURN_ORIENTATION_THRESHOLD_DEG}°"
echo "   MP1+MP2 retries: 20 (hardcoded), MP3 retries: 5 (hardcoded)"
echo "   Augmentation iterations per demo: ${NUM_AUGMENTATION_ITERATIONS}"
idx=1  # zsh arrays are 1-indexed
for i in {0..4}; do
    echo "   Instance $((i+1)): GPU ${GPU_IDS[$idx]}, Seed ${SEEDS[$idx]}, Output ${VERSIONS[$idx]}"
    idx=$((idx + 1))
done

# Launch each instance in tmux session
cd "$BASE_DIR"

# Counter for array iteration (zsh arrays are 1-indexed)
idx=1
for i in {0..4}; do
    GPU_ID=${GPU_IDS[$idx]}
    SEED=${SEEDS[$idx]}
    VERSION=${VERSIONS[$idx]}
    idx=$((idx + 1))

    # Debug: Check if variables are properly set
    if [[ -z "$GPU_ID" || -z "$SEED" || -z "$VERSION" ]]; then
        echo "❌ Error: Variables not properly set for iteration $i"
        echo "   GPU_ID='$GPU_ID', SEED='$SEED', VERSION='$VERSION'"
        continue
    fi

    SESSION_NAME="phase2_${VERSION}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${VERSION}"
    CHECKPOINT_FILE="${BASE_OUTPUT_DIR}/checkpoint_reaching_${VERSION}.pkl"

    echo ""
    echo "🚀 Launching instance $((i+1))/5: ${VERSION}"
    echo "   GPU: ${GPU_ID}, Seed: ${SEED}"
    echo "   Processing: ALL skills (atomic + pick + place)"
    echo "   Output: ${OUTPUT_DIR}"
    echo "   Checkpoint: ${CHECKPOINT_FILE}"

    # Kill existing tmux session if it exists
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "   🔄 Killing existing tmux session: $SESSION_NAME"
        tmux kill-session -t "$SESSION_NAME"
        sleep 1
    fi

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
        --num_augmentation_iterations ${NUM_AUGMENTATION_ITERATIONS} \
        --position_threshold ${POSITION_THRESHOLD} \
        --orientation_threshold_deg ${ORIENTATION_THRESHOLD_DEG} \
        --far_position_threshold ${FAR_POSITION_THRESHOLD} \
        --far_orientation_threshold_deg ${FAR_ORIENTATION_THRESHOLD_DEG} \
        --return_position_threshold ${RETURN_POSITION_THRESHOLD} \
        --return_orientation_threshold_deg ${RETURN_ORIENTATION_THRESHOLD_DEG} \
        --family_distances 0.08 0.10 0.12 \
        --random_seed ${SEED} \
        --checkpoint_file ${CHECKPOINT_FILE} \
        --resume" Enter

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
echo "   tmux attach -t reaching_v1  # (or v2, v3, v4, v5)"
echo ""
echo "🔍 Check GPU usage with:"
echo "   nvidia-smi"
echo ""
echo "📁 Output locations:"
echo "   ${BASE_OUTPUT_DIR}/v1/"
echo "   ${BASE_OUTPUT_DIR}/v2/"
echo "   ${BASE_OUTPUT_DIR}/v3/"
echo "   ${BASE_OUTPUT_DIR}/v4/"
echo "   ${BASE_OUTPUT_DIR}/v5/"
echo ""
echo "💾 Checkpoint files:"
echo "   ${BASE_OUTPUT_DIR}/checkpoint_reaching_*_v*.pkl"
echo ""
echo "⚙️  Motion Planning Configuration:"
echo "   MP3 (Correction): ${POSITION_THRESHOLD}m, ${ORIENTATION_THRESHOLD_DEG}°"
echo "   MP2 (Return): ${RETURN_POSITION_THRESHOLD}m, ${RETURN_ORIENTATION_THRESHOLD_DEG}°"
echo "   MP1 (Far-away): ${FAR_POSITION_THRESHOLD}m, ${FAR_ORIENTATION_THRESHOLD_DEG}°"
echo "   Family distances: 8cm, 10cm, 12cm"
echo ""
echo "⚡ Estimated time: ~24 hours per instance"
echo "🔄 Use --resume flag to restart from checkpoints if needed"
