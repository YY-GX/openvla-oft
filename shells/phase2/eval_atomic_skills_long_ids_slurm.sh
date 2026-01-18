#!/bin/zsh

# ID-based Atomic Skills Evaluation via Slurm (matches training script attributes)
# Submits three separate jobs for ID1/ID2/ID3 evaluations

CPUS_PER_TASK=32
GPUS=1
LOG_DIR="$ENDPOINT/pkgs_baselines/openvla-oft/logs"

mkdir -p "$LOG_DIR"

# ---------- ID1 ----------
JOB_NAME_ID1="eval_openvla_atomic_skills_long_id1"
LOG_FILE_ID1="eval_atomic_skills_long_id1_%j.out"

sbatch \
  --nodelist=arcee.ib \
  --cpus-per-task=$CPUS_PER_TASK \
  --gpus=$GPUS \
  -o "$LOG_DIR/$LOG_FILE_ID1" \
  -J $JOB_NAME_ID1 \
  --wrap="zsh -lc 'source ~/.zshrc && conda activate openvla-oft && cd /mnt/arc/yygx/pkgs_baselines/openvla-oft && \
    python scripts/phase2/more_real_simplified_pose_shift_eval.py \
      --pretrained_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/libero_atomic_skills_augmented_long_id1/1.0.0/openvla-7b+libero_atomic_skills_augmented_long_id1+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_long_id1--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt \
      --ID_eval_mode True \
      --init_files_path datasets/hdf5_datasets/atomic_local_demos_augmented/combined/full_by_language \
      --shift_position_std 0.02 \
      --shift_orientation_std 0.5235'"

# ---------- ID2 ----------
JOB_NAME_ID2="eval_openvla_atomic_skills_long_id2"
LOG_FILE_ID2="eval_atomic_skills_long_id2_%j.out"

sbatch \
  --nodelist=arcee.ib \
  --cpus-per-task=$CPUS_PER_TASK \
  --gpus=$GPUS \
  -o "$LOG_DIR/$LOG_FILE_ID2" \
  -J $JOB_NAME_ID2 \
  --wrap="zsh -lc 'source ~/.zshrc && conda activate openvla-oft && cd /mnt/arc/yygx/pkgs_baselines/openvla-oft && \
    python scripts/phase2/more_real_simplified_pose_shift_eval.py \
      --pretrained_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/libero_atomic_skills_augmented_long_id2/1.0.0/openvla-7b+libero_atomic_skills_augmented_long_id2+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_long_id2--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt \
      --ID_eval_mode True \
      --init_files_path datasets/hdf5_datasets/atomic_local_demos_augmented/combined/full_by_language \
      --shift_position_std 0.02 \
      --shift_orientation_std 0.5235'"

# ---------- ID3 ----------
JOB_NAME_ID3="eval_openvla_atomic_skills_long_id3"
LOG_FILE_ID3="eval_atomic_skills_long_id3_%j.out"

sbatch \
  --nodelist=arcee.ib \
  --cpus-per-task=$CPUS_PER_TASK \
  --gpus=$GPUS \
  -o "$LOG_DIR/$LOG_FILE_ID3" \
  -J $JOB_NAME_ID3 \
  --wrap="zsh -lc 'source ~/.zshrc && conda activate openvla-oft && cd /mnt/arc/yygx/pkgs_baselines/openvla-oft && \
    python scripts/phase2/more_real_simplified_pose_shift_eval.py \
      --pretrained_checkpoint /mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/libero_atomic_skills_augmented_long_id3/1.0.0/openvla-7b+libero_atomic_skills_augmented_long_id3+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--atomic_skills_augmented_long_id3--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt \
      --ID_eval_mode True \
      --init_files_path datasets/hdf5_datasets/atomic_local_demos_augmented/combined/full_by_language \
      --shift_position_std 0.02 \
      --shift_orientation_std 0.5235'"
