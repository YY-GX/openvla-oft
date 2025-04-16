#!/bin/zsh

# No OSS
#CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval_local.py \
#  --pretrained_checkpoint "/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/view_wrist/1.0.1/openvla-7b+libero_local2+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--20000_chkpt" \
#  --task_suite_name "libero_local2" \
#  --wrist_only True &
#
#CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval_local.py \
#  --pretrained_checkpoint "/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/view_wrist/1.0.1/openvla-7b+libero_local2+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt" \
#  --task_suite_name "libero_local2" \
#  --wrist_only True &

#CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval_local.py \
#  --pretrained_checkpoint "/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/view_wrist/1.0.1/openvla-7b+libero_local2+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt" \
#  --task_suite_name "libero_local2" \
#  --wrist_only True &
#
#CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval_local.py \
#  --pretrained_checkpoint "/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/view_wrist/1.0.1/openvla-7b+libero_local2+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--150000_chkpt" \
#  --task_suite_name "libero_local2" \
#  --wrist_only True &

# With OSS
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval_local.py \
  --pretrained_checkpoint "/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/view_wrist/1.0.1/openvla-7b+libero_local2+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--20000_chkpt" \
  --task_suite_name "libero_local2" \
  --wrist_only True \
  --is_oss True &

CUDA_VISIBLE_DEVICES=1 python experiments/robot/libero/run_libero_eval_local.py \
  --pretrained_checkpoint "/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/view_wrist/1.0.1/openvla-7b+libero_local2+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--50000_chkpt" \
  --task_suite_name "libero_local2" \
  --wrist_only True \
  --is_oss True &

CUDA_VISIBLE_DEVICES=2 python experiments/robot/libero/run_libero_eval_local.py \
  --pretrained_checkpoint "/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/view_wrist/1.0.1/openvla-7b+libero_local2+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--100000_chkpt" \
  --task_suite_name "libero_local2" \
  --wrist_only True \
  --is_oss True &

CUDA_VISIBLE_DEVICES=3 python experiments/robot/libero/run_libero_eval_local.py \
  --pretrained_checkpoint "/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/view_wrist/1.0.1/openvla-7b+libero_local2+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--wrist_img--proprio_state--150000_chkpt" \
  --task_suite_name "libero_local2" \
  --wrist_only True \
  --is_oss True &

# Wait for both processes to finish
wait


