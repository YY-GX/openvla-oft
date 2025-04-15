#!/bin/zsh

# No OSS
CUDA_VISIBLE_DEVICES=4 python experiments/robot/libero/run_libero_eval_local.py \
  --pretrained_checkpoint "/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/view_2/1.0.1/openvla-7b+libero_local2+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--20000_chkpt" \
  --task_suite_name "libero_local2" \
  --wrist_only False &

CUDA_VISIBLE_DEVICES=5 python experiments/robot/libero/run_libero_eval_local.py \
  --pretrained_checkpoint "/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/view_2/1.0.1/openvla-7b+libero_local2+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--50000_chkpt" \
  --task_suite_name "libero_local2" \
  --wrist_only False &


# With OSS
CUDA_VISIBLE_DEVICES=6 python experiments/robot/libero/run_libero_eval_local.py \
  --pretrained_checkpoint "/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/view_2/1.0.1/openvla-7b+libero_local2+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--20000_chkpt" \
  --task_suite_name "libero_local2" \
  --wrist_only False \
  --is_oss True &

CUDA_VISIBLE_DEVICES=7 python experiments/robot/libero/run_libero_eval_local.py \
  --pretrained_checkpoint "/mnt/arc/yygx/pkgs_baselines/openvla-oft/runs/view_2/1.0.1/openvla-7b+libero_local2+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state--50000_chkpt" \
  --task_suite_name "libero_local2" \
  --wrist_only False \
  --is_oss True &

# Wait for both processes to finish
wait


