# Pi0.5 Training Scripts

This directory contains training scripts for Pi0.5 LoRA fine-tuning on the atomic skills dataset.

## Available Scripts

### 1. A6000 Training (8 GPUs, 48GB each)
**Script**: `train_pi05_atomic_skills_wrist_only.sh`

```bash
bash shells/phase2_pi05/train_pi05_atomic_skills_wrist_only.sh
```

**Configuration**:
- Hardware: 8x A6000 (48GB VRAM each)
- Node: megatron.ib
- Default batch size: 64 per device
- Effective batch size: 512 (64 × 8)
- FSDP devices: 8
- Estimated time: 6-12 hours

**Usage with custom batch size**:
```bash
bash shells/phase2_pi05/train_pi05_atomic_skills_wrist_only.sh 32   # effective=256
bash shells/phase2_pi05/train_pi05_atomic_skills_wrist_only.sh 64   # effective=512 (default)
bash shells/phase2_pi05/train_pi05_atomic_skills_wrist_only.sh 128  # effective=1024
```

---

### 2. H100 Training (4 GPUs, 80GB each)
**Script**: `train_pi05_atomic_skills_wrist_only_h100.sh`

```bash
bash shells/phase2_pi05/train_pi05_atomic_skills_wrist_only_h100.sh
```

**Configuration**:
- Hardware: 4x H100 (80GB VRAM each)
- Partition: h100
- Default batch size: 64 per device
- Effective batch size: 256 (64 × 4)
- FSDP devices: 4
- Estimated time: 3-5 hours (**2-3x faster than A6000**)

**Recommended H100 settings**:
Edit the script and change `BATCH_SIZE=128` for better GPU utilization:
- Effective batch size: 512 (128 × 4)
- Memory usage: ~25GB / 80GB (still plenty of headroom)

---

## Quick Comparison

| Feature | A6000 Script | H100 Script |
|---------|-------------|-------------|
| **GPUs** | 8x A6000 | 4x H100 |
| **VRAM per GPU** | 48GB | 80GB |
| **Default Effective Batch** | 512 | 256 |
| **Recommended Batch Size** | 64 per device | 128 per device |
| **Speed** | Baseline | 2-3x faster |
| **Time (30k steps)** | 6-12 hours | 3-5 hours |
| **Queue Time** | Usually shorter | May be longer |

---

## Configuration Used

Both scripts use the same training config:
- **Config name**: `pi05_atomic_skills_wrist_only_closer_lora`
- **Model**: Pi0.5 with LoRA adapters
  - `paligemma_variant="gemma_2b_lora"`
  - `action_expert_variant="gemma_300m_lora"`
- **Dataset**: `atomic_skills_wrist_only_closer`
- **Camera**: Wrist camera only (base camera masked)
- **State**: 8D (EE pose + gripper)
- **Actions**: 7D (delta EE pose + gripper)
- **Training steps**: 30,000
- **Learning rate**: 5e-5 (cosine decay)

---

## Which Script to Use?

### Use **A6000** when:
- ✅ H100 partition is busy (long queue)
- ✅ Running overnight/weekend jobs
- ✅ Don't need results urgently
- ✅ Want to use default settings

### Use **H100** when:
- ✅ Need faster iteration (development/debugging)
- ✅ H100 partition is available
- ✅ Want to use larger batch sizes (128-256)
- ✅ Time-sensitive experiments

---

## Output Locations

### Checkpoints
```
externals/openpi/checkpoints/pi05_atomic_skills_wrist_only_closer_lora/
  ├── atomic_skills_wrist_closer_lora_bs64_8gpu/    # A6000
  └── atomic_skills_wrist_closer_lora_h100_bs64_4gpu/  # H100
```

### Logs
```
logs/pi05/
  ├── train_pi05_atomic_skills_wrist_lora_bs64_<job_id>.out      # A6000
  └── train_pi05_atomic_skills_wrist_lora_h100_bs64_<job_id>.out # H100
```

---

## Documentation

- **Training config details**: `prompts/from_ai/phase2_pi05/pi05_lora_training_config.md`
- **Quick start guide**: `prompts/from_ai/phase2_pi05/TRAINING_USAGE.md`
- **H100 specific guide**: `prompts/from_ai/phase2_pi05/H100_TRAINING_GUIDE.md`
- **Main training plan**: `prompts/from_ai/phase2_pi05/phase2_pi05_training_plan.md`

---

## Monitoring

### Check job status
```bash
# A6000
squeue -u $USER

# H100
squeue -u $USER --partition=h100
```

### View logs
```bash
# A6000
tail -f logs/pi05/train_pi05_atomic_skills_wrist_lora_bs64_*.out

# H100
tail -f logs/pi05/train_pi05_atomic_skills_wrist_lora_h100_bs64_*.out
```

### Check GPU usage
```bash
# A6000
ssh megatron.ib nvidia-smi

# H100 (find node from squeue first)
ssh <h100_node> nvidia-smi
```
