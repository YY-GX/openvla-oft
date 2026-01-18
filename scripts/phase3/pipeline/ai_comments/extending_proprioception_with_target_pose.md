# Extending Proprioception with Target Object Pose

**Author**: Claude Code Investigation
**Date**: 2025-12-10
**Purpose**: Add target object pose (7D) to proprioception input as baseline comparison method

---

## Executive Summary

### Goal
Extend proprioception from **8D** (EE pose + gripper) to **15D** (EE pose + gripper + target object pose) to provide the VLA model with explicit target object information as a baseline comparison.

### Key Finding: Architecture Already Supports This! ✅

The OpenVLA-OFT architecture is **already flexible** and supports variable proprioception dimensions. The `ProprioProjector` class accepts `proprio_dim` as a parameter and automatically adjusts its input dimension.

**No architecture modifications needed** - only data pipeline and configuration updates.

---

## Current Architecture Analysis

### 1. Proprioception Flow

**Training Path**:
```
HDF5 → RLDS Dataset (tfds) → Transform → ProprioProjector → LLM Embedding Space
```

**Inference Path**:
```
Env Observation → Proprio Construction → ProprioProjector → VLA Model
```

### 2. Current Proprioception (8D)

**Components**:
- **EE Position**: 3D (xyz)
- **EE Orientation**: 3D (euler angles)
- **Gripper State**: 2D

**Data Storage** (`*_dataset_builder.py:332`):
```python
'state': np.asarray(
    np.concatenate((states[i], gripper_states[i]), axis=-1),  # 6D + 2D = 8D
    np.float32
)
```

**Transform Split** (`transforms.py:839-840`):
```python
trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -2:]
```

### 3. ProprioProjector Architecture

**File**: `prismatic/models/projectors.py:6-24`

```python
class ProprioProjector(nn.Module):
    """Projects proprio state inputs into the LLM's embedding space."""

    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim  # ← FLEXIBLE INPUT DIMENSION

        # Two fully-connected layers
        self.fc1 = nn.Linear(self.proprio_dim, self.llm_dim, bias=True)  # Projects proprio to LLM space
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)      # Additional transformation
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor = None) -> torch.Tensor:
        # proprio: (bsz, proprio_dim) ← Works with ANY dimension
        projected_features = self.fc1(proprio)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features
```

**Key Insight**: The architecture uses **linear layers** with flexible input dimension, so it naturally supports any proprioception size.

### 4. Current Dimension Configuration

**Constants** (`prismatic/vla/constants.py:26-31`):
```python
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,  # ← Currently hardcoded to 8
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}
```

**Training** (`vla-scripts/finetune.py:922`):
```python
ProprioProjector(llm_dim=vla.module.llm_dim, proprio_dim=PROPRIO_DIM)  # Uses constant
```

**Deployment** (`experiments/robot/libero/run_libero_eval.py:154`):
```python
proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)  # Hardcoded
```

**Updated Deployment** (`scripts/phase3/pipeline/evaluation/evaluate_above.py:351`):
```python
self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, proprio_dim=8)  # Needs update
```

---

## Proposed Extension: 15D Proprioception

### New Dimension Breakdown

**Total: 15D = 8D (current) + 7D (target object pose)**

| Component | Dimension | Description |
|-----------|-----------|-------------|
| EE Position | 3D | Robot end-effector xyz position |
| EE Orientation | 3D | Robot end-effector euler angles |
| Gripper State | 2D | Gripper joint positions |
| **Target Object Position** | **3D** | **Target object xyz position** |
| **Target Object Orientation** | **4D** | **Target object quaternion (wxyz)** |

**Rationale for Quaternion**:
- MuJoCo natively stores orientations as quaternions (wxyz format)
- Quaternions avoid gimbal lock issues
- Consistent with simulator's internal representation

---

## Implementation Plan

### Phase 1: Data Preparation

#### Task 1.1: Modify Demo Generation Scripts

**Files to Update**:
- `scripts/phase3/pipeline/data_generation/generate_above_augmented_demos.py`
- `scripts/phase3/pipeline/data_generation/generate_recovery_demos.py`
- Any other demo generation scripts

**Changes**:
```python
# Add to demo collection loop
def save_demo_to_hdf5(demo_group, obs_history, action_history, env, target_object_name):
    # Existing observations
    agentview_images = [obs['agentview_image'] for obs in obs_history]
    wrist_images = [obs['robot0_eye_in_hand_image'] for obs in obs_history]
    ee_states = [obs['robot0_eef_pos'] + obs['robot0_eef_euler'] for obs in obs_history]  # 6D
    gripper_states = [obs['robot0_gripper_qpos'] for obs in obs_history]  # 2D

    # NEW: Collect target object poses
    target_object_poses = []
    target_body_id = env.sim.model.body_name2id(target_object_name)

    for _ in obs_history:
        # Get target object pose from simulator state
        target_pos = env.sim.data.body_xpos[target_body_id].copy()  # 3D (xyz)
        target_quat = env.sim.data.body_xquat[target_body_id].copy()  # 4D (wxyz - MuJoCo format)
        target_pose = np.concatenate([target_pos, target_quat], axis=-1)  # 7D
        target_object_poses.append(target_pose)

    # Save to HDF5
    demo_group.create_dataset("obs/ee_states", data=np.array(ee_states))
    demo_group.create_dataset("obs/gripper_states", data=np.array(gripper_states))
    demo_group.create_dataset("obs/target_object_pose", data=np.array(target_object_poses))  # NEW
    demo_group.create_dataset("obs/agentview_rgb", data=np.array(agentview_images))
    demo_group.create_dataset("obs/eye_in_hand_rgb", data=np.array(wrist_images))
    # ... other datasets
```

**Important Notes**:
- Target object name must be tracked throughout demo generation
- For pick skills: target = object to pick
- For place skills: target = receptacle/destination
- Ensure target_body_id is correct for hierarchical objects (e.g., use main body, not sub-parts)

#### Task 1.2: Update RLDS Dataset Builder

**File**: `externals/rlds_dataset_builder/LIBERO_Above_Atomic_Long_ID_10/LIBERO_Above_Atomic_Long_ID_10_dataset_builder.py`

**Change 1** - Load target object pose (line ~199):
```python
def _parse_example(episode_path, demo_id):
    with h5py.File(episode_path, "r") as F:
        demo_group = F['data'][f"demo_{demo_id}"]
        actions = demo_group["actions"][()]
        states = demo_group["obs"]["ee_states"][()]
        gripper_states = demo_group["obs"]["gripper_states"][()]
        joint_states = demo_group["obs"]["joint_states"][()]
        images = demo_group["obs"]["agentview_rgb"][()]
        wrist_images = demo_group["obs"]["eye_in_hand_rgb"][()]

        # NEW: Load target object pose
        target_object_poses = demo_group["obs"]["target_object_pose"][()]  # shape: (T, 7)
```

**Change 2** - Concatenate into 15D state (line ~332):
```python
# Assemble episode
episode = []
for i in range(actions.shape[0]):
    episode.append({
        'observation': {
            'image': images[i][::-1,::-1],
            'wrist_image': wrist_images[i][::-1,::-1],
            # NEW: Concatenate EE (6D) + gripper (2D) + target pose (7D) = 15D
            'state': np.asarray(
                np.concatenate((states[i], gripper_states[i], target_object_poses[i]), axis=-1),
                np.float32
            ),
            'joint_state': np.asarray(joint_states[i], dtype=np.float32),
        },
        'action': np.asarray(actions[i], dtype=np.float32),
        # ... rest of episode data
    })
```

**Change 3** - Update schema (line ~409):
```python
def _info(self) -> tfds.core.DatasetInfo:
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'steps': tfds.features.Dataset({
                'observation': tfds.features.FeaturesDict({
                    'image': tfds.features.Image(...),
                    'wrist_image': tfds.features.Image(...),
                    # UPDATED: Change shape from (8,) to (15,)
                    'state': tfds.features.Tensor(
                        shape=(15,),  # Changed from (8,)
                        dtype=np.float32,
                        doc='Robot state: EEF pose (6D) + gripper (2D) + target object pose (7D)',
                    ),
                    'joint_state': tfds.features.Tensor(...),
                }),
                # ... rest of schema
            }),
        })
    )
```

---

### Phase 2: Data Transform Updates

#### Task 2.1: Update Dataset Transform

**File**: `prismatic/vla/datasets/rlds/oxe/transforms.py`

**Change** - Split 15D state into components (line ~839):
```python
def libero_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # Gripper action processing (unchanged)
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        [trajectory["action"][:, :6], gripper_action],
        axis=1,
    )

    # Split 15D state into three components
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]       # First 6D
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, 6:8]  # Next 2D
    trajectory["observation"]["target_object_pose"] = trajectory["observation"]["state"][:, 8:15]  # Last 7D (NEW)

    return trajectory
```

#### Task 2.2: Update Dataset Config

**File**: `prismatic/vla/datasets/rlds/oxe/configs.py`

**Change** - Add target_object_pose to state_obs_keys (line ~816):
```python
"libero_above_atomic_long_id10": {
    "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    # UPDATED: Add "target_object_pose" to state_obs_keys
    "state_obs_keys": ["EEF_state", "gripper_state", "target_object_pose"],
    "state_encoding": StateEncoding.POS_EULER,
    "action_encoding": ActionEncoding.EEF_POS,
},
```

---

### Phase 3: Constants Update

#### Task 3.1: Update PROPRIO_DIM Constant

**File**: `prismatic/vla/constants.py`

**Change** - Increase PROPRIO_DIM (line ~29):
```python
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 15,  # Changed from 8 to 15 (8 + 7)
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}
```

**Important**: This change affects both training and inference if code uses the constant.

---

### Phase 4: Deployment Updates

#### Task 4.1: Update Evaluation Scripts

**Files to Update**:
1. `experiments/robot/libero/run_libero_eval.py`
2. `scripts/phase3/pipeline/evaluation/evaluate_above.py`
3. Any other deployment scripts

**Changes**:

**File 1**: `experiments/robot/libero/run_libero_eval.py` (line ~154):
```python
proprio_projector = get_proprio_projector(
    cfg,
    model.llm_dim,
    proprio_dim=15,  # Changed from 8 to 15
)
```

**File 2**: `scripts/phase3/pipeline/evaluation/evaluate_above.py` (line ~351):
```python
if cfg.use_proprio:
    self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, proprio_dim=15)  # Changed from 8
```

#### Task 4.2: Update Observation Construction

**Create or Update**: Helper function to construct 15D proprioception at inference time

**File**: `experiments/robot/libero/libero_utils.py` (add new function):

```python
def get_libero_proprio_with_target(obs, env, target_object_name: str) -> np.ndarray:
    """
    Extract 15D proprioception for OpenVLA-OFT inference.

    Args:
        obs: Observation dict from LIBERO environment
        env: LIBERO environment instance
        target_object_name: Name of target object (e.g., "akita_black_bowl_1_main")

    Returns:
        np.ndarray: 15D proprioception vector
            - EE position (3D): xyz
            - EE orientation (3D): euler angles
            - Gripper state (2D): gripper joint positions
            - Target object position (3D): xyz
            - Target object orientation (4D): quaternion (wxyz)
    """
    # Extract robot state (8D)
    ee_pos = obs['robot0_eef_pos']  # 3D
    ee_euler = obs['robot0_eef_euler']  # 3D
    gripper_state = obs['robot0_gripper_qpos']  # 2D

    # Extract target object pose (7D)
    target_body_id = env.sim.model.body_name2id(target_object_name)
    target_pos = env.sim.data.body_xpos[target_body_id].copy()  # 3D (xyz)
    target_quat = env.sim.data.body_xquat[target_body_id].copy()  # 4D (wxyz - MuJoCo format)

    # Concatenate: 3 + 3 + 2 + 3 + 4 = 15D
    proprio = np.concatenate([
        ee_pos,         # 3D
        ee_euler,       # 3D
        gripper_state,  # 2D
        target_pos,     # 3D
        target_quat     # 4D
    ], axis=-1)

    return proprio.astype(np.float32)
```

**Usage in Evaluation Scripts**:
```python
# In VLA execution loop
obs = env._get_observations()
target_object = skill_data['target_object']  # From skill config

# OLD: proprio = get_libero_proprio(obs)  # Returns 8D
# NEW:
proprio = get_libero_proprio_with_target(obs, env, target_object)  # Returns 15D
```

**Integration with `evaluate_above.py`**:

Add target object tracking to the `_execute_vla()` method:
```python
def _execute_vla(self, skill_language: str, target_object_name: str, max_steps: int = 85):
    """Execute skill using VLA policy."""
    from experiments.robot.libero.libero_utils import get_libero_proprio_with_target

    for step in range(max_steps):
        obs = self.env._get_observations()

        # Get wrist image
        wrist_image = self.get_libero_wrist_image(obs)

        # Get 15D proprioception (includes target object pose)
        proprio = get_libero_proprio_with_target(obs, self.env, target_object_name)

        # VLA inference (same as before)
        predicted_action_chunk = self.vla.predict_action(
            unnorm_key=self.cfg.unnorm_key,
            proprio_obs=proprio,  # Now 15D
            image_primary=wrist_image,
            language_instruction=skill_language,
        )
        # ... rest of execution
```

---

### Phase 5: Training and Validation

#### Task 5.1: Regenerate HDF5 Demos with Target Pose

**Command**:
```bash
# Run updated demo generation scripts
python scripts/phase3/pipeline/data_generation/generate_above_augmented_demos.py \
    --output_dir datasets/hdf5_datasets/atomic_above_with_target_pose/v1 \
    --num_demos_per_skill 100
```

**Validation**:
```python
import h5py

# Check HDF5 structure
with h5py.File("datasets/hdf5_datasets/.../pick_moka_pot.hdf5", "r") as f:
    demo = f['data']['demo_0']

    # Verify new field exists
    assert "obs/target_object_pose" in demo, "Missing target_object_pose!"

    target_pose = demo["obs/target_object_pose"][()]
    print(f"Target pose shape: {target_pose.shape}")  # Should be (T, 7)
    print(f"Target pose sample: {target_pose[0]}")     # [x, y, z, qw, qx, qy, qz]
```

#### Task 5.2: Rebuild RLDS Dataset

**Command**:
```bash
cd externals/rlds_dataset_builder/LIBERO_Above_Atomic_Long_ID_10

# Rebuild with new schema
tfds build --overwrite --data_dir /mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/rlds_datasets
```

**Validation**:
```python
import tensorflow_datasets as tfds

# Load dataset
ds = tfds.load('libero_above_atomic_long_id10:1.0.2', data_dir='datasets/rlds_datasets', split='train')

# Check first episode
for episode in ds.take(1):
    for step in episode['steps'].take(1):
        state = step['observation']['state']
        print(f"State shape: {state.shape}")  # Should be (15,)
        print(f"State sample: {state.numpy()[:15]}")

        # After transform
        # EEF_state: state[:6]
        # gripper_state: state[6:8]
        # target_object_pose: state[8:15]
```

#### Task 5.3: Train Model

**Update Training Script** (if needed):
- Verify `PROPRIO_DIM=15` is correctly detected from constants
- Update dataset name if using new version

**Command**:
```bash
# Use existing training script - no modifications needed
sbatch shells/phase3/train_oft_above_atomic_id_10_h100.sh
```

**Expected Training Log**:
```
Using LIBERO constants:
  NUM_ACTIONS_CHUNK = 8
  ACTION_DIM = 7
  PROPRIO_DIM = 15          ← Should show 15, not 8
  ...

Initializing ProprioProjector with:
  llm_dim = 4096
  proprio_dim = 15          ← Confirms 15D input
```

#### Task 5.4: Verify Trained Model

**Check Projector Weights**:
```python
import torch

checkpoint_path = "runs/libero_above_atomic_long_id10/1.0.2/checkpoints/step-100000-proprio_projector.pt"
state_dict = torch.load(checkpoint_path)

# Check input layer shape
fc1_weight = state_dict['fc1.weight']
print(f"FC1 input dim: {fc1_weight.shape[1]}")  # Should be 15
print(f"FC1 output dim: {fc1_weight.shape[0]}")  # Should be llm_dim (4096)
```

---

## File Summary

### Files to Modify (8 categories)

#### 1. Demo Generation Scripts
- `scripts/phase3/pipeline/data_generation/generate_above_augmented_demos.py`
- `scripts/phase3/pipeline/data_generation/generate_recovery_demos.py`
- Add: `obs/target_object_pose` field to HDF5

#### 2. RLDS Dataset Builder
- `externals/rlds_dataset_builder/LIBERO_Above_Atomic_Long_ID_10/LIBERO_Above_Atomic_Long_ID_10_dataset_builder.py`
- Changes: Load target pose, concatenate to 15D state, update schema

#### 3. Data Transform
- `prismatic/vla/datasets/rlds/oxe/transforms.py`
- Change: Split 15D state into 3 components

#### 4. Dataset Config
- `prismatic/vla/datasets/rlds/oxe/configs.py`
- Change: Add `"target_object_pose"` to `state_obs_keys`

#### 5. Constants
- `prismatic/vla/constants.py`
- Change: `PROPRIO_DIM: 8 → 15`

#### 6. Deployment Scripts
- `experiments/robot/libero/run_libero_eval.py`
- `scripts/phase3/pipeline/evaluation/evaluate_above.py`
- Change: `proprio_dim=8 → proprio_dim=15`

#### 7. Observation Construction
- `experiments/robot/libero/libero_utils.py`
- Add: `get_libero_proprio_with_target()` function

#### 8. Evaluation Integration
- `scripts/phase3/pipeline/evaluation/evaluate_above.py`
- Change: Pass `target_object_name` to VLA execution, use new proprio function

---

## Architecture Insights

### Why This Works Without Model Changes

**1. Linear Layers are Dimension-Agnostic**

The `ProprioProjector` uses `nn.Linear(proprio_dim, llm_dim)`, which creates a weight matrix of shape `(llm_dim, proprio_dim)`. When you initialize with a different `proprio_dim`, PyTorch automatically creates the correct sized matrix.

**Old Model** (8D):
```
FC1 weights: (4096, 8)
FC2 weights: (4096, 4096)
```

**New Model** (15D):
```
FC1 weights: (4096, 15)  ← Different size
FC2 weights: (4096, 4096)
```

**2. No Shared Weights**

The proprioception projector is **separate** from the vision backbone and LLM. It's initialized from scratch during training, so changing its input dimension doesn't affect other components.

**3. Training from Scratch**

When you train with the new 15D data:
- ProprioProjector is initialized with `proprio_dim=15`
- FC1 layer has 15 input neurons (one per dimension)
- Model learns to map all 15 dimensions to LLM space
- Vision backbone and LLM remain unchanged

---

## Dimension Tracking Checklist

Use this checklist to ensure dimensions are correct throughout the pipeline:

- [ ] **HDF5 Data**: `obs/target_object_pose` has shape `(T, 7)`
- [ ] **RLDS State**: `observation/state` has shape `(T, 15)`
- [ ] **Transform Output**:
  - [ ] `EEF_state`: shape `(T, 6)`
  - [ ] `gripper_state`: shape `(T, 2)`
  - [ ] `target_object_pose`: shape `(T, 7)`
- [ ] **Config**: `state_obs_keys = ["EEF_state", "gripper_state", "target_object_pose"]`
- [ ] **Constants**: `PROPRIO_DIM = 15`
- [ ] **Training Log**: Shows "proprio_dim = 15"
- [ ] **Deployment**: `proprio_dim=15` passed to `get_proprio_projector()`
- [ ] **Inference**: Proprio construction returns 15D array
- [ ] **Model Checkpoint**: FC1 weight shape is `(4096, 15)`

---

## Potential Issues and Solutions

### Issue 1: Dimension Mismatch During Inference

**Symptom**: Runtime error about tensor shape mismatch

**Cause**: Deployment code still constructing 8D proprio but model expects 15D

**Solution**:
- Verify `get_libero_proprio_with_target()` returns 15D
- Check all evaluation scripts use updated proprio_dim
- Print proprio shape before VLA inference to debug

### Issue 2: Target Object Name Not Available

**Symptom**: Cannot find target object body in simulator

**Cause**: Target object name not tracked or incorrect format

**Solution**:
- Ensure skill config contains `target_object` field
- For hierarchical objects, use main body name (e.g., `"flat_stove_2_main"`, not `"flat_stove_2_burner"`)
- Add fallback to extract from skill language description

### Issue 3: Quaternion Convention Mismatch

**Symptom**: Model performs poorly despite correct dimensions

**Cause**: Training data uses different quaternion format than inference

**Solution**:
- **Always use MuJoCo's native format**: `body_xquat` returns `(w, x, y, z)`
- **Don't convert** unless necessary
- Document quaternion format in code comments

### Issue 4: Training Data Has Wrong Dimension

**Symptom**: Training crashes with dimension error

**Cause**: RLDS dataset not rebuilt after schema change

**Solution**:
- Delete old dataset: `rm -rf datasets/rlds_datasets/libero_above_atomic_long_id10`
- Rebuild: `tfds build --overwrite`
- Verify state shape is (15,) before training

---

## Testing Plan

### Unit Tests

**Test 1**: HDF5 Data Integrity
```python
def test_hdf5_target_pose():
    with h5py.File("demo.hdf5", "r") as f:
        demo = f['data']['demo_0']
        target_pose = demo["obs/target_object_pose"][()]
        assert target_pose.shape[1] == 7, "Target pose must be 7D"
        assert not np.any(np.isnan(target_pose)), "Target pose contains NaN"
```

**Test 2**: RLDS State Dimension
```python
def test_rlds_state_dim():
    ds = tfds.load('libero_above_atomic_long_id10:1.0.2', split='train')
    for episode in ds.take(1):
        for step in episode['steps'].take(1):
            state = step['observation']['state']
            assert state.shape[0] == 15, f"Expected 15D state, got {state.shape[0]}D"
```

**Test 3**: Proprio Construction
```python
def test_proprio_construction():
    obs = env._get_observations()
    proprio = get_libero_proprio_with_target(obs, env, "akita_black_bowl_1_main")
    assert proprio.shape == (15,), f"Expected (15,), got {proprio.shape}"
    assert proprio.dtype == np.float32
```

### Integration Tests

**Test 4**: End-to-End Data Pipeline
```bash
# 1. Generate demo
python scripts/phase3/pipeline/data_generation/generate_above_augmented_demos.py --num_demos 1

# 2. Build RLDS
tfds build

# 3. Load and inspect
python -c "
import tensorflow_datasets as tfds
ds = tfds.load('libero_above_atomic_long_id10:1.0.2', split='train')
for ep in ds.take(1):
    for step in ep['steps'].take(1):
        print('State shape:', step['observation']['state'].shape)
"
```

**Test 5**: Training Initialization
```python
# Check model accepts 15D input
from prismatic.models.projectors import ProprioProjector

projector = ProprioProjector(llm_dim=4096, proprio_dim=15)
dummy_input = torch.randn(1, 15)
output = projector(dummy_input)
assert output.shape == (1, 4096), "Projector output shape incorrect"
```

---

## Baseline vs. Your Method Comparison

### Baseline (This Implementation)
- **Input**: 15D = EE pose (6D) + gripper (2D) + target pose (7D)
- **Data**: Original skill demos (non-augmented)
- **Hypothesis**: Explicit target pose helps model learn object-centric behaviors

### Your Method (Current)
- **Input**: 8D = EE pose (6D) + gripper (2D)
- **Data**: Augmented demos with pose shifts
- **Hypothesis**: Data augmentation improves generalization

### Experimental Design
1. Train both models with **same number of demos** (100 per skill)
2. Evaluate on **same test conditions**:
   - Clean (set_init): No perturbations
   - Object shift: Target object position shifted
   - Above shift: Initial EE position shifted
3. Compare **success rates** and **failure modes**

### Metrics to Track
- Overall task success rate
- Reaching accuracy (distance to target)
- Grasping success rate (for picks)
- Placement accuracy (for places)
- Generalization to unseen positions

---

## Migration Checklist

Use this checklist when implementing the changes:

### Data Preparation
- [ ] Update demo generation scripts to save target object pose
- [ ] Regenerate all HDF5 demos with new field
- [ ] Verify HDF5 structure with sample file inspection
- [ ] Update RLDS dataset builder (_parse_example function)
- [ ] Update RLDS schema (_info function)
- [ ] Rebuild RLDS dataset with new version number
- [ ] Validate RLDS dataset loading

### Code Updates
- [ ] Update libero_dataset_transform in transforms.py
- [ ] Update dataset config in configs.py
- [ ] Update PROPRIO_DIM in constants.py
- [ ] Add get_libero_proprio_with_target() to libero_utils.py
- [ ] Update run_libero_eval.py (proprio_dim parameter)
- [ ] Update evaluate_above.py (proprio_dim parameter)
- [ ] Update evaluate_above.py (_execute_vla method to use new proprio)

### Testing
- [ ] Test HDF5 data integrity
- [ ] Test RLDS dataset loading
- [ ] Test proprio construction at inference
- [ ] Test model initialization with 15D
- [ ] Run unit tests
- [ ] Run integration tests

### Training
- [ ] Launch training job
- [ ] Monitor training logs for correct dimension
- [ ] Verify model checkpoint shapes
- [ ] Save baseline model for comparison

### Deployment
- [ ] Test inference with trained model
- [ ] Compare performance with your augmented model
- [ ] Document results

---

## References

### Key Files Locations

**Data Pipeline**:
- HDF5 demos: `datasets/hdf5_datasets/atomic_above_fewer/`
- RLDS builder: `externals/rlds_dataset_builder/LIBERO_Above_Atomic_Long_ID_10/`
- RLDS dataset: `datasets/rlds_datasets/libero_above_atomic_long_id10/`

**Model Code**:
- ProprioProjector: `prismatic/models/projectors.py`
- Constants: `prismatic/vla/constants.py`
- Transforms: `prismatic/vla/datasets/rlds/oxe/transforms.py`
- Configs: `prismatic/vla/datasets/rlds/oxe/configs.py`

**Training**:
- Training script: `vla-scripts/finetune.py`
- Training launch: `shells/phase3/train_oft_above_atomic_id_10_h100.sh`

**Deployment**:
- Evaluation: `scripts/phase3/pipeline/evaluation/evaluate_above.py`
- Utilities: `experiments/robot/libero/libero_utils.py`
- OpenVLA utils: `experiments/robot/openvla_utils.py`

---

## Conclusion

The OpenVLA-OFT architecture is **already designed to support variable proprioception dimensions** through its flexible ProprioProjector class. Extending from 8D to 15D requires:

1. **Data changes**: Add target pose to HDF5 and RLDS datasets
2. **Config changes**: Update constants and dataset config
3. **Code changes**: Update data loading, transforms, and inference code

**No architecture modifications are needed** - the model will automatically adapt to the new dimension during training.

This baseline will provide a valuable comparison to your augmentation-based approach!
