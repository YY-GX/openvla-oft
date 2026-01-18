# VLA Real Robot Deployment - Context & Roadmap

## System Overview

### Hardware Setup
We have a Franka Panda robot controlled through a 2-PC architecture:

**PC1 (Ubuntu 20.04):**
- Connected to 2 ZED cameras for vision
- Connected to school network (for cluster access)
- Connected to PC2 (NUC) via local network
- Runs Docker

**PC2 (NUC, Ubuntu 22.04):**
- Connected to Franka Panda arm + gripper
- Runs Polymetis for robot control
- Runs Docker
- Not directly accessible from outside network

### Current Goal
Deploy a Vision-Language-Action (VLA) model trained on this codebase to control the real robot.

---

## Architecture Design

### Data Flow
```
ZED Cameras (PC1) ──┐
                    ├─► Middleware (PC1) ──► VLA Server (Cluster)
Robot State (NUC) ──┘                              │
                                                   │
                    Actions                        │
                      ▲                            │
                      │                            │
                      └────────────────────────────┘
                      │
                      ▼
              Polymetis (NUC) ──► Robot
```

### Middleware Responsibilities (PC1)
1. **Observation Collection:**
   - Capture ZED camera images (2 cameras at 10Hz)
   - Receive robot state from NUC (joint positions, gripper state at 20Hz)
   - Combine into observation dictionary

2. **VLA Communication:**
   - Send observations to VLA server on cluster
   - Receive action chunks (e.g., 8 actions at once)
   - Maintain action buffer

3. **Action Execution:**
   - Send actions to NUC at 20Hz (one action per control cycle)
   - Handle replanning (request new action chunk every N steps)

### Control Frequencies
- **Robot Control (Polymetis):** 20Hz (standard for Franka)
- **Middleware Main Loop:** 20Hz (matches robot)
- **VLA Replanning:** Every 5 steps (~4Hz, similar to OpenPI)
- **Camera Capture:** 10Hz

---

## Implementation Roadmap

### Phase 1: Component Testing (Week 1)

#### 1.1 NUC → PC1 State Communication
**Goal:** Establish reliable robot state streaming from NUC to PC1

**Tasks:**
- [ ] Create state publisher on NUC (sends dummy 7D joint + 2D gripper state)
- [ ] Create state subscriber on PC1 (receives and logs state)
- [ ] Run at 20Hz for 5 minutes, measure:
  - Average latency
  - Packet drop rate
  - Timing jitter

**Success Criteria:**
- Latency < 10ms
- Zero packet drops
- Stable 20Hz reception

---

#### 1.2 ZED Camera Capture (PC1)
**Goal:** Capture and preprocess camera images reliably

**Tasks:**
- [ ] Initialize 2 ZED cameras
- [ ] Capture at 10Hz, resize to 224x224
- [ ] Apply 180° rotation (match training preprocessing)
- [ ] Log FPS and timing

**Success Criteria:**
- Stable 10Hz capture from both cameras
- No frame drops over 5 minutes
- Images visually correct (not upside down)

---

#### 1.3 Dummy VLA Server (PC1 or Cluster)
**Goal:** Test VLA communication with fixed outputs

**Tasks:**
- [ ] Create dummy VLA server (websocket or REST API)
- [ ] Server returns fixed 8-action chunk on request
- [ ] Client sends dummy observation, receives actions
- [ ] Measure round-trip latency

**Success Criteria:**
- Round-trip < 100ms (local) or < 500ms (cluster)
- Stable communication over 100 requests

---

#### 1.4 PC1 → NUC Action Communication
**Goal:** Send actions from PC1 to NUC reliably

**Tasks:**
- [ ] Create action publisher on PC1
- [ ] Create action receiver on NUC (logs actions, no robot control yet)
- [ ] Send dummy actions at 20Hz for 5 minutes

**Success Criteria:**
- All actions received in order
- Timing matches 20Hz (50ms intervals)
- No buffer overruns

---

### Phase 2: Integration Testing (Week 2)

#### 2.1 Full Pipeline with Dummy VLA
**Goal:** Test complete pipeline without real VLA

**Architecture:**
```python
# Middleware main loop (20Hz)
while True:
    # 1. Get observations
    images = capture_zed_cameras()  # 2 cameras
    robot_state = receive_from_nuc()  # 7D joint + 2D gripper

    # 2. Build observation dict
    obs = {
        "observation/image": images[0],
        "observation/wrist_image": images[1],
        "observation/state": robot_state,
        "prompt": "pick up the red block"
    }

    # 3. Get actions (replan every 5 steps)
    if action_buffer.empty() or step % 5 == 0:
        action_chunk = vla_client.infer(obs)  # Returns 8 actions
        action_buffer.extend(action_chunk)

    # 4. Execute one action
    action = action_buffer.pop_left()
    send_action_to_nuc(action)

    sleep_until_next_cycle()  # Maintain 20Hz
    step += 1
```

**Tasks:**
- [ ] Implement middleware main loop
- [ ] Test with dummy VLA (returns fixed actions)
- [ ] NUC receives actions but doesn't move robot yet
- [ ] Log timing statistics

**Success Criteria:**
- Main loop runs stably at 20Hz
- VLA queries happen every 5 steps
- Action buffer never underflows
- All components synchronized

---

#### 2.2 Safe Robot Motion Test
**Goal:** Execute dummy actions on real robot safely

**Tasks:**
- [ ] Set robot to safe starting position
- [ ] Enable Polymetis control with dummy actions
- [ ] Execute simple motion (e.g., sine wave joint motion)
- [ ] Verify smooth execution
- [ ] Test emergency stop

**Success Criteria:**
- Robot moves smoothly at 20Hz
- No jerky motions or jumps
- Emergency stop works reliably
- Can safely stop and restart

---

### Phase 3: Real VLA Deployment (Week 3)

#### 3.1 VLA Server Setup
**Goal:** Deploy trained VLA model on cluster

**Tasks:**
- [ ] Choose checkpoint (e.g., `runs/libero_above_atomic_long_id10/...`)
- [ ] Create VLA server script (adapt from OpenPI's `serve_policy.py`)
- [ ] Deploy on cluster with GPU
- [ ] Test inference speed with dummy observations

**Success Criteria:**
- VLA loads successfully
- Inference time < 200ms per query
- Server handles concurrent requests
- Server auto-restarts on crash

---

#### 3.2 End-to-End Task Execution
**Goal:** Complete real-world task with VLA

**Tasks:**
- [ ] Connect middleware to real VLA server
- [ ] Set up simple task (e.g., "pick up red block")
- [ ] Place object in workspace
- [ ] Execute full episode
- [ ] Record video from ZED cameras

**Success Criteria:**
- Robot completes task successfully (>50% success over 10 trials)
- No safety violations
- Smooth motion throughout episode
- System recovers from failures gracefully

---

## Docker Considerations

### Recommendation: Start Middleware Outside Docker

**Reasons:**
1. Camera access (ZED SDK) easier outside Docker
2. Network debugging simpler
3. Faster iteration during development

**If using Docker:**
```bash
# Use host networking for simplicity
docker run --network=host \
           --device=/dev/video0 \
           --device=/dev/video1 \
           --gpus all \
           my-middleware
```

### NUC Docker
Keep Polymetis in Docker as currently working. Expose ports for action receiver.

---

## Key Technical Details

### Observation Format (to VLA)
```python
{
    "observation/image": np.ndarray (224, 224, 3),      # Primary camera
    "observation/wrist_image": np.ndarray (224, 224, 3), # Wrist camera
    "observation/state": np.ndarray (9,),                # 7D joint + 2D gripper
    "prompt": str                                        # Language instruction
}
```

### Action Format (from VLA)
```python
{
    "actions": np.ndarray (8, 7)  # 8 timesteps, 7D action (6D pose + 1D gripper)
}
```

### Coordinate Frame Matching
- **Training data:** Uses specific robot base frame and EEF conventions
- **Real robot:** Ensure Polymetis outputs match training data format
- **Common issues:**
  - Gripper convention: -1 (open) vs +1 (close)
  - EEF orientation: quaternion vs euler angles
  - Action space: absolute vs delta

---

## Communication Protocols

### Recommended: ZeroMQ or gRPC

**ZeroMQ (Simpler):**
```python
# NUC: State publisher
import zmq
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")

while True:
    state = get_robot_state()
    socket.send_pyobj(state)
    time.sleep(0.05)  # 20Hz
```

```python
# PC1: State subscriber
socket = context.socket(zmq.SUB)
socket.connect("tcp://nuc_ip:5555")
socket.setsockopt(zmq.SUBSCRIBE, b'')

state = socket.recv_pyobj()
```

**VLA Server:** Use WebSocket (like OpenPI) or HTTP REST API

---

## Safety Considerations

1. **Joint Limits:** Always check action validity before sending to robot
2. **Velocity Limits:** Clip actions to safe velocities
3. **Emergency Stop:** Implement hardware e-stop and software kill switch
4. **Collision Avoidance:** Monitor force/torque sensors
5. **Workspace Bounds:** Reject actions that move outside safe zone
6. **Watchdog Timer:** Stop robot if no commands received for >100ms

---

## Testing Checklist

Before each phase:
- [ ] Verify all network connections
- [ ] Check camera feeds are live
- [ ] Confirm robot is in safe starting position
- [ ] Test emergency stop
- [ ] Clear workspace of obstacles
- [ ] Have someone ready to hit physical e-stop

---

## Debugging Tips

**High Latency:**
- Check network bandwidth (use `iperf3`)
- Profile VLA inference time
- Monitor CPU/GPU usage

**Jerky Robot Motion:**
- Verify 20Hz control loop timing
- Check for action buffer underflow
- Ensure actions are smooth (no jumps)

**Camera Issues:**
- Check ZED SDK version
- Verify USB3 connection
- Test camera independently first

**State Mismatch:**
- Log both sent and received states
- Check for coordinate frame differences
- Verify timestamp synchronization

---

## Reference Code Locations

- **OpenPI Server Example:** `externals/openpi/scripts/serve_policy.py`
- **OpenPI Client Example:** `externals/openpi/examples/libero/main.py`
- **VLA Training Code:** `vla-scripts/finetune.py`
- **Robot Utils:** Check for any existing Polymetis integration code

---

## Next Steps After Roadmap

1. Collect real-world evaluation data
2. Fine-tune VLA on failure cases
3. Implement online learning/adaptation
4. Scale to more complex tasks
5. Add multi-step reasoning

---

## Questions to Resolve

1. What is the exact action space of the real robot?
   - Absolute joint positions? Delta positions? EEF pose?

2. Which trained checkpoint to use?
   - Wrist camera only? Both views? Mask augmentation?

3. What tasks to test first?
   - Pick-and-place? Push? More complex?

4. Network topology:
   - Can cluster directly reach NUC? Or must route through PC1?

5. Gripper control:
   - Binary open/close? Continuous position? Force control?

---

**Generated:** 2025-12-17
**For:** VLA real robot deployment on Franka Panda
**System:** 2-PC setup with Polymetis + ZED cameras
