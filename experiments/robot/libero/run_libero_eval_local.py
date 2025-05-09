"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
import h5py
import pickle
import traceback
import re

import wandb


# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"
    LIBERO_LOCAL1 = "libero_local1"
    LIBERO_LOCAL2 = "libero_local2"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
    TaskSuite.LIBERO_LOCAL1: 400,  # longest training demo has 373 steps
    TaskSuite.LIBERO_LOCAL2: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # TODO - Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_local1"           # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 20                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "yygx"                       # Name of WandB entity
    wandb_project: str = "openvla-oft-libero-eval"   # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    #################################################################################################################
    # Locality
    #################################################################################################################
    local_demo_and_inits_path: str = "/mnt/arc/yygx/pkgs_baselines/openvla-oft/datasets/hdf5_datasets/local_demos_libero_90_openvla_no_noops_pre_3/"
                                                     # Path to save local demos and initial states
    wrist_only: bool = False                         # TODO: change to True while using it
    agent_only: bool = False                         # TODO: change to True while using it
    pro_only: bool = False                           # TODO: change to True while using it
    is_oss: bool = False                             # Whether to OSS happens TODO: change to True while using it



    # fmt: on







######################################################################################################################
################################################## yy added methods ##################################################

def set_local_inits(cfg, env, task_name):
    """
    - For task_name: KITCHEN_SCENE1_put_the_black_bowl_on_the_plate
    - obs.keys() = ['robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos',
    'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'agentview_image', 'robot0_eye_in_hand_image',
    'akita_black_bowl_1_pos', 'akita_black_bowl_1_quat', 'akita_black_bowl_1_to_robot0_eef_pos',
    'akita_black_bowl_1_to_robot0_eef_quat', 'plate_1_pos', 'plate_1_quat', 'plate_1_to_robot0_eef_pos',
    'plate_1_to_robot0_eef_quat', 'robot0_proprio-state', 'object-state']
    - useful obs keys: ['robot0_joint_pos', 'robot0_gripper_qpos', 'agentview_image', 'robot0_eye_in_hand_image']
    - [(7,), (7,), (7,), (7,), (3,), (4,), (2,), (2,), (256, 256, 3), (256, 256, 3), (3,), (4,), (3,), (4,), (3,), (4,),
    (3,), (4,), (39,), (28,)]
    - [7, 7, 7, 7, 3, 4, 2, 2, 3, 4, 3, 4, 3, 4, 3, 4, 39, 28]
    - At least these are in the env_state: ['robot0_joint_pos', 'robot0_joint_vel', 'robot0_gripper_qpos',
    'robot0_gripper_qvel', 'akita_black_bowl_1_pos', 'akita_black_bowl_1_quat', 'plate_1_pos', 'plate_1_quat']
    (i.e., robot joint pos/vel + gripper pos/vel + obj pos/vel)

    Debug:
    - I found that even if I set the joint/gripper vel as 0, the gripper still rotate for a bit angel. My guess: this is
    caused by gravity or sth. in sim.
    """
    if cfg.is_oss:
        task_name = task_name.split("_with_")[0]
        init_path = os.path.join(cfg.local_demo_and_inits_path, "oss_local_init", f"{task_name}_local_init.init")
    else:
        init_path = os.path.join(cfg.local_demo_and_inits_path, f"{task_name}_local_init.init")
    with open(init_path, 'rb') as f:
        all_states = pickle.load(f)  # shape: [num_demos, 9+*]
    if len(all_states) == 0:
        return None
    idx = np.random.randint(len(all_states))
    sim_state = all_states[idx][9:]  # states - containing all the information, including objects placements/robot proprio

    # sampled_robot_local_state = all_states[idx]  # shape: [9,]
    #
    # sim_state = env.get_sim_state().copy()  # sim_state -> [77,]
    # sim_state[1:10] = sampled_robot_local_state
    # # set joint and gripper vel as 0
    # sim_state[34:36] = np.zeros([2])  # gripper vel
    # sim_state[27:34] = np.zeros([7])  # joint vel

    obs = env.set_init_state(sim_state)
    return obs

# yy: commented cuz not used
# def obtain_local_pose(task_name):
#     """
#     Extracts the initial joint_states and gripper_states from all demos
#     and saves them into an `.init` file for the given task.
#     Resulting state dim: 9 (7 joint + 2 gripper)
#     ---
#     Important information:
#     1. gripper open -> close: -1 -> 1
#     2. to access action: data_dict['data']['demo_0']['actions'] [116, 7]
#     3. to access obs: data_dict['data']['demo_0']['obs']['agentview_rgb'] [116, 256, 256, 3]
#     4. to access obs: data_dict['data']['demo_0']['obs']['eye_in_hand_rgb'] [116, 256, 256, 3]
#     5. to access ee: data_dict['data']['demo_0']['obs']['ee_pos'] [116, 3]; data_dict['data']['demo_0']['obs']['ee_ori'] [116, 3]
#     6. to access joint: data_dict['data']['demo_0']['obs']['joint_states'] [116, 7]
#     6. to access gripper: data_dict['data']['demo_0']['obs']['gripper_states'] [116, 2]
#     """
#     base_path = "/mnt/arc/yygx/pkgs_baselines/LIBERO/libero/datasets/hdf5_datasets/local_demos_libero_90_openvla_no_noops_pre_3"
#     demo_path = os.path.join(base_path, f"{task_name}_demo.hdf5")
#     init_save_path = os.path.join("local_inits", f"{task_name}_local_init.init")
#     os.makedirs("local_inits", exist_ok=True)
#
#     # Skip if already exists
#     if os.path.exists(init_save_path):
#         print(f"Init already exists for {task_name}, skipping.")
#         return
#
#     all_states = []
#
#     with h5py.File(demo_path, 'r') as f:
#         for demo_key in f['data'].keys():
#             obs = f['data'][demo_key]['obs']
#             joint = obs['joint_states'][0]  # shape [7]
#             gripper = obs['gripper_states'][0]  # shape [2]
#             state = np.concatenate([joint, gripper])  # shape [9]
#             all_states.append(state)
#
#     all_states = np.array(all_states)  # shape [num_demos, 9]
#
#     # Save as a pickle file
#     with open(init_save_path, 'wb') as f:
#         pickle.dump(all_states, f)
#
#     print(f"Saved initial states ({all_states.shape[0]} demos) to {init_save_path}")


# yy: create new folders
def get_eval_results_folder(cfg):
    base = Path("./runs/eval_results")
    version = 1

    match = re.search(r"/(\d+\.\d+\.\d+)/.*?--(\d+)_chkpt/?$", cfg.pretrained_checkpoint)
    if match:
        ckpt_version = match.group(1)
        ckpt_steps = match.group(2)
    else:
        raise ValueError(f"Failed to parse checkpoint path: {cfg.pretrained_checkpoint}")

    while True:
        folder = base / cfg.task_suite_name / f"ckpt_{ckpt_version}_{ckpt_steps}" \
                 f"_wrist_only_{cfg.wrist_only}_agent_only_{cfg.agent_only}_is_oss_{cfg.is_oss}_v{version}"
        if not folder.exists():
            break
        version += 1

    folder.mkdir(parents=True, exist_ok=True)
    return folder



# yy: save succ info
def save_all_success_rates_to(results_folder, task_success_rates):
    path = results_folder / "success_rates.pkl"
    with open(path, "wb") as f:
        pickle.dump(task_success_rates, f)
    print(f"[✔] Saved success rates to: {path}")


# yy: save rollouts for vis
def maybe_save_rollout_obs(
        obs_list,
        actions,
        task_name: str,
        episode_idx: int,
        success: bool,
        results_folder: Path,
        success_count: int,
        failure_count: int,
        max_failures=3,
        max_successes=1
):
    if success and success_count >= max_successes:
        return success_count, failure_count
    if not success and failure_count >= max_failures:
        return success_count, failure_count

    # Extract fields
    joint_states = np.array([o["robot0_joint_pos"] for o in obs_list])
    gripper_states = np.array([o["robot0_gripper_qpos"] for o in obs_list])
    agent_imgs = np.array([o["agentview_image"] for o in obs_list])
    wrist_imgs = np.array([o["robot0_eye_in_hand_image"] for o in obs_list])
    actions_np = np.array(actions)

    task_dir = results_folder / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    tag = "succ" if success else "fail"
    file_path = task_dir / f"{tag}_{episode_idx}.npz"

    np.savez_compressed(
        file_path,
        joint_states=joint_states,
        gripper_states=gripper_states,
        agentview_image=agent_imgs,
        wrist_image=wrist_imgs,
        actions=actions_np,
        success=success,
    )
    print(f"[💾] Saved rollout to: {file_path}")

    if success:
        success_count += 1
    else:
        failure_count += 1
    return success_count, failure_count


    def generate_gaussian_noise_image(resize_size):
        """Generate a grayscale Gaussian noise image with values in [0, 255]."""
        noise = np.random.normal(loc=127.5, scale=50.0, size=(resize_size, resize_size, 3))
        noise = np.clip(noise, 0, 255).astype(np.uint8)
        return noise

################################################## yy added methods ##################################################
######################################################################################################################


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)
    print(f"[INFO] model.norm_stats: {model.norm_stats}; cfg.task_suite_name: {cfg.task_suite_name};")
    """
    printed result：
    {'libero_local1': 
        {'action': 
            {'mean': [0.002507842378690839, -0.15118049085140228, 0.02066560834646225, 0.002950476948171854, 0.005715579725801945, -0.0015340133104473352, 0.43849873542785645], 
            'std': [0.21942110359668732, 0.524631679058075, 0.5088357925415039, 0.0518932044506073, 0.058938167989254, 0.04254530742764473, 0.4962540566921234], 
            'max': [0.7875000238418579, 0.918749988079071, 0.9375, 0.23678570985794067, 0.24321427941322327, 0.2442857176065445, 1.0], 
            'min': [-0.8464285731315613, -0.9375, -0.9375, -0.21642857789993286, -0.29571428894996643, -0.23035714030265808, 0.0], 
            'q01': [-0.5330356955528259, -0.9375, -0.9241071343421936, -0.1339285671710968, -0.1574999988079071, -0.11142857372760773, 0.0], 
            'q99': [0.5383928418159485, 0.8223214149475098, 0.9375, 0.1371428519487381, 0.15857142210006714, 0.13285714387893677, 1.0], 
            'mask': [True, True, True, True, True, True, False]}, 
        'proprio': 
            {'mean': [-0.015013529919087887, 0.024858880788087845, 1.075886845588684, 3.0118942260742188, -0.07162269949913025, -0.05059882253408432, 0.02390718087553978, -0.024048639461398125], 
            'std': [0.04551268741488457, 0.1278114765882492, 0.10391157120466232, 0.15138517320156097, 0.2320321649312973, 0.2017555683851242, 0.014469162560999393, 0.014209154061973095], 
            'max': [0.1261719912290573, 0.3274403214454651, 1.2957615852355957, 3.485107660293579, 0.8342941999435425, 0.6297973990440369, 0.04103553295135498, 0.0005067858728580177], 
            'min': [-0.21906019747257233, -0.20677441358566284, 0.908033549785614, 2.4297289848327637, -1.2787442207336426, -0.729426383972168, -0.0010300194844603539, -0.04060375690460205], 
            'q01': [-0.13745781242847444, -0.1743734085559845, 0.9096850061416626, 2.6473849964141847, -0.6773141145706176, -0.5456978797912597, 0.0015204141987487673, -0.040014040023088455], 
            'q99': [0.06525037258863453, 0.3018227553367615, 1.2476234674453737, 3.2997582054138186, 0.5103225588798526, 0.4014424371719362, 0.040100044161081316, -0.002178712487220764]}, 
        'num_transitions': 27097, 'num_trajectories': 237}}
    """

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-IS_OSS_{cfg.is_oss}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


# yy: modify to adapt to only wrist_camera view
def prepare_observation(obs, resize_size, cfg):
    """Prepare observation for policy input."""
    if cfg.wrist_only:
        # Get preprocessed images
        wrist_img = get_libero_wrist_image(obs)

        # Resize images to size expected by model
        wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

        # Prepare observations dict
        observation = {
            "full_image": wrist_img_resized,
            "state": np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
        }

        img = wrist_img
    elif cfg.agent_only:
        # Get preprocessed images
        img = get_libero_image(obs)

        # Resize images to size expected by model
        img_resized = resize_image_for_policy(img, resize_size)

        # Prepare observations dict
        observation = {
            "full_image": img_resized,
            "state": np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
        }
    else:
        # Get preprocessed images
        img = get_libero_image(obs)
        wrist_img = get_libero_wrist_image(obs)

        # Resize images to size expected by model
        img_resized = resize_image_for_policy(img, resize_size)
        wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

        # Prepare observations dict
        observation = {
            "full_image": img_resized,
            "wrist_image": wrist_img_resized,
            "state": np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
        }

    if cfg.pro_only:
        noise_img = generate_gaussian_noise_image(resize_size)
        observation["full_image"] = noise_img
        if "wrist_image" in observation:
            observation["wrist_image"] = noise_img
        img = noise_img


    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_name: str,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    # yy: could delete these cuz no need to run dummy actions
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()



    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
               "{NUM_ACTIONS_CHUNK} constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    all_obs, all_actions = [], []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    try:
        # yy: I delete these cuz no need to run dummy actions
        # while t < max_steps + cfg.num_steps_wait:
        #     all_obs.append(obs)
        #
        #     # Do nothing for the first few timesteps to let objects stabilize
        #     if t < cfg.num_steps_wait:
        #         obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
        #         t += 1
        #
        #         if t == cfg.num_steps_wait:
        #             # yy: special initialization: make robot move to local pose
        #             # yy: I make this set_pose after dummy actions, cuz dummy actions can cause a bit movement.
        #             if cfg.is_oss:
        #                 task_name = task_name.split("_with_")[0]
        #             obs = set_robot_local_pose(cfg, env, task_name)
        #         continue

        obs = set_local_inits(cfg, env, task_name)
        if obs is None:
            print("[WARNING] This task doesn't contain local init!!")
            return None, None, None, None  # this means this task doesn't contain local init
        while t < max_steps:
            all_obs.append(obs)

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size, cfg)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())

            all_actions.append(action.tolist())

            if done:
                success = True
                break
            t += 1

    except Exception as e:
        # log_message(f"Episode error: {e}", log_file)
        error_details = traceback.format_exc()
        log_message(f"Episode error:\n{error_details}", log_file)
        exit(1)

    # return success, replay_images
    return success, replay_images, all_obs, all_actions



def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    results_folder,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    saved_successes, saved_failures = 0, 0

    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    task_name = task.name

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx % len(initial_states)]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images, obs_list, actions = run_episode(
            cfg,
            env,
            task_name,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
        )

        # yy: jump the task if it doesn't contain local init
        if success is None and replay_images is None:
            task_episodes = None
            break


        # Save rollout for rerun (via encapsulated maybe_save_rollout_obs(...) )
        saved_successes, saved_failures = maybe_save_rollout_obs(
            obs_list,
            actions,
            task_name,
            episode_idx,
            success,
            results_folder,
            saved_successes,
            saved_failures
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    if task_episodes is None:
        task_success_rate = -1
    else:
        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes, task_success_rate


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""

    # yy: modify config based on existing param
    if cfg.wrist_only or cfg.agent_only:
        cfg.num_images_in_input = 1

    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    if cfg.is_oss:
        task_suite = benchmark_dict[f"ch1_{cfg.task_suite_name}"]()
    else:
        task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # # yy: save all local inits
    # for task_id in range(num_tasks):
    #     task_name = task_suite.get_task(task_id).name
    #     if cfg.is_oss:
    #         task_name = task_name.split("_with_")[0]
    #     obtain_local_pose(task_name)

    # yy: pre-check whether the tasks are all in boss-ch1
    if cfg.is_oss:
        task_id_ls = []
        for task_id in range(num_tasks):
            task_name = task_suite.get_task(task_id).name.split("_with_")[0]
            init_path = os.path.join(cfg.local_demo_and_inits_path, "oss_local_init", f"{task_name}_local_init.init")
            if not os.path.exists(init_path):
                print(f"[ERROR] {task_suite.get_task(task_id).name} does not have corresponding oss local init.")
            else:
                task_id_ls.append(task_id)
    else:
        task_id_ls = list(range(num_tasks))

    results_folder = get_eval_results_folder(cfg)
    task_success_rates = {}

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_id_ls):
        total_episodes, total_successes, task_success_rate = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            results_folder,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
        )

        task_name = task_suite.get_task(task_id).name
        task_success_rates[task_name] = task_success_rate

    # Save succ info
    save_all_success_rates_to(results_folder, task_success_rates)

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
