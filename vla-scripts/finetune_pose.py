"""
finetune_pose.py

Fine-tunes PoseVLM for pose prediction from vision and language inputs.
"""

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    update_auto_map,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models import load_pose_vla
from prismatic.models.vlms.pose_vlm import PoseVLM
from prismatic.models.pose_heads import GMMPoseHead, SimplePoseHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import (
    NoisyActionProjector,
    ProprioProjector,
)
from prismatic.util.data_utils import PaddedCollatorForPosePrediction
from prismatic.vla.datasets.pose_dataset import create_pose_dataset
from prismatic.util.pose_augmentation import create_pose_augmentation

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class PoseFinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"             # Path to OpenVLA model (on HuggingFace Hub or stored locally)

    # Dataset
    data_root_dir: Path = Path("datasets/local_pairs_datasets")  # Directory containing pose datasets
    dataset_name: str = "pose_dataset"               # Name of fine-tuning dataset
    run_root_dir: Path = Path("runs")                # Path to directory to store logs & checkpoints
    shuffle_buffer_size: int = 100_000               # Dataloader shuffle buffer size (can reduce if OOM errors occur)

    # Algorithm and architecture
    pose_head_type: str = "gmm"                      # Type of pose head: "gmm" or "simple"
    gmm_num_components: int = 3                      # Number of GMM components (when pose_head_type="gmm")
    pose_dim: int = 6                                # Dimension of pose (3D position + 3D orientation)
    num_pose_tokens: int = 6                         # Number of pose tokens
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                        # If True, includes robot proprioceptive state in input (disabled for pose)
    max_length: int = 512                            # Maximum sequence length for text tokens

    # Training configuration
    batch_size: int = 8                              # Batch size per device (total batch size = batch_size * num GPUs)
    learning_rate: float = 5e-4                      # Learning rate
    lr_warmup_steps: int = 0                         # Number of steps to warm up learning rate (from 10%% to 100%%)
    num_steps_before_decay: int = 100_000            # Number of steps before LR decays by 10x
    grad_accumulation_steps: int = 1                 # Number of gradient accumulation steps
    max_steps: int = 200_000                         # Max number of training steps
    use_val_set: bool = False                        # If True, uses validation set and log validation metrics
    val_freq: int = 10_000                           # (When `use_val_set==True`) Validation set logging frequency in steps
    val_time_limit: int = 180                        # (When `use_val_set==True`) Time limit for computing validation metrics
    save_freq: int = 10_000                          # Checkpoint saving frequency in steps
    save_latest_checkpoint_only: bool = False        # If True, saves only 1 checkpoint, overwriting latest checkpoint
                                                     #   (If False, saves all checkpoints)
    resume: bool = False                             # If True, resumes from checkpoint
    resume_step: Optional[int] = None                # (When `resume==True`) Step number that we are resuming from
    image_aug: bool = True                           # If True, trains with image augmentations (HIGHLY RECOMMENDED)
    pose_aug: bool = True                            # If True, trains with pose augmentations
    pose_aug_position_std: float = 0.02              # Standard deviation for position augmentation (meters)
    pose_aug_orientation_std: float = 0.1            # Standard deviation for orientation augmentation (radians)

    # LoRA
    use_lora: bool = True                            # If True, uses LoRA fine-tuning
    lora_rank: int = 32                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                        # Dropout applied to LoRA weights
    merge_lora_during_training: bool = True          # If True, merges LoRA weights and saves result during training
                                                     #   Note: Merging can be very slow on some machines. If so, set to
                                                     #         False and merge final checkpoint offline!

    # Logging
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    run_id_override: Optional[str] = None            # Optional string to override the run ID with
    wandb_log_freq: int = 10                         # WandB logging frequency in steps

    # fmt: on


def remove_ddp_in_checkpoint(state_dict) -> dict:
    """
    Removes the 'module.' prefix from parameter names in a PyTorch model state dictionary that was saved using
    DistributedDataParallel (DDP).

    When a model is trained using PyTorch's DistributedDataParallel, the saved state dictionary contains parameters
    prefixed with 'module.'. This function removes these prefixes to make the state dictionary compatible when
    loading into models that are not yet wrapped in DDP.

    Args:
        state_dict (dict): PyTorch model state dictionary.

    Returns:
        dict: A new state dictionary with the same contents but with 'module.' prefixes removed from parameter names.
              Parameters without the 'module.' prefix remain unchanged.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_run_id(cfg) -> str:
    """
    Generates or retrieves an identifier string for an experiment run.

    Args:
        cfg (PoseFinetuneConfig): Training configuration.

    Returns:
        str: Experiment run ID.
    """
    if cfg.run_id_override is not None:
        # Override the run ID with the user-provided ID
        run_id = cfg.run_id_override
    elif cfg.resume:
        # Override run ID with the previous resumed run's ID
        run_id = cfg.vla_path.split("/")[-1]
        # Remove the "--XXX_chkpt" suffix from the run ID if it exists
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = (
            f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
            f"+pose_{cfg.pose_head_type}"
        )
        if cfg.pose_head_type == "gmm":
            run_id += f"_c{cfg.gmm_num_components}"
        if cfg.use_lora:
            run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.pose_aug:
            run_id += "--pose_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """
    Loads a checkpoint for a given module.

    Args:
        module_name (str): Name of model component to load checkpoint for.
        path (str): Path to checkpoint directory.
        step (int): Gradient step number of saved checkpoint.
        device (str): String specifying how to remap storage locations (default = "cpu").

    Returns:
        dict: PyTorch model state dictionary.
    """
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """
    Wraps a PyTorch module with DistributedDataParallel (DDP).

    Args:
        module (nn.Module): PyTorch module to wrap.
        device_id (int): Device ID to place the module on.
        find_unused (bool): Whether to find unused parameters in the module.

    Returns:
        DDP: Wrapped module.
    """
    module = module.to(device_id)
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused)


def count_parameters(module: nn.Module, name: str) -> None:
    """
    Counts and prints the number of parameters in a PyTorch module.

    Args:
        module (nn.Module): PyTorch module.
        name (str): Name of the module for logging.
    """
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"{name}: {total_params:,} total parameters, {trainable_params:,} trainable parameters")


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: PoseFinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    """
    Initializes a PyTorch module with DDP wrapping.

    Args:
        module_class (Type[nn.Module]): Class of the module to initialize.
        module_name (str): Name of the module.
        cfg (PoseFinetuneConfig): Training configuration.
        device_id (int): Device ID to place the module on.
        module_args (dict): Arguments to pass to the module constructor.
        to_bf16 (bool): Whether to convert the module to bfloat16.
        find_unused_params (bool): Whether to find unused parameters in DDP.

    Returns:
        DDP: Wrapped module.
    """
    module = module_class(**module_args)
    if to_bf16:
        module = module.to(torch.bfloat16)
    count_parameters(module, module_name)
    return wrap_ddp(module, device_id, find_unused_params)


def run_forward_pass(
    vla,
    pose_head,
    batch,
    device_id,
    cfg,
    use_film,
    num_patches,
    pose_augmentation=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Run a forward pass through the model.

    Args:
        vla: PoseVLM model.
        pose_head: Pose prediction head (not used, included for compatibility).
        batch: Input batch.
        device_id: Device ID.
        cfg: Training configuration.
        use_film: Whether to use FiLM (not used, included for compatibility).
        num_patches: Number of image patches (not used, included for compatibility).
        pose_augmentation: Pose augmentation object.

    Returns:
        Tuple[torch.Tensor, Dict[str, float]]: Loss and metrics.
    """
    # Extract batch components
    images = batch["pixel_values"].to(device_id)
    text = batch["input_ids"].to(device_id)
    text_attention_mask = batch["attention_mask"].to(device_id)
    target_poses = batch["pose_targets"].to(device_id)
    
    # Apply pose augmentation if enabled
    if pose_augmentation is not None and pose_augmentation.enabled:
        target_poses = pose_augmentation(target_poses)
    
    # Compute loss using PoseVLM's built-in method
    loss = vla.compute_loss(
        images=images,
        text=text,
        text_attention_mask=text_attention_mask,
        target_poses=target_poses,
    )
    
    # Compute additional metrics
    with torch.no_grad():
        # Get predictions for metrics
        predictions = vla.predict_pose(
            images=images,
            text=text,
            text_attention_mask=text_attention_mask,
            num_samples=1,
        )
        
        if cfg.pose_head_type == "gmm":
            sampled_poses = predictions['sampled_poses'].squeeze(1)
            l1_error = torch.mean(torch.abs(sampled_poses - target_poses))
            
            # Compute component weights entropy (diversity measure)
            weights = predictions['weights']
            entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1).mean()
            
            metrics = {
                "loss": loss.item(),
                "l1_error": l1_error.item(),
                "entropy": entropy.item(),
            }
        else:
            predicted_poses = predictions['predicted_poses']
            l1_error = torch.mean(torch.abs(predicted_poses - target_poses))
            metrics = {
                "loss": loss.item(),
                "l1_error": l1_error.item(),
            }
    
    return loss, metrics


def compute_smoothened_metrics(metrics_deques) -> dict:
    """
    Computes smoothed metrics from deques of metric values.

    Args:
        metrics_deques (dict): Dictionary of deques containing metric values.

    Returns:
        dict: Smoothed metrics.
    """
    smoothed_metrics = {}
    for metric_name, deque_obj in metrics_deques.items():
        if len(deque_obj) > 0:
            smoothed_metrics[metric_name] = sum(deque_obj) / len(deque_obj)
    return smoothed_metrics


def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    """
    Logs metrics to WandB.

    Args:
        metrics (dict): Metrics to log.
        prefix (str): Prefix for metric names.
        step (int): Current step.
        wandb_entity (str): WandB entity name.
    """
    if wandb_entity != "your-wandb-entity":
        log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()}
        log_dict[f"{prefix}/step"] = step
        wandb.log(log_dict)


def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    pose_head,
    train_dataset,
    distributed_state,
) -> None:
    """
    Saves a training checkpoint.

    Args:
        cfg (PoseFinetuneConfig): Training configuration.
        run_dir (Path): Directory to save checkpoint in.
        log_step (int): Current step.
        vla: OpenVLA model.
        processor: Data processor.
        pose_head: Pose prediction head.
        train_dataset: Training dataset.
        distributed_state: Distributed training state.
    """
    # Save VLA checkpoint
    if distributed_state.is_main_process:
        vla_checkpoint_path = run_dir / f"vla--{log_step}_checkpoint.pt"
        torch.save(
            {
                "model": vla.state_dict(),
                "step": log_step,
                "config": cfg,
            },
            vla_checkpoint_path,
        )
        print(f"Saved VLA checkpoint to {vla_checkpoint_path}")
    
    # Save pose head checkpoint
    if distributed_state.is_main_process:
        pose_head_checkpoint_path = run_dir / f"pose_head--{log_step}_checkpoint.pt"
        torch.save(
            {
                "model": pose_head.state_dict(),
                "step": log_step,
                "config": cfg,
            },
            pose_head_checkpoint_path,
        )
        print(f"Saved pose head checkpoint to {pose_head_checkpoint_path}")
    
    # Save processor
    if distributed_state.is_main_process:
        processor_checkpoint_path = run_dir / f"processor--{log_step}_checkpoint.pt"
        torch.save(
            {
                "processor": processor,
                "step": log_step,
                "config": cfg,
            },
            processor_checkpoint_path,
        )
        print(f"Saved processor checkpoint to {processor_checkpoint_path}")
    
    # Save latest checkpoint info
    if distributed_state.is_main_process:
        latest_checkpoint_path = run_dir / "latest_checkpoint.txt"
        with open(latest_checkpoint_path, "w") as f:
            f.write(f"{log_step}")
    
    # Remove old checkpoints if save_latest_checkpoint_only is True
    if cfg.save_latest_checkpoint_only and distributed_state.is_main_process:
        for checkpoint_file in run_dir.glob("*_checkpoint.pt"):
            if checkpoint_file.name != f"vla--{log_step}_checkpoint.pt" and \
               checkpoint_file.name != f"pose_head--{log_step}_checkpoint.pt" and \
               checkpoint_file.name != f"processor--{log_step}_checkpoint.pt":
                checkpoint_file.unlink()


def run_validation(
    vla,
    pose_head,
    val_dataloader,
    device_id,
    cfg,
    num_patches,
    log_step,
    distributed_state,
    val_time_limit,
    pose_augmentation=None,
) -> None:
    """
    Runs validation.

    Args:
        vla: OpenVLA model.
        pose_head: Pose prediction head.
        val_dataloader: Validation data loader.
        device_id: Device ID.
        cfg (PoseFinetuneConfig): Training configuration.
        num_patches: Number of image patches.
        log_step (int): Current step.
        distributed_state: Distributed training state.
        val_time_limit (int): Time limit for validation.
        pose_augmentation: Pose augmentation object.
    """
    vla.eval()
    pose_head.eval()
    
    val_metrics_deques = {
        "loss": deque(maxlen=100),
        "l1_error": deque(maxlen=100),
    }
    if cfg.pose_head_type == "gmm":
        val_metrics_deques["entropy"] = deque(maxlen=100)
    
    start_time = time.time()
    num_val_batches = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            if time.time() - start_time > val_time_limit:
                break
            
            loss, metrics = run_forward_pass(
                vla=vla,
                pose_head=pose_head,
                batch=batch,
                device_id=device_id,
                cfg=cfg,
                use_film=cfg.use_film,
                num_patches=num_patches,
                pose_augmentation=pose_augmentation,
            )
            
            for metric_name, value in metrics.items():
                if metric_name in val_metrics_deques:
                    val_metrics_deques[metric_name].append(value)
            
            num_val_batches += 1
    
    # Compute smoothed metrics
    val_metrics = compute_smoothened_metrics(val_metrics_deques)
    
    # Log validation metrics
    if distributed_state.is_main_process:
        print(f"Validation at step {log_step}: {val_metrics}")
        log_metrics_to_wandb(val_metrics, "val", log_step, cfg.wandb_entity)
    
    vla.train()
    pose_head.train()


# Create a wrapper to make OpenVLA's language_model compatible with LLMBackbone interface
class OpenVLAWrapper(nn.Module):
    def __init__(self, language_model, processor):
        super().__init__()
        self.llm = language_model
        self.tokenizer = processor.tokenizer  # Get tokenizer from processor
        self.config = language_model.config
        self.embed_dim = language_model.config.hidden_size
    
    def forward(self, **kwargs):
        return self.llm(**kwargs)
    
    def get_tokenizer(self):
        return self.tokenizer


@draccus.wrap()
def finetune_pose(cfg: PoseFinetuneConfig) -> None:
    """
    Main function for fine-tuning PoseVLM.

    Args:
        cfg (PoseFinetuneConfig): Training configuration.
    """
    # Initialize distributed training
    distributed_state = PartialState()
    device_id = distributed_state.device.index
    
    # Set up run directory
    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    if distributed_state.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run directory: {run_dir}")
    
    # Initialize WandB
    if distributed_state.is_main_process and cfg.wandb_entity != "your-wandb-entity":
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
            config=vars(cfg),
        )
    
    # Two options:
    # (1) Base model is on Hugging Face Hub
    #   - Then download it and record the path to the download directory
    # (2) Base model is stored locally
    #   - Then register model config in HF Auto Classes
    # In both cases, we want to check whether any changes have been made to
    # the `modeling_prismatic.py` file in this codebase; if so, we will copy
    # the file to the downloaded or locally stored checkpoint directory so
    # that the user's changes to the VLA class logic go into effect
    if model_is_on_hf_hub(cfg.vla_path):
        # Download model directly from Hugging Face Hub
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        # Overwrite VLA path
        cfg.vla_path = vla_download_path
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Update config.json and sync model files
    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)

    # Wait for model files to be synced (only in distributed mode)
    if distributed_state.num_processes > 1:
        dist.barrier()

    # Load processor and VLA
    if cfg.resume:
        processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True, local_files_only=True)
    else:
        processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)

    # Load base VLA model first
    base_vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)

    # Create wrapper for language model
    llm_backbone = OpenVLAWrapper(base_vla.language_model, processor)

    # Create PoseVLM from base VLA
    vla = PoseVLM(
        model_id=f"pose_vlm_{cfg.pose_head_type}",
        vision_backbone=base_vla.vision_backbone,
        llm_backbone=llm_backbone,  # Use the wrapper
        pose_head_type=cfg.pose_head_type,
        pose_dim=cfg.pose_dim,
        gmm_num_components=cfg.gmm_num_components,
        enable_mixed_precision_training=False,  # We want full precision for training
    ).to(device_id)
    
    # === LoRA setup ===
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)
    vla.print_trainable_parameters()
    
    # Freeze VLA backbone (only train pose head)
    for param in vla.parameters():
        param.requires_grad = False
    
    # Initialize pose head
    hidden_dim = vla.config.hidden_size
    if cfg.pose_head_type == "gmm":
        pose_head = GMMPoseHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            pose_dim=cfg.pose_dim,
            num_components=cfg.gmm_num_components,
        )
    else:
        pose_head = SimplePoseHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            pose_dim=cfg.pose_dim,
        )
    
    # Move models to device
    vla = vla.to(device_id)
    pose_head = pose_head.to(device_id)
    
    # Wrap with DDP if using distributed training
    if distributed_state.num_processes > 1:
        vla = wrap_ddp(vla, device_id)
        pose_head = wrap_ddp(pose_head, device_id)
    
    # Count parameters
    if distributed_state.is_main_process:
        count_parameters(vla, "VLA")
        count_parameters(pose_head, "Pose Head")
    
    # Create datasets
    train_dataset = create_pose_dataset(
        data_root=str(cfg.data_root_dir),
        split="train",
        num_images_in_input=cfg.num_images_in_input,
        use_image_augmentation=cfg.image_aug,
        tokenizer_name=processor.tokenizer.name_or_path,  # Use the tokenizer from processor like original code
    )
    
    if cfg.use_val_set:
        val_dataset = create_pose_dataset(
            data_root=str(cfg.data_root_dir),
            split="val",
            num_images_in_input=cfg.num_images_in_input,
            use_image_augmentation=False,  # No augmentation for validation
            tokenizer_name=processor.tokenizer.name_or_path,  # Use the tokenizer from processor like original code
        )
    
    # Create data loaders
    train_collator = PaddedCollatorForPosePrediction(
        model_max_length=cfg.max_length,
        pad_token_id=processor.tokenizer.pad_token_id,
        padding_side=processor.tokenizer.padding_side,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=0,  # Reduce workers to avoid hanging
        pin_memory=False,  # Disable pin_memory to avoid issues
    )
    
    if cfg.use_val_set:
        val_collator = PaddedCollatorForPosePrediction(
            model_max_length=cfg.max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
            padding_side=processor.tokenizer.padding_side,
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=val_collator,
            num_workers=0,  # Reduce workers to avoid hanging
            pin_memory=False,  # Disable pin_memory to avoid issues
        )
    
    # Create pose augmentation
    pose_augmentation = None
    if cfg.pose_aug:
        pose_augmentation = create_pose_augmentation(
            augmentation_type="euler",
            position_std=cfg.pose_aug_position_std,
            orientation_std=cfg.pose_aug_orientation_std,
            enabled=True,
        )
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        pose_head.parameters(),
        lr=cfg.learning_rate,
        weight_decay=0.01,
    )
    
    scheduler = MultiStepLR(
        optimizer,
        milestones=[cfg.num_steps_before_decay],
        gamma=0.1,
    )
    
    # Resume from checkpoint if specified
    if cfg.resume:
        if cfg.resume_step is None:
            # Load latest checkpoint
            latest_checkpoint_path = run_dir / "latest_checkpoint.txt"
            if latest_checkpoint_path.exists():
                with open(latest_checkpoint_path, "r") as f:
                    cfg.resume_step = int(f.read().strip())
            else:
                raise ValueError("No latest checkpoint found")
        
        # Load checkpoints
        vla_state_dict = load_checkpoint("vla", str(run_dir), cfg.resume_step, device_id)
        pose_head_state_dict = load_checkpoint("pose_head", str(run_dir), cfg.resume_step, device_id)
        
        vla.load_state_dict(vla_state_dict)
        pose_head.load_state_dict(pose_head_state_dict)
        
        print(f"Resumed from step {cfg.resume_step}")
    
    # Training loop
    vla.train()
    pose_head.train()
    
    print("Starting training loop...")
    
    # Test loading one batch first
    print("Testing batch loading...")
    try:
        test_batch = next(iter(train_dataloader))
        print(f"Successfully loaded test batch with keys: {list(test_batch.keys())}")
        print(f"Batch shapes: pixel_values={test_batch['pixel_values'].shape}, input_ids={test_batch['input_ids'].shape}")
    except Exception as e:
        print(f"Error loading test batch: {e}")
        return
    
    metrics_deques = {
        "loss": deque(maxlen=100),
        "l1_error": deque(maxlen=100),
    }
    if cfg.pose_head_type == "gmm":
        metrics_deques["entropy"] = deque(maxlen=100)
    
    log_step = cfg.resume_step if cfg.resume else 0
    optimizer.zero_grad()
    
    print(f"Training for {cfg.max_steps} steps...")
    
    for epoch in range(1000):  # Large number of epochs
        print(f"Starting epoch {epoch}...")
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx == 0:
                print(f"Processing first batch of epoch {epoch}...")
            
            if log_step >= cfg.max_steps:
                break
            
            loss, metrics = run_forward_pass(
                vla=vla,
                pose_head=pose_head,
                batch=batch,
                device_id=device_id,
                cfg=cfg,
                use_film=cfg.use_film,
                num_patches=train_dataset.num_patches,
                pose_augmentation=pose_augmentation,
            )
            
            # Backward pass
            loss = loss / cfg.grad_accumulation_steps
            loss.backward()
            
            # Update metrics
            for metric_name, value in metrics.items():
                if metric_name in metrics_deques:
                    metrics_deques[metric_name].append(value)
            
            # Gradient accumulation
            if (log_step + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Logging
            if log_step % cfg.wandb_log_freq == 0 and distributed_state.is_main_process:
                smoothed_metrics = compute_smoothened_metrics(metrics_deques)
                print(f"Step {log_step}: {smoothed_metrics}")
                log_metrics_to_wandb(smoothed_metrics, "train", log_step, cfg.wandb_entity)
            
            # Validation
            if cfg.use_val_set and log_step % cfg.val_freq == 0:
                run_validation(
                    vla=vla,
                    pose_head=pose_head,
                    val_dataloader=val_dataloader,
                    device_id=device_id,
                    cfg=cfg,
                    num_patches=val_dataset.num_patches,
                    log_step=log_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                    pose_augmentation=pose_augmentation,
                )
            
            # Save checkpoint
            if log_step % cfg.save_freq == 0 and log_step > 0:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    pose_head=pose_head,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                )
            
            log_step += 1
        
        if log_step >= cfg.max_steps:
            break
    
    # Save final checkpoint
    if distributed_state.is_main_process:
        save_training_checkpoint(
            cfg=cfg,
            run_dir=run_dir,
            log_step=log_step,
            vla=vla,
            processor=processor,
            pose_head=pose_head,
            train_dataset=train_dataset,
            distributed_state=distributed_state,
        )
        print("Training completed!")


if __name__ == "__main__":
    finetune_pose() 