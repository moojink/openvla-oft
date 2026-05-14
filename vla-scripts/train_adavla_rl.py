"""
train_adavla_rl.py

Second-stage AdaVLA-RL training for the adaptive compute policy.

The base OpenVLA/OFT policy is frozen. The compute policy is optimized with a
PPO-style clipped objective using an offline reward proxy:

    reward = - action_l1_error - lambda * compute_cost

If an environment wrapper supplies task rewards, this script can be extended by
replacing the action-error reward with rollout success/reward.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import draccus
import torch
import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader

from experiments.robot.openvla_utils import (
    DEVICE,
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
)
from prismatic.models.adaptive_compute import (
    AdaptiveComputePolicy,
    ComputeCostConfig,
    FastPathActionHead,
    compute_policy_state_dim,
    ppo_actor_critic_loss,
)
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class AdaVLARLConfig:
    # Model and dataset
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    data_root_dir: Path = Path("datasets/rlds")
    dataset_name: str = "aloha_scoop_x_into_bowl"
    run_root_dir: Path = Path("runs")
    run_id: str = "adavla_rl_compute_policy"
    shuffle_buffer_size: int = 100_000

    # Base OFT components
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = False
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    lora_rank: int = 32

    # Compute policy
    compute_policy_checkpoint: Union[str, Path] = ""
    compute_policy_hidden_dim: int = 512
    compute_token_budgets: Tuple[int, ...] = (16, 32, 64, 128)
    compute_layer_depths: Tuple[int, ...] = (4, 8, 12)
    compute_cost_lambda: float = 0.03
    compute_slow_path_cost_weight: float = 0.1
    train_fast_path: bool = False
    fast_path_checkpoint: Union[str, Path] = ""
    fast_path_hidden_dim: int = 512
    fast_path_learning_rate: float = 1e-4
    fast_path_loss_coef: float = 1.0

    # PPO
    batch_size: int = 1
    learning_rate: float = 1e-4
    max_steps: int = 20_000
    ppo_epochs: int = 4
    ppo_clip_ratio: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    max_grad_norm: float = 1.0

    # Logging/checkpointing
    log_freq: int = 10
    save_freq: int = 2_000
    image_aug: bool = True


def _load_or_init_compute_policy(cfg: AdaVLARLConfig, llm_dim: int) -> AdaptiveComputePolicy:
    state_dim = compute_policy_state_dim(
        llm_dim,
        previous_action_dim=ACTION_DIM,
        uncertainty_dim=1,
        include_intermediate=True,
    )
    compute_policy = AdaptiveComputePolicy(
        state_dim=state_dim,
        hidden_dim=cfg.compute_policy_hidden_dim,
        token_budgets=cfg.compute_token_budgets,
        layer_depths=cfg.compute_layer_depths,
    ).to(DEVICE)

    if cfg.compute_policy_checkpoint:
        checkpoint = torch.load(cfg.compute_policy_checkpoint, map_location=DEVICE)
        checkpoint = {k.removeprefix("module."): v for k, v in checkpoint.items()}
        compute_policy.load_state_dict(checkpoint)
        print(f"Loaded compute policy checkpoint: {cfg.compute_policy_checkpoint}")

    return compute_policy


def _load_or_init_fast_path(cfg: AdaVLARLConfig, llm_dim: int) -> FastPathActionHead:
    state_dim = compute_policy_state_dim(
        llm_dim,
        previous_action_dim=ACTION_DIM,
        uncertainty_dim=1,
        include_intermediate=True,
    )
    fast_path = FastPathActionHead(
        state_dim=state_dim,
        hidden_dim=cfg.fast_path_hidden_dim,
        action_dim=ACTION_DIM,
    ).to(DEVICE)

    if cfg.fast_path_checkpoint:
        checkpoint = torch.load(cfg.fast_path_checkpoint, map_location=DEVICE)
        checkpoint = {k.removeprefix("module."): v for k, v in checkpoint.items()}
        fast_path.load_state_dict(checkpoint)
        print(f"Loaded fast-path checkpoint: {cfg.fast_path_checkpoint}")

    return fast_path


def _prepare_batch(batch, cfg: AdaVLARLConfig):
    prepared = {
        "input_ids": batch["input_ids"].to(DEVICE),
        "attention_mask": batch["attention_mask"].to(DEVICE),
        "pixel_values": batch["pixel_values"].to(DEVICE, dtype=torch.bfloat16),
        "labels": batch["labels"].to(DEVICE),
        "actions": batch["actions"].to(DEVICE, dtype=torch.bfloat16),
    }
    prepared["proprio"] = batch["proprio"].to(DEVICE) if cfg.use_proprio and batch["proprio"] is not None else None
    prepared["previous_action"] = torch.zeros(
        prepared["actions"].shape[0],
        ACTION_DIM,
        device=DEVICE,
        dtype=prepared["actions"].dtype,
    )
    return prepared


def _extract_action_l1(output, action_head, batch) -> torch.Tensor:
    labels = batch["labels"]
    ground_truth_actions = batch["actions"]
    ground_truth_token_ids = labels[:, 1:]
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
    action_mask = current_action_mask | next_actions_mask

    effective_num_patches = output.projector_features.shape[1]
    last_hidden_states = output.hidden_states[-1]
    text_hidden_states = last_hidden_states[:, effective_num_patches:-1]
    actions_hidden_states = text_hidden_states[action_mask].reshape(
        labels.shape[0],
        NUM_ACTIONS_CHUNK * ACTION_DIM,
        -1,
    )
    predicted_actions = action_head.predict_action(actions_hidden_states.to(torch.bfloat16))
    return (predicted_actions.float() - ground_truth_actions.float()).abs().mean(dim=(1, 2))


@torch.no_grad()
def _sample_compute_rollout(vla, action_head, proprio_projector, compute_policy, fast_path, batch, cfg):
    cost_config = ComputeCostConfig(slow_path_weight=cfg.compute_slow_path_cost_weight)
    autocast_enabled = DEVICE.type == "cuda"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        output = vla(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"],
            proprio_projector=proprio_projector if cfg.use_proprio else None,
            use_film=cfg.use_film,
            compute_policy=compute_policy,
            use_adaptive_compute=True,
            adaptive_compute_mode="sample",
            previous_action=batch["previous_action"],
            compute_cost_config=cost_config,
            allow_fast_path=fast_path is not None,
        )

        slow_action_l1 = _extract_action_l1(output, action_head, batch)
        if fast_path is not None:
            fast_actions = fast_path(output.compute_state)
            fast_action_l1 = (fast_actions.float() - batch["actions"].float()).abs().mean(dim=(1, 2))
            action_l1 = torch.where(output.compute_metrics["path"].bool(), slow_action_l1, fast_action_l1)
        else:
            fast_action_l1 = slow_action_l1
            action_l1 = slow_action_l1
        reward = -action_l1 - cfg.compute_cost_lambda * output.compute_cost.float()

    rollout = {
        "state": output.compute_state.detach(),
        "actions": {k: v.detach() for k, v in output.compute_action.items()},
        "old_log_prob": output.compute_log_prob.detach(),
        "old_value": output.compute_value.detach(),
        "reward": reward.detach(),
        "action_l1": action_l1.detach(),
        "slow_action_l1": slow_action_l1.detach(),
        "fast_action_l1": fast_action_l1.detach(),
        "ground_truth_actions": batch["actions"].detach(),
        "compute_cost": output.compute_cost.detach(),
        "vision_cost": output.compute_metrics["vision_cost"].detach(),
        "llm_cost": output.compute_metrics["llm_cost"].detach(),
        "token_budget": output.compute_metrics["token_budget"].detach(),
        "layer_depth": output.compute_metrics["layer_depth"].detach(),
        "effective_token_budget": output.compute_metrics["effective_token_budget"].detach(),
        "effective_layer_depth": output.compute_metrics["effective_layer_depth"].detach(),
        "path": output.compute_metrics["path"].detach(),
    }
    return rollout


def _fast_path_update(fast_path, optimizer, rollout, cfg):
    if fast_path is None:
        return {}
    predicted_actions = fast_path(rollout["state"])
    loss = cfg.fast_path_loss_coef * (
        predicted_actions.float() - rollout["ground_truth_actions"].float()
    ).abs().mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(fast_path.parameters(), cfg.max_grad_norm)
    optimizer.step()
    return {"fast_path_loss": loss.item()}


def _ppo_update(compute_policy, optimizer, rollout, cfg):
    last_metrics = {}
    returns = rollout["reward"].float()
    advantages = returns - rollout["old_value"].float()

    for _ in range(cfg.ppo_epochs):
        policy_output = compute_policy(rollout["state"], action_indices=rollout["actions"])
        loss, metrics = ppo_actor_critic_loss(
            new_log_probs=policy_output.log_prob,
            old_log_probs=rollout["old_log_prob"],
            advantages=advantages,
            values=policy_output.value,
            returns=returns,
            entropy=policy_output.entropy,
            clip_ratio=cfg.ppo_clip_ratio,
            value_coef=cfg.ppo_value_coef,
            entropy_coef=cfg.ppo_entropy_coef,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(compute_policy.parameters(), cfg.max_grad_norm)
        optimizer.step()
        last_metrics = {name: value.item() for name, value in metrics.items()}
        last_metrics["ppo_loss"] = loss.item()

    return last_metrics


def _build_dataloader(cfg: AdaVLARLConfig, processor):
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    use_wrist_image = cfg.num_images_in_input > 1
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
    )
    dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(processor.image_processor.input_sizes[0][-2:]),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    return DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=collator, num_workers=0)


@draccus.wrap()
def train_adavla_rl(cfg: AdaVLARLConfig) -> None:
    assert cfg.use_l1_regression and not cfg.use_diffusion, "AdaVLA-RL PPO script currently expects L1 OFT actions."
    assert cfg.batch_size == 1, (
        "AdaVLA-RL PPO uses per-decision compute rewards; keep --batch_size=1 so sampled token/layer/path costs match "
        "the actual single-query deployment path."
    )

    run_dir = cfg.run_root_dir / cfg.run_id
    os.makedirs(run_dir, exist_ok=True)

    vla = get_vla(cfg)
    vla.eval()
    vla.requires_grad_(False)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    processor = get_processor(cfg)
    action_head = get_action_head(cfg, vla.llm_dim)
    action_head.eval()
    action_head.requires_grad_(False)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, vla.llm_dim, PROPRIO_DIM)
        proprio_projector.eval()
        proprio_projector.requires_grad_(False)

    compute_policy = _load_or_init_compute_policy(cfg, vla.llm_dim)
    optimizer = AdamW(compute_policy.parameters(), lr=cfg.learning_rate)
    fast_path = _load_or_init_fast_path(cfg, vla.llm_dim) if cfg.train_fast_path else None
    fast_path_optimizer = (
        AdamW(fast_path.parameters(), lr=cfg.fast_path_learning_rate) if fast_path is not None else None
    )
    dataloader = _build_dataloader(cfg, processor)

    progress = tqdm.tqdm(total=cfg.max_steps, leave=False)
    for step, raw_batch in enumerate(dataloader, start=1):
        batch = _prepare_batch(raw_batch, cfg)
        rollout = _sample_compute_rollout(vla, action_head, proprio_projector, compute_policy, fast_path, batch, cfg)
        fast_path_metrics = _fast_path_update(fast_path, fast_path_optimizer, rollout, cfg)
        metrics = _ppo_update(compute_policy, optimizer, rollout, cfg)
        metrics.update(fast_path_metrics)

        if step % cfg.log_freq == 0:
            print(
                f"step={step} "
                f"reward={rollout['reward'].mean().item():.4f} "
                f"action_l1={rollout['action_l1'].mean().item():.4f} "
                f"fast_l1={rollout['fast_action_l1'].mean().item():.4f} "
                f"cost={rollout['compute_cost'].mean().item():.4f} "
                f"llm_cost={rollout['llm_cost'].mean().item():.4f} "
                f"k={rollout['token_budget'].float().mean().item():.1f} "
                f"d={rollout['layer_depth'].float().mean().item():.1f} "
                f"slow={rollout['path'].float().mean().item():.2f} "
                f"ppo_loss={metrics.get('ppo_loss', 0.0):.4f}"
            )

        if step % cfg.save_freq == 0:
            checkpoint_path = run_dir / f"compute_policy--{step}_checkpoint.pt"
            torch.save(compute_policy.state_dict(), checkpoint_path)
            print(f"Saved compute policy checkpoint: {checkpoint_path}")
            if fast_path is not None:
                fast_path_path = run_dir / f"fast_path--{step}_checkpoint.pt"
                torch.save(fast_path.state_dict(), fast_path_path)
                print(f"Saved fast-path checkpoint: {fast_path_path}")

        progress.update()
        if step >= cfg.max_steps:
            break

    final_path = run_dir / "compute_policy--latest_checkpoint.pt"
    torch.save(compute_policy.state_dict(), final_path)
    print(f"Saved final compute policy checkpoint: {final_path}")
    if fast_path is not None:
        fast_path_path = run_dir / "fast_path--latest_checkpoint.pt"
        torch.save(fast_path.state_dict(), fast_path_path)
        print(f"Saved final fast-path checkpoint: {fast_path_path}")


if __name__ == "__main__":
    train_adavla_rl()
