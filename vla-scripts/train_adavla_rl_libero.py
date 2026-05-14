"""
train_adavla_rl_libero.py

Online AdaVLA-RL training on LIBERO rollouts.

This script optimizes only the compute policy with true environment feedback:

    reward = success_reward - lambda * compute_cost

It keeps the base OpenVLA/OFT action policy frozen. Use the offline
`train_adavla_rl.py` stage first when you also need to train the optional
fast-path action head.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import draccus
import numpy as np
import torch
import tqdm
from libero.libero import benchmark
from torch.optim import AdamW

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.libero.run_libero_eval import (
    TASK_MAX_STEPS,
    TaskSuite,
    prepare_observation,
    process_action,
)
from experiments.robot.openvla_utils import (
    DEVICE,
    get_action_head,
    get_fast_path_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    normalize_proprio,
    prepare_images_for_vla,
)
from prismatic.models.adaptive_compute import (
    AdaptiveComputePolicy,
    compute_policy_state_dim,
    ppo_actor_critic_loss,
)
from prismatic.vla.constants import ACTION_DIM, PROPRIO_DIM

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class AdaVLALiberoRLConfig:
    # Model
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    lora_rank: int = 32
    unnorm_key: Union[str, Path] = ""

    # Compute policy
    compute_policy_checkpoint: Union[str, Path] = ""
    compute_policy_hidden_dim: int = 512
    compute_token_budgets: Tuple[int, ...] = (16, 32, 64, 128)
    compute_layer_depths: Tuple[int, ...] = (4, 8, 12)
    compute_cost_lambda: float = 0.03
    adaptive_compute_mode: str = "sample"
    use_fast_path: bool = False
    fast_path_checkpoint: Union[str, Path] = ""
    fast_path_hidden_dim: int = 512

    # LIBERO
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL
    num_steps_wait: int = 10
    num_episodes: int = 100
    env_img_res: int = 256
    initial_states_path: str = "DEFAULT"

    # PPO
    learning_rate: float = 1e-5
    ppo_epochs: int = 4
    ppo_clip_ratio: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_entropy_coef: float = 0.01
    max_grad_norm: float = 1.0

    # Logging/checkpointing
    run_root_dir: Path = Path("runs")
    run_id: str = "adavla_rl_libero"
    save_freq: int = 10
    seed: int = 7


def _load_or_init_compute_policy(cfg: AdaVLALiberoRLConfig, llm_dim: int) -> AdaptiveComputePolicy:
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
        state_dict = torch.load(cfg.compute_policy_checkpoint, map_location=DEVICE)
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        compute_policy.load_state_dict(state_dict)
    return compute_policy


def _check_unnorm_key(cfg: AdaVLALiberoRLConfig, vla) -> None:
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in vla.norm_stats and f"{unnorm_key}_no_noops" in vla.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in vla.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
    cfg.unnorm_key = unnorm_key


def _query_action_with_compute(
    cfg: AdaVLALiberoRLConfig,
    vla,
    processor,
    observation: Dict,
    task_description: str,
    action_head,
    proprio_projector,
    compute_policy,
    fast_path_action_head,
) -> Tuple[List[np.ndarray], Dict[str, torch.Tensor]]:
    all_images = [observation["full_image"]]
    if cfg.num_images_in_input > 1:
        all_images.extend([observation[k] for k in observation.keys() if "wrist" in k])
    all_images = prepare_images_for_vla(all_images, cfg)
    primary_image = all_images.pop(0)
    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
    inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)
    if all_images:
        wrist_inputs = [processor(prompt, wrist).to(DEVICE, dtype=torch.bfloat16) for wrist in all_images]
        inputs["pixel_values"] = torch.cat(
            [inputs["pixel_values"], *[wrist_input["pixel_values"] for wrist_input in wrist_inputs]],
            dim=1,
        )

    proprio = None
    if cfg.use_proprio:
        proprio = observation["state"]
        proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
        proprio = normalize_proprio(proprio, proprio_norm_stats)

    with torch.inference_mode():
        actions, _, compute_metrics = vla.predict_action(
            **inputs,
            unnorm_key=cfg.unnorm_key,
            proprio=proprio,
            proprio_projector=proprio_projector if cfg.use_proprio else None,
            action_head=action_head,
            use_film=cfg.use_film,
            compute_policy=compute_policy,
            fast_path_action_head=fast_path_action_head,
            use_adaptive_compute=True,
            adaptive_compute_mode=cfg.adaptive_compute_mode,
            return_compute_metrics=True,
        )

    return [actions[i] for i in range(len(actions))], compute_metrics


def _ppo_update(compute_policy, optimizer, transitions, episode_success: bool, cfg: AdaVLALiberoRLConfig):
    states = torch.cat([t["state"] for t in transitions], dim=0).to(DEVICE)
    action_indices = {
        "token_index": torch.cat([t["token_index"] for t in transitions], dim=0).to(DEVICE),
        "layer_index": torch.cat([t["layer_index"] for t in transitions], dim=0).to(DEVICE),
        "path_index": torch.cat([t["path_index"] for t in transitions], dim=0).to(DEVICE),
    }
    old_log_probs = torch.cat([t["log_prob"] for t in transitions], dim=0).to(DEVICE)
    old_values = torch.cat([t["value"] for t in transitions], dim=0).to(DEVICE)
    compute_cost = torch.cat([t["compute_cost"] for t in transitions], dim=0).to(DEVICE)
    task_reward = torch.full_like(compute_cost, 1.0 if episode_success else 0.0)
    returns = task_reward - cfg.compute_cost_lambda * compute_cost
    advantages = returns - old_values

    last_metrics = {}
    for _ in range(cfg.ppo_epochs):
        output = compute_policy(states, action_indices=action_indices, allow_fast_path=cfg.use_fast_path)
        loss, metrics = ppo_actor_critic_loss(
            new_log_probs=output.log_prob,
            old_log_probs=old_log_probs,
            advantages=advantages,
            values=output.value,
            returns=returns,
            entropy=output.entropy,
            clip_ratio=cfg.ppo_clip_ratio,
            value_coef=cfg.ppo_value_coef,
            entropy_coef=cfg.ppo_entropy_coef,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(compute_policy.parameters(), cfg.max_grad_norm)
        optimizer.step()
        last_metrics = {k: v.item() for k, v in metrics.items()}
        last_metrics["loss"] = loss.item()
    return last_metrics


def _run_episode(
    cfg: AdaVLALiberoRLConfig,
    env,
    task_description: str,
    initial_state,
    vla,
    processor,
    resize_size,
    action_head,
    proprio_projector,
    compute_policy,
    fast_path_action_head,
):
    env.reset()
    obs = env.set_init_state(initial_state)
    action_queue = []
    transitions = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    for t in range(max_steps + cfg.num_steps_wait):
        if t < cfg.num_steps_wait:
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
            continue

        observation, _ = prepare_observation(obs, resize_size)
        if len(action_queue) == 0:
            actions, compute_metrics = _query_action_with_compute(
                cfg,
                vla,
                processor,
                observation,
                task_description,
                action_head,
                proprio_projector,
                compute_policy,
                fast_path_action_head,
            )
            action_queue.extend(actions)
            transitions.append({k: v.detach() for k, v in compute_metrics.items() if isinstance(v, torch.Tensor)})

        action = process_action(action_queue.pop(0), cfg.model_family)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return True, transitions

    return False, transitions


@draccus.wrap()
def train_adavla_rl_libero(cfg: AdaVLALiberoRLConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    run_dir = cfg.run_root_dir / cfg.run_id
    os.makedirs(run_dir, exist_ok=True)

    vla = get_vla(cfg)
    vla.eval()
    vla.requires_grad_(False)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    _check_unnorm_key(cfg, vla)

    processor = get_processor(cfg)
    action_head = get_action_head(cfg, vla.llm_dim)
    action_head.eval()
    action_head.requires_grad_(False)

    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, PROPRIO_DIM) if cfg.use_proprio else None
    if proprio_projector is not None:
        proprio_projector.eval()
        proprio_projector.requires_grad_(False)

    compute_policy = _load_or_init_compute_policy(cfg, vla.llm_dim)
    optimizer = AdamW(compute_policy.parameters(), lr=cfg.learning_rate)

    fast_path_action_head = None
    if cfg.use_fast_path:
        fast_path_action_head = get_fast_path_action_head(cfg, vla.llm_dim)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    resize_size = 224
    successes = 0

    for episode_idx in tqdm.tqdm(range(cfg.num_episodes)):
        task_id = episode_idx % task_suite.n_tasks
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        initial_state = initial_states[episode_idx % len(initial_states)]
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

        success, transitions = _run_episode(
            cfg,
            env,
            task_description,
            initial_state,
            vla,
            processor,
            resize_size,
            action_head,
            proprio_projector,
            compute_policy,
            fast_path_action_head,
        )
        successes += int(success)
        if transitions:
            metrics = _ppo_update(compute_policy, optimizer, transitions, success, cfg)
        else:
            metrics = {"loss": 0.0}

        print(
            f"episode={episode_idx + 1} success={success} "
            f"sr={successes / (episode_idx + 1):.3f} ppo_loss={metrics['loss']:.4f}"
        )

        if (episode_idx + 1) % cfg.save_freq == 0:
            checkpoint_path = run_dir / f"compute_policy--{episode_idx + 1}_checkpoint.pt"
            torch.save(compute_policy.state_dict(), checkpoint_path)
            print(f"Saved compute policy checkpoint: {checkpoint_path}")

    final_path = run_dir / "compute_policy--latest_checkpoint.pt"
    torch.save(compute_policy.state_dict(), final_path)
    print(f"Saved final compute policy checkpoint: {final_path}")


if __name__ == "__main__":
    train_adavla_rl_libero()
