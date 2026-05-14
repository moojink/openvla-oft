"""Adaptive compute components for AdaVLA-RL.

This module contains the small policy network and helper utilities used to
choose how much VLA inference compute to spend at each decision step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

DEFAULT_TOKEN_BUDGETS: Tuple[int, ...] = (16, 32, 64, 128)
DEFAULT_LAYER_DEPTHS: Tuple[int, ...] = (4, 8, 12)


@dataclass(frozen=True)
class ComputeCostConfig:
    """Weights for the compute-aware penalty C_t."""

    vision_weight: float = 0.0
    flops_weight: float = 1.0
    latency_weight: float = 0.1
    token_weight: float = 0.1
    slow_path_weight: float = 0.0


@dataclass
class AdaptiveComputeOutput:
    """Structured output from the compute policy."""

    token_budget: torch.LongTensor
    layer_depth: torch.LongTensor
    path: torch.LongTensor
    token_index: torch.LongTensor
    layer_index: torch.LongTensor
    path_index: torch.LongTensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    token_logits: torch.Tensor
    layer_logits: torch.Tensor
    path_logits: torch.Tensor

    def as_action_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "token_index": self.token_index,
            "layer_index": self.layer_index,
            "path_index": self.path_index,
            "token_budget": self.token_budget,
            "layer_depth": self.layer_depth,
            "path": self.path,
        }


def compute_policy_state_dim(
    llm_dim: int,
    previous_action_dim: int = ACTION_DIM,
    uncertainty_dim: int = 1,
    include_intermediate: bool = True,
) -> int:
    """Return the expected AdaVLA-RL compute-policy state size."""

    pooled_blocks = 2 + int(include_intermediate)
    return pooled_blocks * llm_dim + previous_action_dim + uncertainty_dim


class AdaptiveComputePolicy(nn.Module):
    """Categorical actor-critic policy over (token budget, layer depth, path)."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 512,
        token_budgets: Sequence[int] = DEFAULT_TOKEN_BUDGETS,
        layer_depths: Sequence[int] = DEFAULT_LAYER_DEPTHS,
        dropout: float = 0.0,
        token_dim: Optional[int] = None,
        previous_action_dim: int = ACTION_DIM,
        uncertainty_dim: int = 1,
        include_intermediate: bool = True,
    ) -> None:
        super().__init__()
        if len(token_budgets) == 0:
            raise ValueError("`token_budgets` must contain at least one budget.")
        if len(layer_depths) == 0:
            raise ValueError("`layer_depths` must contain at least one depth.")

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.token_budgets = tuple(int(k) for k in token_budgets)
        self.layer_depths = tuple(int(d) for d in layer_depths)
        if token_dim is None:
            pooled_blocks = 2 + int(include_intermediate)
            inferred_dim = state_dim - previous_action_dim - uncertainty_dim
            if inferred_dim <= 0 or inferred_dim % pooled_blocks != 0:
                raise ValueError(
                    "`token_dim` could not be inferred from state_dim; pass token_dim explicitly when constructing "
                    "AdaptiveComputePolicy."
                )
            token_dim = inferred_dim // pooled_blocks
        self.token_dim = int(token_dim)

        self.trunk = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.token_head = nn.Linear(hidden_dim, len(self.token_budgets))
        self.layer_head = nn.Linear(hidden_dim, len(self.layer_depths))
        self.path_head = nn.Linear(hidden_dim, 2)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.token_scorer = nn.Sequential(
            nn.LayerNorm(2 * self.token_dim),
            nn.Linear(2 * self.token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def build_state(
        self,
        visual_embeddings: torch.Tensor,
        language_embeddings: torch.Tensor,
        intermediate_embeddings: Optional[torch.Tensor] = None,
        previous_action: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool multimodal features into the state vector s_t.

        The state is `[vision_pool, language_pool, intermediate_pool?,
        previous_action, uncertainty]`, padded with zeros when optional inputs
        are absent so the policy can be used consistently during training and
        deployment.
        """

        batch_size = visual_embeddings.shape[0]
        dtype = visual_embeddings.dtype
        device = visual_embeddings.device
        vision_pool = visual_embeddings.mean(dim=1)
        language_pool = language_embeddings.mean(dim=1)

        parts = [vision_pool, language_pool]
        remaining_dim = self.state_dim - vision_pool.shape[-1] - language_pool.shape[-1]

        llm_dim = vision_pool.shape[-1]
        needs_intermediate = remaining_dim >= llm_dim + 1
        if needs_intermediate:
            if intermediate_embeddings is None:
                intermediate_pool = torch.zeros(batch_size, llm_dim, dtype=dtype, device=device)
            else:
                intermediate_pool = intermediate_embeddings.mean(dim=1)
            parts.append(intermediate_pool)
            remaining_dim -= llm_dim

        if remaining_dim < 0:
            raise ValueError(
                f"Compute policy state_dim={self.state_dim} is too small for pooled feature dimension {llm_dim}."
            )

        if previous_action is None:
            prev_action = torch.zeros(batch_size, max(remaining_dim - 1, 0), dtype=dtype, device=device)
        else:
            prev_action = previous_action.reshape(batch_size, -1).to(device=device, dtype=dtype)
            if prev_action.shape[-1] > max(remaining_dim - 1, 0):
                prev_action = prev_action[:, : max(remaining_dim - 1, 0)]
            elif prev_action.shape[-1] < max(remaining_dim - 1, 0):
                pad = torch.zeros(
                    batch_size,
                    max(remaining_dim - 1, 0) - prev_action.shape[-1],
                    dtype=dtype,
                    device=device,
                )
                prev_action = torch.cat([prev_action, pad], dim=-1)
        parts.append(prev_action)
        remaining_dim -= prev_action.shape[-1]

        if remaining_dim > 0:
            if uncertainty is None:
                uncertainty_features = torch.zeros(batch_size, remaining_dim, dtype=dtype, device=device)
            else:
                uncertainty_features = uncertainty.reshape(batch_size, -1).to(device=device, dtype=dtype)
                if uncertainty_features.shape[-1] > remaining_dim:
                    uncertainty_features = uncertainty_features[:, :remaining_dim]
                elif uncertainty_features.shape[-1] < remaining_dim:
                    pad = torch.zeros(
                        batch_size,
                        remaining_dim - uncertainty_features.shape[-1],
                        dtype=dtype,
                        device=device,
                    )
                    uncertainty_features = torch.cat([uncertainty_features, pad], dim=-1)
            parts.append(uncertainty_features)

        state = torch.cat(parts, dim=-1).float()
        if state.shape[-1] != self.state_dim:
            raise RuntimeError(f"Built compute state with dim {state.shape[-1]}, expected {self.state_dim}.")
        return state

    def forward(
        self,
        state: torch.Tensor,
        action_indices: Optional[Dict[str, torch.Tensor]] = None,
        deterministic: bool = False,
        allow_fast_path: bool = True,
    ) -> AdaptiveComputeOutput:
        hidden = self.trunk(state.float())
        token_logits = self.token_head(hidden)
        layer_logits = self.layer_head(hidden)
        path_logits = self.path_head(hidden)
        if not allow_fast_path:
            path_logits = path_logits.clone()
            path_logits[:, 0] = torch.finfo(path_logits.dtype).min

        token_dist = Categorical(logits=token_logits)
        layer_dist = Categorical(logits=layer_logits)
        path_dist = Categorical(logits=path_logits)

        if action_indices is None:
            if deterministic:
                token_index = torch.argmax(token_logits, dim=-1)
                layer_index = torch.argmax(layer_logits, dim=-1)
                path_index = torch.argmax(path_logits, dim=-1)
            else:
                token_index = token_dist.sample()
                layer_index = layer_dist.sample()
                path_index = path_dist.sample()
        else:
            token_index = action_indices["token_index"].to(state.device).long()
            layer_index = action_indices["layer_index"].to(state.device).long()
            path_index = action_indices["path_index"].to(state.device).long()

        token_values = torch.tensor(self.token_budgets, device=state.device, dtype=torch.long)
        layer_values = torch.tensor(self.layer_depths, device=state.device, dtype=torch.long)
        token_budget = token_values[token_index]
        layer_depth = layer_values[layer_index]

        log_prob = token_dist.log_prob(token_index) + layer_dist.log_prob(layer_index) + path_dist.log_prob(path_index)
        entropy = token_dist.entropy() + layer_dist.entropy() + path_dist.entropy()
        value = self.value_head(hidden).squeeze(-1)

        return AdaptiveComputeOutput(
            token_budget=token_budget,
            layer_depth=layer_depth,
            path=path_index,
            token_index=token_index,
            layer_index=layer_index,
            path_index=path_index,
            log_prob=log_prob,
            entropy=entropy,
            value=value,
            token_logits=token_logits,
            layer_logits=layer_logits,
            path_logits=path_logits,
        )

    def score_tokens(self, visual_embeddings: torch.Tensor, language_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute learned token-importance scores conditioned on language."""

        language_pool = language_embeddings.mean(dim=1, keepdim=True).expand(-1, visual_embeddings.shape[1], -1)
        scorer_inputs = torch.cat([visual_embeddings, language_pool], dim=-1)
        return self.token_scorer(scorer_inputs.float()).squeeze(-1)


class FastPathActionHead(nn.Module):
    """Lightweight action decoder for r_t = fast path."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 512,
        action_dim: int = ACTION_DIM,
        num_actions_chunk: int = NUM_ACTIONS_CHUNK,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.num_actions_chunk = num_actions_chunk
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions_chunk * action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        actions = self.net(state.float())
        return actions.reshape(state.shape[0], self.num_actions_chunk, self.action_dim)


def select_topk_tokens(
    embeddings: torch.Tensor,
    token_budget: torch.Tensor,
    importance_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select top-k visual tokens and return selected embeddings plus a mask.

    For a batch with different budgets, the physical sequence length is
    `max(token_budget)`. Entries beyond each sample's own budget are masked out.
    """

    if embeddings.ndim != 3:
        raise ValueError(f"Expected embeddings with shape [B, N, D], got {tuple(embeddings.shape)}")

    batch_size, num_tokens, hidden_dim = embeddings.shape
    if token_budget.ndim == 0:
        token_budget = token_budget.expand(batch_size)
    token_budget = token_budget.to(device=embeddings.device).long().clamp(min=1, max=num_tokens)
    max_budget = int(token_budget.max().item())

    if importance_scores is None:
        importance_scores = embeddings.float().norm(dim=-1)
    if importance_scores.shape != (batch_size, num_tokens):
        raise ValueError(
            f"Expected importance scores with shape {(batch_size, num_tokens)}, got {tuple(importance_scores.shape)}"
        )

    topk_indices = torch.topk(importance_scores, k=max_budget, dim=1, largest=True, sorted=True).indices
    gather_indices = topk_indices.unsqueeze(-1).expand(batch_size, max_budget, hidden_dim)
    selected = torch.gather(embeddings, dim=1, index=gather_indices)
    selected_mask = torch.arange(max_budget, device=embeddings.device).unsqueeze(0) < token_budget.unsqueeze(1)
    selected = selected * selected_mask.unsqueeze(-1).to(selected.dtype)
    return selected, selected_mask, topk_indices


def estimate_compute_breakdown(
    token_budget: torch.Tensor,
    layer_depth: torch.Tensor,
    path: torch.Tensor,
    base_num_tokens: int,
    base_num_layers: int,
    cost_config: Optional[ComputeCostConfig] = None,
) -> Dict[str, torch.Tensor]:
    """Estimate normalized compute-cost components for the sampled action c_t."""

    cost_config = cost_config or ComputeCostConfig()
    token_ratio = token_budget.float() / max(float(base_num_tokens), 1.0)
    layer_ratio = layer_depth.float() / max(float(base_num_layers), 1.0)
    flops_proxy = token_ratio.square() * layer_ratio
    latency_proxy = 0.5 * token_ratio + 0.5 * layer_ratio
    vision_proxy = torch.ones_like(flops_proxy)
    slow_path_proxy = path.float()
    total = (
        cost_config.vision_weight * vision_proxy
        + cost_config.flops_weight * flops_proxy
        + cost_config.latency_weight * latency_proxy
        + cost_config.token_weight * token_ratio
        + cost_config.slow_path_weight * slow_path_proxy
    )

    return {
        "vision": vision_proxy,
        "flops": flops_proxy,
        "latency": latency_proxy,
        "tokens": token_ratio,
        "slow_path": slow_path_proxy,
        "total": total,
    }


def estimate_compute_cost(
    token_budget: torch.Tensor,
    layer_depth: torch.Tensor,
    path: torch.Tensor,
    base_num_tokens: int,
    base_num_layers: int,
    cost_config: Optional[ComputeCostConfig] = None,
) -> torch.Tensor:
    """Estimate normalized total compute cost for the sampled action c_t."""

    return estimate_compute_breakdown(
        token_budget,
        layer_depth,
        path,
        base_num_tokens=base_num_tokens,
        base_num_layers=base_num_layers,
        cost_config=cost_config,
    )["total"]


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if advantages.numel() <= 1:
        return advantages
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + eps)


def ppo_actor_critic_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute a compact PPO clipped objective for the compute policy."""

    advantages = normalize_advantages(advantages.detach())
    ratio = torch.exp(new_log_probs - old_log_probs.detach())
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    actor_loss = -torch.min(unclipped, clipped).mean()
    value_loss = nn.functional.mse_loss(values, returns.detach())
    entropy_bonus = entropy.mean()
    loss = actor_loss + value_coef * value_loss - entropy_coef * entropy_bonus

    metrics = {
        "ppo_actor_loss": actor_loss.detach(),
        "ppo_value_loss": value_loss.detach(),
        "ppo_entropy": entropy_bonus.detach(),
        "ppo_ratio": ratio.mean().detach(),
        "ppo_clip_fraction": (torch.abs(ratio - 1.0) > clip_ratio).float().mean().detach(),
    }
    return loss, metrics
