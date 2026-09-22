"""Regression tests for distributed continuous-action-head training."""

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 8
_MISSING_MODULE = object()
_SYNTHETIC_MODULE_NAMES = (
    "prismatic",
    "prismatic.vla",
    "prismatic.vla.constants",
)


def _load_action_heads_module():
    """Load the module without importing the heavyweight ``prismatic`` package."""
    constants = types.ModuleType("prismatic.vla.constants")
    constants.ACTION_DIM = ACTION_DIM
    constants.ACTION_TOKEN_BEGIN_IDX = 31743
    constants.IGNORE_INDEX = -100
    constants.NUM_ACTIONS_CHUNK = NUM_ACTIONS_CHUNK
    constants.PROPRIO_DIM = 8
    constants.STOP_INDEX = 2

    previous_modules = {name: sys.modules.get(name, _MISSING_MODULE) for name in _SYNTHETIC_MODULE_NAMES}
    try:
        prismatic = types.ModuleType("prismatic")
        prismatic.__path__ = []
        vla = types.ModuleType("prismatic.vla")
        vla.__path__ = []
        sys.modules["prismatic"] = prismatic
        sys.modules["prismatic.vla"] = vla
        sys.modules["prismatic.vla.constants"] = constants

        path = Path(__file__).parents[1] / "prismatic" / "models" / "action_heads.py"
        spec = importlib.util.spec_from_file_location("action_heads_under_test", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is _MISSING_MODULE:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


_PRISMATIC_MODULES_BEFORE_LOAD = {name: sys.modules.get(name, _MISSING_MODULE) for name in _SYNTHETIC_MODULE_NAMES}
ACTION_HEADS = _load_action_heads_module()
DiffusionActionHead = ACTION_HEADS.DiffusionActionHead
L1RegressionActionHead = ACTION_HEADS.L1RegressionActionHead


def _build_action_head(kind: str) -> torch.nn.Module:
    kwargs = {"input_dim": 4, "hidden_dim": 16, "action_dim": ACTION_DIM}
    if kind == "l1":
        return L1RegressionActionHead(**kwargs)
    if kind == "diffusion":
        return DiffusionActionHead(**kwargs, num_diffusion_steps_train=4)
    raise ValueError(f"unsupported action head: {kind}")


def _ddp_step(rank: int, world_size: int, init_file: str, kind: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(1234)
        action_head = DDP(_build_action_head(kind))
        optimizer = torch.optim.AdamW(action_head.parameters(), lr=1.0e-3)

        generator = torch.Generator().manual_seed(2000 + rank)
        transformer_hidden_dim = 4 if kind == "l1" else 16
        hidden_states = torch.randn(
            2,
            NUM_ACTIONS_CHUNK * ACTION_DIM,
            transformer_hidden_dim,
            generator=generator,
        )
        target = torch.randn(
            2,
            NUM_ACTIONS_CHUNK,
            ACTION_DIM,
            generator=generator,
        )

        optimizer.zero_grad(set_to_none=True)
        prediction = action_head(hidden_states)
        torch.nn.functional.mse_loss(prediction, target).backward()
        optimizer.step()

        parameters = torch.cat([parameter.detach().flatten() for parameter in action_head.module.parameters()])
        gathered = [torch.empty_like(parameters) for _ in range(world_size)]
        dist.all_gather(gathered, parameters)
        for other in gathered[1:]:
            torch.testing.assert_close(gathered[0], other, rtol=0.0, atol=0.0)
    finally:
        dist.destroy_process_group()


class ActionHeadDDPTest(unittest.TestCase):
    def test_action_head_loader_restores_sys_modules(self) -> None:
        for name, previous_module in _PRISMATIC_MODULES_BEFORE_LOAD.items():
            self.assertIs(sys.modules.get(name, _MISSING_MODULE), previous_module)

    def test_training_source_calls_ddp_wrapper(self) -> None:
        source = (Path(__file__).parents[1] / "vla-scripts" / "finetune.py").read_text(encoding="utf-8")

        self.assertNotIn("action_head.module.predict_action(actions_hidden_states)", source)
        self.assertEqual(source.count("action_head(actions_hidden_states)"), 2)
        # Sampling is intentionally outside gradient tracking and still needs access to the scheduler.
        self.assertIn("action_head.module.predict_noise(actions_hidden_states)", source)

    def test_two_rank_optimizer_step_keeps_action_heads_synchronized(self) -> None:
        for kind in ("l1", "diffusion"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                init_file = os.path.join(directory, "process-group")
                mp.spawn(_ddp_step, args=(2, init_file, kind), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
