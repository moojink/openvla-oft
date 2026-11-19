"""Regression test: `vla-scripts/finetune.py` must store its trainable weights in fp32.

PEFT 0.11.1 (pinned in `pyproject.toml`) casts LoRA weights to the dtype of the base model, which `finetune.py` loads in
bf16. PyTorch's AdamW keeps no fp32 copy of a bf16 weight, so it rounds away every update smaller than 1/512 of the
weight. The test runs the real `finetune()` setup on CPU with a tiny stand-in model, up to the point where the optimizer
is built, and downloads nothing.

Run from the repository root with `python -m unittest discover -s tests` (or `pytest tests`).
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_finetune_module():
    spec = importlib.util.spec_from_file_location("finetune_under_test", REPO_ROOT / "vla-scripts" / "finetune.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FINETUNE = _load_finetune_module()


class _TinyVLAConfig(PretrainedConfig):
    model_type = "tiny_vla_for_tests"


class _TinyVisionBackbone(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.featurizer = nn.Linear(dim, dim)
        self.num_images_in_input = 1

    def set_num_images_in_input(self, num_images_in_input):
        self.num_images_in_input = num_images_in_input

    def get_num_images_in_input(self):
        return self.num_images_in_input

    def get_num_patches(self):
        return 4


class _TinyVLA(PreTrainedModel):
    """Stands in for OpenVLA: linear layers that PEFT's `target_modules="all-linear"` wraps with LoRA."""

    config_class = _TinyVLAConfig

    def __init__(self, config):
        super().__init__(config)
        self.llm_dim = 16
        self.vision_backbone = _TinyVisionBackbone(self.llm_dim)
        self.language_model = nn.Sequential(nn.Linear(self.llm_dim, self.llm_dim), nn.Linear(self.llm_dim, self.llm_dim))
        self.lm_head = nn.Linear(self.llm_dim, 32)

    def get_output_embeddings(self):
        return self.lm_head


class _NoDDP(nn.Module):
    """Replaces DistributedDataParallel, which needs a process group."""

    def __init__(self, module):
        super().__init__()
        self.module = module


class _OptimizerBuilt(Exception):
    pass


def _setup_until_optimizer(**cfg_overrides):
    """Runs `finetune()` until it builds the optimizer; returns the trainable parameters and the (unwrapped) model."""
    captured = {}

    def capture_params(params, lr):
        captured["params"] = list(params)
        raise _OptimizerBuilt

    vla = _TinyVLA(_TinyVLAConfig()).to(torch.bfloat16)  # finetune.py loads OpenVLA with torch_dtype=torch.bfloat16
    auto_model = mock.MagicMock()
    auto_model.from_pretrained.return_value = vla
    with tempfile.TemporaryDirectory() as run_root_dir, mock.patch.multiple(
        FINETUNE,
        PartialState=mock.MagicMock(return_value=SimpleNamespace(local_process_index="cpu", is_main_process=True)),
        wandb=mock.MagicMock(),
        dist=mock.MagicMock(),
        model_is_on_hf_hub=mock.MagicMock(return_value=False),
        update_auto_map=mock.MagicMock(),
        check_model_logic_mismatch=mock.MagicMock(),
        AutoConfig=mock.MagicMock(),
        AutoImageProcessor=mock.MagicMock(),
        AutoProcessor=mock.MagicMock(),
        AutoModelForVision2Seq=auto_model,
        wrap_ddp=lambda module, device_id, find_unused=False: _NoDDP(module),
        AdamW=capture_params,
    ), mock.patch("torch.cuda.set_device"), mock.patch("torch.cuda.empty_cache"):
        cfg = FINETUNE.FinetuneConfig(vla_path="tiny-vla", run_root_dir=Path(run_root_dir), **cfg_overrides)
        try:
            FINETUNE.finetune.__wrapped__(cfg)  # `__wrapped__` skips draccus' command-line parsing
        except _OptimizerBuilt:
            pass
    return captured["params"], vla


class TrainableWeightsTest(unittest.TestCase):
    def test_adamw_step_on_bf16_weight_without_fp32_copy_is_rounded_away(self):
        # The bf16 numbers next to 1.0 are 2**-8 and 2**-7 away, so a step of 1e-5 rounds back to 1.0.
        for dtype, should_change in ((torch.bfloat16, False), (torch.float32, True)):
            weight = nn.Parameter(torch.ones(8, dtype=dtype))
            weight.grad = torch.ones_like(weight)
            torch.optim.AdamW([weight], lr=1e-5).step()
            self.assertEqual(bool((weight != 1).all()), should_change, dtype)

    def _check(self, **cfg_overrides):
        params, vla = _setup_until_optimizer(**cfg_overrides)
        self.assertGreater(len(params), 0)
        dtypes = {param.dtype for param in params}

        before = [param.detach().clone() for param in params]
        for param in params:
            param.grad = torch.ones_like(param)
        torch.optim.AdamW(params, lr=1e-5).step()
        unchanged = sum(int((param == old).sum()) for param, old in zip(params, before))
        total = sum(param.numel() for param in params)
        self.assertEqual(unchanged, 0, f"{unchanged} of {total} trainable weights ({dtypes}) ignored a 1e-5 step")
        self.assertEqual(dtypes, {torch.float32})

        # The frozen base model is not upcast: only the trainable weights get an fp32 copy.
        self.assertEqual({p.dtype for p in vla.parameters() if not p.requires_grad}, {torch.bfloat16})

    def test_l1_regression_setup_trains_fp32_weights(self):
        self._check(use_l1_regression=True, use_diffusion=False, use_proprio=True)

    def test_diffusion_setup_trains_fp32_weights(self):
        self._check(use_l1_regression=False, use_diffusion=True, use_proprio=True)


if __name__ == "__main__":
    unittest.main()
