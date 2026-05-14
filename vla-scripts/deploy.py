"""
deploy.py

Starts VLA server which the client can query to get robot actions.
"""

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import draccus
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from experiments.robot.openvla_utils import (
    get_action_head,
    get_compute_policy,
    get_fast_path_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from experiments.robot.robot_utils import (
    get_image_resize_size,
)
from prismatic.vla.constants import PROPRIO_DIM


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class OpenVLAServer:
    def __init__(self, cfg) -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given observation + instruction.
        """
        self.cfg = cfg

        # Load model
        self.vla = get_vla(cfg)
        self.inference_lock = threading.Lock()

        # Load proprio projector
        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, PROPRIO_DIM)

        # Load continuous action head
        self.action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            self.action_head = get_action_head(cfg, self.vla.llm_dim)

        self.compute_policy = None
        if cfg.use_adaptive_compute:
            self.compute_policy = get_compute_policy(cfg, self.vla.llm_dim)

        self.fast_path_action_head = None
        if cfg.use_fast_path:
            self.fast_path_action_head = get_fast_path_action_head(cfg, self.vla.llm_dim)

        # Check that the model contains the action un-normalization key
        assert cfg.unnorm_key in self.vla.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

        # Get Hugging Face processor
        self.processor = None
        self.processor = get_processor(cfg)

        # Get expected image dimensions
        self.resize_size = get_image_resize_size(cfg)


    def get_server_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            observation = payload
            instruction = observation["instruction"]

            with self.inference_lock:
                action = get_vla_action(
                    self.cfg,
                    self.vla,
                    self.processor,
                    observation,
                    instruction,
                    action_head=self.action_head,
                    proprio_projector=self.proprio_projector,
                    compute_policy=self.compute_policy,
                    fast_path_action_head=self.fast_path_action_head,
                    use_film=self.cfg.use_film,
                )

            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'observation': dict, 'instruction': str}\n"
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8777) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.get_server_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8777                                                    # Host Port

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 3                     # Number of images in the VLA input (default: 3)
    use_proprio: bool = True                         # Whether to include proprio state in input
    use_adaptive_compute: bool = False               # If True, enables AdaVLA-RL dynamic compute at inference
    adaptive_compute_mode: str = "greedy"            # "greedy" for serving, "sample" for exploration
    compute_policy_checkpoint: Union[str, Path] = "" # Optional compute_policy checkpoint file or directory
    compute_policy_hidden_dim: int = 512             # Must match training
    compute_token_budgets: tuple = (16, 32, 64, 128) # Candidate visual token budgets
    compute_layer_depths: tuple = (4, 8, 12)         # Candidate language-model depths
    use_fast_path: bool = False                      # If True, load and use optional fast-path action head
    fast_path_checkpoint: Union[str, Path] = ""      # Optional fast-path checkpoint file or directory
    fast_path_hidden_dim: int = 512                  # Must match fast-path training

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key
    use_relative_actions: bool = False               # Whether to use relative actions (delta joint angles)

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7                                    # Random Seed (for reproducibility)
    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OpenVLAServer(cfg)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
