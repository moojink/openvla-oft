# Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success

**Project website: https://openvla-oft.github.io/**

**Paper: https://arxiv.org/abs/2502.19645**

**Summary video: https://youtu.be/T3Zkkr_NTSA**

## System Requirements

Inference:
* 1 GPU with ~16 GB VRAM for LIBERO sim benchmark tasks
* 1 GPU with ~18 GB VRAM for ALOHA robot tasks

Training:
* Between 1-8 GPUs with 27-80 GB, depending on the desired training setup (with default bfloat16 data type). See [this FAQ on our project website](https://openvla-oft.github.io/#train-compute) for details.

## Quick Start

First, set up a conda environment (see instructions in [SETUP.md](SETUP.md)).

Then, run the Python script below to download a pretrained OpenVLA-OFT checkpoint and run inference to generate an action chunk:

```python
import pickle
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

# Instantiate config (see class GenerateConfig in experiments/robot/libero/run_libero_eval.py for definitions)
cfg = GenerateConfig(
    pretrained_checkpoint = "moojink/openvla-7b-oft-finetuned-libero-spatial",
    use_l1_regression = True,
    use_diffusion = False,
    use_film = False,
    num_images_in_input = 2,
    use_proprio = True,
    load_in_8bit = False,
    load_in_4bit = False,
    center_crop = True,
    num_open_loop_steps = NUM_ACTIONS_CHUNK,
    unnorm_key = "libero_spatial_no_noops",
)

# Load OpenVLA-OFT policy and inputs processor
vla = get_vla(cfg)
processor = get_processor(cfg)

# Load MLP action head to generate continuous actions (via L1 regression)
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

# Load proprio projector to map proprio to language embedding space
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

# Load sample observation:
#   observation (dict): {
#     "full_image": primary third-person image,
#     "wrist_image": wrist-mounted camera image,
#     "state": robot proprioceptive state,
#     "task_description": task description,
#   }
with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
    observation = pickle.load(file)

# Generate robot action chunk (sequence of future actions)
actions = get_vla_action(cfg, vla, processor, observation, observation["task_description"], action_head, proprio_projector)
print("Generated action chunk:")
for act in actions:
    print(act)
```

## Installation

See [SETUP.md](SETUP.md) for instructions on setting up the conda environment.

### Run on Intel GPU (XPU)

The inference code (the Quick Start above, the LIBERO evaluation in [LIBERO.md](LIBERO.md) and `vla-scripts/deploy.py`) runs on an Intel GPU (XPU) when no NVIDIA GPU is present: it picks the device by itself, so the commands stay the same. Set up the environment with Python 3.12 and the XPU build of PyTorch instead of the steps in [SETUP.md](SETUP.md):

```bash
conda create -n openvla-oft-xpu python=3.12 -y
conda activate openvla-oft-xpu
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e . --no-deps
pip install -r requirements-xpu.txt
pip install --no-deps "dlimp @ git+https://github.com/moojink/dlimp_openvla" "tensorflow_graphics==2021.12.3"
```

For the LIBERO evaluation (and the Quick Start, which imports it), install LIBERO as in [LIBERO.md](LIBERO.md) with these lines instead:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO --config-settings editable_mode=compat  # so that `libero` imports with recent setuptools
pip install -r experiments/robot/libero/libero_requirements.txt "numpy<2" "mujoco==3.2.7"  # tensorflow 2.16 needs NumPy 1, robosuite 1.4.1 does not run with MuJoCo 4
```

Training, Flash Attention 2, 8-bit and 4-bit loading and `vla-scripts/merge_lora_weights_and_save.py` stay CUDA-only.

## Training and Evaluation

See [LIBERO.md](LIBERO.md) for fine-tuning/evaluating on LIBERO simulation benchmark task suites.

See [ALOHA.md](ALOHA.md) for fine-tuning/evaluating on real-world ALOHA robot tasks.

## Support

If you run into any issues, please open a new GitHub issue. If you do not receive a response within 2 business days, please email Moo Jin Kim (moojink@cs.stanford.edu) to bring the issue to his attention.

## Citation

If you use our code in your work, please cite [our paper](https://arxiv.org/abs/2502.19645):

```bibtex
@article{kim2025fine,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
  journal={arXiv preprint arXiv:2502.19645},
  year={2025}
}
```
