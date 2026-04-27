# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""
SFT (Supervised Fine-Tuning) config for the ACE CONUS 100km→25km distillation spike.

SFT is the sanity-check step: no step reduction, no adversarial loss — just
flow-matching on teacher trajectories.  A decreasing loss and visually plausible
samples at teacher step count confirm the adapter wiring is correct before we
run the more expensive distillation methods.

Gate: student DSM loss decreases; samples generated at the teacher's 18-step
count are visually indistinguishable from teacher samples.

Adapted from:
  fastgen/fastgen/configs/experiments/EDM/config_sft_edm_in64.py
"""

import os

from omegaconf import DictConfig

import fastgen.configs.methods.config_sft as config_sft_default
from fastgen.configs.callbacks import EMA_CONST_CALLBACKS

# Path to the pre-trained ACE teacher checkpoint.
# Override via: config.model.pretrained_model_path = "/path/to/teacher.ckpt"
TEACHER_CKPT_PATH = os.environ.get(
    "ACE_TEACHER_CKPT",
    "",  # must be set before training
)

# Output dimensions: ACE CONUS 100km→25km outputs ~C variables at 512×512
# (coarse 16×16, downscale_factor=32 → fine 512×512).
# Set C_out to the number of output channels from the teacher checkpoint.
C_OUT = int(os.environ.get("ACE_C_OUT", "1"))
H_FINE = int(os.environ.get("ACE_H_FINE", "512"))
W_FINE = int(os.environ.get("ACE_W_FINE", "512"))


def create_config():
    """Return the SFT training config.

    Set ``ACE_TEACHER_CKPT`` in the environment before calling, or update
    ``config.model.pretrained_model_path`` after this call returns.
    To inject a live ``AceDiffusionTeacher`` object, do::

        cfg = create_config()
        cfg.model.net = teacher   # direct assignment, no LazyCall needed
        cfg.model.pretrained_model_path = ""
    """
    config = config_sft_default.create_config()

    # ------------------------------------------------------------------ model
    config.model.input_shape = [C_OUT, H_FINE, W_FINE]
    config.model.precision_amp = "bfloat16"
    config.model.grad_scaler_enabled = False  # bf16 doesn't need grad scaler

    # ACE uses log-normal noise sampling (p_mean=-1.2, p_std=1.2 typical).
    # These match the SFT defaults; override from the teacher config if needed.
    config.model.sample_t_cfg.time_dist_type = "lognormal"
    config.model.sample_t_cfg.train_p_mean = -1.2
    config.model.sample_t_cfg.train_p_std = 1.2

    config.model.pretrained_model_path = TEACHER_CKPT_PATH

    # Optimizer: lower LR for fine-tuning.
    config.model.net_optimizer.optim_type = "adam"
    config.model.net_optimizer.lr = 2e-5
    config.model.net_optimizer.betas = (0.9, 0.999)
    config.model.net_optimizer.weight_decay = 0.0

    # EMA
    config.model.use_ema = ["ema_9999", "ema_99995"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_CONST_CALLBACKS)

    # Sampling steps used for visualisation during training.
    # Keep at the teacher step count for the SFT sanity check.
    config.model.student_sample_steps = 18

    # ----------------------------------------------------------------- trainer
    config.trainer.ddp = True
    config.trainer.batch_size_global = 32  # start small for GPU budget
    config.trainer.max_iter = 50_000
    config.trainer.save_ckpt_iter = 5_000
    config.trainer.logging_iter = 500

    config.log_config.group = ""

    return config
