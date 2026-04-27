# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""
sCM (score Consistency Model) config for ACE CONUS 100km→25km.

sCM has no adversarial term; it uses score-consistency training which has
strong theoretical mass-coverage properties without mode-seeking pressure.
A viable alternative to f-distill if GAN training instability is a concern.

Target: 2 student steps (sCM is designed for very low step counts).

Adapted from:
  fastgen/fastgen/configs/experiments/EDM/config_scd_in64.py
"""

import os

from omegaconf import DictConfig

import fastgen.configs.methods.config_scm as config_scm_default
from fastgen.configs.callbacks import EMA_CONST_CALLBACKS
from fastgen.utils import LambdaInverseSquareRootScheduler
from fastgen.utils import LazyCall as L

TEACHER_CKPT_PATH = os.environ.get("ACE_TEACHER_CKPT", "")

C_OUT = int(os.environ.get("ACE_C_OUT", "1"))
H_FINE = int(os.environ.get("ACE_H_FINE", "512"))
W_FINE = int(os.environ.get("ACE_W_FINE", "512"))

STUDENT_STEPS = int(os.environ.get("ACE_STUDENT_STEPS", "2"))


def create_config():
    """Return the sCM training config.

    Set ``ACE_TEACHER_CKPT`` in the environment before calling, or update
    ``config.model.pretrained_model_path`` after this call returns.
    """
    config = config_scm_default.create_config()

    # ------------------------------------------------------------------ model
    config.model.input_shape = [C_OUT, H_FINE, W_FINE]
    config.model.precision_amp = "bfloat16"
    config.model.precision_amp_jvp = "float32"  # JVP stability

    config.model.pretrained_model_path = TEACHER_CKPT_PATH

    # Noise distribution matching ACE's EDM training.
    config.model.sample_t_cfg.sigma_data = 1.0  # ACE always uses sigma_data=1
    config.model.sample_t_cfg.train_p_mean = -1.2
    config.model.sample_t_cfg.train_p_std = 1.2

    # Optimizer (Adam with conservative LR for sCM)
    config.model.net_optimizer.optim_type = "adam"
    config.model.net_optimizer.lr = 7e-5
    config.model.net_optimizer.betas = (0.9, 0.99)
    config.model.net_optimizer.eps = 1e-11
    config.model.net_optimizer.weight_decay = 0.0
    config.model.net_scheduler = L(LambdaInverseSquareRootScheduler)(
        warm_up_steps=0,
        decay_steps=35_000,
    )

    # Consistency distillation loss (JVP-based, recommended for EDM teacher)
    config.model.loss_config.use_cd = True

    # EMA
    config.model.use_ema = ["ema_9999", "ema_99995"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_CONST_CALLBACKS)
    config.trainer.callbacks.grad_clip.grad_norm = 10

    # Student inference: 2-step (recommended for sCM with EDM teacher)
    config.model.student_sample_steps = STUDENT_STEPS

    # ----------------------------------------------------------------- trainer
    config.trainer.ddp = True
    config.trainer.batch_size_global = 32
    config.trainer.max_iter = 200_000
    config.trainer.save_ckpt_iter = 10_000
    config.trainer.logging_iter = 500

    config.log_config.group = ""

    return config
