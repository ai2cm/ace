# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""
f-distill with forward-KL divergence for ACE CONUS 100km→25km.

Forward KL (f_div="kl") is mass-covering, so the student tends to place weight
on all modes of the teacher rather than collapsing to the most probable mode.
This is the primary distillation candidate for preserving extreme-event tails.

Target: 2-4 student denoising steps.

Primary go/no-go metric:
``generation/histogram/prediction_frac_of_target/99.9999th-percentile``
within ~5% of the teacher.

Adapted from:
  fastgen/fastgen/configs/experiments/EDM/config_f_distill_in64.py
"""

import os

import fastgen.configs.methods.config_f_distill as config_f_distill_default
from fastgen.configs.callbacks import EMA_CONST_CALLBACKS
from omegaconf import DictConfig

TEACHER_CKPT_PATH = os.environ.get("ACE_TEACHER_CKPT", "")

C_OUT = int(os.environ.get("ACE_C_OUT", "1"))
H_FINE = int(os.environ.get("ACE_H_FINE", "512"))
W_FINE = int(os.environ.get("ACE_W_FINE", "512"))

# Number of student steps.  Start at 4; if tail coverage holds, try 2.
STUDENT_STEPS = int(os.environ.get("ACE_STUDENT_STEPS", "4"))


def create_config():
    """Return the f-distill forward-KL training config.

    Set ``ACE_TEACHER_CKPT`` in the environment before calling, or update
    ``config.model.pretrained_model_path`` after this call returns.
    """
    config = config_f_distill_default.create_config()

    # ------------------------------------------------------------------ model
    config.model.input_shape = [C_OUT, H_FINE, W_FINE]
    config.model.precision_amp = "bfloat16"
    config.model.grad_scaler_enabled = False

    # Forward KL (mass-covering) — the critical switch for tail preservation.
    config.model.f_distill.f_div = "kl"
    config.model.f_distill.ratio_ema_rate = 0.5
    config.model.f_distill.ratio_lower = 0.1
    config.model.f_distill.ratio_upper = 20.0
    config.model.f_distill.ratio_normalization = True

    # Noise distribution matching ACE's training distribution.
    config.model.sample_t_cfg.time_dist_type = "lognormal"
    config.model.sample_t_cfg.train_p_mean = -1.2
    config.model.sample_t_cfg.train_p_std = 1.8

    config.model.pretrained_model_path = TEACHER_CKPT_PATH

    # Optimizers
    config.model.net_optimizer.lr = 2e-6
    config.model.discriminator_optimizer.lr = 2e-6
    config.model.fake_score_optimizer.lr = 2e-6

    # GAN loss weight — start conservative for stable training.
    config.model.gan_loss_weight_gen = 1e-3

    # EMA
    config.model.use_ema = ["ema_9999", "ema_99995"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_CONST_CALLBACKS)

    # Student sampling
    config.model.student_sample_steps = STUDENT_STEPS

    # ----------------------------------------------------------------- trainer
    config.trainer.ddp = True
    config.trainer.batch_size_global = 16
    config.trainer.max_iter = 100_000
    config.trainer.save_ckpt_iter = 130
    config.trainer.logging_iter = 130

    config.log_config.group = ""

    return config
