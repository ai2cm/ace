# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""
DMD2 (reverse-KL / VSD) baseline config for ACE CONUS 100km→25km.

DMD2 uses the reverse-KL divergence which is mode-seeking.  We expect this
to *fail* the 99.9999th-percentile tail metric.  Running it confirms the
tail-loss hypothesis and provides a baseline comparison vs. f-distill (KL)
and sCM.

Target: 2-4 student steps.

Adapted from:
  fastgen/fastgen/configs/experiments/EDM/config_dmd2_in64.py
"""

import os

from omegaconf import DictConfig

import fastgen.configs.methods.config_dmd2 as config_dmd2_default
from fastgen.configs.callbacks import EMA_CONST_CALLBACKS

TEACHER_CKPT_PATH = os.environ.get("ACE_TEACHER_CKPT", "")

C_OUT = int(os.environ.get("ACE_C_OUT", "1"))
H_FINE = int(os.environ.get("ACE_H_FINE", "512"))
W_FINE = int(os.environ.get("ACE_W_FINE", "512"))

STUDENT_STEPS = int(os.environ.get("ACE_STUDENT_STEPS", "1"))

# Teacher sigma range.  Defaults match the original single-model CONUS
# teacher (sigma in [0.002, 150]); the multivariate MoE teachers use
# [0.005, 200].  run.sh sets these env vars for --moe-teacher runs.
SIGMA_MIN = float(os.environ.get("ACE_SIGMA_MIN", "0.002"))
SIGMA_MAX = float(os.environ.get("ACE_SIGMA_MAX", "150.0"))


def create_config():
    """Return the DMD2 reverse-KL baseline training config.

    Set ``ACE_TEACHER_CKPT`` in the environment before calling, or update
    ``config.model.pretrained_model_path`` after this call returns.
    """
    config = config_dmd2_default.create_config()

    # ------------------------------------------------------------------ model
    config.model.input_shape = [C_OUT, H_FINE, W_FINE]
    config.model.precision_amp = "bfloat16"
    config.model.grad_scaler_enabled = False

    # Sample t from the Karras sigma grid, which the AceDiffusionTeacher
    # builds over the teacher checkpoint's [sigma_min, sigma_max].  min_t and
    # max_t must still span the teacher's full sigma range — the
    # SampleTConfig defaults of [0.002, 80] would otherwise clamp sampled
    # sigmas at 80.
    config.model.sample_t_cfg.time_dist_type = "polynomial"
    config.model.sample_t_cfg.min_t = SIGMA_MIN
    config.model.sample_t_cfg.max_t = SIGMA_MAX

    config.model.pretrained_model_path = TEACHER_CKPT_PATH

    # Optimizers
    config.model.net_optimizer.lr = 2e-6
    config.model.discriminator_optimizer.lr = 2e-6
    config.model.fake_score_optimizer.lr = 2e-6

    # Conservative GAN weight; expected to mode-seek at higher values.
    config.model.gan_loss_weight_gen = 3e-3

    # EMA
    config.model.use_ema = ["ema_9999", "ema_99995"]
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(EMA_CONST_CALLBACKS)

    config.model.student_sample_steps = STUDENT_STEPS

    # ----------------------------------------------------------------- trainer
    config.trainer.ddp = True
    config.trainer.batch_size_global = 32
    config.trainer.max_iter = 100_000
    config.trainer.save_ckpt_iter = 130
    config.trainer.logging_iter = 130

    config.log_config.group = ""

    return config
