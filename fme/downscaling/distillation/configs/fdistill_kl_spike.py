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

# Number of student steps.  Default 2 — the "intended fdistill" recipe.
# Combined with gan_loss_weight_gen=1e-3 (below) and the loosened
# ratio_upper=100, this lets the forward-KL term actually drive training
# instead of the GAN.  Override with ACE_STUDENT_STEPS to try 1 or 4.
STUDENT_STEPS = int(os.environ.get("ACE_STUDENT_STEPS", "2"))

# Teacher training-noise parameters.  Defaults match the original
# single-model CONUS teacher, trained with sigma ~ lognormal(p_mean=-1.2,
# p_std=1.8) on sigma in [0.002, 150].  The multivariate MoE teachers are
# trained with sigma ~ loguniform on [0.005, 200]; run.sh sets these env
# vars for --moe-teacher runs.
NOISE_DIST = os.environ.get("ACE_NOISE_DIST", "lognormal")
SIGMA_MIN = float(os.environ.get("ACE_SIGMA_MIN", "0.002"))
SIGMA_MAX = float(os.environ.get("ACE_SIGMA_MAX", "150.0"))

# --------------------------------------------------------------- GAN stabilizers
# The MoE distillation runs collapse the coarse PRMSL spectra late in training
# with a classic discriminator-winning signature (gan_loss_disc down,
# gan_loss_gen up, fake_score_loss spike); see MOE_DISTILLATION_STATUS.md.
# These env knobs expose the standard counter-levers so they can be A/B'd
# without code edits.  Defaults reproduce the previous (un-stabilized) behavior.
#
#  - ACE_GAN_R1_REG_WEIGHT: R1 gradient penalty on the discriminator (the
#    textbook fix for a disc that's overpowering the generator).  0.0 = off.
#  - ACE_GAN_LOSS_WEIGHT_GEN: generator-side GAN loss weight (was hard-coded
#    1e-3); lower it to let forward-KL carry more of the signal.
#  - ACE_LR_DECAY_STEPS > 0: linearly decay all three LRs to ACE_LR_F_MIN over
#    this many iters and cap max_iter at the same value (banks the gains before
#    the GAN tips).  0 = no decay (constant LR, the prior behavior).
GAN_R1_REG_WEIGHT = float(os.environ.get("ACE_GAN_R1_REG_WEIGHT", "0.0"))
GAN_LOSS_WEIGHT_GEN = float(os.environ.get("ACE_GAN_LOSS_WEIGHT_GEN", "1e-3"))
LR_DECAY_STEPS = int(os.environ.get("ACE_LR_DECAY_STEPS", "0"))
LR_F_MIN = float(os.environ.get("ACE_LR_F_MIN", "0.05"))


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
    # ratio_upper bumped 20 → 100 to keep the forward-KL gradient alive on
    # extreme-tail samples (where p_target / p_student blows up); ratio_ema_rate
    # bumped 0.5 → 0.9 to keep the EMA stable as the clip loosens.
    config.model.f_distill.f_div = "kl"
    config.model.f_distill.ratio_ema_rate = 0.9
    config.model.f_distill.ratio_lower = 0.1
    config.model.f_distill.ratio_upper = 100.0
    config.model.f_distill.ratio_normalization = True

    # Noise distribution matching the teacher's training distribution.
    # min_t/max_t truncate the lognormal and parameterize the loguniform;
    # they must span the teacher's full sigma range — the SampleTConfig
    # defaults of [0.002, 80] would otherwise clamp sampling at sigma=80.
    config.model.sample_t_cfg.time_dist_type = NOISE_DIST
    config.model.sample_t_cfg.train_p_mean = -1.2  # lognormal only
    config.model.sample_t_cfg.train_p_std = 1.8  # lognormal only
    config.model.sample_t_cfg.min_t = SIGMA_MIN
    config.model.sample_t_cfg.max_t = SIGMA_MAX

    config.model.pretrained_model_path = TEACHER_CKPT_PATH

    # Optimizers at lr=1e-5 — paired with STUDENT_STEPS=1 above this gives
    # fdistill the same training-input distribution as DMD2 (pure noise *
    # sigma_max), where the 1e-5 LR has been observed to train stably.
    config.model.net_optimizer.lr = 1e-5
    config.model.discriminator_optimizer.lr = 1e-5
    config.model.fake_score_optimizer.lr = 1e-5

    # GAN loss weight (default 1e-3, the conservative original).  With
    # STUDENT_STEPS=2 and the loosened ratio clip, the forward-KL term should
    # carry the training signal and we want the GAN to be a stabilizer, not
    # the dominant gradient (which is what 3e-3 + STUDENT_STEPS=1 made it).
    # Lower via ACE_GAN_LOSS_WEIGHT_GEN to further tame the GAN.
    config.model.gan_loss_weight_gen = GAN_LOSS_WEIGHT_GEN

    # R1 discriminator regularization (off by default).  Turning it on is the
    # standard fix for the disc-winning collapse documented for the MoE runs;
    # logs a `gan_loss_ar1` term when both weights are > 0.  Applied in the
    # inherited DMD2 fake-score/discriminator update step.
    config.model.gan_r1_reg_weight = GAN_R1_REG_WEIGHT

    # Optional LR decay (ACE_LR_DECAY_STEPS > 0): linearly decay all three
    # optimizers' LRs to LR_F_MIN over LR_DECAY_STEPS iters.  Cap max_iter at
    # the cycle length so LambdaLinearScheduler never indexes past its single
    # cycle (find_in_interval returns None otherwise → crash) and so the decay
    # completes by the end of the run.
    if LR_DECAY_STEPS > 0:
        for scheduler in (
            config.model.net_scheduler,
            config.model.discriminator_scheduler,
            config.model.fake_score_scheduler,
        ):
            scheduler.warm_up_steps = [0]
            scheduler.cycle_lengths = [LR_DECAY_STEPS]
            scheduler.f_start = [1.0]
            scheduler.f_max = [1.0]
            scheduler.f_min = [LR_F_MIN]

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

    # When LR decay is requested, end the run at the cycle length so the decay
    # reaches LR_F_MIN and the scheduler never indexes past its single cycle.
    if LR_DECAY_STEPS > 0:
        config.trainer.max_iter = LR_DECAY_STEPS

    config.log_config.group = ""

    return config
