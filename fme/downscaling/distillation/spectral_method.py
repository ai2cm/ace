# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""ACE-side f-distill method with an auxiliary spectral-matching loss.

Rather than modify the pinned upstream FastGen submodule, we subclass
``FdistillModel`` and add a band-weighted, per-variable spectral term to the
generator objective. The subclass is selected by pointing ``config.model_class``
at it from the distillation entry point (``fastgen_train.py``); the loss module
and its weight are attached after instantiation via :meth:`set_spectral_loss`.

The spectral term matches the student's zonal power spectrum to that of the
teacher's clean EDM *sample* (``data["real"]``) -- NOT the teacher's x0
*prediction* ``teacher_x0``, which is a conditional mean and hence too smooth.
Both operands live in the same network-output space at the point they are
compared, so the residual/denormalization bookkeeping that the full-field
generation path needs does not apply here.

See ``fme/downscaling/distillation/specs/11-spectral-matching-loss.md``.
"""

from __future__ import annotations

from typing import Any

import torch

# FastGen is an optional, deferred dependency (see spec 01). Importing this
# module requires it, so it is only imported from within the entry point after
# the FastGen availability check, exactly like fastgen_teacher.
from fastgen.methods.common_loss import (
    gan_loss_generator,
    variational_score_distillation_loss,
)
from fastgen.methods.distribution_matching.f_distill import FdistillModel

from fme.downscaling.spectral_loss import SpectralMatchingLoss


class AceFdistillModel(FdistillModel):
    """``FdistillModel`` plus an optional auxiliary spectral-matching loss.

    Behaves identically to the parent until :meth:`set_spectral_loss` is called;
    thereafter the generator loss gains ``weight * spectral_loss(gen_data,
    teacher_sample)`` where ``teacher_sample`` is the clean EDM target
    ``data["real"]``.
    """

    def set_spectral_loss(
        self, spectral_loss: SpectralMatchingLoss, weight: float
    ) -> None:
        # Stored inside a tuple so ``nn.Module.__setattr__`` does not register
        # it as a submodule: the loss carries no trainable parameters and must
        # stay out of the optimizer, DDP parameter list, and state_dict.
        self._ace_spectral = (spectral_loss.to(self.device), float(weight))

    def _student_update_step(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        data: dict[str, Any],
        condition: Any | None = None,
        neg_condition: Any | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        spectral = getattr(self, "_ace_spectral", None)
        if spectral is None or spectral[1] == 0.0:
            return super()._student_update_step(
                input_student,
                t_student,
                t,
                eps,
                data,
                condition=condition,
                neg_condition=neg_condition,
            )
        spectral_loss_module, spectral_weight = spectral

        # Body mirrors FdistillModel._student_update_step (FastGen submodule
        # pinned at 123e6a2), with the spectral term added to the loss. Kept in
        # sync manually because upstream exposes no hook at the loss= line.
        gen_data = self.gen_data_from_net(input_student, t_student, condition=condition)
        perturbed_data = self.net.noise_scheduler.forward_process(gen_data, eps, t)

        with torch.no_grad():
            fake_score_x0 = self.fake_score(
                perturbed_data, t, condition=condition, fwd_pred_type="x0"
            )

        teacher_x0, fake_feat = self.teacher(
            perturbed_data,
            t,
            condition=condition,
            feature_indices=self.discriminator.feature_indices,
            fwd_pred_type="x0",
        )
        fake_logits = self.discriminator(fake_feat)
        gan_loss_gen = gan_loss_generator(fake_logits)

        if self.config.guidance_scale is not None:
            teacher_x0 = self._apply_classifier_free_guidance(
                perturbed_data, t, teacher_x0, neg_condition=neg_condition
            )

        h = self._get_f_div_weighting_h(fake_logits, t)
        f_distill_loss = variational_score_distillation_loss(
            gen_data, teacher_x0, fake_score_x0, additional_scale=h
        )

        # Spectral target is the teacher's clean EDM *sample* (data["real"]),
        # NOT teacher_x0. teacher_x0 is the teacher's x0 *prediction*
        # E[x0 | x_t] -- a conditional mean, which is smoother (less
        # high-wavenumber power) than a sample, so matching its spectrum would
        # push the student toward over-smoothing. The clean sample carries the
        # correct ensemble high-k power. _prepare_training_data is deterministic
        # (just returns data["real"]); augmentation already happened upstream.
        real_data, _, _ = self._prepare_training_data(data)
        spectral_loss = spectral_loss_module(gen_data, real_data)
        weighted_spectral_loss = spectral_weight * spectral_loss
        loss = (
            f_distill_loss
            + self.config.gan_loss_weight_gen * gan_loss_gen
            + weighted_spectral_loss
        )

        rkl = self.config.f_distill.f_div == "rkl"
        one = torch.tensor(1).to(self.device)
        loss_map = {
            "total_loss": loss,
            "f_distill_loss": f_distill_loss,
            "gan_loss_gen": gan_loss_gen,
            "spectral_loss": spectral_loss,
            "spectral_loss_weighted": weighted_spectral_loss,
            "min_h": h.min() if not rkl else one,
            "avg_h": h.mean() if not rkl else one,
            "max_h": h.max() if not rkl else one,
        }
        outputs = self._get_outputs(gen_data, input_student, condition=condition)
        return loss_map, outputs
