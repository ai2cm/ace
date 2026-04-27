# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""
Data adapter that wraps ACE's DataLoaderConfig (coarse-only) and emits the
``{"real": x0_teacher, "condition": cond_tensor}`` dicts expected by
FastGen training loops.

The fine-resolution ground-truth zarr is not required. Instead, the teacher
runs its EDM sampler at each iteration to generate clean x0 targets from the
coarse condition. This is the correct approach for all FastGen distillation
methods (SFT, f-distill, sCM, DMD2): in all cases the student is learning
the teacher's output distribution, not the ground-truth fine distribution.

Usage
-----
    from fme.downscaling.distillation.fastgen_loader import AceConditionBuilder

    builder = AceConditionBuilder(model, teacher)
    for batch in builder.iter_fastgen_batches(gridded_data):
        # batch["real"]      : [B, C_out, H_fine, W_fine]  (teacher sample)
        # batch["condition"] : [B, C_cond, H_fine, W_fine]
        ...
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

from fme.downscaling.data.datasets import BatchData, GriddedData

if TYPE_CHECKING:
    from fme.downscaling.distillation.fastgen_teacher import AceDiffusionTeacher
    from fme.downscaling.models import DiffusionModel


class AceConditionBuilder:
    """Builds FastGen-style ``{real, condition}`` dicts from ACE coarse data.

    The teacher EDM sampler is run at each call to produce clean x0 targets,
    so the fine-resolution zarr dataset is not needed.

    Args:
        model: Fully loaded ``DiffusionModel`` instance (e.g. from
            ``CheckpointModelConfig.build()``).
        teacher: ``AceDiffusionTeacher`` wrapping the same model, used to
            run the EDM sampler for x0 target generation.
    """

    def __init__(
        self,
        model: DiffusionModel,
        teacher: AceDiffusionTeacher,
        teacher_num_steps: int = 0,
    ) -> None:
        self._model = model
        self._teacher = teacher
        self._teacher_num_steps = teacher_num_steps

    def build_fastgen_batch(
        self,
        batch: BatchData,
    ) -> dict[str, torch.Tensor]:
        """Convert one ACE coarse batch to a FastGen training dict.

        Builds the condition tensor from coarse inputs, then runs the teacher
        sampler to generate a clean x0 target.

        Returns:
            ``{"real": x0, "condition": cond}`` where both tensors have shape
            ``[B, C, H_fine, W_fine]``.
        """
        m = self._model
        static_inputs = m._subset_static_if_available(batch)
        condition = m._get_input_from_coarse(batch.data, static_inputs)

        B = condition.shape[0]
        C_out = len(m.out_packer.names)
        H_fine, W_fine = condition.shape[-2:]
        noise = torch.randn(B, C_out, H_fine, W_fine, device=condition.device)

        with torch.no_grad():
            x0 = self._teacher.sample(
                noise, condition, num_steps=self._teacher_num_steps
            )

        return {
            "real": x0,
            "condition": condition,
            "neg_condition": torch.zeros_like(condition),
        }

    def iter_fastgen_batches(
        self,
        data: GriddedData,
        patch_extent_yx: tuple[int, int] | None = None,
        shuffle: bool = False,
        random_offset: bool = False,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Iterate over a ``GriddedData`` loader, yielding FastGen dicts.

        Args:
            data: Built ``GriddedData`` object from
                ``DataLoaderConfig.build()``.
            patch_extent_yx: If provided, yield patched batches of this
                coarse-grid extent (lat, lon). Suitable for patch-based
                training.
            shuffle: If True and patch_extent_yx is set, shuffle patches
                within each time step before yielding.
            random_offset: If True and patch_extent_yx is set, apply a
                random spatial offset to each patch extraction so patch
                boundaries vary across epochs.
        """
        import fastgen.utils.logging_utils as _logger

        if patch_extent_yx is not None:
            _logger.debug(
                f"iter_fastgen_batches: using patch extent yx={patch_extent_yx}"
            )
            gen = data.get_patched_generator(
                yx_patch_extent=patch_extent_yx,
                shuffle=shuffle,
                random_offset=random_offset,
            )
        else:
            _logger.debug("iter_fastgen_batches: using full-domain generator")
            gen = data.get_generator()

        _logger.debug(
            "iter_fastgen_batches: generator created, waiting for first batch..."
        )
        for batch in gen:
            yield self.build_fastgen_batch(batch)
