# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""
Data adapter that wraps ACE's PairedDataLoaderConfig and emits the
``{"real": fine_tensor, "condition": cond_tensor}`` dicts expected by
FastGen training loops.

Usage
-----
    from fme.downscaling.distillation.fastgen_loader import AceConditionBuilder

    builder = AceConditionBuilder(model)
    for batch in builder.iter_fastgen_batches(data_config, train=True):
        # batch["real"]      : [B, C_fine, H_fine, W_fine]
        # batch["condition"] : [B, C_cond, H_fine, W_fine]
        ...
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

from fme.downscaling.condition import build_condition_tensor
from fme.downscaling.data import StaticInputs
from fme.downscaling.data.datasets import PairedBatchData, PairedGriddedData

if TYPE_CHECKING:
    from fme.downscaling.models import DiffusionModel


class AceConditionBuilder:
    """Builds FastGen-style ``{real, condition}`` dicts from ACE paired data.

    Wraps a loaded ``DiffusionModel`` to reuse its packer, normalizer, and
    conditioning config, ensuring teacher and any student share bit-identical
    conditioning tensors.

    Args:
        model: Fully loaded ``DiffusionModel`` instance (e.g. from
            ``CheckpointModelConfig.build()``).
    """

    def __init__(self, model: DiffusionModel) -> None:
        self._model = model

    def build_fastgen_batch(
        self,
        batch: PairedBatchData,
    ) -> dict[str, torch.Tensor]:
        """Convert one ACE paired batch to a FastGen training dict.

        Returns:
            ``{"real": fine_tensor, "condition": cond_tensor}`` where both
            tensors have shape ``[B, C, H_fine, W_fine]``.
        """
        m = self._model
        static_inputs: StaticInputs | None = m._subset_static_if_available(batch.coarse)

        condition = build_condition_tensor(
            coarse=batch.coarse.data,
            packer=m.in_packer,
            coarse_normalizer=m.normalizer.coarse,
            downscale_factor=m.downscale_factor,
            use_fine_topography=m.config.use_fine_topography,
            interpolate_input=bool(m.config._interpolate_input),
            static_inputs=static_inputs,
            channel_axis=m._channel_axis,
        )

        fine_norm = m.out_packer.pack(
            m.normalizer.fine.normalize(dict(batch.fine.data)),
            axis=m._channel_axis,
        )

        if m.config.predict_residual:
            from fme.downscaling.metrics_and_maths import interpolate

            base = interpolate(
                m.out_packer.pack(
                    m.normalizer.coarse.normalize(
                        {k: batch.coarse.data[k] for k in m.out_packer.names}
                    ),
                    axis=m._channel_axis,
                ),
                m.downscale_factor,
            )
            fine_norm = fine_norm - base

        return {"real": fine_norm, "condition": condition}

    def iter_fastgen_batches(
        self,
        data: PairedGriddedData,
        patch_extent_yx: tuple[int, int] | None = None,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Iterate over a ``PairedGriddedData`` loader, yielding FastGen dicts.

        Args:
            data: Built ``PairedGriddedData`` object from
                ``PairedDataLoaderConfig.build()``.
            patch_extent_yx: If provided, yield patched batches of this
                coarse-grid extent (lat, lon). Suitable for patch-based training.
        """
        if patch_extent_yx is not None:
            gen = data.get_patched_generator(
                coarse_yx_patch_extent=patch_extent_yx,
                shuffle=True,
            )
        else:
            gen = data.get_generator()

        for batch in gen:
            yield self.build_fastgen_batch(batch)
