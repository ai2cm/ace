# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""
FastGenNetwork subclass that adapts an ACE DiffusionModel for use as a
teacher (or student initialisation) inside FastGen training loops.

The ACE downscaling model uses:
  - EDM noise schedule:  x_t = x_0 + sigma * eps,  sigma ∈ [sigma_min, sigma_max]
  - sigma_data = 1.0 (standard-score normalisation of outputs)
  - Net prediction type: x0  (EDMPrecond applies c_skip/c_out preconditioning
    so the raw network output is already a clean x0 estimate)
  - Conditioning: a dense tensor [B, C_cond, H_fine, W_fine] built by
    build_condition_tensor() and passed as the second argument to
    EDMPrecond.forward(x, condition, sigma)

This adapter bridges the two APIs so FastGen's SFT / f-distill / sCM / DMD2
training loops can drive teacher forward passes and teacher-guided sampling
without any changes to the ACE model weights.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import torch
from fastgen.networks.network import FastGenNetwork
from torch.nn.parallel import DistributedDataParallel as DDP

if TYPE_CHECKING:
    from fme.downscaling.models import DiffusionModel


class AceDiffusionTeacher(FastGenNetwork):
    """FastGenNetwork wrapping an ACE DiffusionModel for distillation.

    Parameters
    ----------
    model:
        A fully loaded ``DiffusionModel`` (e.g. from
        ``CheckpointModelConfig.build()``).  Weights are **not frozen** by
        default; call ``freeze()`` before distillation training (f-distill /
        sCM / DMD2) and omit the call for SFT fine-tuning.

    Notes:
    -----
    - ``forward()`` returns an x0 prediction (``net_pred_type = "x0"``).
    - The EDM noise schedule is initialised with ``sigma_min``/``sigma_max``
      from the ACE model config, so FastGen's noise sampling aligns with the
      teacher's training distribution.
    - ``sample()`` delegates to ACE's ``stochastic_sampler`` so visualisation
      during distillation training uses the teacher's own Heun-sampler.
    """

    def __init__(self, model: DiffusionModel) -> None:
        cfg = model.config
        super().__init__(
            net_pred_type="x0",
            schedule_type="edm",
            min_t=cfg.sigma_min,
            max_t=cfg.sigma_max,
        )

        # Unwrap DDP so FastGen can re-wrap the bare module itself.
        raw_module = (
            model.module.module if isinstance(model.module, DDP) else model.module
        )
        self._ace_module = raw_module

        # Stash sampling hyperparameters from the teacher config.
        self._sigma_min = cfg.sigma_min
        self._sigma_max = cfg.sigma_max
        self._churn = cfg.churn
        self._default_num_steps = cfg.num_diffusion_generation_steps
        self._sigma_data = model.sigma_data

    def freeze(self) -> None:
        """Freeze teacher weights (call this for f-distill / sCM / DMD2)."""
        self._ace_module.requires_grad_(False)
        self._ace_module.eval()

    def unfreeze(self) -> None:
        """Allow teacher weights to be updated (call this for SFT fine-tuning)."""
        self._ace_module.requires_grad_(True)
        self._ace_module.train()

    # ------------------------------------------------------------------
    # Encoder feature introspection (used by DMD2 discriminator wiring)
    # ------------------------------------------------------------------

    def encoder_feature_info(self) -> list[tuple[str, int, int]]:
        """Return ``(block_key, out_channels, resolution)`` for the last encoder
        block at each UNet level, ordered finest→coarsest.

        Both ``SongUNet`` (v1) and ``SongUNetv2`` name their encoder blocks
        ``"{res}x{res}_block{idx}"``.  The last block at each resolution is the
        one whose activations are used as discriminator features.
        """
        unet = self._ace_module.model
        res_to_info: dict[int, tuple[str, int, int]] = {}
        for key, block in unet.enc.items():
            if "_block" not in key or "aux" in key:
                continue
            parts = key.rsplit("_block", 1)
            try:
                res = int(parts[0].split("x")[0])
                block_idx = int(parts[1])
            except (ValueError, IndexError):
                continue
            channels = getattr(block, "out_channels", None)
            if channels is None:
                continue
            if res not in res_to_info or block_idx > res_to_info[res][2]:
                res_to_info[res] = (key, channels, block_idx)
        result = []
        for res in sorted(res_to_info.keys(), reverse=True):
            key, channels, _ = res_to_info[res]
            result.append((key, channels, res))
        return result

    @contextlib.contextmanager
    def _capture_encoder_features(
        self, feature_indices: set[int]
    ) -> Generator[dict[int, torch.Tensor], None, None]:
        """Context manager: register hooks, run forward, yield captured dict."""
        info = self.encoder_feature_info()
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for fi in sorted(feature_indices):
            if fi < len(info):
                block_key, _, _ = info[fi]
                block = self._ace_module.model.enc[block_key]

                def _make_hook(idx: int):
                    def _hook(
                        module: torch.nn.Module,
                        inp: tuple,
                        out: torch.Tensor,
                    ) -> None:
                        captured[idx] = out

                    return _hook

                handles.append(block.register_forward_hook(_make_hook(fi)))
        try:
            yield captured
        finally:
            for h in handles:
                h.remove()

    # ------------------------------------------------------------------
    # FastGenNetwork abstract interface
    # ------------------------------------------------------------------

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Any = None,
        r: torch.Tensor | None = None,
        return_features_early: bool = False,
        feature_indices: set[int] | None = None,
        return_logvar: bool = False,
        fwd_pred_type: str | None = None,
        **fwd_kwargs,
    ) -> torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        """Denoise ``x_t`` conditioned on ``condition`` at noise level ``t``.

        In FastGen's EDM schedule ``t == sigma``, so we pass ``t`` directly
        as the sigma argument to ACE's ``EDMPrecond``.

        Args:
            x_t: Noisy fine-resolution data, shape ``[B, C_out, H, W]``.
            t: Sigma (noise level), shape ``[B]`` or broadcastable.
            condition: Dense conditioning tensor ``[B, C_cond, H, W]`` built
                by ``build_condition_tensor`` / ``AceConditionBuilder``.
            r: Unused; accepted for FastGenNetwork interface compatibility.
            return_features_early: If True with ``feature_indices``, returns
                just the captured encoder feature list (no x0).
            feature_indices: Set of encoder-level indices whose last block
                activations to capture.  Index 0 = finest resolution.
            return_logvar: Unused; FastGenNetwork interface compat.
            fwd_pred_type: Unused; FastGenNetwork interface compat.
            **fwd_kwargs: Unused; FastGenNetwork interface compat.

        Returns:
            - Normal: x0 prediction ``[B, C_out, H, W]``.
            - ``return_features_early=True``: list of captured feature tensors.
            - ``feature_indices`` only: ``[x0, [feat_0, feat_1, ...]]``.
        """
        if not feature_indices:
            with torch.amp.autocast(x_t.device.type, dtype=torch.bfloat16):
                return self._ace_module(x_t, condition, t)

        with self._capture_encoder_features(feature_indices) as captured:
            with torch.amp.autocast(x_t.device.type, dtype=torch.bfloat16):
                x0 = self._ace_module(x_t, condition, t)

        feat_list = [captured[i] for i in sorted(feature_indices) if i in captured]

        if return_features_early:
            return feat_list
        return [x0, feat_list]

    # ------------------------------------------------------------------
    # Sampling (used by FastGen for visualisation during training)
    # ------------------------------------------------------------------

    def sample(
        self,
        noise: torch.Tensor,
        condition: Any | None = None,
        neg_condition: Any | None = None,
        guidance_scale: float | None = None,
        num_steps: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Generate a sample using ACE's Heun/EDM sampler.

        Args:
            noise: Initial Gaussian noise, shape ``[B, C_out, H, W]``.
            condition: Dense conditioning tensor ``[B, C_cond, H, W]``.
            neg_condition: Unused; FastGenNetwork interface compat.
            guidance_scale: Unused; FastGenNetwork interface compat.
            num_steps: Number of Heun steps. Falls back to the teacher's
                configured step count when 0 or not provided.
            **kwargs: Unused; FastGenNetwork interface compat.
        """
        from fme.downscaling.samplers import stochastic_sampler

        n_steps = num_steps if num_steps > 0 else self._default_num_steps
        device_type = noise.device.type
        with torch.no_grad(), torch.amp.autocast(device_type, dtype=torch.bfloat16):
            generated, _ = stochastic_sampler(
                self._ace_module,
                noise,
                condition,
                num_steps=n_steps,
                sigma_min=self._sigma_min,
                sigma_max=self._sigma_max,
                S_churn=self._churn,
            )
        return generated
