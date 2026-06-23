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
import math
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import torch
from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import EDMNoiseSchedule
from torch.nn.parallel import DistributedDataParallel as DDP

if TYPE_CHECKING:
    from fme.downscaling.models import DiffusionModel
    from fme.downscaling.predictors.serial_denoising import DenoisingMoEPredictor


class AceEDMNoiseSchedule(EDMNoiseSchedule):
    """EDM noise schedule extended with a ``"loguniform"`` time distribution.

    ACE teachers trained with ``LogUniformNoiseDistribution`` (e.g. the
    multivariate MoE teachers) draw sigma log-uniformly over
    ``[sigma_min, sigma_max]``.  FastGen's stock EDM schedule has no
    equivalent ``time_dist_type``, so it is added here rather than patching
    the vendored FastGen package.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._supported_time_dist_types = tuple(  # type: ignore
            list(self._supported_time_dist_types) + ["loguniform"]  # type: ignore
        )

    def sample_t(
        self,
        n: int,
        time_dist_type: str = "polynomial",
        min_t: float | None = 0.002,
        max_t: float | None = 80.0,
        device: torch.device | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if time_dist_type != "loguniform":
            return super().sample_t(
                n,
                time_dist_type=time_dist_type,
                min_t=min_t,
                max_t=max_t,
                device=device,
                **kwargs,
            )
        min_t = max(min_t, self.min_t) if min_t is not None else self.min_t
        max_t = min(max_t, self.max_t) if max_t is not None else self.max_t
        target_device = device or self._sigmas.device
        log_min = math.log(max(min_t, self.clamp_min))
        log_max = math.log(max_t)  # type: ignore
        u = torch.rand(n, device=target_device, dtype=self.t_precision)
        t = torch.exp(u * (log_max - log_min) + log_min)
        return self.safe_clamp(t, min_t, max_t)


class AceDiffusionTeacher(FastGenNetwork):
    """FastGenNetwork wrapping an ACE DiffusionModel (or DenoisingMoEPredictor)
    for distillation.

    Parameters
    ----------
    model:
        A fully loaded ``DiffusionModel`` or ``DenoisingMoEPredictor``.
        Weights are **not frozen** by default; call ``freeze()`` before
        distillation training (f-distill / sCM / DMD2) and omit for SFT.

    Notes:
    -----
    - ``forward()`` returns an x0 prediction (``net_pred_type = "x0"``).
    - For a ``DenoisingMoEPredictor`` the original teacher instance routes each
      sample to the expert whose sigma range contains its noise level (see
      ``_dispatch``), so x0-target generation and scoring use the correct
      expert at every noise level.  Deepcopies (student / frozen teacher copy)
      drop the expert list and use the single high-noise expert they were
      initialised from.  That high-noise expert's UNet is also used for
      encoder-feature introspection (DMD2 discriminator wiring).
    """

    def __init__(self, model: DiffusionModel | DenoisingMoEPredictor) -> None:
        from fme.core.distributed.non_distributed import DummyWrapper
        from fme.downscaling.predictors.serial_denoising import DenoisingMoEPredictor

        def _unwrap(m: torch.nn.Module) -> torch.nn.Module:
            # Unwrap DDP / DummyWrapper to reach the bare UNetDiffusionModule.
            raw = m.module if isinstance(m, DDP) else m
            if isinstance(raw, DummyWrapper):
                raw = raw.module
            return raw

        if isinstance(model, DenoisingMoEPredictor):
            sigma_min = model._sigma_schedule_min
            sigma_max = model._sigma_schedule_max
            churn = model._churn
            num_steps = model._num_diffusion_generation_steps
            # Keep every expert's bare module alongside its sigma range so
            # target generation and scoring route to the correct expert at
            # every noise level (see ``_dispatch``).  ``model._experts`` and
            # ``model._sigma_ranges`` are sorted by sigma ascending.
            expert_modules = [_unwrap(e.module) for e in model._experts]
            moe_sigma_ranges: list[tuple[float, float]] | None = list(
                model._sigma_ranges
            )
            # Initialise the student / discriminator feature extractor from the
            # HIGH-noise expert (the last, highest-sigma range).  The student's
            # first generation step starts near sigma_max, so the high-noise
            # expert is the better single-expert initialisation; it is also the
            # larger network, which the auto-derived discriminator inherits.
            ace_module: torch.nn.Module = expert_modules[-1]
            sigma_data = model._experts[-1].sigma_data
            moe_experts: list[torch.nn.Module] | None = expert_modules
        else:
            cfg = model.config
            sigma_min = cfg.sigma_min
            sigma_max = cfg.sigma_max
            churn = cfg.churn
            num_steps = cfg.num_diffusion_generation_steps
            sigma_data = model.sigma_data
            # Module chain: DiffusionModel.module → DummyWrapper → UNetDiffusionModule
            ace_module = _unwrap(model.module)
            moe_experts = None
            moe_sigma_ranges = None

        super().__init__(
            net_pred_type="x0",
            schedule_type="edm",
            min_t=sigma_min,
            max_t=sigma_max,
        )
        self._ace_module = ace_module
        self._primary_ace_module = ace_module
        # Store the MoE experts ONLY on the original instance so that
        # deepcopies (student / teacher copy) don't carry all expert weights.
        self._moe_experts = moe_experts
        self._moe_sigma_ranges = moe_sigma_ranges

        # Stash sampling hyperparameters.
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._churn = churn
        self._default_num_steps = num_steps
        self._sigma_data = sigma_data

        # Learnable per-sample log-variance used by sCM's adaptive loss
        # weighting.  A global scalar (not noise-level-dependent) is sufficient
        # for the EDM teacher.  FastGen loads with strict=False so this new
        # parameter initialises fresh without disturbing teacher weights.
        self.logvar_scalar = torch.nn.Parameter(torch.full((1,), -9.0))

    def set_noise_schedule(
        self, schedule_type: str | None = None, **noise_schedule_kwargs
    ) -> None:
        """Build the noise schedule with ``"loguniform"`` support.

        FastGen's ``reset_parameters`` re-invokes this with no arguments,
        which under the base implementation would rebuild the schedule with
        default EDM bounds (sigma in [0.002, 80]) and silently lose the
        teacher's sigma range, so the construction kwargs are remembered.
        """
        if schedule_type is not None and schedule_type != "edm":
            raise ValueError(
                f"AceDiffusionTeacher only supports the 'edm' noise schedule, "
                f"got {schedule_type!r}"
            )
        self.schedule_type = "edm"
        if noise_schedule_kwargs:
            self._noise_schedule_kwargs = noise_schedule_kwargs
        self.noise_scheduler = AceEDMNoiseSchedule(
            **getattr(self, "_noise_schedule_kwargs", {})
        )

    def __deepcopy__(self, memo: dict) -> AceDiffusionTeacher:
        """Return a copy that does NOT carry all MoE expert weights.

        When FastGen deepcopies this teacher (for student and frozen-teacher
        initialisation), we want only the primary expert's weights — not the
        full expert list.  ``_moe_experts`` is set to ``None`` in the copy so
        that ``freeze``/``unfreeze`` operate only on ``_ace_module``.
        """
        import copy

        cls = type(self)
        result = object.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_moe_experts":
                object.__setattr__(result, k, None)
            else:
                object.__setattr__(result, k, copy.deepcopy(v, memo))
        return result

    def freeze(self) -> None:
        """Freeze teacher weights (call this for f-distill / sCM / DMD2).

        Only the ACE UNet weights are frozen; logvar_scalar stays trainable
        so sCM can learn the adaptive loss weighting.
        """
        if self._moe_experts is not None:
            for m in self._moe_experts:
                m.requires_grad_(False)
                m.eval()
        self._ace_module.requires_grad_(False)
        self._ace_module.eval()

    def unfreeze(self) -> None:
        """Allow teacher weights to be updated (call this for SFT fine-tuning)."""
        if self._moe_experts is not None:
            for m in self._moe_experts:
                m.requires_grad_(True)
                m.train()
        self._ace_module.requires_grad_(True)
        self._ace_module.train()

    # ------------------------------------------------------------------
    # Encoder feature introspection (used by DMD2 discriminator wiring)
    # ------------------------------------------------------------------

    def _get_songunet(self) -> torch.nn.Module:
        """Traverse wrappers to reach the SongUNet (the module with `.enc`).

        Actual chain: UNetDiffusionModule → .unet (EDMPrecond) → .model (SongUNet).
        We step through known wrapper attributes until we find a module that
        exposes `.enc`, so this is robust to future wrapper changes.

        For MoE teachers the primary expert's module is used (the dispatch
        module is not a nn.Module and cannot be traversed).
        """
        m = self._primary_ace_module
        for attr in ("unet", "model", "module"):
            candidate = getattr(m, attr, None)
            if candidate is not None and hasattr(candidate, "enc"):
                return candidate
            if candidate is not None:
                m = candidate
        if hasattr(m, "enc"):
            return m
        raise AttributeError(
            f"Cannot find SongUNet with .enc; "
            f"outermost module is {type(self._ace_module).__name__}"
        )

    def encoder_feature_info(self) -> list[tuple[str, int, int]]:
        """Return ``(block_key, out_channels, resolution)`` for the last encoder
        block at each UNet level, ordered finest→coarsest.

        Both ``SongUNet`` (v1) and ``SongUNetv2`` name their encoder blocks
        ``"{res}x{res}_block{idx}"``.  The last block at each resolution is the
        one whose activations are used as discriminator features.
        """
        unet = self._get_songunet()
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
                block = self._get_songunet().enc[block_key]

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
    # MoE sigma dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        x_t: torch.Tensor,
        condition: Any,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Route each sample to the expert whose sigma range contains its
        noise level.

        Used by MoE teachers so x0-target generation and the teacher score use
        the correct expert at every sigma.  Matches ``_SigmaDispatchModule``
        semantics: ranges are inclusive, a sigma on a shared boundary routes to
        the lower-sigma expert, and a sigma outside the union of ranges routes
        to the nearest boundary expert.

        Handles both per-sample ``t`` (shape ``[B]`` or ``[B, 1, 1, 1]``) and a
        single scalar ``t`` broadcast across the batch (the EDM sampler path).
        """
        assert self._moe_experts is not None and self._moe_sigma_ranges is not None
        t_flat = t.reshape(-1)
        broadcast = t_flat.numel() == 1
        route = t_flat.expand(x_t.shape[0]) if broadcast else t_flat
        out = torch.empty_like(x_t)
        remaining = torch.ones(x_t.shape[0], dtype=torch.bool, device=x_t.device)

        def _run(module: torch.nn.Module, sel: torch.Tensor) -> None:
            cond_sel = condition[sel] if condition is not None else None
            # Pass the scalar t directly when broadcasting (it broadcasts over
            # the selected subset); otherwise index t to preserve its shape.
            sigma_sel = t if broadcast else t[sel]
            out[sel] = module(x_t[sel], cond_sel, sigma_sel)

        for (lo, hi), module in zip(self._moe_sigma_ranges, self._moe_experts):
            sel = remaining & (route >= lo) & (route <= hi)
            if sel.any():
                _run(module, sel)
                remaining = remaining & ~sel
        if remaining.any():
            below = remaining & (route < self._moe_sigma_ranges[0][0])
            if below.any():
                _run(self._moe_experts[0], below)
            above = remaining & ~below
            if above.any():
                _run(self._moe_experts[-1], above)
        return out

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
            return_logvar: If True, return ``(x0, logvar)`` where ``logvar``
                is a ``[B]`` tensor from a learnable global scalar.  Required
                by sCM's adaptive loss weighting.
            fwd_pred_type: Unused; FastGenNetwork interface compat.
            **fwd_kwargs: Unused; FastGenNetwork interface compat.

        Returns:
            - Normal: x0 prediction ``[B, C_out, H, W]``.
            - ``return_logvar=True``: ``(x0 [B, C_out, H, W], logvar [B])``.
            - ``return_features_early=True``: list of captured feature tensors.
            - ``feature_indices`` only: ``[x0, [feat_0, feat_1, ...]]``.
        """
        if not feature_indices:
            with torch.amp.autocast(x_t.device.type, dtype=torch.bfloat16):
                if self._moe_experts is not None:
                    x0 = self._dispatch(x_t, condition, t)
                else:
                    x0 = self._ace_module(x_t, condition, t)
            if return_logvar:
                logvar = self.logvar_scalar.expand(x_t.shape[0])
                return x0, logvar
            return x0

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
        # MoE teachers route each sampler step to the expert covering its sigma;
        # single-model teachers (and student copies) use their one net.
        net = self._dispatch if self._moe_experts is not None else self._ace_module
        with torch.no_grad(), torch.amp.autocast(device_type, dtype=torch.bfloat16):
            generated, _ = stochastic_sampler(
                net,
                noise,
                condition,
                num_steps=n_steps,
                sigma_min=self._sigma_min,
                sigma_max=self._sigma_max,
                S_churn=self._churn,
            )
        return generated
