import dataclasses

import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.typing_ import TensorDict, TensorMapping
from fme.downscaling.data import BatchData, PairedBatchData, StaticInputs
from fme.downscaling.metrics_and_maths import filter_tensor_mapping
from fme.downscaling.models import (
    CheckpointModelConfig,
    DiffusionModel,
    ModelOutputs,
    _repeat_batch_by_samples,
)
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.samplers import stochastic_sampler as edm_sampler


@dataclasses.dataclass
class DenoisingExpertCheckpointConfig:
    """
    One expert checkpoint and the sigma interval (inclusive) it handles.
    """

    checkpoint_config: CheckpointModelConfig
    sigma_min: float
    sigma_max: float


def _validate_sigma_ranges(sigma_ranges: list[tuple[float, float]]) -> None:
    if not sigma_ranges:
        raise ValueError("sigma_ranges must contain at least one entry.")
    for s_min, s_max in sigma_ranges:
        if s_min >= s_max:
            raise ValueError(
                f"Each range needs sigma_min < sigma_max; got " f"[{s_min}, {s_max}]."
            )
    for i in range(len(sigma_ranges) - 1):
        if sigma_ranges[i][0] >= sigma_ranges[i + 1][0]:
            raise ValueError("sigma_ranges must be sorted by sigma_min ascending.")
    for i in range(len(sigma_ranges) - 1):
        if sigma_ranges[i][1] != sigma_ranges[i + 1][0]:
            raise ValueError(
                "Sigma ranges must be contiguous: "
                f"sigma_ranges[{i}] max ({sigma_ranges[i][1]}) must equal "
                f"sigma_ranges[{i + 1}] min ({sigma_ranges[i + 1][0]}). "
                "List ranges in ascending sigma order."
            )


def _validate_experts_compatible(experts: list[DiffusionModel]) -> None:
    primary = experts[0]
    for m in experts[1:]:
        if m.in_packer.names != primary.in_packer.names:
            raise ValueError(
                "All experts must share the same input names; "
                f"got {m.in_packer.names} vs {primary.in_packer.names}."
            )
        if m.out_packer.names != primary.out_packer.names:
            raise ValueError(
                "All experts must share the same output names; "
                f"got {m.out_packer.names} vs {primary.out_packer.names}."
            )
        if m.coarse_shape != primary.coarse_shape:
            raise ValueError(
                "All experts must share the same coarse_shape; "
                f"got {m.coarse_shape} vs {primary.coarse_shape}."
            )
        if m.downscale_factor != primary.downscale_factor:
            raise ValueError(
                "All experts must share the same downscale_factor; "
                f"got {m.downscale_factor} vs {primary.downscale_factor}."
            )
        if m.config.predict_residual != primary.config.predict_residual:
            raise ValueError(
                "All experts must share the same predict_residual setting."
            )
        if (m.static_inputs is None) != (primary.static_inputs is None):
            raise ValueError(
                "All experts must either have static_inputs=None or all "
                "must have static_inputs; got a mix."
            )
        if (
            m.static_inputs is not None
            and primary.static_inputs is not None
            and m.static_inputs.coords != primary.static_inputs.coords
        ):
            raise ValueError(
                "All experts must share the same static_inputs coordinates."
            )


class _SigmaDispatchModule:
    """
    Routes ``net(x, x_lr, sigma)`` to the expert whose inclusive sigma range
    contains ``sigma``. At a boundary shared by two experts, the one with the
    smaller ``sigma_max`` (lower-noise segment) is used. If sigma is outside
    the range of all experts, the nearest boundary expert is used.

    Assumes the sigma ranges and corresponding modules are sorted by
    sigma_min ascending.
    """

    def __init__(
        self,
        sigma_ranges: list[tuple[float, float]],
        modules: list[torch.nn.Module],
    ) -> None:
        if len(sigma_ranges) != len(modules):
            raise ValueError("sigma_ranges and modules must have the same length.")
        self._entries: list[tuple[float, float, torch.nn.Module]] = list(
            zip(
                [lo for lo, _ in sigma_ranges],
                [hi for _, hi in sigma_ranges],
                modules,
                strict=True,
            )
        )
        _validate_sigma_ranges(sigma_ranges)

        self._min_sigma = sigma_ranges[0][0]
        self._max_sigma = sigma_ranges[-1][1]

    def __call__(
        self,
        x: torch.Tensor,
        x_lr: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        s = float(sigma.item())
        candidates: list[tuple[float, float, torch.nn.Module]] = [
            (lo, hi, module) for lo, hi, module in self._entries if lo <= s <= hi
        ]
        if not candidates:
            # Clamp to the nearest boundary expert when sigma falls outside
            # all registered ranges (below the global min or above the global max).
            if s < min(lo for lo, _, _ in self._entries):
                _, _, module = self._entries[0]
                return module(x, x_lr, sigma)
            else:
                _, _, module = self._entries[-1]
                return module(x, x_lr, sigma)
        # if on boundary, uses expert with the smallest sigma_max
        _lo, _hi, module = candidates[0]
        return module(x, x_lr, sigma)


@dataclasses.dataclass
class DenoisingMoEConfig:
    """
    Configuration for mixture of experts specializing in different parts of
    the denoising schedule.
    Loads multiple checkpoints and route denoising steps to the expert whose
    sigma range contains the current noise level.

    ``denoising_range_configs`` must list non-overlapping contiguous intervals.
    The overall schedule bounds are the minimum ``sigma_min`` and maximum ``sigma_max``
    across all ranges.

    Parameters:
        denoising_expert_configs: One entry per expert (checkpoint + sigma range).
        num_diffusion_generation_steps: EDM sampler step count for the full schedule.
        churn: EDM sampler churn (stochasticity) for the full schedule. Default 0.0.
    """

    denoising_expert_configs: list[DenoisingExpertCheckpointConfig]
    num_diffusion_generation_steps: int
    churn: float = 0.0

    def __post_init__(self) -> None:
        self.denoising_expert_configs = sorted(
            self.denoising_expert_configs, key=lambda c: c.sigma_min
        )

    def build(self) -> "DenoisingScheduleMoEPredictor":
        experts = [rc.checkpoint_config.build() for rc in self.denoising_expert_configs]
        _validate_experts_compatible(experts)
        sigma_ranges = [
            (rc.sigma_min, rc.sigma_max) for rc in self.denoising_expert_configs
        ]
        return DenoisingScheduleMoEPredictor(
            experts=experts,
            sigma_ranges=sigma_ranges,
            num_diffusion_generation_steps=self.num_diffusion_generation_steps,
            churn=self.churn,
        )

    @property
    def data_requirements(self) -> DataRequirements:
        return self.denoising_expert_configs[0].checkpoint_config.data_requirements


class DenoisingScheduleMoEPredictor:
    """
    Mixture of ``DiffusionModel`` experts, each used for part of the EDM sigma
    schedule. Behaves like ``DiffusionModel`` for generation and patching.
    """

    def __init__(
        self,
        experts: list[DiffusionModel],
        sigma_ranges: list[tuple[float, float]],
        num_diffusion_generation_steps: int,
        churn: float,
    ) -> None:
        if not experts:
            raise ValueError("experts must be non-empty.")
        if len(experts) != len(sigma_ranges):
            raise ValueError("experts and sigma_ranges must have the same length.")
        self._experts = experts
        self._primary = experts[0]
        self._sigma_ranges = sigma_ranges
        _validate_sigma_ranges(sigma_ranges)
        self._sigma_schedule_min = sigma_ranges[0][0]
        self._sigma_schedule_max = sigma_ranges[-1][1]
        self._num_diffusion_generation_steps = num_diffusion_generation_steps
        self._churn = churn
        self._dispatch_module = _SigmaDispatchModule(
            sigma_ranges,
            [e.module for e in experts],
        )

    @property
    def modules(self) -> torch.nn.ModuleList:
        return torch.nn.ModuleList([e.module for e in self._experts])

    @property
    def coarse_shape(self) -> tuple[int, int]:
        return self._primary.coarse_shape

    @property
    def downscale_factor(self) -> int:
        return self._primary.downscale_factor

    @property
    def fine_shape(self) -> tuple[int, int]:
        return self._primary.fine_shape

    @property
    def full_fine_coords(self) -> LatLonCoordinates:
        return self._primary.full_fine_coords

    @property
    def static_inputs(self) -> StaticInputs | None:
        return self._primary.static_inputs

    def get_fine_coords_for_batch(self, batch: BatchData) -> LatLonCoordinates:
        return self._primary.get_fine_coords_for_batch(batch)

    @torch.no_grad()
    def generate(
        self,
        coarse_data: TensorMapping,
        static_inputs: StaticInputs | None,
        n_samples: int = 1,
    ) -> tuple[TensorDict, torch.Tensor, list[torch.Tensor]]:
        latents, inputs = self._primary.prepare_generation_inputs(
            coarse_data, static_inputs, n_samples
        )
        generated_norm, latent_steps = edm_sampler(
            self._dispatch_module,
            latents,
            inputs,
            S_churn=self._churn,
            sigma_min=self._sigma_schedule_min,
            sigma_max=self._sigma_schedule_max,
            num_steps=self._num_diffusion_generation_steps,
        )
        generated, generated_norm = self._primary.postprocess_generated(
            generated_norm, coarse_data, n_samples
        )
        return generated, generated_norm, latent_steps

    @torch.no_grad()
    def generate_on_batch_no_target(
        self,
        batch: BatchData,
        n_samples: int = 1,
    ) -> TensorDict:
        _static_inputs = self._primary._subset_static_if_available(batch)
        generated, _, _ = self.generate(batch.data, _static_inputs, n_samples)
        return generated

    @torch.no_grad()
    def generate_on_batch(
        self,
        batch: PairedBatchData,
        n_samples: int = 1,
    ) -> ModelOutputs:
        _static_inputs = self._primary._subset_static_if_available(batch.coarse)
        coarse, fine = batch.coarse.data, batch.fine.data
        generated, generated_norm, latent_steps = self.generate(
            coarse, _static_inputs, n_samples
        )

        targets_norm = self._primary.out_packer.pack(
            self._primary.normalizer.fine.normalize(dict(fine)),
            axis=self._primary._channel_axis,
        )
        targets_norm = _repeat_batch_by_samples(targets_norm, n_samples)

        targets = filter_tensor_mapping(
            batch.fine.data, set(self._primary.out_packer.names)
        )
        targets = {k: v.unsqueeze(1) for k, v in targets.items()}

        loss = self._primary.loss(generated_norm, targets_norm)
        return ModelOutputs(
            prediction=generated, target=targets, loss=loss, latent_steps=latent_steps
        )
