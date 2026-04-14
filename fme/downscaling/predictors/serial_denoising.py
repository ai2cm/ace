import dataclasses

import torch

from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.core.typing_ import TensorDict, TensorMapping
from fme.downscaling.data import BatchData, PairedBatchData, StaticInputs
from fme.downscaling.metrics_and_maths import filter_tensor_mapping, interpolate
from fme.downscaling.models import (
    CheckpointModelConfig,
    DiffusionModel,
    ModelOutputs,
    _repeat_batch_by_samples,
    _separate_interleaved_samples,
)
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.samplers import stochastic_sampler as edm_sampler


@dataclasses.dataclass
class DenoisingRangeModelConfig:
    """
    One expert checkpoint and the sigma interval (inclusive) it handles.

    Ranges across experts must be sorted by ``sigma_min`` ascending, contiguous
    (each ``sigma_max`` equals the next ``sigma_min``), and non-overlapping.
    """

    checkpoint_config: CheckpointModelConfig
    sigma_min: float
    sigma_max: float


def _validate_sigma_ranges(configs: list[DenoisingRangeModelConfig]) -> None:
    if not configs:
        raise ValueError("denoising_range_configs must contain at least one entry.")
    for c in configs:
        if c.sigma_min >= c.sigma_max:
            raise ValueError(
                f"Each range needs sigma_min < sigma_max; got "
                f"[{c.sigma_min}, {c.sigma_max}]."
            )
    for i in range(len(configs) - 1):
        if configs[i].sigma_min >= configs[i + 1].sigma_min:
            raise ValueError(
                "denoising_range_configs must be sorted by sigma_min ascending."
            )
    for i in range(len(configs) - 1):
        if configs[i].sigma_max != configs[i + 1].sigma_min:
            raise ValueError(
                "Sigma ranges must be contiguous: "
                f"configs[{i}].sigma_max ({configs[i].sigma_max}) must equal "
                f"configs[{i + 1}].sigma_min ({configs[i + 1].sigma_min}). "
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


class _SigmaDispatchModule:
    """
    Routes ``net(x, x_lr, sigma)`` to the expert whose inclusive sigma range
    contains ``sigma``. At a boundary shared by two experts, the one with the
    smaller ``sigma_max`` (lower-noise segment) is used.
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

    def __call__(
        self,
        x: torch.Tensor,
        x_lr: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        s = float(sigma.detach().float().cpu().reshape(-1)[0].item())
        candidates: list[tuple[float, float, torch.nn.Module]] = [
            (lo, hi, mod) for lo, hi, mod in self._entries if lo <= s <= hi
        ]
        if not candidates:
            # Clamp to the nearest boundary expert when sigma falls outside
            # all registered ranges (below the global min or above the global max).
            if s < min(lo for lo, _, _ in self._entries):
                _, _, mod = min(self._entries, key=lambda t: t[0])
                return mod(x, x_lr, sigma)
            else:
                _, _, mod = max(self._entries, key=lambda t: t[1])
                return mod(x, x_lr, sigma)
        # Prefer the expert with the smallest sigma_max (lower-noise range).
        _lo, _hi, mod = min(candidates, key=lambda t: (t[1], t[0]))
        return mod(x, x_lr, sigma)


@dataclasses.dataclass
class DenoisingMoECheckpointConfig:
    """
    Load multiple checkpoints and route denoising steps to the expert whose
    sigma range contains the current noise level.

    ``denoising_range_configs`` must list non-overlapping contiguous intervals
    in ascending ``sigma_min`` order. Overall schedule bounds are the minimum
    ``sigma_min`` and maximum ``sigma_max`` across ranges.

    Parameters:
        denoising_range_configs: One entry per expert (checkpoint + sigma range).
        num_diffusion_generation_steps: EDM sampler step count for the full schedule.
        churn: EDM sampler churn (stochasticity) for the full schedule.
    """

    denoising_range_configs: list[DenoisingRangeModelConfig]
    num_diffusion_generation_steps: int
    churn: float = 0.0

    def __post_init__(self) -> None:
        _validate_sigma_ranges(self.denoising_range_configs)

    def build(self) -> "DenoisingScheduleSequentialPredictor":
        experts = [rc.checkpoint_config.build() for rc in self.denoising_range_configs]
        _validate_experts_compatible(experts)
        sigma_ranges = [
            (rc.sigma_min, rc.sigma_max) for rc in self.denoising_range_configs
        ]
        return DenoisingScheduleSequentialPredictor(
            experts=experts,
            sigma_ranges=sigma_ranges,
            num_diffusion_generation_steps=self.num_diffusion_generation_steps,
            churn=self.churn,
        )

    @property
    def data_requirements(self) -> DataRequirements:
        return self.denoising_range_configs[0].checkpoint_config.data_requirements


class DenoisingScheduleSequentialPredictor:
    """
    Multiple ``DiffusionModel`` experts, each used for part of the EDM sigma
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
        self._sigma_schedule_min = min(lo for lo, _ in sigma_ranges)
        self._sigma_schedule_max = max(hi for _, hi in sigma_ranges)
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
        inputs_ = self._primary._get_input_from_coarse(coarse_data, static_inputs)
        inputs_ = _repeat_batch_by_samples(inputs_, n_samples)
        coarse_input_shape = next(iter(coarse_data.values())).shape[-2:]

        outputs_shape = (
            inputs_.shape[0],
            len(self._primary.out_packer.names),
            *self._primary._get_fine_shape(coarse_input_shape),
        )
        latents = torch.randn(outputs_shape).to(device=get_device())

        generated_norm, latent_steps = edm_sampler(
            self._dispatch_module,
            latents,
            inputs_,
            S_churn=self._churn,
            sigma_min=self._sigma_schedule_min,
            sigma_max=self._sigma_schedule_max,
            num_steps=self._num_diffusion_generation_steps,
        )

        if self._primary.config.predict_residual:
            base_prediction = interpolate(
                self._primary.out_packer.pack(
                    self._primary.normalizer.coarse.normalize(
                        {k: coarse_data[k] for k in self._primary.out_packer.names}
                    ),
                    axis=self._primary._channel_axis,
                ),
                self._primary.downscale_factor,
            )
            generated_norm = generated_norm + _repeat_batch_by_samples(
                base_prediction, n_samples
            )

        generated_norm_reshaped = _separate_interleaved_samples(
            generated_norm, n_samples
        )
        generated = self._primary.normalizer.fine.denormalize(
            self._primary.out_packer.unpack(
                generated_norm_reshaped, axis=self._primary._channel_axis
            )
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
