import dataclasses

from fme.downscaling.models import CheckpointModelConfig
from fme.downscaling.requirements import DataRequirements


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
    churn: float

    def __post_init__(self) -> None:
        _validate_sigma_ranges(self.denoising_range_configs)

    @property
    def data_requirements(self) -> DataRequirements:
        return self.denoising_range_configs[0].checkpoint_config.data_requirements
