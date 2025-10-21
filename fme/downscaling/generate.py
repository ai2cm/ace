"""Generate and save downscaled data using a trained FME model."""

from dataclasses import dataclass, field

from fme.core.dataset.time import RepeatedInterval, TimeSlice
from fme.core.logging_utils import LoggingConfig
from fme.core.typing_ import Slice

from .data import ClosedInterval, DataLoaderConfig
from .models import CheckpointModelConfig
from .predictors import CascadePredictorConfig, PatchPredictionConfig


@dataclass
class OutputTargetConfig:
    name: str
    n_ens: int
    save_vars: list[str]
    time: str | int | RepeatedInterval | Slice | TimeSlice | None = None
    time_format: str = "%Y-%m-%dT%H:%M:%S"
    lat_extent: ClosedInterval = field(
        default_factory=lambda: ClosedInterval(-90.0, 90.0)
    )
    lon_extent: ClosedInterval = field(
        default_factory=lambda: ClosedInterval(float("-inf"), float("inf"))
    )
    zarr_chunks: dict[str, int] | None = None
    zarr_shards: dict[str, int] | None = None


@dataclass
class GenerationConfig:
    model: CheckpointModelConfig | CascadePredictorConfig
    data: DataLoaderConfig
    output_dir: str
    regions: list[OutputTargetConfig]
    logging: LoggingConfig
    patch: PatchPredictionConfig = field(default_factory=PatchPredictionConfig)
