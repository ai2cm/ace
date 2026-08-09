import abc
import dataclasses
import math
from typing import Literal

import dacite
import fsspec
import torch
import xarray as xr


class FieldTransform(abc.ABC):
    """A pointwise, invertible transform applied to a single field before
    mean/std normalization and inverted after denormalization.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def inverse(self, z: torch.Tensor) -> torch.Tensor: ...


class Log1pTransform(FieldTransform):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.clamp(x, min=0.0))

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return torch.expm1(z)


class LogitTransform(FieldTransform):
    def __init__(self, epsilon: float, scale: float):
        self._epsilon = epsilon
        self._scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(x / self._scale, self._epsilon, 1.0 - self._epsilon)
        return torch.log(p / (1.0 - p))

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(z) * self._scale


def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation of x against knots (xp, fp), clamped to
    the end values outside the knot range. xp must be strictly increasing.
    """
    xp = xp.to(x.device, x.dtype)
    fp = fp.to(x.device, x.dtype)
    idx = torch.clamp(torch.searchsorted(xp, x.contiguous()), 1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    frac = (torch.clamp(x, xp[0], xp[-1]) - x0) / (x1 - x0)
    return f0 + frac * (f1 - f0)


class GaussianRankTransform(FieldTransform):
    """Monotone map through fitted (x, z) knots, where z values are standard
    normal quantiles of the training CDF at the x knots.
    """

    def __init__(self, x_knots: list[float], z_knots: list[float]):
        self._x = torch.tensor(x_knots, dtype=torch.float64)
        self._z = torch.tensor(z_knots, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _interp(x, self._x, self._z)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return _interp(z, self._z, self._x)


@dataclasses.dataclass
class Log1pTransformConfig:
    """Encode a non-negative field as log1p(x); negative inputs are clamped
    to zero before encoding.

    Parameters:
        type: Must be "log1p".
    """

    type: Literal["log1p"] = "log1p"

    def load(self):
        pass

    def build(self) -> Log1pTransform:
        return Log1pTransform()


@dataclasses.dataclass
class LogitTransformConfig:
    """Encode a bounded field as logit(clip(x / scale, epsilon, 1 - epsilon)),
    decoded through sigmoid(z) * scale so predictions stay within bounds.

    Parameters:
        type: Must be "logit".
        epsilon: Clip margin away from the 0 and 1 bounds.
        scale: Physical value of full saturation (1.0 for a 0-1 fraction,
            100.0 for a percent field).
    """

    type: Literal["logit"] = "logit"
    epsilon: float = 1e-4
    scale: float = 1.0

    def __post_init__(self):
        if not 0.0 < self.epsilon < 0.5:
            raise ValueError(f"epsilon must be in (0, 0.5), got {self.epsilon}")
        if self.scale <= 0.0:
            raise ValueError(f"scale must be positive, got {self.scale}")

    def load(self):
        pass

    def build(self) -> LogitTransform:
        return LogitTransform(epsilon=self.epsilon, scale=self.scale)


@dataclasses.dataclass
class GaussianRankTransformConfig:
    """Encode a field through its training-data empirical CDF mapped onto
    standard normal quantiles, via a fitted monotone knot table.

    Either table_path (a netCDF file with 1-D variables ``x_knots`` and
    ``z_knots``) or explicit knot lists must be provided. ``load()`` embeds
    the table so checkpoints do not depend on the file.

    Parameters:
        type: Must be "gaussian_rank".
        table_path: Path to a netCDF file containing the knot table.
        x_knots: Physical-space knot values, strictly increasing.
        z_knots: Normal-quantile knot values, strictly increasing.
    """

    type: Literal["gaussian_rank"] = "gaussian_rank"
    table_path: str | None = None
    x_knots: list[float] = dataclasses.field(default_factory=list)
    z_knots: list[float] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        using_path = self.table_path is not None
        using_explicit = len(self.x_knots) > 0
        if using_path == using_explicit:
            raise ValueError(
                "Provide exactly one of table_path or explicit x_knots/z_knots."
            )
        if using_explicit:
            self._validate_knots()

    def _validate_knots(self):
        if len(self.x_knots) != len(self.z_knots):
            raise ValueError(
                "x_knots and z_knots must have the same length, got "
                f"{len(self.x_knots)} and {len(self.z_knots)}"
            )
        if len(self.x_knots) < 2:
            raise ValueError("At least two knots are required.")
        for name, knots in (("x_knots", self.x_knots), ("z_knots", self.z_knots)):
            if any(b <= a for a, b in zip(knots[:-1], knots[1:])):
                raise ValueError(f"{name} must be strictly increasing.")
        if any(not math.isfinite(v) for v in self.x_knots + self.z_knots):
            raise ValueError("Knots must be finite.")

    def load(self):
        if self.table_path is not None:
            with fsspec.open(self.table_path, "rb") as f:
                ds = xr.load_dataset(f)
            self.x_knots = [float(v) for v in ds["x_knots"].values]
            self.z_knots = [float(v) for v in ds["z_knots"].values]
            self.table_path = None
            self._validate_knots()

    def build(self) -> GaussianRankTransform:
        if self.table_path is not None:
            self.load()
        return GaussianRankTransform(x_knots=self.x_knots, z_knots=self.z_knots)


FieldTransformConfig = (
    Log1pTransformConfig | LogitTransformConfig | GaussianRankTransformConfig
)

_CONFIG_TYPES: dict[str, type] = {
    "log1p": Log1pTransformConfig,
    "logit": LogitTransformConfig,
    "gaussian_rank": GaussianRankTransformConfig,
}


def transform_config_from_dict(data: dict) -> FieldTransformConfig:
    """Rebuild a transform config from its dataclasses.asdict form."""
    try:
        config_class = _CONFIG_TYPES[data["type"]]
    except KeyError:
        raise ValueError(
            f"Unknown field transform type {data.get('type')!r}; "
            f"expected one of {sorted(_CONFIG_TYPES)}"
        )
    return dacite.from_dict(
        data_class=config_class, data=data, config=dacite.Config(strict=True)
    )
