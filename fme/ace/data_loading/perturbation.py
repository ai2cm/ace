import abc
import dataclasses
from collections.abc import Callable, Mapping

# we use Type to distinguish from type attr of PerturbationSelector
from typing import Any, ClassVar, Type  # noqa: UP035

import dacite
import numpy as np
import torch

from fme.core.registry.registry import Registry


@dataclasses.dataclass
class PerturbationConfig(abc.ABC):
    """
    Returns a perturbation function config class.
    """

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "PerturbationConfig":
        """
        Create a PerturbationSelector from a dictionary containing all the information
        needed to build a PerturbationConfig.
        """
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    @abc.abstractmethod
    def apply_perturbation(
        self,
        data: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        ocean_fraction: torch.Tensor,
    ) -> None: ...

    def reset(self) -> None:
        """Reset any internal state. Override for stateful perturbations."""
        pass


@dataclasses.dataclass
class PerturbationSelector:
    type: str
    config: Mapping[str, Any]
    registry: ClassVar[Registry[PerturbationConfig]] = Registry[PerturbationConfig]()

    def __post_init__(self):
        if not isinstance(self.registry, Registry):
            raise ValueError("PerturbationSelector.registry should not be set manually")

    @classmethod
    def register(
        cls, type_name
    ) -> Callable[[Type[PerturbationConfig]], Type[PerturbationConfig]]:  # noqa: UP006
        return cls.registry.register(type_name)

    def build(self) -> PerturbationConfig:
        return self.registry.get(self.type, self.config)

    @classmethod
    def get_available_types(cls):
        """This class method is used to expose all available types of Perturbations."""
        return cls(type="", config={}).registry._types.keys()


@dataclasses.dataclass
class SSTPerturbation:
    """
    Configuration for sea surface temperature perturbations
    applied to initial condition and forcing data.
    Currently, this is strictly applied to both.

    Parameters:
        sst: List of perturbation selectors for SST perturbations.
    """

    sst: list[PerturbationSelector]

    def __post_init__(self):
        self.perturbations: list[PerturbationConfig] = [
            perturbation.build() for perturbation in self.sst
        ]


@dataclasses.dataclass
class ForcingPerturbation:
    """
    Configuration for perturbations applied to arbitrary forcing variables
    during inference.

    Parameters:
        variables: Mapping from variable name to list of perturbation selectors.
            Each variable can have multiple perturbations applied sequentially.
        stds: Optional mapping from variable name to its standard deviation.
            When provided, perturbation amplitudes specified as fractions
            are scaled by the variable's std. This is typically populated
            from the model checkpoint's normalization statistics.
    """

    variables: dict[str, list[PerturbationSelector]]
    stds: dict[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.built_perturbations: dict[str, list[PerturbationConfig]] = {}
        for var_name, selectors in self.variables.items():
            perturbations = [s.build() for s in selectors]
            std = self.stds.get(var_name)
            if std is not None:
                for p in perturbations:
                    if isinstance(p, StdScaledPerturbationConfig):
                        p.set_std(std)
            self.built_perturbations[var_name] = perturbations

    def set_stds_from_normalizer(
        self, normalizer_stds: dict[str, torch.Tensor]
    ) -> None:
        """Populate stds from a normalizer's stds dict (e.g. from checkpoint).

        Only sets stds for variables that have StdScaledPerturbationConfig
        perturbations and don't already have a std set via the config.
        """
        for var_name, perturbations in self.built_perturbations.items():
            if var_name in self.stds:
                continue  # already set from config, don't override
            if var_name in normalizer_stds:
                std = float(normalizer_stds[var_name])
                self.stds[var_name] = std
                for p in perturbations:
                    if isinstance(p, StdScaledPerturbationConfig):
                        p.set_std(std)

    def reset(self) -> None:
        """Reset all stateful perturbations (e.g. at start of new IC window)."""
        for perturbations in self.built_perturbations.values():
            for p in perturbations:
                p.reset()


def _get_ocean_mask(ocean_fraction: torch.Tensor, cutoff: float = 0.5) -> torch.Tensor:
    return ocean_fraction > cutoff


@dataclasses.dataclass
class StdScaledPerturbationConfig(PerturbationConfig):
    """Base class for perturbations whose amplitude is a fraction of the
    variable's standard deviation.
    """

    amplitude: float = 0.1

    def __post_init__(self):
        self._std: float | None = None

    def set_std(self, std: float) -> None:
        self._std = std

    @property
    def physical_amplitude(self) -> float:
        if self._std is not None:
            return self.amplitude * self._std
        return self.amplitude


@PerturbationSelector.register("constant")
@dataclasses.dataclass
class ConstantConfig(PerturbationConfig):
    """
    Configuration for a constant perturbation.
    """

    amplitude: float = 1.0

    def apply_perturbation(
        self,
        data: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        ocean_fraction: torch.Tensor,
    ):
        ocean_mask = _get_ocean_mask(ocean_fraction)
        data[ocean_mask] += self.amplitude  # type: ignore


@PerturbationSelector.register("greens_function")
@dataclasses.dataclass
class GreensFunctionConfig(PerturbationConfig):
    """
    Configuration for a single sinusoidal patch of a Green's function perturbation.
    See equation 1 in Bloch‐Johnson, J., et al. (2024).

    Parameters:
        amplitude: The amplitude of the perturbation,
            maximum is reached at (lat_center, lon_center).
        lat_center: The latitude at the center of the patch in degrees.
        lon_center: The longitude at the center of the patch in degrees.
        lat_width: latitudinal width of the patch in degrees.
        lon_width: longitudinal width of the patch in degrees.
    """

    amplitude: float = 1.0
    lat_center: float = 0.0
    lon_center: float = 0.0
    lat_width: float = 10.0
    lon_width: float = 10.0

    def __post_init__(self):
        self._lat_center_rad = np.deg2rad(self.lat_center)
        self._lon_center_rad = np.deg2rad(self.lon_center)
        self._lat_width_rad = np.deg2rad(self.lat_width)
        self._lon_width_rad = np.deg2rad(self.lon_width)

    def _wrap_longitude_discontinuity(
        self,
        lon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Assume longitude is in the range [0, 360) degrees.
        If the patch crosses the discontinuity at 0/360 degrees,
        shift the longitude accordingly.
        """
        lon_min = self.lon_center - self.lon_width / 2.0
        lon_max = self.lon_center + self.lon_width / 2.0
        if lon_min < 0:
            lon_shifted = ((lon + 180) % 360) - 180
            lon_in_patch = (lon_shifted > lon_min) & (lon_shifted < lon_max)
        elif lon_max > 360:
            lon_in_patch = (lon > lon_min) | (lon < lon_max % 360)
            lon_shifted = ((lon + 180) % 360) + 180
        else:
            lon_in_patch = (lon > lon_min) & (lon < lon_max)
            lon_shifted = lon
        return lon_in_patch, lon_shifted

    def apply_perturbation(
        self,
        data: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        ocean_fraction: torch.Tensor,
    ):
        lat_in_patch = torch.abs(lat - self.lat_center) < self.lat_width / 2.0
        lon_in_patch, lon_shifted = self._wrap_longitude_discontinuity(lon)
        mask = lat_in_patch & lon_in_patch
        ocean_mask = _get_ocean_mask(ocean_fraction)
        perturbation = self.amplitude * (
            torch.cos(
                torch.pi
                / 2
                * (lat.deg2rad() - self._lat_center_rad)
                / (self._lat_width_rad / 2.0)
            )
            ** 2
            * torch.cos(
                torch.pi
                / 2
                * (lon_shifted.deg2rad() - self._lon_center_rad)
                / (self._lon_width_rad / 2.0)
            )
            ** 2
        )
        mask = mask.expand(data.shape)
        perturbation = perturbation.expand(data.shape)
        data[mask & ocean_mask] += perturbation[mask & ocean_mask]


@PerturbationSelector.register("white_noise")
@dataclasses.dataclass
class WhiteNoiseConfig(StdScaledPerturbationConfig):
    """
    i.i.d. Gaussian noise applied at each timestep.

    Parameters:
        amplitude: Noise standard deviation as a fraction of the variable's
            standard deviation. If no std is provided, used as absolute value.
        ocean_mask: Whether to apply noise only over ocean points.
        seed: Optional random seed for reproducibility.
    """

    ocean_mask: bool = True
    seed: int | None = None

    def apply_perturbation(
        self,
        data: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        ocean_fraction: torch.Tensor,
    ):
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=data.device)
            generator.manual_seed(self.seed)
        noise = torch.randn(data.shape, device=data.device, generator=generator)
        noise *= self.physical_amplitude
        if self.ocean_mask:
            mask = _get_ocean_mask(ocean_fraction).expand(data.shape)
            data[mask] += noise[mask]
        else:
            data += noise


@PerturbationSelector.register("red_noise")
@dataclasses.dataclass
class RedNoiseConfig(StdScaledPerturbationConfig):
    """
    Temporally correlated AR(1) noise. At each timestep, the noise is:
        noise_t = alpha * noise_{t-1} + sqrt(1 - alpha^2) * white_noise
    where alpha = exp(-1 / decorrelation_steps), ensuring the stationary
    variance of the noise process equals (physical_amplitude)^2.

    Parameters:
        amplitude: Noise standard deviation as a fraction of the variable's
            standard deviation. If no std is provided, used as absolute value.
        decorrelation_steps: Number of timesteps for the autocorrelation
            to decay to 1/e. For example, at 6-hour timesteps:
            10 steps = 2.5 days, 40 steps = 10 days, 120 steps = 30 days.
        ocean_mask: Whether to apply noise only over ocean points.
        seed: Optional random seed for reproducibility.
    """

    decorrelation_steps: float = 10.0
    ocean_mask: bool = True
    seed: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.decorrelation_steps <= 0:
            raise ValueError("decorrelation_steps must be positive")
        self._alpha = np.exp(-1.0 / self.decorrelation_steps)
        self._innovation_scale = np.sqrt(1.0 - self._alpha**2)
        self._previous_noise: torch.Tensor | None = None
        self._generator: torch.Generator | None = None

    def reset(self) -> None:
        self._previous_noise = None

    def apply_perturbation(
        self,
        data: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        ocean_fraction: torch.Tensor,
    ):
        if self._generator is None and self.seed is not None:
            self._generator = torch.Generator(device=data.device)
            self._generator.manual_seed(self.seed)

        innovation = torch.randn(
            data.shape, device=data.device, generator=self._generator
        )

        if self._previous_noise is None:
            # First timestep: draw from stationary distribution
            noise = innovation
        else:
            if self._previous_noise.shape != data.shape:
                # Shape changed (e.g. last window is shorter), reset
                noise = innovation
            else:
                noise = (
                    self._alpha * self._previous_noise
                    + self._innovation_scale * innovation
                )

        self._previous_noise = noise.clone()

        scaled_noise = noise * self.physical_amplitude
        if self.ocean_mask:
            mask = _get_ocean_mask(ocean_fraction).expand(data.shape)
            data[mask] += scaled_noise[mask]
        else:
            data += scaled_noise
