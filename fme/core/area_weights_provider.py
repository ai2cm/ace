import abc
from typing import Any

import torch

from fme.core.constants import SPHERE_AREA_M2
from fme.core.device import get_device
from fme.core.distributed import Distributed

_RTOL = 1e-5


class AreaWeightsProviderABC(abc.ABC):
    @property
    @abc.abstractmethod
    def area_weights(self) -> torch.Tensor | None:
        """2-D (lat, lon) horizontal weights, or None to use latitude weights."""
        ...

    @abc.abstractmethod
    def to(self, device: str) -> "AreaWeightsProviderABC": ...

    @abc.abstractmethod
    def localize(self) -> "AreaWeightsProviderABC":
        """Return a copy with weights sliced to the local spatial chunk."""
        ...

    @abc.abstractmethod
    def get_state(self) -> dict[str, Any] | None: ...

    def merge(self, other: "AreaWeightsProviderABC") -> "AreaWeightsProviderABC":
        """Combine the providers of two datasets merged along variables.

        Null + P -> P, P + P -> P, P + Q (unequal) -> ValueError.
        """
        if other.area_weights is None:
            return self
        if self.area_weights is None:
            return other
        if self != other:
            raise ValueError(
                "Merged datasets have different area weights (areacello * "
                "sea_surface_fraction)."
            )
        return self

    def assert_compatible_with(self, other: "AreaWeightsProviderABC") -> None:
        if self != other:
            raise AssertionError(f"area weights differ: {self} != {other}")


class _NullAreaWeightsProvider(AreaWeightsProviderABC):
    @property
    def area_weights(self) -> None:
        return None

    def to(self, device: str) -> "_NullAreaWeightsProvider":
        return self

    def localize(self) -> "_NullAreaWeightsProvider":
        return self

    def get_state(self) -> None:
        return None

    def __eq__(self, other) -> bool:
        return isinstance(other, _NullAreaWeightsProvider)

    def __hash__(self) -> int:
        return hash(_NullAreaWeightsProvider)

    def __repr__(self) -> str:
        return "NullAreaWeightsProvider"


NullAreaWeightsProvider = _NullAreaWeightsProvider()


class AreaWeightsProvider(AreaWeightsProviderABC):
    """
    Ocean-area weights for area-weighted reductions on a lat-lon grid.

    Built from the time-invariant dataset variables ``areacello`` (cell area,
    m2) and ``sea_surface_fraction`` (1)::

        J: horizontal cell index
        area_weights_J = areacello_J * sea_surface_fraction_J / SPHERE_AREA_M2
        <x>_name = sum_J(area_weights_J * mask_J * x_J)
                   / sum_J(area_weights_J * mask_J)

    where ``mask_J`` is the mask the ``SpatialMaskProvider`` returns for the
    variable name. The same weights apply to every variable and level;
    ``area_weighted_sum(x) * SPHERE_AREA_M2`` is the ocean-area integral in m2.

    Caveat: names whose values are per full cell area rather than per ocean
    area (e.g. ``hfds_total_area``, ``sea_ice_fraction``,
    ``interval_average_sea_ice_fraction``, and fme-derived variables that
    multiply by ``sea_surface_fraction``) count ``sea_surface_fraction``
    twice under these weights. The error is confined to cells with
    0 < sea_surface_fraction < 1.

    Args:
        area_weights: 2-D (lat, lon) tensor of non-negative, finite weights.
    """

    def __init__(self, area_weights: torch.Tensor):
        if area_weights.ndim != 2:
            raise ValueError(
                f"area_weights must be 2-D (lat, lon), got shape "
                f"{tuple(area_weights.shape)}"
            )
        if not torch.isfinite(area_weights).all():
            raise ValueError("area_weights must be finite.")
        self._area_weights = area_weights

    @classmethod
    def from_cell_area_and_fraction(
        cls, areacello: torch.Tensor, sea_surface_fraction: torch.Tensor
    ) -> "AreaWeightsProvider":
        return cls(areacello * sea_surface_fraction / SPHERE_AREA_M2)

    @property
    def area_weights(self) -> torch.Tensor:
        return self._area_weights

    def to(self, device: str) -> "AreaWeightsProvider":
        return AreaWeightsProvider(self._area_weights.to(device))

    def localize(self) -> "AreaWeightsProvider":
        slices = Distributed.get_instance().get_local_slices(self._area_weights.shape)
        return AreaWeightsProvider(self._area_weights[slices].contiguous())

    def get_state(self) -> dict[str, Any]:
        return {"area_weights": self._area_weights}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "AreaWeightsProvider":
        return cls(state["area_weights"].to(get_device(), copy=True))

    def __eq__(self, other) -> bool:
        if not isinstance(other, AreaWeightsProvider):
            return False
        if self._area_weights.shape != other._area_weights.shape:
            return False
        # atol=0: weights are O(1/n_cells), far below default atol
        return torch.allclose(
            self._area_weights.cpu(),
            other._area_weights.cpu().to(self._area_weights.dtype),
            rtol=_RTOL,
            atol=0.0,
        )

    def __repr__(self) -> str:
        return f"AreaWeightsProvider(shape={tuple(self._area_weights.shape)})"


def area_weights_provider_from_state(
    state: dict[str, Any] | None,
) -> AreaWeightsProviderABC:
    if state is None:
        return NullAreaWeightsProvider
    return AreaWeightsProvider.from_state(state)
