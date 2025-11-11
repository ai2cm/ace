import abc
from collections.abc import Callable
from typing import Any, TypeVar, final

import torch
import torch_harmonics
from torch import nn

from fme.core import metrics
from fme.core.cuhpx.sht import SHT as CuHpxSHT
from fme.core.cuhpx.sht import iSHT as CuHpxiSHT
from fme.core.device import get_device
from fme.core.hpx.reorder import get_reordering_xy_to_ring
from fme.core.mask_provider import MaskProviderABC, NullMaskProvider
from fme.core.tensors import assert_dict_allclose
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.distributed import Distributed


class GriddedOperations(abc.ABC):
    def __eq__(self, other) -> bool:
        if not isinstance(other, GriddedOperations):
            return False
        try:
            assert_dict_allclose(self.to_state(), other.to_state())
        except AssertionError:
            return False
        return True

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            + ", ".join(
                [f"{k}={v}" for k, v in self.get_initialization_kwargs().items()]
            )
            + ")"
        )

    @property
    @abc.abstractmethod
    def zonal_mean(self) -> Callable[[torch.Tensor], torch.Tensor] | None: ...

    @abc.abstractmethod
    def area_weighted_sum(
        self,
        data: torch.Tensor,
        keepdim: bool = False,
        name: str | None = None,
    ) -> torch.Tensor: ...

    @final
    def area_weighted_sum_dict(
        self, data: TensorMapping, keepdim: bool = False
    ) -> TensorDict:
        result = {}
        for name in data:
            result[name] = self.area_weighted_sum(
                data=data[name],
                keepdim=keepdim,
                name=name,
            )
        return result

    @abc.abstractmethod
    def area_weighted_mean(
        self,
        data: torch.Tensor,
        keepdim: bool = False,
        name: str | None = None,
    ) -> torch.Tensor: ...

    @final
    def area_weighted_mean_dict(
        self, data: TensorMapping, keepdim: bool = False
    ) -> TensorDict:
        result = {}
        for name in data:
            result[name] = self.area_weighted_mean(
                data=data[name],
                keepdim=keepdim,
                name=name,
            )
        return result

    def area_weighted_mean_bias(
        self,
        truth: torch.Tensor,
        predicted: torch.Tensor,
        name: str | None = None,
    ) -> torch.Tensor:
        return self.area_weighted_mean(predicted - truth, name=name)

    @final
    def area_weighted_mean_bias_dict(
        self, truth: TensorMapping, predicted: TensorMapping
    ) -> TensorDict:
        result = {}
        for name in truth:
            result[name] = self.area_weighted_mean_bias(
                truth=truth[name],
                predicted=predicted[name],
                name=name,
            )
        return result

    def area_weighted_rmse(
        self,
        truth: torch.Tensor,
        predicted: torch.Tensor,
        name: str | None = None,
    ) -> torch.Tensor:
        return torch.sqrt(self.area_weighted_mean((predicted - truth) ** 2, name=name))

    @final
    def area_weighted_rmse_dict(
        self, truth: TensorMapping, predicted: TensorMapping
    ) -> TensorDict:
        result = {}
        for name in truth:
            result[name] = self.area_weighted_rmse(
                truth=truth[name],
                predicted=predicted[name],
                name=name,
            )
        return result

    def area_weighted_std(
        self,
        data: torch.Tensor,
        keepdim: bool = False,
        name: str | None = None,
    ):
        return self.area_weighted_mean(
            (data - self.area_weighted_mean(data, keepdim=True, name=name)) ** 2,
            keepdim=keepdim,
            name=name,
        ).sqrt()

    @final
    def area_weighted_std_dict(
        self,
        data: TensorMapping,
        keepdim: bool = False,
    ) -> TensorDict:
        result = {}
        for name in data:
            result[name] = self.area_weighted_std(
                data=data[name],
                keepdim=keepdim,
                name=name,
            )
        return result

    @abc.abstractmethod
    def area_weighted_gradient_magnitude_percent_diff(
        self,
        truth: torch.Tensor,
        predicted: torch.Tensor,
        name: str | None = None,
    ): ...

    @final
    def area_weighted_gradient_magnitude_percent_diff_dict(
        self, truth: TensorMapping, predicted: TensorMapping
    ) -> TensorDict:
        result = {}
        for name in truth:
            result[name] = self.area_weighted_gradient_magnitude_percent_diff(
                truth=truth[name],
                predicted=predicted[name],
                name=name,
            )
        return result

    @abc.abstractmethod
    def regional_area_weighted_mean(
        self,
        data: torch.Tensor,
        regional_weights: torch.Tensor,
        keepdim: bool = False,
        name: str | None = None,
    ) -> torch.Tensor: ...

    @final
    def regional_area_weighted_mean_dict(
        self,
        data: TensorMapping,
        regional_weights: torch.Tensor,
        keepdim: bool = False,
    ) -> TensorDict:
        result = {}
        for name in data:
            result[name] = self.regional_area_weighted_mean(
                data=data[name],
                regional_weights=regional_weights,
                keepdim=keepdim,
                name=name,
            )
        return result

    @abc.abstractmethod
    def get_real_sht(
        self,
    ) -> nn.Module: ...

    @abc.abstractmethod
    def get_real_isht(
        self,
    ) -> nn.Module: ...

    def to_state(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "state": self.get_initialization_kwargs(),
        }

    @abc.abstractmethod
    def get_initialization_kwargs(self) -> dict[str, Any]:
        """
        Get the keyword arguments needed to initialize the instance.
        """
        ...

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "GriddedOperations":
        """
        Given a dictionary with a "type" key and a "state" key, return
        the GriddedOperations it describes.

        The "type" key should be the name of a subclass of GriddedOperations,
        and the "state" key should be a dictionary specific to
        that subclass.

        Args:
            state: A dictionary with a "type" key and a "state" key.

        Returns:
            An instance of the subclass.
        """
        if cls is not GriddedOperations:
            raise RuntimeError(
                "This method should be called on GriddedOperations, "
                "not on its subclasses."
            )
        subclasses = get_all_subclasses(cls)
        for subclass in subclasses:
            if subclass.__name__ == state["type"]:
                return subclass(**state["state"])
        raise ValueError(
            f"Unknown subclass type: {state['type']}, "
            f"available: {[s.__name__ for s in subclasses]}"
        )


T = TypeVar("T")


def get_all_subclasses(cls: type[T]) -> list[type[T]]:
    """
    Gets all subclasses of a given class, including their subclasses etc.
    """
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses


def _mask_area_weights(
    area_weights: torch.Tensor,
    mask_provider: MaskProviderABC,
    name: str | None,
) -> torch.Tensor:
    if name is None:
        return area_weights
    mask = mask_provider.get_mask_tensor_for(name)
    if mask is None:
        return area_weights
    return area_weights * mask


class LatLonOperations(GriddedOperations):
    HORIZONTAL_DIMS = (-2, -1)

    def __init__(
        self,
        area_weights: torch.Tensor,
        mask_provider: MaskProviderABC = NullMaskProvider,
        grid: str = "legendre-gauss",
    ):
        # requires weights are longitudinally uniform
        if not torch.allclose(area_weights, area_weights[..., :1]):
            raise ValueError(
                "Area weights must be longitudinally uniform, "
                "as assumed for zonal mean."
            )

        dist = Distributed.get_instance()
        if dist.spatial_parallelism:
          area_weights = area_weights[*dist.get_local_slices(area_weights.shape)]

        self._device_area = area_weights.to(get_device())
        #NOTE: we do not need the *.to("cpu") lines.
        self._cpu_area = area_weights.to("cpu")
        self._device_mask_provider = mask_provider.to(get_device())
        self._cpu_mask_provider = mask_provider.to("cpu")
        self._grid = "legendre-gauss"

    @property
    def zonal_mean(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self._zonal_mean

    def _zonal_mean(self, data: torch.Tensor) -> torch.Tensor:
        return data.nanmean(dim=self.HORIZONTAL_DIMS[1])

    def _get_area_weights(
        self,
        data: torch.Tensor,
        name: str | None = None,
        regional_weights: torch.Tensor | None = None,
    ):
        if data.device == torch.device("cpu"):
            area_weights = self._cpu_area
            mask_provider = self._cpu_mask_provider
        else:
            area_weights = self._device_area
            mask_provider = self._device_mask_provider
        area_weights = _mask_area_weights(area_weights, mask_provider, name)
        if regional_weights is None:
            return area_weights
        if regional_weights.device.type != data.device.type:
            regional_weights = regional_weights.to(data.device)
        return regional_weights * area_weights

    def area_weighted_sum(
        self,
        data: torch.Tensor,
        keepdim: bool = False,
        name: str | None = None,
    ) -> torch.Tensor:
        area_weights = self._get_area_weights(data, name)
        return metrics.weighted_sum(
            data, area_weights, dim=self.HORIZONTAL_DIMS, keepdim=keepdim
        )

    def area_weighted_mean(
        self,
        data: torch.Tensor,
        keepdim: bool = False,
        name: str | None = None,
    ) -> torch.Tensor:
        area_weights = self._get_area_weights(data, name)
        return metrics.weighted_mean(
            data, area_weights, dim=self.HORIZONTAL_DIMS, keepdim=keepdim
        )

    def regional_area_weighted_mean(
        self,
        data: torch.Tensor,
        regional_weights: torch.Tensor,
        keepdim: bool = False,
        name: str | None = None,
    ) -> torch.Tensor:
        regional_area_weights = self._get_area_weights(data, name, regional_weights)
        return metrics.weighted_mean(
            data,
            regional_area_weights,
            dim=self.HORIZONTAL_DIMS,
            keepdim=keepdim,
        )

    def area_weighted_gradient_magnitude_percent_diff(
        self,
        truth: torch.Tensor,
        predicted: torch.Tensor,
        name: str | None = None,
    ):
        area_weights = self._get_area_weights(truth, name)
        return metrics.gradient_magnitude_percent_diff(
            truth,
            predicted,
            weights=area_weights,
            dim=self.HORIZONTAL_DIMS,
        )

    def get_real_sht(self) -> torch_harmonics.RealSHT:
        return torch_harmonics.RealSHT(
            nlat=self._cpu_area.shape[-2],
            nlon=self._cpu_area.shape[-1],
            grid=self._grid,
        ).to(get_device())

    def get_real_isht(self) -> torch_harmonics.InverseRealSHT:
        return torch_harmonics.InverseRealSHT(
            nlat=self._cpu_area.shape[-2],
            nlon=self._cpu_area.shape[-1],
            grid=self._grid,
        ).to(get_device())

    def get_initialization_kwargs(self) -> dict[str, Any]:
        return {"area_weights": self._cpu_area}


class HEALPixSHT(nn.Module):
    def __init__(self, nside: int, lmax: int, mmax: int, grid: str):
        super().__init__()
        self.nside = nside
        self.lmax = lmax
        self.mmax = mmax
        self.grid = grid
        self.sht = CuHpxSHT(nside, lmax=lmax, mmax=mmax, grid=grid)
        self._reordering = get_reordering_xy_to_ring(nside, device=get_device())
        self._reordering_cpu = self._reordering.to("cpu")

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if data.shape[-2] == 1:  # ring ordering, stored as [..., 1, npix]
            return self.sht(data[..., 0, :])
        else:  # face ordering, stored as [..., 12, n_channel, ny, nx]
            n_face, ny, nx = data.shape[-3:]
            if n_face != 12:
                raise ValueError(
                    f"Expected 12 faces, got {n_face} in shape {data.shape}"
                )
            if ny != nx:
                raise ValueError(
                    f"Expected square grid, got {ny}x{nx} in shape {data.shape}"
                )
            if ny != self.nside:
                raise ValueError(
                    f"Expected nside {self.nside}, got {ny} in shape {data.shape}"
                )
            data = data.reshape(*data.shape[:-3], 12 * self.nside * self.nside)
            if data.device.type == "cpu":
                data = data[..., self._reordering_cpu]
            else:
                data = data[..., self._reordering]
            return self.sht(data)


class HEALPixInverseSHT(nn.Module):
    def __init__(self, nside: int, lmax: int, mmax: int, grid: str):
        super().__init__()
        self.nside = nside
        self.lmax = lmax
        self.mmax = mmax
        self.grid = grid
        self.isht = CuHpxiSHT(nside, lmax=lmax, mmax=mmax, grid=grid)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.isht(data).unsqueeze(-2)


class HEALPixOperations(GriddedOperations):
    HORIZONTAL_DIMS = (-3, -2, -1)

    def __init__(self, nside: int | None = None):
        """
        Args:
            nside: The nside of the HEALPix grid. nside must be specified in order to
                use the SHT. It is allowed to be None only for backwards compatibility.
        """
        self.nside = nside

    @property
    def zonal_mean(self) -> None:
        # not implemented, though we definitely could
        # as HEALPix rings are constant-latitude
        return None

    def area_weighted_sum(
        self,
        data: torch.Tensor,
        keepdim: bool = False,
        name: str | None = None,
    ) -> torch.Tensor:
        # For HEALPix, area weights are uniform, so sum is sufficient
        return data.sum(dim=self.HORIZONTAL_DIMS, keepdim=keepdim)

    def area_weighted_mean(
        self,
        data: torch.Tensor,
        keepdim: bool = False,
        name: str | None = None,
    ) -> torch.Tensor:
        # For HEALPix, area weights are uniform, so mean is sufficient
        return data.mean(dim=self.HORIZONTAL_DIMS, keepdim=keepdim)

    def area_weighted_gradient_magnitude_percent_diff(
        self,
        truth: torch.Tensor,
        predicted: torch.Tensor,
        name: str | None = None,
    ) -> torch.Tensor:
        return metrics.gradient_magnitude_percent_diff(
            truth, predicted, weights=None, dim=self.HORIZONTAL_DIMS
        )

    def regional_area_weighted_mean(
        self,
        data: torch.Tensor,
        weights: torch.Tensor,
        keepdim: bool = False,
        name: str | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Regional area weighted mean is not implemented for HEALPix."
        )

    def get_real_sht(self) -> nn.Module:
        if self.nside is None:
            raise ValueError("nside must be specified for SHT.")
        lmax = 2 * self.nside - 1
        return HEALPixSHT(self.nside, lmax=lmax, mmax=lmax, grid="healpix")

    def get_real_isht(self) -> nn.Module:
        if self.nside is None:
            raise ValueError("nside must be specified for SHT.")
        lmax = 2 * self.nside - 1
        return HEALPixInverseSHT(self.nside, lmax=lmax, mmax=lmax, grid="healpix")

    def get_initialization_kwargs(self) -> dict[str, Any]:
        if self.nside is None:
            return {}
        return {"nside": self.nside}
