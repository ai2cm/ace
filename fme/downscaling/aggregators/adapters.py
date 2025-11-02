import abc
from collections.abc import Mapping
from typing import Any, Protocol

import torch
import xarray as xr

from fme.core.histogram import ComparedDynamicHistograms, DynamicHistogramAggregator
from fme.core.typing_ import TensorMapping


class _HistogramInterface(Protocol):
    def record_batch(self, *args, **kwargs): ...
    def get_wandb(self) -> Mapping[str, Any]: ...
    def get_dataset(self) -> dict[str, Any]: ...


def _ensure_trailing_slash(key: str) -> str:
    if key and not key.endswith("/"):
        key += "/"
    return key


class _HistogramsAdapter(abc.ABC):
    def __init__(self, histograms: _HistogramInterface, name: str = "") -> None:
        self._histograms = histograms
        self._name = _ensure_trailing_slash(name)

    @abc.abstractmethod
    def record_batch(self, *args, **kwargs): ...

    def get_wandb(self, prefix: str = "") -> Mapping[str, Any]:
        """
        Get the histogram logs for wandb.
        """
        prefix = _ensure_trailing_slash(prefix)
        histograms = self._histograms.get_wandb()
        return {f"{prefix}{self._name}{k}": v for k, v in histograms.items()}

    def get_dataset(self) -> dict[str, Any]:
        """
        Get the histogram dataset.
        """
        ds = self._histograms.get_dataset()
        return xr.Dataset({f"{self._name}{k}": v for k, v in ds.items()})


class DynamicHistogramsAdapter(_HistogramsAdapter):
    """
    Adapter to use DynamicHistogramAggregator with the naming and prefix
    scheme used by downscaling aggregators.

    Args:
        histograms: The DynamicHistogramAggregator object to adapt.
        name: The name to use for the histograms in the wandb output.
    """

    def __init__(self, histograms: DynamicHistogramAggregator, name: str = "") -> None:
        super().__init__(histograms=histograms, name=name)

    @torch.no_grad()
    def record_batch(
        self, prediction: TensorMapping, coarse: TensorMapping, time: xr.DataArray
    ) -> None:
        """
        Record the histograms for the current batch comparison.
        Adapter signature works with no target aggregator convention.

            prediction: The predicted fine data for the current batch.
            coarse: unused, the coarse data for the current batch.
            time: unused, datatime information corresponding to leading dim of tensors.
        """
        self._histograms.record_batch(prediction)


class ComparedDynamicHistogramsAdapter(_HistogramsAdapter):
    def __init__(self, histograms: ComparedDynamicHistograms, name: str = "") -> None:
        super().__init__(histograms=histograms, name=name)

    @torch.no_grad()
    def record_batch(self, target: TensorMapping, prediction: TensorMapping) -> None:
        """
        Record the histograms for the current batch comparison.
        """
        self._histograms.record_batch(target=target, prediction=prediction)
