import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

import torch
import xarray as xr

from fme.ace.aggregator.inference.data import InferenceBatchData, make_dummy_time
from fme.ace.aggregator.inference.spectrum import (
    PairedSphericalPowerSpectrumAggregator as _InferencePairedSpectrumAgg,
)
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.fill import SmoothFloodFill
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorMapping

from .build_context import (
    MetricNotSupportedError,
    OneStepBuildContext,
    OneStepMetricBuildResult,
)


class PairedSphericalPowerSpectrumAggregator:
    """Wraps the inference PairedSphericalPowerSpectrumAggregator for one-step use."""

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        report_plot: bool,
        nan_fill_fn: Callable[[torch.Tensor, str], torch.Tensor] = lambda x, _: x,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        self._inner = _InferencePairedSpectrumAgg(
            gridded_operations=gridded_operations,
            report_plot=report_plot,
            nan_fill_fn=nan_fill_fn,
            variable_metadata=variable_metadata,
        )

    def record_batch(self, target_data: TensorMapping, gen_data: TensorMapping) -> None:
        first_tensor = next(iter(gen_data.values()))
        n_sample, n_time = first_tensor.shape[0], first_tensor.shape[1]
        batch = InferenceBatchData(
            prediction=gen_data,
            target=target_data,
            time=make_dummy_time(n_sample=n_sample, n_time=n_time),
            i_time_start=0,
        )
        self._inner.record_batch(batch)

    def get_logs(self, label: str) -> dict[str, Any]:
        return self._inner.get_logs(label)

    def get_dataset(self) -> xr.Dataset:
        return self._inner.get_dataset()


class SpectrumAggregator:
    def __init__(
        self,
        gridded_operations: GriddedOperations,
        target_time: int = 1,
        nan_fill_fn: Callable[[torch.Tensor, str], torch.Tensor] = lambda x, _: x,
    ):
        self._wrapped = _InferencePairedSpectrumAgg(
            gridded_operations=gridded_operations,
            report_plot=False,
            nan_fill_fn=nan_fill_fn,
        )
        self._target_time = target_time

    @torch.no_grad()
    def record_batch(
        self,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        target_data_norm: TensorMapping,
        gen_data_norm: TensorMapping,
        loss: float = float("nan"),
        i_time_start: int = 0,
    ):
        n_timesteps = next(iter(target_data.values())).shape[1]
        if (
            self._target_time >= i_time_start
            and self._target_time < i_time_start + n_timesteps
        ):
            i_time_target = self._target_time - i_time_start
            target_data = {
                key: value[:, i_time_target : i_time_target + 1, ...]
                for key, value in target_data.items()
            }
            gen_data = {
                key: value[:, i_time_target : i_time_target + 1, ...]
                for key, value in gen_data.items()
            }
            self._wrapped.record_paired_data(
                prediction=gen_data,
                target=target_data,
            )

    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        return self._wrapped.get_logs(label)

    def get_dataset(self) -> xr.Dataset:
        return self._wrapped.get_dataset()


@dataclasses.dataclass
class OneStepSpectrumMetricConfig:
    name: str = "power_spectrum"
    enabled: bool = True
    strict: bool = False

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: OneStepBuildContext) -> OneStepMetricBuildResult:
        try:
            flood_fill = SmoothFloodFill(num_steps=4)
            agg = SpectrumAggregator(ctx.ops, nan_fill_fn=flood_fill)
        except NotImplementedError as e:
            raise MetricNotSupportedError(str(e)) from e
        return OneStepMetricBuildResult(deterministic=agg)
