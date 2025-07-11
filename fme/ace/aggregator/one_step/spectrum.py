import numpy as np
import torch
import xarray as xr

from fme.ace.aggregator.inference.spectrum import PairedSphericalPowerSpectrumAggregator
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorMapping


class SpectrumAggregator:
    def __init__(
        self,
        gridded_operations: GriddedOperations,
        target_time: int = 1,
    ):
        self._wrapped = PairedSphericalPowerSpectrumAggregator(
            gridded_operations=gridded_operations,
            report_plot=False,
        )
        self._target_time = target_time

    @torch.no_grad()
    def record_batch(
        self,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        target_data_norm: TensorMapping,
        gen_data_norm: TensorMapping,
        loss: torch.Tensor = torch.tensor(np.nan),
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
            self._wrapped.record_batch(
                target_data,
                gen_data,
                target_data_norm,
                gen_data_norm,
                i_time_start,
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
