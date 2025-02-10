import torch
import xarray as xr

from fme.core.histogram import ComparedDynamicHistograms
from fme.core.typing_ import TensorMapping


class HistogramAggregator:
    def __init__(self):
        self._histograms = ComparedDynamicHistograms(n_bins=200, percentiles=[99.9999])

    @torch.no_grad()
    def record_batch(
        self,
        target_data: TensorMapping,
        gen_data: TensorMapping,
        target_data_norm: TensorMapping,
        gen_data_norm: TensorMapping,
        i_time_start: int = 0,
    ):
        self._histograms.record_batch(target_data, gen_data)

    @torch.no_grad()
    def get_logs(self, label: str):
        logs = self._histograms.get_wandb()
        if label != "":
            logs = {f"{label}/{k}": v for k, v in logs.items()}
        return logs

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        return self._histograms.get_dataset()
