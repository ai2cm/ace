import torch
import xarray as xr

from fme.core.histogram import ComparedDynamicHistograms

from .data import InferenceBatchData


class HistogramAggregator:
    def __init__(self):
        self._histograms = ComparedDynamicHistograms(n_bins=200, percentiles=[99.9999])

    @torch.no_grad()
    def record_batch(
        self,
        data: InferenceBatchData,
    ):
        self._histograms.record_batch(data.target, data.prediction)

    @torch.no_grad()
    def get_logs(self, label: str):
        logs = self._histograms.get_wandb()
        if label != "":
            logs = {f"{label}/{k}": v for k, v in logs.items()}
        return logs

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        return self._histograms.get_dataset()
