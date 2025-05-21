import logging
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch
import xarray as xr

from fme.ace.aggregator.plotting import plot_paneled_data
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorMapping
from fme.core.wandb import Image


class SeasonalAggregator:
    def __init__(
        self,
        ops: GriddedOperations,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        self._area_weighted_mean = ops.area_weighted_mean
        self._variable_metadata = variable_metadata
        self._target_dataset: xr.Dataset | None = None
        self._gen_dataset: xr.Dataset | None = None

    @torch.no_grad()
    def record_batch(
        self,
        time: xr.DataArray,
        target_data: TensorMapping,
        gen_data: TensorMapping,
    ):
        """Record a batch of data for computing time variability statistics."""
        target_data = {name: value.cpu() for name, value in target_data.items()}
        gen_data = {name: value.cpu() for name, value in gen_data.items()}
        target_ds = _to_dataset(target_data, time)
        gen_ds = _to_dataset(gen_data, time)

        # must keep a separate dataset for each sample to avoid averaging across
        # samples when we groupby year
        if self._target_dataset is None:
            self._target_dataset = target_ds.groupby(
                target_ds.valid_time.dt.season
            ).sum(dim="stacked_sample_time", skipna=False)
        else:
            self._target_dataset = _add_dataarray(
                self._target_dataset,
                target_ds.groupby(target_ds.valid_time.dt.season).sum(
                    dim="stacked_sample_time", skipna=False
                ),
            )

        if self._gen_dataset is None:
            self._gen_dataset = gen_ds.groupby(gen_ds.valid_time.dt.season).sum(
                dim="stacked_sample_time", skipna=False
            )
        else:
            self._gen_dataset = _add_dataarray(
                self._gen_dataset,
                gen_ds.groupby(gen_ds.valid_time.dt.season).sum(
                    dim="stacked_sample_time", skipna=False
                ),
            )

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, Any]:
        if self._target_dataset is None or self._gen_dataset is None:
            raise ValueError("No data has been recorded yet.")
        dist = Distributed.get_instance()
        if dist.world_size > 1:
            target = _reduce_datasets(dist, self._target_dataset)
            gen = _reduce_datasets(dist, self._gen_dataset)
        else:
            target = self._target_dataset
            gen = self._gen_dataset
        if not dist.is_root():
            return {}
        if target is None or gen is None:
            raise RuntimeError("we are on root but no data was collected")

        if len(gen.season) < 4 or len(target.season) < 4:
            return {}  # seasonal metrics undefined when not all seasons are recorded

        target = cast(xr.Dataset, target / target["counts"])  # type: ignore
        gen = cast(xr.Dataset, gen / gen["counts"])  # type: ignore
        bias = gen - target
        plots: dict[str, Image] = {}
        metric_logs: dict[str, float] = {}

        for name in gen.data_vars.keys():
            if name == "counts":
                continue

            if self._variable_metadata is not None and name in self._variable_metadata:
                long_name = self._variable_metadata[name].long_name
                units = self._variable_metadata[name].units
                caption_name = f"{long_name} ({units})"
            else:
                caption_name = name

            target_mean_pattern = target[name].mean(dim="season")
            gen_anomaly = gen[name] - target_mean_pattern
            target_anomaly = target[name] - target_mean_pattern
            r2 = get_r2(gen_anomaly, target_anomaly)

            image = plot_paneled_data(
                [
                    [
                        target_anomaly.sel(season="DJF").values,
                        target_anomaly.sel(season="MAM").values,
                        target_anomaly.sel(season="JJA").values,
                        target_anomaly.sel(season="SON").values,
                    ],
                    [
                        gen_anomaly.sel(season="DJF").values,
                        gen_anomaly.sel(season="MAM").values,
                        gen_anomaly.sel(season="JJA").values,
                        gen_anomaly.sel(season="SON").values,
                    ],
                ],
                diverging=True,
                caption=(
                    f"Seasonal time-mean anomaly of {caption_name} for target (top) "
                    f"and gen (bottom) starting with DJF, R2={r2:.4f}. "
                    "Time-mean of target is subtracted from predictions and target."
                ),
            )
            plots[f"anomaly/{name}"] = image

            image_err = plot_paneled_data(
                [
                    [
                        bias[name].sel(season="DJF").values,
                        bias[name].sel(season="MAM").values,
                    ],
                    [
                        bias[name].sel(season="JJA").values,
                        bias[name].sel(season="SON").values,
                    ],
                ],
                diverging=True,
                caption=(
                    f"Seasonal bias of {caption_name} for DJF (Upper-Left), "
                    "MAM (UR), JJA (LL), and SON (LR). "
                    f"Seasonal anomaly R2={r2:.4f} (excludes time-mean of target)."
                ),
            )
            plots[f"bias/{name}"] = image_err

            mse_tensor = self._area_weighted_mean(
                torch.as_tensor(bias[name].values ** 2),
                name=name,
            )
            for i, season in enumerate(bias[name].season.values):
                rmse = float(mse_tensor[i].sqrt().numpy())
                metric_logs[f"time-mean-rmse/{name}-{season}"] = rmse
            rmse = float(
                # must compute area mean and then mean across seasons
                # before sqrt, so we can't use metrics.root_mean_squared_error
                mse_tensor.mean().sqrt().numpy()
            )
            metric_logs[f"time-mean-rmse/{name}"] = rmse

        if len(label) > 0:
            label = label + "/"
        logs: dict[str, Image | float] = {}
        logs.update({f"{label}{name}": plots[name] for name in plots.keys()})
        logs.update({f"{label}{name}": val for name, val in metric_logs.items()})
        return logs

    def get_dataset(self) -> xr.Dataset:
        logging.debug(
            "get_dataset not implemented for SeasonalAggregator. "
            "Returning an empty dataset."
        )
        return xr.Dataset()


ALL_SEASONS = np.asarray(["DJF", "MAM", "JJA", "SON"])


def _add_dataarray(da1: xr.DataArray, da2: xr.DataArray):
    """
    Perform dataarray addition, assuming any missing season indices
    have zero values.
    """
    if len(da1.season) < 4:
        da1 = da1.reindex(season=ALL_SEASONS, fill_value=0)
    if len(da2.season) < 4:
        da2 = da2.reindex(season=ALL_SEASONS, fill_value=0)
    return da1 + da2


def get_r2(da: xr.DataArray, target: xr.DataArray) -> float:
    """Compute the R2 value of the target compared to the reference."""
    SS_ref = np.sum((target.values - np.mean(target.values)) ** 2)
    SS_pred = np.sum((da - target).values ** 2)
    return float(1 - SS_pred / SS_ref)


def _reduce_datasets(dist: Distributed, dataset: xr.Dataset) -> xr.Dataset | None:
    """
    Add the dataset across all processes.

    Requires all dataset variables have the same shape.
    """
    # collect all data into one torch.Tensor for gathering, sort for determinism
    names = sorted(list(dataset.data_vars))
    # 'counts' must be present in the data, but we don't want to pack it with the others
    names.remove("counts")
    for name in names:
        if dataset[name].shape != dataset[names[0]].shape:
            raise ValueError(
                f"Variable {name} has shape {dataset[name].shape} "
                f"which is not equal to {dataset[names[0]].shape}"
            )
    tensor = torch.stack(
        [torch.as_tensor(dataset[name].values) for name in names],
        dim=0,
    ).to(get_device())
    reduced = dist.reduce_sum(tensor).cpu()
    reduced_counts = dist.reduce_sum(
        torch.as_tensor(dataset["counts"].values).to(get_device())
    ).cpu()
    dataset_out = xr.Dataset(
        {name: (["season", "lat", "lon"], reduced[i]) for i, name in enumerate(names)},
        coords=dataset.coords,
    )
    dataset_out["counts"] = xr.DataArray(reduced_counts, dims=["season"])
    return dataset_out


@torch.no_grad()
def _to_dataset(data: TensorMapping, time: xr.DataArray) -> xr.Dataset:
    """Convert a dictionary of data to an xarray dataset."""
    assert time.dims == ("sample", "time")  # must be consistent with this module
    data_vars = {}
    for name, tensor in data.items():
        data_vars[name] = (["sample", "time", "lat", "lon"], tensor)
    data_vars["counts"] = (["sample", "time"], np.ones(shape=time.shape))
    return xr.Dataset(data_vars, coords={"valid_time": time})
