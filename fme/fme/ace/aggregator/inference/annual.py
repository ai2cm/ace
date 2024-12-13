import dataclasses
import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import xarray as xr
from matplotlib.figure import Figure

from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorMapping


class GlobalMeanAnnualAggregator:
    def __init__(
        self,
        ops: GriddedOperations,
        timestep: datetime.timedelta,
        variable_metadata: Optional[Mapping[str, VariableMetadata]] = None,
        monthly_reference_data: Optional[xr.Dataset] = None,
    ):
        self._area_weighted_mean = ops.area_weighted_mean
        self.timestep = timestep
        self.variable_metadata = variable_metadata
        self._target_datasets: Optional[List[xr.Dataset]] = None
        self._gen_datasets: Optional[List[xr.Dataset]] = None
        self._monthly_reference_data = monthly_reference_data
        self._variable_reference_data: Dict[str, VariableReferenceData] = {}

    def _get_reference(self, name: str) -> Optional["VariableReferenceData"]:
        if self._monthly_reference_data is None:
            return None
        if name not in self._variable_reference_data:
            if name not in self._monthly_reference_data:
                return None
            self._variable_reference_data[name] = process_monthly_reference(
                self._monthly_reference_data, self._area_weighted_mean, name
            )
        return self._variable_reference_data[name]

    @torch.no_grad()
    def record_batch(
        self,
        time: xr.DataArray,
        target_data: TensorMapping,
        gen_data: TensorMapping,
    ):
        """Record a batch of data for computing time variability statistics."""
        target_data_area_mean, gen_data_area_mean = {}, {}
        for name in gen_data.keys():
            target_data_area_mean[name] = self._area_weighted_mean(
                target_data[name]
            ).cpu()
            gen_data_area_mean[name] = self._area_weighted_mean(gen_data[name]).cpu()
        target_ds = to_dataset(target_data_area_mean, time)
        gen_ds = to_dataset(gen_data_area_mean, time)

        # must keep a separate dataset for each sample to avoid averaging across
        # samples when we groupby year
        if self._target_datasets is None:
            self._target_datasets = []
            for i_sample in range(target_ds.sizes["sample"]):
                self._target_datasets.append(
                    target_ds.isel(sample=i_sample)
                    .groupby(target_ds["valid_time"].isel(sample=i_sample).dt.year)
                    .sum()
                )
        else:
            for i_sample in range(target_ds.sizes["sample"]):
                self._target_datasets[i_sample] = _add_dataarray(
                    self._target_datasets[i_sample],
                    target_ds.isel(sample=i_sample)
                    .groupby(target_ds["valid_time"].isel(sample=i_sample).dt.year)
                    .sum(),
                )

        if self._gen_datasets is None:
            self._gen_datasets = []
            for i_sample in range(gen_ds.sizes["sample"]):
                self._gen_datasets.append(
                    gen_ds.isel(sample=i_sample)
                    .groupby(gen_ds["valid_time"].isel(sample=i_sample).dt.year)
                    .sum()
                )
        else:
            for i_sample in range(gen_ds.sizes["sample"]):
                self._gen_datasets[i_sample] = _add_dataarray(
                    self._gen_datasets[i_sample],
                    gen_ds.isel(sample=i_sample)
                    .groupby(gen_ds["valid_time"].isel(sample=i_sample).dt.year)
                    .sum(),
                )

    def _get_gathered_means(self) -> Optional[Tuple[xr.Dataset, xr.Dataset]]:
        """
        Gather the mean target and generated data across all processes.

        Returns:
            A tuple of the target and generated datasets, or None if this is not the
            root rank.
        """
        if self._target_datasets is None or self._gen_datasets is None:
            raise ValueError("No data has been recorded yet.")
        dist = Distributed.get_instance()
        target = xr.concat(self._target_datasets, dim="sample")
        gen = xr.concat(self._gen_datasets, dim="sample")
        if dist.world_size > 1:
            target = _gather_sample_datasets(dist, target)
            gen = _gather_sample_datasets(dist, gen)
        if target is None or gen is None:
            return None  # we are not root rank
        # filter out data with insufficient samples
        min_samples = _get_min_samples(self.timestep)
        target = target.where(target["counts"] > min_samples, drop=True)
        gen = gen.where(gen["counts"] > min_samples, drop=True)
        target = target / target["counts"]
        gen = gen / gen["counts"]
        return target, gen

    @torch.no_grad()
    def get_logs(self, label: str) -> Dict[str, Any]:
        gathered = self._get_gathered_means()
        if gathered is None:  # not the root rank
            return {}
        target, gen = gathered
        plots = {}
        metrics = {}

        for name in gen.data_vars.keys():
            if name == "counts":
                continue
            fig = Figure()  # create directly for cleanup when it leaves scope
            ax = fig.add_subplot(1, 1, 1)  # Add an axes to the figure
            ref = self._get_reference(name)
            if ref is not None:
                if ref.mean.sizes["year"] > 1:
                    # dataarray.plot() does not work for singleton dimensions
                    ref.mean.plot(ax=ax, x="year", label="ref_mean", color="black")
                    ref.min.plot(
                        ax=ax, x="year", label="ref_min", color="grey", linestyle="--"
                    )
                    ref.max.plot(
                        ax=ax, x="year", label="ref_max", color="grey", linestyle="--"
                    )
            if gen.sizes["year"] > 1:
                target_ensemble_mean = target[name].mean("sample")
                gen_ensemble_mean = gen[name].mean("sample")
                # compute R2 values
                if ref is not None:
                    r2_target = get_r2(target_ensemble_mean, ref.mean)
                    r2_gen = get_r2(gen_ensemble_mean, ref.mean)
                    metrics[f"r2/{name}_target"] = r2_target
                    metrics[f"r2/{name}_gen"] = r2_gen
                    target_label = f"target R2: {r2_target:.2f}"
                    gen_label = f"gen R2: {r2_gen:.2f}"
                else:
                    target_label = "target"
                    gen_label = "gen"
                if "year" in target_ensemble_mean.dims:
                    target_ensemble_mean.plot(
                        ax=ax, x="year", label=target_label, color="orange"
                    )
                if "year" in gen_ensemble_mean.dims:
                    gen_ensemble_mean.plot(
                        ax=ax, x="year", label=gen_label, color="cornflowerblue"
                    )
                if "year" in gen[name].dims:
                    for i_sample in range(gen.sizes["sample"]):
                        gen[name].isel(sample=i_sample).plot(
                            ax=ax,
                            x="year",
                            color="cornflowerblue",
                            alpha=0.4,
                            linestyle="-.",
                            marker="x",
                        )
            ax.set_title(f"{name}")
            ax.legend()
            fig.tight_layout()
            plots[name] = fig
        if len(label) > 0:
            label = label + "/"
        logs = {}
        logs.update({f"{label}{name}": plots[name] for name in plots.keys()})
        logs.update({f"{label}{name}": metrics[name] for name in metrics.keys()})
        return logs

    def get_dataset(self) -> xr.Dataset:
        gathered = self._get_gathered_means()
        if gathered is None:
            return xr.Dataset()
        target, gen = gathered
        return xr.concat(
            [
                target.expand_dims({"source": ["target"]}),
                gen.expand_dims({"source": ["prediction"]}),
            ],
            dim="source",
        )


@dataclasses.dataclass
class VariableReferenceData:
    mean: xr.DataArray
    min: xr.DataArray
    max: xr.DataArray


def process_monthly_reference(
    monthly_reference_data: xr.Dataset,
    area_weighted_mean: Callable[[torch.Tensor], torch.Tensor],
    name: str,
) -> VariableReferenceData:
    ref_global_mean = xr.DataArray(
        area_weighted_mean(torch.as_tensor(monthly_reference_data[name].values)),
        dims=monthly_reference_data[name].dims[:-2],
        coords={"time": monthly_reference_data[name].coords["time"]},
    )
    valid_time_0 = monthly_reference_data.valid_time.isel(sample=0)
    for i in range(1, monthly_reference_data.sizes["sample"]):
        valid_time_i = monthly_reference_data.valid_time.isel(sample=i)
        if not valid_time_0.equals(valid_time_i):
            raise ValueError("All time axes must be the same")
    # need to select just one time axis so we don't lose sample dim
    ref_annual_coarsened = (ref_global_mean * monthly_reference_data["counts"]).groupby(
        valid_time_0.dt.year
    ).sum() / monthly_reference_data["counts"].groupby(valid_time_0.dt.year).sum()
    return VariableReferenceData(
        mean=ref_annual_coarsened.mean("sample"),
        min=ref_annual_coarsened.min("sample"),
        max=ref_annual_coarsened.max("sample"),
    )


def _add_dataarray(da1: xr.DataArray, da2: xr.DataArray):
    """
    Perform dataarray addition, assuming any missing year indices
    have zero values.
    """
    union_index = np.union1d(da1.year.values, da2.year.values)
    da1 = da1.reindex(year=union_index, fill_value=0)
    da2 = da2.reindex(year=union_index, fill_value=0)
    return da1 + da2


def get_r2(da: xr.DataArray, reference: xr.DataArray) -> float:
    """Compute the R2 value of the target compared to the reference."""
    ref_data = reference.sel(year=da.year)
    SS_ref = np.sum((ref_data.values - np.mean(ref_data.values)) ** 2)
    SS_pred = np.sum((da - ref_data).values ** 2)
    return float(1 - SS_pred / SS_ref)


def _gather_sample_datasets(
    dist: Distributed, dataset: xr.Dataset
) -> Optional[xr.Dataset]:
    """
    Gather the dataset across all processes, concatenating on the sample dimension.

    Assumes all dataset variables have the same dimensions and shape, and that the
    first dimension is "sample".
    """
    # collect all data into one torch.Tensor for gathering, sort for determinism
    names = sorted(list(dataset.data_vars))
    tensor = torch.cat(
        [torch.asarray(np.expand_dims(dataset[name].values, axis=0)) for name in names],
        dim=0,
    ).to(get_device())
    years = torch.asarray(dataset.year.values).to(get_device())
    gathered_tensors = dist.gather_irregular(tensor)
    gathered_years = dist.gather_irregular(years)
    if gathered_tensors is None or gathered_years is None:
        return None
    datasets = []
    for tensor, years in zip(gathered_tensors, gathered_years):
        single_rank_dataset = xr.Dataset(
            {
                name: (["sample", "year"], tensor[i].cpu())
                for i, name in enumerate(names)
            },
            coords={"year": years.cpu()},
        )
        datasets.append(single_rank_dataset)
    # concat ranks along sample dim
    dataset_out = xr.concat(datasets, dim="sample")
    return dataset_out


@torch.no_grad()
def to_dataset(data: TensorMapping, time: xr.DataArray) -> xr.Dataset:
    """Convert a dictionary of data to an xarray dataset."""
    assert time.dims == ("sample", "time")  # must be consistent with this module
    data_vars = {}
    for name, tensor in data.items():
        data_vars[name] = (["sample", "time"], tensor)
    data_vars["counts"] = (["sample", "time"], np.ones(shape=time.shape))
    return xr.Dataset(data_vars, coords={"valid_time": time})


def _get_min_samples(timestep: datetime.timedelta) -> int:
    steps_per_day = datetime.timedelta(days=1) // timestep
    return 362 * steps_per_day
