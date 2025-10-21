import logging
import os
from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

import fme.core.logging_utils as logging_utils
from fme.core.device import get_device
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.logging_utils import LoggingConfig
from fme.downscaling.data import (
    BatchData,
    BatchedLatLonCoordinates,
    ContiguousDistributedSampler,
    LatLonCoordinates,
    PairedBatchData,
)
from fme.downscaling.evaluator import CheckpointModelConfig
from fme.downscaling.writer import ZarrWriter


@dataclass
class Region:
    lat_upper: float
    lat_lower: float
    lon_upper: float
    lon_lower: float


MASKS = {
    "conus": Region(
        lat_upper=50,
        lat_lower=22,
        lon_upper=299,
        lon_lower=227,
    ),
    "bayou": Region(
        lat_upper=32,
        lat_lower=28,
        lon_upper=272,
        lon_lower=268,
    ),
    "hurricane_segment": Region(
        lat_upper=31,
        lat_lower=26,
        lon_upper=287,
        lon_lower=282,
    ),
    "florida": Region(
        lat_upper=38,
        lat_lower=22,
        lon_upper=289,
        lon_lower=273,
    ),
    "california": Region(
        lat_upper=45,
        lat_lower=29,
        lon_upper=248,
        lon_lower=232,
    ),
    "singapore": Region(
        lat_upper=9,
        lat_lower=-7,
        lon_upper=112,
        lon_lower=96,
    ),
}


def get_latlon_mask(ds, region: Region):
    lon = ds.longitude.load()
    lat = ds.latitude.load()
    return (
        (lat > region.lat_lower)
        & (lat < region.lat_upper)
        & (lon > region.lon_lower)
        & (lon < region.lon_upper)
    )


def get_masked_for_region(ds_25, ds_3, region: str):
    return (
        ds_25.where(get_latlon_mask(ds_25, MASKS[region]), drop=True),
        ds_3.where(get_latlon_mask(ds_3, MASKS[region]), drop=True),
    )


def time_slice_to_batch_data(
    fine: xr.Dataset,
    coarse: xr.Dataset,
    time_slice: slice,
    var_keys: list[str],
    region: str | None = None,
):
    """
    Convert a time slice of fine and coarse data to a PairedBatchData object.
    """

    # Select the time slice
    fine = fine.isel(time=time_slice)
    coarse = coarse.isel(time=time_slice)
    if region is not None:
        fine, coarse = get_masked_for_region(fine, coarse, region)
    topo = fine["normalized_topography"].values[None]
    topo = np.repeat(topo, len(fine.time), axis=0)

    ntimes = len(fine.time)
    # Create coordinates
    fine_latlon_coords = LatLonCoordinates(
        lat=torch.from_numpy(fine.latitude.values).to(get_device()),
        lon=torch.from_numpy(fine.longitude.values).to(get_device()),
    )
    coarse_latlon_coords = LatLonCoordinates(
        lat=torch.from_numpy(coarse.latitude.values).to(get_device()),
        lon=torch.from_numpy(coarse.longitude.values).to(get_device()),
    )
    fine_latlon_coords_batch = BatchedLatLonCoordinates.from_sequence(
        [fine_latlon_coords] * ntimes,
    )
    coarse_latlon_coords_batch = BatchedLatLonCoordinates.from_sequence(
        [coarse_latlon_coords] * ntimes,
    )

    fine_tensor = {
        k: torch.from_numpy(fine[k].values).to(get_device()) for k in var_keys
    }
    coarse_tensor = {
        k: torch.from_numpy(coarse[k].values).to(get_device()) for k in var_keys
    }

    fine_batch = BatchData(
        data=fine_tensor,
        latlon_coordinates=fine_latlon_coords_batch,
        time=fine.time,
    )

    coarse_batch = BatchData(
        data=coarse_tensor,
        latlon_coordinates=coarse_latlon_coords_batch,
        time=coarse.time,
    )

    return PairedBatchData(fine_batch, coarse_batch)


@dataclass
class SliceItem:
    time: slice
    ens: slice

    def __post_init__(self):
        self.n_ens = self.ens.stop - self.ens.start

    @property
    def ens_coord(self):
        return np.arange(self.ens.start, self.ens.stop)


class SliceDataset(torch.utils.data.Dataset):
    """
    Dataset returns a slice to use for the data
    """

    def __init__(self, time_len: int, n_ens: int, n_item_per_gpu: int = 4):
        self.time_len = time_len
        self.n_ens = n_ens
        self._n_item_per_gpu = n_item_per_gpu
        self._create_slices()

    def _create_slices(self):
        n_ens_per_slice = min(self.n_ens, self._n_item_per_gpu)
        n_time_per_slice = max(1, self._n_item_per_gpu // n_ens_per_slice)

        ens_slices = []
        start = 0
        while start < self.n_ens:
            end = min(start + n_ens_per_slice, self.n_ens)
            ens_slices.append(slice(start, end))
            start = end

        time_slices = []
        start = 0
        while start < self.time_len:
            end = min(start + n_time_per_slice, self.time_len)
            time_slices.append(slice(start, end))
            start = end
        self._slices = [
            SliceItem(t_sl, ens_sl)
            for (t_sl, ens_sl) in product(time_slices, ens_slices)
        ]

    def __len__(self):
        return len(self._slices)

    def __getitem__(self, idx):
        return self._slices[idx]


@dataclass
class SaveConfig:
    model_path: str
    fine_zarr_path: str
    coarse_zarr_path: str
    experiment_dir: str
    save_vars: list[str]
    region: str | None = None
    time_slice: slice | None = None
    n_ensemble: int = 1
    n_item_per_gpu: int = 4


def _get_ds_coarse_fine(config: SaveConfig):
    ds_coarse = xr.open_zarr(config.coarse_zarr_path, chunks=None)
    ds_fine = xr.open_zarr(config.fine_zarr_path, chunks=None)
    ds_fine["normalized_topography"] = (
        (ds_fine["HGTsfc"] - ds_fine["HGTsfc"].mean()) / ds_fine["HGTsfc"].std()
    ).compute()
    return ds_coarse, ds_fine


def run(config: SaveConfig):
    dist = Distributed.get_instance()
    ds_coarse, ds_fine = _get_ds_coarse_fine(config)

    model = CheckpointModelConfig(
        checkpoint_path=config.model_path,
    ).build()

    if config.time_slice is not None:
        ds_coarse = ds_coarse.sel(time=config.time_slice)
        ds_fine = ds_fine.sel(time=config.time_slice)

    n_times = len(ds_fine.time)
    slice_ds = SliceDataset(
        n_times, config.n_ensemble, n_item_per_gpu=config.n_item_per_gpu
    )
    if dist.is_distributed():
        sampler = ContiguousDistributedSampler(slice_ds)
    else:
        sampler = None

    dataloader = DataLoader(
        slice_ds,
        batch_size=1,
        sampler=sampler,
        num_workers=1,
        collate_fn=lambda x: x,
    )

    ds_fine_item = ds_fine.isel(time=0)
    ds_coarse_item = ds_coarse.isel(time=0)
    if config.region is not None:
        ds_fine_item, ds_coarse_item = get_masked_for_region(
            ds_fine_item, ds_coarse_item, config.region
        )

    writer = ZarrWriter(
        path=f"{config.experiment_dir}/generated.zarr",
        dims=("time", "ensemble", "latitude", "longitude"),
        chunks={"time": 1, "ensemble": 1, "latitude": 448, "longitude": 1152},
        coords={
            "latitude": ds_fine_item.latitude.values,
            "longitude": ds_fine_item.longitude.values,
            "time": ds_fine.time.values,
            "ensemble": np.arange(config.n_ensemble),
        },
        data_vars=config.save_vars,
    )

    try:
        for i, batch in enumerate(dataloader):
            slice_item = batch[0]
            time_slice, ens_slice, n_ens = (
                slice_item.time,
                slice_item.ens,
                slice_item.n_ens,
            )
            logging.info(
                f"Rank {dist.rank} - Batch {i} - Time slice {time_slice} -"
                f" Ensemble slice {ens_slice}"
            )

            batch = time_slice_to_batch_data(
                fine=ds_fine,
                coarse=ds_coarse,
                time_slice=slice_item.time,
                var_keys=model.config.in_names,
                region=config.region,
            )
            outputs = model.generate_on_batch(batch, None, n_samples=n_ens)

            generated = {
                k: outputs.prediction[k].cpu().numpy() for k in config.save_vars
            }

            writer.record_batch(generated, {"time": time_slice, "ensemble": ens_slice})
    finally:
        if dist.is_distributed():
            torch.distributed.destroy_process_group()


EXPERIMENT_DIR = "/pscratch/sd/a/andrep/diffusion/experiments"
ZARR_DIR = "/pscratch/sd/a/andrep/diffusion/data/hires-merged"
WEKA_DIR = "/climate-default/2025-02-27-downscaling-data"


SINGAPORE_CONFIG = SaveConfig(
    model_path=f"{EXPERIMENT_DIR}/train-singapore-25-to-3km-multivar/checkpoints/best.ckpt",
    fine_zarr_path=f"{ZARR_DIR}/3km.zarr",
    coarse_zarr_path=f"{ZARR_DIR}/25km.zarr",
    experiment_dir="save_generation_script/singapore_dec21_u10",
    save_vars=[
        "PRATEsfc",
        "eastward_wind_at_ten_meters",
        "northward_wind_at_ten_meters",
    ],
    region="singapore",
    time_slice=slice("2021-12-20T00:00", "2021-12-31T00:00"),
    n_ensemble=1,
    n_item_per_gpu=4,
)

CONUS_100km_25km_CONFIG = SaveConfig(
    model_path=f"{EXPERIMENT_DIR}/train-conus-pr-100km-to-25km-wind-in-v1/checkpoints/ema_ckpt.tar",
    fine_zarr_path=f"{ZARR_DIR}/25km.zarr",
    coarse_zarr_path=f"{ZARR_DIR}/100km.zarr",
    experiment_dir=f"{EXPERIMENT_DIR}/conus-100km-to-25km-save-generated",
    save_vars=["PRATEsfc"],
    region="conus",
    n_ensemble=16,
    n_item_per_gpu=256,
)

CONUS_100km_25km_CONFIG_AUGUSTA = SaveConfig(
    model_path=f"/checkpoints/ema_ckpt.tar",
    fine_zarr_path=f"gs://vcm-ml-intermediate/2025-03-10-downscaling-2yr-XSHiELD/25km.zarr",
    coarse_zarr_path=f"gs://vcm-ml-intermediate/2025-03-10-downscaling-2yr-XSHiELD/100km.zarr",
    experiment_dir=f"/results/conus-100km-to-25km-save-generated-augusta",
    save_vars=["PRATEsfc"],
    region="conus",
    n_ensemble=16,
    n_item_per_gpu=256,
)

CONUS_25km_3km_CONFIG_WEKA = SaveConfig(
    model_path=f"{WEKA_DIR}/models/model-25-to-3/checkpoints/ema_ckpt.tar",
    fine_zarr_path=f"{WEKA_DIR}/3km.zarr",
    coarse_zarr_path=f"{WEKA_DIR}/25km.zarr",
    experiment_dir=f"/climate-default/home/andrep/multivar-july-oct-2021-v3",
    region="conus",
    save_vars=[
        "PRATEsfc",
        "eastward_wind_at_ten_meters",
        "northward_wind_at_ten_meters",
    ],
    time_slice=slice("2021-07-01T00:00", "2021-10-31T18:00"),
    n_ensemble=8,
    n_item_per_gpu=1,
)

# fl_convection
# time_start = "2021-08-20T00:00"
# time_end = "2021-08-30T00:00"

# fl_hurricane
# time_start = "2020-07-06T00:00"
# time_end = "2020-07-16T00:00"

# ca_low
# time_start = "2021-12-10T12:00"
# time_end = "2021-12-20T12:00"

if __name__ == "__main__":
    log_config = LoggingConfig(
        project="downscaling",
        log_to_screen=True,
        log_to_file=False,
        log_to_wandb=True,
        level=logging.INFO,
    )
    log_config.configure_logging("not_used", "")
    env_vars = logging_utils.retrieve_env_vars()
    logging_utils.log_versions()
    url = logging_utils.log_beaker_url()

    use_config = CONUS_25km_3km_CONFIG_WEKA

    os.makedirs(use_config.experiment_dir, exist_ok=True)

    log_config.configure_wandb(
        config=to_flat_dict(asdict(use_config)),
        env_vars=env_vars,
        notes=url,
    )

    run(use_config)
