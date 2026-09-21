"""Shared definitions for the masked, per-land-area snow channels.

The two daily parent stores define snow differently. CM4's ``surface_snow_amount``
and ``surface_snow_area_fraction`` are land-model diagnostics per unit LAND area
(``cell_measures: area: land_area``), with cover in percent. ERA5's are grid-box
averages: the native snow depth is a grid-box mean, so both the native field and
its conservative regrid are per unit CELL area, diluted by the land fraction in
coastal cells, with cover as a fraction.

``land_snow_fields`` maps both onto one definition, per unit land area with cover
as a fraction in [0, 1], and applies the shared static snow mask (NaN outside):

- ERA5: SWE / land_fraction; cover = min(1, cover / land_fraction). The SWE
  division is exact (regrid(lsm * sd_land) / regrid(lsm)). The cover division is
  exact for a single native point and approximate across the native points of a
  1-degree cell, because ERA5 clips cover at 1 on the native grid before
  regridding. An exact cover needs the native fields (see README).
- CM4: SWE unchanged; cover / 100.

Used by ``build_masked_snow_channels.py`` (store) and ``fit_masked_snow_stats.py``
(normalization statistics) so the data and its statistics cannot drift apart.
"""

import dataclasses
import os
import subprocess

import numpy as np
import xarray as xr

SWE = "surface_snow_amount"
SCF = "surface_snow_area_fraction"
MASKED = {SWE: f"{SWE}_masked", SCF: f"{SCF}_masked"}
OUTPUT_SUFFIX = "land-snow-masked"
HERE = os.path.dirname(os.path.abspath(__file__))
MASK_FILE = os.path.join(HERE, "snow_mask.nc")
SHARD_STEPS = 360
STATS_FILENAMES = (
    "centering.nc",
    "scaling-full-field.nc",
    "scaling-residual.nc",
    "time-mean.nc",
)


@dataclasses.dataclass(frozen=True)
class Parent:
    name: str
    directory: str
    stats_url: str
    cover_scale: float
    divide_by_land_fraction: bool
    stats_start: str | None
    stats_stop: str | None
    stats_pair_stride_days: int

    @property
    def url(self) -> str:
        return f"{self.directory}/{self.name}.zarr"

    @property
    def output_name(self) -> str:
        return f"{self.name}-{OUTPUT_SUFFIX}"

    @property
    def output_url(self) -> str:
        return f"{self.directory}/{self.output_name}.zarr"


PARENTS = {
    "era5": Parent(
        name="2026-08-07-era5-1deg-8layer-daily-1940-2025",
        directory="gs://vcm-ml-intermediate/2026-08-07-era5-1deg-8layer-daily-1940-2025",
        stats_url=(
            "gs://vcm-ml-intermediate/"
            "2026-08-07-era5-1deg-8layer-daily-stats-1990-2019/combined"
        ),
        cover_scale=1.0,
        divide_by_land_fraction=True,
        stats_start="1990-01-01",
        stats_stop="2019-12-31",
        stats_pair_stride_days=2,
    ),
    "cm4": Parent(
        name="2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily",
        directory=(
            "gs://vcm-ml-intermediate/"
            "2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily"
        ),
        stats_url=(
            "gs://vcm-ml-intermediate/"
            "2025-03-21-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily-stats/"
            "combined"
        ),
        cover_scale=100.0,
        divide_by_land_fraction=False,
        stats_start=None,
        stats_stop=None,
        stats_pair_stride_days=8,
    ),
}


def patch_gcsfs_suffix_ranges() -> None:
    """gcsfs 2026.8.0 mishandles negative (suffix) byte ranges, which the zarr
    sharding codec uses to read shard indexes, so reads of the sharded parent
    stores fail with checksum errors. Rewrite suffix ranges as explicit ranges."""
    import gcsfs

    if getattr(gcsfs.GCSFileSystem, "_suffix_ranges_patched", False):
        return
    original = gcsfs.GCSFileSystem._cat_file

    async def _cat_file(self, path, start=None, end=None, **kwargs):
        if start is not None and start < 0:
            size = (await self._info(path))["size"]
            start, end = max(0, size + start), size
        return await original(self, path, start=start, end=end, **kwargs)

    gcsfs.GCSFileSystem._cat_file = _cat_file
    gcsfs.GCSFileSystem._suffix_ranges_patched = True


def gcs_credentials():
    from google.oauth2.credentials import Credentials

    token = subprocess.run(
        ["gcloud", "auth", "print-access-token"], capture_output=True, text=True
    ).stdout.strip()
    return Credentials(token=token)


def open_parent(parent: Parent) -> xr.Dataset:
    patch_gcsfs_suffix_ranges()
    return xr.open_zarr(
        parent.url, consolidated=False, storage_options={"token": gcs_credentials()}
    )


def load_mask() -> np.ndarray:
    """Static validity mask, 1 on land cells with land_fraction >= 0.5 that are
    not ice sheet, 0 elsewhere. Same array for both datasets."""
    return xr.load_dataset(MASK_FILE)["mask"].values


def land_fraction(ds: xr.Dataset) -> np.ndarray:
    field = ds["land_fraction"]
    if "time" in field.dims:
        field = field.isel(time=0)
    return field.values.astype(np.float64)


def land_snow_fields(
    swe: np.ndarray,
    scf: np.ndarray,
    parent: Parent,
    land_frac: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-land-area SWE [kg/m2] and cover fraction [0, 1], NaN outside the mask.

    ``swe`` and ``scf`` may carry leading dimensions (time); ``land_frac`` and
    ``valid`` are the 2D spatial fields.
    """
    swe = np.asarray(swe, dtype=np.float64)
    scf = np.asarray(scf, dtype=np.float64) / parent.cover_scale
    if parent.divide_by_land_fraction:
        divisor = np.where(valid, land_frac, 1.0)
        if np.any(divisor[valid] < 0.5):
            raise ValueError("mask admits cells with land_fraction below 0.5")
        swe = swe / divisor
        scf = np.minimum(scf / divisor, 1.0)
    swe = np.where(valid, swe, np.nan)
    scf = np.where(valid, scf, np.nan)
    return swe.astype(np.float32), scf.astype(np.float32)
