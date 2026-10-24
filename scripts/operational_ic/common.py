"""Shared pieces of the operational initial-condition pipeline.

The pipeline reuses the per-chunk functions of ``scripts/glorys`` (GLO12 ->
4-degree ocean states) and ``scripts/era5`` (ERA5 -> 4-degree atmosphere)
without Apache Beam: both pipeline modules are loaded here by path, the way
``scripts/glorys/make_regrid_weights.py`` does, so one day of data is processed
by exactly the code that built the training stores.
"""

import functools
import importlib.util
import logging
import os
import pathlib
import types

import pandas as pd
import xarray as xr

HERE = pathlib.Path(__file__).resolve().parent
SCRIPTS = HERE.parent

OUTPUT_GRID = "F22.5"  # 4 degrees; the grid of the deployed GLO12 fine-tune

# Where downloaded one-off inputs live (regrid weights, N320 coordinates, the
# TOA-insolation climatology). Override with SAMUDRACE_OPER_CACHE.
CACHE_DIR = pathlib.Path(
    os.environ.get("SAMUDRACE_OPER_CACHE", "~/.cache/samudrace-oper")
).expanduser()

_INTERMEDIATE = "gs://vcm-ml-intermediate"
# xESMF conservative weights 1/12 degree -> F22.5, built once by
# scripts/glorys/make_regrid_weights.py and used for the training store.
REGRID_WEIGHTS_URL = (
    f"{_INTERMEDIATE}/2026-09-17-GLORYS-trial-run/glorys-F22.5-weights.nc"
)
# The GLO12 4-degree training store (GLORYS pipeline output): the source of
# every static ocean field (masks, fractions, deptho, hfgeou, idepth) so the
# initial condition sits on exactly the geometry the checkpoint was trained on.
REFERENCE_OCEAN_STORE = (
    f"{_INTERMEDIATE}/2026-10-22-glo12-ocean-4deg-19level-5day-2022-2025.zarr"
)
# The ERA5 4-degree daily store the ACE2S atmosphere was trained on: the source
# of HGTsfc and ak/bk, of the TOA-insolation climatology, and the regression
# reference for the atmosphere sample.
REFERENCE_ATMOS_STORE = (
    f"{_INTERMEDIATE}/2026-09-08-era5-4deg-8layer-daily-1940-2025.zarr"
)
# Annual-mean CO2 interpolated to daily values, as read by scripts/era5.
CO2_URL = "gs://vcm-ml-raw-flexible-retention/2026-03-12-co2-annual-mean-for-era5.zarr"

# Output states of the training store lie on this grid (scripts/glorys
# TIME_ORIGIN + 5 days); an initial condition need not, but it is reported.
OCEAN_TIME_ORIGIN = pd.Timestamp("1993-01-01")
OCEAN_STRIDE_DAYS = 5

log = logging.getLogger("operational_ic")


def _load_module(name: str, path: pathlib.Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=None)
def glorys_pipeline() -> types.ModuleType:
    """``scripts/glorys/pipeline/glorys-pipeline.py`` as a module."""
    return _load_module(
        "glorys_pipeline", SCRIPTS / "glorys" / "pipeline" / "glorys-pipeline.py"
    )


@functools.lru_cache(maxsize=None)
def era5_pipeline() -> types.ModuleType:
    """``scripts/era5/pipeline/xr-beam-pipeline.py`` as a module."""
    return _load_module(
        "era5_pipeline", SCRIPTS / "era5" / "pipeline" / "xr-beam-pipeline.py"
    )


def cached_gcs_file(url: str) -> pathlib.Path:
    """Local copy of a ``gs://`` file under CACHE_DIR, downloaded once."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dst = CACHE_DIR / url.rsplit("/", 1)[-1]
    if not dst.exists():
        import gcsfs

        log.info("downloading %s -> %s", url, dst)
        gcsfs.GCSFileSystem(token="cloud").get(url, str(dst))
    return dst


def open_zarr(url: str) -> xr.Dataset:
    """Open a zarr store by local path or ``gs://`` URL (ADC credentials)."""
    if url.startswith("gs://"):
        return xr.open_dataset(
            url,
            engine="zarr",
            chunks=None,
            cache=False,
            storage_options={"token": "cloud"},
        )
    return xr.open_zarr(url, chunks=None)


def on_training_time_grid(day: pd.Timestamp) -> bool:
    return (day - OCEAN_TIME_ORIGIN).days % OCEAN_STRIDE_DAYS == 0


def set_time_encoding(ds: xr.Dataset) -> xr.Dataset:
    """Encode ``time`` the way the training stores do (proleptic Gregorian)."""
    for v in list(ds.variables):
        ds[v].encoding = {}
    ds["time"].encoding = {
        "units": "days since 1990-01-01",
        "calendar": "proleptic_gregorian",
    }
    return ds
