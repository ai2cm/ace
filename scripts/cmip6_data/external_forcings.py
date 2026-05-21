"""Download and stage external (input4MIPs/LUH2) forcings per scenario.

External forcings are prescribed boundary conditions shared across all
CMIP6 models within an experiment. We stage them once per scenario in a
small zarr at ``<output_directory>/external_forcings/<experiment>.zarr``,
then ``process.py`` joins the relevant scenario's forcings into each
per-model dataset at processing time.

Currently implements:

- ``input4mips_co2`` (annual global-mean CO2 concentration, ppm)

  - **Historical (≥1959)**: NOAA Mauna Loa annual mean CSV. Differs from
    the CMIP6-prescribed Meinshausen et al. historical CO2 by <1 ppm at
    any year; functionally equivalent for emulator training.
  - **Pre-1959 historical**: not available from NOAA. Datasets covering
    this window get the earliest NOAA value (1959, 315.97 ppm) as a
    constant-extrapolation fallback. CMIP6 prescribed values drift down
    to ~278 ppm at 1850 — not a great match. Use the time-subset config
    to start ≥1959 if scientific exactness pre-1959 matters.
  - **SSP245 / SSP585 (2015-2500)**: UoM input4MIPs annual files from
    ESGF (Meinshausen et al. 2017). For 2015 onwards the SSP file's
    values supersede the NOAA observations.

- ``input4mips_so2`` and ``input4mips_bc`` (gridded monthly aerosol
  emission flux, kg m⁻² s⁻¹) — summed across all sectors before regrid

  - **Historical (1950-2023)**: CMIP7-vintage CEDS-CMIP-2025-04-18 on
    the gn (native ~0.5°) grid. The CMIP6-vintage CEDS-2017-05-18 dataset
    has been retracted from every ESGF node — see README's source-vintage
    note for why we use CMIP7-era values for the CMIP6 historical window.
  - **SSP245 / SSP585 (2015-2100)**: CMIP6-vintage IAMC scenario files
    on the native ~0.5° grid.
  - All files regridded conservatively to F22.5.

Planned (deferred to a follow-up commit):

- ``luh2_forest`` (annual forest fraction)

Storage layout::

    <output_directory>/external_forcings/
      historical.zarr/
        co2(time_annual)             # ppm, NOAA Mauna Loa
        so2(time_monthly, lat, lon)  # kg m-2 s-1, CMIP7 CEDS
        bc(time_monthly, lat, lon)   # kg m-2 s-1, CMIP7 CEDS
      ssp245.zarr/
        co2(time_annual)             # ppm, UoM Meinshausen
        so2(time_monthly, lat, lon)  # kg m-2 s-1, IAMC MESSAGE-GLOBIOM
        bc(time_monthly, lat, lon)   # kg m-2 s-1, IAMC MESSAGE-GLOBIOM
      ssp585.zarr/                   # similar, IAMC REMIND-MAGPIE

Each forcing has its own cadence and lives on its own time dimension
(``time_annual`` / ``time_monthly``) inside the per-scenario zarr.
``attach_external_forcings`` renames the dim to ``time`` before calling
the relevant causal helper.

Usage::

    python external_forcings.py --output-directory ./data/cmip6-daily-pilot/v0
    python external_forcings.py --output-directory ... --experiments historical ssp585
    python external_forcings.py --output-directory ... --force  # rebuild
"""

import argparse
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# CO2 sources
# ---------------------------------------------------------------------------

NOAA_MLO_ANNUAL_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.csv"

# UoM input4MIPs CO2 file URLs (ESGF mirror at ORNL). Each SSP file is
# ~42 KB and covers 2015-2500 with (time, sector) where sector 0 = global.
UOM_CO2_SSP_URLS: dict[str, str] = {
    "ssp126": (
        "http://esgf-node.ornl.gov/thredds/fileServer/user_pub_work/"
        "input4MIPs/CMIP6/ScenarioMIP/UoM/UoM-IMAGE-ssp126-1-2-1/"
        "atmos/yr/mole_fraction_of_carbon_dioxide_in_air/gr1-GMNHSH/"
        "v20181127/"
        "mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_"
        "ScenarioMIP_UoM-IMAGE-ssp126-1-2-1_gr1-GMNHSH_2015-2500.nc"
    ),
    "ssp245": (
        "http://esgf-node.ornl.gov/thredds/fileServer/user_pub_work/"
        "input4MIPs/CMIP6/ScenarioMIP/UoM/UoM-MESSAGE-GLOBIOM-ssp245-1-2-1/"
        "atmos/yr/mole_fraction_of_carbon_dioxide_in_air/gr1-GMNHSH/"
        "v20181127/"
        "mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_"
        "ScenarioMIP_UoM-MESSAGE-GLOBIOM-ssp245-1-2-1_gr1-GMNHSH_2015-2500.nc"
    ),
    "ssp370": (
        "http://esgf-node.ornl.gov/thredds/fileServer/user_pub_work/"
        "input4MIPs/CMIP6/ScenarioMIP/UoM/UoM-AIM-ssp370-1-2-1/"
        "atmos/yr/mole_fraction_of_carbon_dioxide_in_air/gr1-GMNHSH/"
        "v20181127/"
        "mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_"
        "ScenarioMIP_UoM-AIM-ssp370-1-2-1_gr1-GMNHSH_2015-2500.nc"
    ),
    "ssp585": (
        "http://esgf-node.ornl.gov/thredds/fileServer/user_pub_work/"
        "input4MIPs/CMIP6/ScenarioMIP/UoM/UoM-REMIND-MAGPIE-ssp585-1-2-1/"
        "atmos/yr/mole_fraction_of_carbon_dioxide_in_air/gr1-GMNHSH/"
        "v20181127/"
        "mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_"
        "ScenarioMIP_UoM-REMIND-MAGPIE-ssp585-1-2-1_gr1-GMNHSH_2015-2500.nc"
    ),
}

# Anthropogenic SO2/BC emission flux (kg m-2 s-1). All files are gridded
# monthly at ~0.5° native resolution with a ``sector`` dimension we sum
# across to get total anthropogenic flux.
#
# Historical uses **CMIP7-vintage** CEDS-CMIP-2025-04-18 because the
# CMIP6-vintage CEDS-2017-05-18 dataset has been retracted from every
# ESGF node we tried. The CMIP7 update is a re-run of the same
# methodology on the same calendar years — values agree to within a
# few percent. See README's source-vintage note. We use the ``gn``
# (native) grid; pre-1950 chunks exist but are deferred since the
# pilot's time-subset is post-1979.
_CEDS_BASE = (
    "https://esgf-node.ornl.gov/thredds/fileServer/user_pub_work/"
    "input4MIPs/CMIP7/CMIP/PNNL-JGCRI/CEDS-CMIP-2025-04-18/atmos/mon/"
    "{var_id}/gn/v20250421/"
    "{var_dash}_input4MIPs_emissions_CMIP_CEDS-CMIP-2025-04-18_gn_{years}.nc"
)
CEDS_HISTORICAL_CHUNKS = ("195001-199912", "200001-202312")

CEDS_HISTORICAL_URLS: dict[str, list[str]] = {
    var_id: [
        _CEDS_BASE.format(
            var_id=var_id,
            var_dash=var_id.replace("_", "-"),
            years=chunk,
        )
        for chunk in CEDS_HISTORICAL_CHUNKS
    ]
    for var_id in ("SO2_em_anthro", "BC_em_anthro")
}


# IAMC SSP emission files. ``(experiment, var_id) -> URL``. Each SSP
# has a single file covering 2015-2100 monthly at ~0.5° native.
_IAMC_SSP_SOURCES = {
    "ssp126": "IAMC-IMAGE-ssp126-1-1",
    "ssp245": "IAMC-MESSAGE-GLOBIOM-ssp245-1-1",
    "ssp370": "IAMC-AIM-ssp370-1-1",
    "ssp585": "IAMC-REMIND-MAGPIE-ssp585-1-1",
}
_IAMC_BASE = (
    "https://esgf-node.ornl.gov/thredds/fileServer/user_pub_work/"
    "input4MIPs/CMIP6/ScenarioMIP/IAMC/{source}/atmos/mon/{var_id}/gn/"
    "v20180628/"
    "{var_dash}_input4MIPs_emissions_ScenarioMIP_{source}_gn_201501-210012.nc"
)

IAMC_SSP_EMISSION_URLS: dict[tuple[str, str], str] = {
    (exp, var_id): _IAMC_BASE.format(
        source=src, var_id=var_id, var_dash=var_id.replace("_", "-")
    )
    for exp, src in _IAMC_SSP_SOURCES.items()
    for var_id in ("SO2_em_anthro", "BC_em_anthro")
}


# Experiments that don't have a dedicated SSP CO2 file fall back to NOAA-only
# (effectively historical-style observed CO2). For historical, NOAA is the
# only source we use. For SSPs, we splice NOAA (≤2014) with UoM (≥2015).
_HISTORICAL_EXPERIMENTS = frozenset({"historical"})


def _download(url: str, dest: Path) -> Path:
    """Stream a URL to a local file, replacing on success."""
    logging.info("Downloading %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    req = urllib.request.Request(url, headers={"User-Agent": "ace-cmip6-pilot"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        with open(partial, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    partial.rename(dest)
    return dest


def fetch_noaa_mlo_annual(cache_dir: Path) -> pd.DataFrame:
    """Return a DataFrame with columns ``year, co2_ppm`` from NOAA's
    Mauna Loa annual mean CO2 record (1959 onwards).
    """
    csv = cache_dir / "co2_annmean_mlo.csv"
    if not csv.exists():
        _download(NOAA_MLO_ANNUAL_URL, csv)
    # The NOAA CSV has comment lines starting with '#'; the header row is
    # ``year,mean,unc``.
    df = pd.read_csv(csv, comment="#")
    df = df.rename(columns={"mean": "co2_ppm"})[["year", "co2_ppm"]]
    df["year"] = df["year"].astype(int)
    df["co2_ppm"] = df["co2_ppm"].astype(float)
    return df


def fetch_uom_ssp_co2(scenario: str, cache_dir: Path) -> pd.DataFrame:
    """Return a DataFrame with columns ``year, co2_ppm`` from the UoM
    input4MIPs annual file for an SSP scenario (2015 onwards).
    Uses the global-mean sector (sector index 0).
    """
    if scenario not in UOM_CO2_SSP_URLS:
        raise ValueError(
            f"No UoM CO2 source URL configured for scenario {scenario!r}; "
            f"available: {sorted(UOM_CO2_SSP_URLS)}"
        )
    url = UOM_CO2_SSP_URLS[scenario]
    nc = cache_dir / Path(url).name
    if not nc.exists():
        _download(url, nc)
    ds = xr.open_dataset(nc, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    var = "mole_fraction_of_carbon_dioxide_in_air"
    da = ds[var].isel(sector=0)  # global mean
    years = np.array([int(t.year) for t in ds["time"].values])
    df = pd.DataFrame({"year": years, "co2_ppm": np.asarray(da.values, dtype=float)})
    ds.close()
    return df


def build_co2_series(experiment: str, cache_dir: Path) -> pd.DataFrame:
    """Construct an annual CO2 time series for one experiment.

    Historical: NOAA Mauna Loa annual (1959+; constant-extrapolation
    fallback for pre-1959). SSPs: NOAA up to 2014 + UoM ssp{N} from 2015.
    Returned DataFrame is sorted by year, with no duplicate years.
    """
    noaa = fetch_noaa_mlo_annual(cache_dir)
    if experiment in _HISTORICAL_EXPERIMENTS:
        return noaa.sort_values("year").reset_index(drop=True)
    ssp = fetch_uom_ssp_co2(experiment, cache_dir)
    # Splice: NOAA for years < first SSP year, SSP from there on.
    cutover = int(ssp["year"].min())
    noaa_part = noaa[noaa["year"] < cutover]
    combined = pd.concat([noaa_part, ssp], ignore_index=True)
    return combined.sort_values("year").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-scenario zarr writer
# ---------------------------------------------------------------------------


def _ceds_emission_urls(experiment: str, var_id: str) -> list[str]:
    """Return the list of source URLs needed to cover ``experiment`` for
    one of ``SO2_em_anthro`` / ``BC_em_anthro``.
    """
    if experiment in _HISTORICAL_EXPERIMENTS:
        return list(CEDS_HISTORICAL_URLS[var_id])
    key = (experiment, var_id)
    if key in IAMC_SSP_EMISSION_URLS:
        return [IAMC_SSP_EMISSION_URLS[key]]
    raise ValueError(f"No emission URL configured for ({experiment!r}, {var_id!r})")


def _open_concat_emission_files(
    paths: list[Path],
) -> xr.DataArray:
    """Open one or more CMIP6-format emission netCDFs, concatenate in
    time, sum across the ``sector`` dimension, and return the
    `(time, lat, lon)` DataArray with its lat/lon bounds preserved for
    conservative regridding.
    """
    pieces: list[xr.Dataset] = []
    for p in paths:
        ds = xr.open_dataset(
            p,
            chunks={"time": 12},
            decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
        )
        pieces.append(ds)
    merged = xr.concat(pieces, dim="time", data_vars="minimal")
    var_id = next(v for v in merged.data_vars if v.endswith("_em_anthro"))
    summed = merged[var_id].sum(dim="sector", keep_attrs=True)
    summed.attrs.setdefault("units", "kg m-2 s-1")
    summed.attrs["sectors_summed"] = int(merged.sizes["sector"])
    # Bring back lat/lon bounds so the regridder can do conservative.
    # ``xr.concat`` along time gives the bounds a spurious time dim
    # (they're time-invariant per file but the concat broadcasts them);
    # collapse to the first timestep so xesmf sees the expected
    # (lat, 2) / (lon, 2) shape.
    out = summed.to_dataset(name=var_id)
    for b in ("lat_bnds", "lon_bnds"):
        if b in merged.variables:
            bnds = merged[b]
            if "time" in bnds.dims:
                bnds = bnds.isel(time=0, drop=True)
            out[b] = bnds
    return out


def _regrid_emissions_to_target(ds: xr.Dataset, target: xr.Dataset) -> xr.DataArray:
    """Conservative regrid of a summed-emission ``(time, lat, lon)``
    variable to the target Gauss-Legendre grid. Falls back to bilinear
    if the source bounds aren't accepted by xesmf — same logic as the
    main pipeline's ``regrid_variables``.
    """
    # ``make_regridder`` normalizes the source dataset internally and
    # the regridder applies to the un-normalized dataset (xesmf uses
    # the coords frozen at build time). Match the existing
    # ``regrid_variables`` pattern.
    from processing import make_regridder

    var_id = next(v for v in ds.data_vars if v.endswith("_em_anthro"))
    regridder, method = make_regridder(ds, target, "conservative")
    logging.info("  emission regrid method=%s for %s", method, var_id)
    regridded = regridder(ds[[var_id]], keep_attrs=True)
    return regridded[var_id]


def fetch_anthro_emissions(
    experiment: str,
    var_id: str,
    cache_dir: Path,
    target: xr.Dataset,
) -> xr.DataArray:
    """End-to-end: download the source files for one ``var_id``
    (``SO2_em_anthro`` / ``BC_em_anthro``) for one ``experiment``, sum
    across sector, regrid conservatively to ``target``, return a
    ``(time, lat, lon)`` DataArray at the target grid.
    """
    urls = _ceds_emission_urls(experiment, var_id)
    local_paths: list[Path] = []
    for url in urls:
        name = Path(url).name
        dest = cache_dir / name
        if not dest.exists():
            _download(url, dest)
        local_paths.append(dest)
    ds = _open_concat_emission_files(local_paths)
    out = _regrid_emissions_to_target(ds, target)
    # Drop the load to memory at this point — the regridded array is
    # tiny (months × 45 × 90 × 4 bytes ≪ 100 MB) so eager is fine.
    out = out.compute()
    out.attrs.setdefault(
        "source_vintage",
        "CMIP7 CEDS-CMIP-2025-04-18"
        if experiment in _HISTORICAL_EXPERIMENTS
        else "CMIP6 IAMC SSP",
    )
    return out


def co2_series_to_dataset(df: pd.DataFrame, experiment: str) -> xr.Dataset:
    """Wrap a (year, co2_ppm) DataFrame as an xarray Dataset with a cftime
    annual time axis (midyear timestamps) and a ``co2`` variable in ppm.
    """
    import cftime

    times = [cftime.DatetimeNoLeap(int(y), 7, 1, 12) for y in df["year"]]
    ds = xr.Dataset(
        {"co2": (("time",), df["co2_ppm"].to_numpy(dtype=np.float32))},
        coords={"time": times},
        attrs={
            "experiment": experiment,
            "title": f"External forcings for {experiment}: annual CO2",
            "source_historical": "NOAA Mauna Loa annual mean (≥1959)",
            "source_ssp": (
                "UoM input4MIPs GHGConcentrations (Meinshausen et al. 2017)"
                if experiment not in _HISTORICAL_EXPERIMENTS
                else "n/a"
            ),
            "units": "ppm",
        },
    )
    ds["co2"].attrs["units"] = "ppm"
    ds["co2"].attrs["long_name"] = "global mean CO2 mole fraction in air"
    return ds


def external_forcings_zarr_path(output_directory: str, experiment: str) -> Path:
    """Path to the per-scenario external-forcings zarr."""
    return Path(output_directory) / "external_forcings" / f"{experiment}.zarr"


def attach_external_forcings(
    day_dataset: xr.Dataset,
    row,
    output_directory: str,
    experiment: str,
) -> None:
    """Open the per-scenario external-forcings zarr for ``experiment``
    and add its variables to ``day_dataset`` in-place, mapped onto the
    daily time axis via the appropriate causal transform.

    Currently handles:
    - ``co2``: annual scalar → broadcast to ``(time, lat, lon)`` via
      :func:`processing.causal_annual_to_daily`. Output name
      ``input4mips_co2``.

    If the per-scenario zarr is missing the function records a warning
    on ``row`` and returns silently — datasets without staged external
    forcings still produce valid output (just without input4mips_*).
    """
    # Imported here to avoid a circular import at module-load time
    # (processing.py imports from config.py which has no dependency on
    # external_forcings.py, but external_forcings.py uses xarray for
    # writes which is also imported by processing.py).
    from processing import causal_annual_to_daily

    path = external_forcings_zarr_path(output_directory, experiment)
    if not path.exists():
        row.warnings.append(
            f"no external forcings staged at {path} for {experiment}; "
            "input4mips_* variables omitted"
        )
        return
    ef = xr.open_zarr(path, consolidated=True)
    daily_time = day_dataset["time"]

    if "co2" in ef.data_vars:
        # ``co2`` lives on ``time_annual`` in the staged zarr; rename
        # to ``time`` so the causal helper finds its expected dim.
        co2_annual = ef["co2"].rename({"time_annual": "time"})
        co2_daily = causal_annual_to_daily(co2_annual, daily_time)
        # Broadcast scalar (time,) to (time, lat, lon) so the output
        # matches the rest of the dataset's shape convention. Build
        # explicitly rather than via ``broadcast_like`` so we don't
        # depend on any specific 2D variable being present (with
        # max_core_missing > 0, even ``tas`` could be absent).
        n_lat = day_dataset.sizes["lat"]
        n_lon = day_dataset.sizes["lon"]
        arr = (
            np.broadcast_to(
                co2_daily.values[:, None, None],
                (co2_daily.sizes["time"], n_lat, n_lon),
            )
            .copy()
            .astype(np.float32)
        )
        day_dataset["input4mips_co2"] = xr.DataArray(
            arr,
            dims=("time", "lat", "lon"),
            coords={
                "time": day_dataset["time"],
                "lat": day_dataset["lat"],
                "lon": day_dataset["lon"],
            },
            name="input4mips_co2",
            attrs={"units": "ppm", "long_name": "global mean CO2 mole fraction"},
        )

    # SO2/BC: gridded monthly. Already at F22.5 (the staging script
    # regridded), so we just need the causal monthly-to-daily mapping
    # and a rename. Lat/lon match the daily dataset's grid exactly.
    from processing import causal_monthly_to_daily

    for short, attrs in (
        (
            "so2",
            {
                "units": "kg m-2 s-1",
                "long_name": "anthropogenic SO2 emission flux",
            },
        ),
        (
            "bc",
            {
                "units": "kg m-2 s-1",
                "long_name": "anthropogenic BC emission flux",
            },
        ),
    ):
        if short not in ef.data_vars:
            continue
        # so2/bc live on ``time_monthly`` in the staged zarr.
        monthly = ef[short].rename({"time_monthly": "time"})
        daily = causal_monthly_to_daily(monthly, daily_time)
        # Drop sector_bnds-style remnants if any survived the staging.
        for d in list(daily.dims):
            if d not in ("time", "lat", "lon"):
                daily = daily.isel({d: 0}, drop=True)
        out_name = f"input4mips_{short}"
        day_dataset[out_name] = (
            daily.rename(out_name).astype(np.float32).assign_attrs(**attrs)
        )

    ef.close()


def stage_for_experiment(
    experiment: str,
    output_directory: str,
    cache_dir: Optional[Path] = None,
    force: bool = False,
    variables: Optional[tuple[str, ...]] = None,
) -> Path:
    """Build the per-scenario external-forcings zarr at
    ``<output_directory>/external_forcings/<experiment>.zarr``,
    containing every forcing variable in ``variables`` (default: all
    implemented). Skips work if the target zarr already exists and
    ``force=False``.

    Available variables: ``co2``, ``so2``, ``bc``.
    """
    if variables is None:
        variables = ("co2", "so2", "bc")
    out_dir = Path(output_directory) / "external_forcings"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{experiment}.zarr"
    if target.exists() and not force:
        logging.info("  [%s] up to date at %s", experiment, target)
        return target
    if cache_dir is None:
        cache_dir = out_dir / ".cache"

    # Target grid for emission regridding — same Gauss-Legendre F22.5
    # the rest of the pipeline uses.
    from grid import make_target_grid

    target_grid = make_target_grid("F22.5")

    # Each forcing has its own cadence (CO2 annual, SO2/BC monthly), so
    # we give them distinct time-dimension names (``time_annual``,
    # ``time_monthly``) and merge as a single zarr. attach_external_forcings
    # renames the relevant dim to ``time`` before calling the causal helper.
    pieces: list[xr.Dataset] = []
    if "co2" in variables:
        df = build_co2_series(experiment, cache_dir)
        co2_ds = co2_series_to_dataset(df, experiment).rename({"time": "time_annual"})
        logging.info(
            "  [%s] CO2 series years %d-%d (%d values)",
            experiment,
            int(df["year"].min()),
            int(df["year"].max()),
            len(df),
        )
        pieces.append(co2_ds)
    for short, source_var in (("so2", "SO2_em_anthro"), ("bc", "BC_em_anthro")):
        if short not in variables:
            continue
        da = fetch_anthro_emissions(experiment, source_var, cache_dir, target_grid)
        logging.info(
            "  [%s] %s gridded series months=%d, range=(%g, %g) %s",
            experiment,
            short,
            da.sizes["time"],
            float(da.min()),
            float(da.max()),
            da.attrs.get("units", "?"),
        )
        pieces.append(da.rename(short).rename({"time": "time_monthly"}).to_dataset())

    ds = xr.merge(pieces, compat="override")
    if target.exists():
        import shutil

        shutil.rmtree(target)
    ds.to_zarr(target, consolidated=True)
    return target


def stage_co2_for_experiment(
    experiment: str,
    output_directory: str,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Backwards-compatible wrapper that stages only CO2."""
    return stage_for_experiment(
        experiment, output_directory, cache_dir, force, variables=("co2",)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        required=True,
        help="Same as the ProcessConfig's output_directory; the staging "
        "zarrs land at <output_directory>/external_forcings/",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["historical", "ssp245", "ssp585"],
        help="Experiments to stage (default: historical ssp245 ssp585)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild even if zarr exists"
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        choices=["co2", "so2", "bc"],
        help="Subset of variables to stage (default: all)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Where to cache downloaded source files (default: "
        "<output_directory>/external_forcings/.cache)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cache = Path(args.cache_dir) if args.cache_dir else None
    for exp in args.experiments:
        stage_for_experiment(
            exp,
            args.output_directory,
            cache_dir=cache,
            force=args.force,
            variables=tuple(args.variables) if args.variables else None,
        )


if __name__ == "__main__":
    main()


__all__ = [
    "NOAA_MLO_ANNUAL_URL",
    "UOM_CO2_SSP_URLS",
    "fetch_noaa_mlo_annual",
    "fetch_uom_ssp_co2",
    "build_co2_series",
    "co2_series_to_dataset",
    "stage_co2_for_experiment",
]
