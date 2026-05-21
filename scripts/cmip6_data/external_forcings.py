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

Planned (deferred to a follow-up commit):

- ``input4mips_so2``, ``input4mips_bc`` (gridded monthly aerosol emissions)
- ``luh2_forest`` (annual forest fraction)

Storage layout::

    <output_directory>/external_forcings/
      historical.zarr/
        co2(time)   # annual time series, units = ppm
      ssp245.zarr/
        co2(time)
      ssp585.zarr/
        co2(time)

The per-scenario zarr is tiny (a few KB per scenario) so we copy it into
each per-model dataset at process time rather than referencing it from
disk at training time.

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
        co2_daily = causal_annual_to_daily(ef["co2"], daily_time)
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

    ef.close()


def stage_co2_for_experiment(
    experiment: str,
    output_directory: str,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Build the per-scenario CO2 zarr at
    ``<output_directory>/external_forcings/<experiment>.zarr``.
    Skips work if the target zarr already exists and ``force=False``.
    """
    out_dir = Path(output_directory) / "external_forcings"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{experiment}.zarr"
    if target.exists() and not force:
        logging.info("  [%s] up to date at %s", experiment, target)
        return target
    if cache_dir is None:
        cache_dir = out_dir / ".cache"
    df = build_co2_series(experiment, cache_dir)
    ds = co2_series_to_dataset(df, experiment)
    logging.info(
        "  [%s] writing CO2 series years %d-%d (%d values)",
        experiment,
        int(df["year"].min()),
        int(df["year"].max()),
        len(df),
    )
    if target.exists():
        import shutil

        shutil.rmtree(target)
    ds.to_zarr(target, consolidated=True)
    return target


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
        "--cache-dir",
        default=None,
        help="Where to cache downloaded source files (default: "
        "<output_directory>/external_forcings/.cache)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cache = Path(args.cache_dir) if args.cache_dir else None
    for exp in args.experiments:
        stage_co2_for_experiment(
            exp, args.output_directory, cache_dir=cache, force=args.force
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
