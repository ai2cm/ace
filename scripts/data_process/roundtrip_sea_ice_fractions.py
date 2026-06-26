"""De-block the sea-ice surface fractions in the *existing* coupled E3SMv3 zarrs
with a real-space Gaussian smoother, writing new versioned zarrs for coupled
fine-tuning.

WHY
---
The atmosphere fields were remapped online by EAM (ne30pg2 -> shifted gaussian
180x360) with a *first-order conservative* scheme ("blocky" piecewise-constant
reconstruction). ACE's offline pipeline applies an SHT roundtrip to the
prognostic atmosphere variables to de-block them, but ICEFRAC / OCNFRAC are NOT
in that list, so the sea-ice fraction (and the derived ocean_sea_ice_fraction
that Samudra predicts) keep the blocky structure. This script smooths the
sea-ice fractions in the already-processed coupled zarrs, *post hoc*, so we can
do a short coupled fine-tune without re-running the whole data pipeline.

WHY GAUSSIAN (not the pipeline's SHT roundtrip)
-----------------------------------------------
ICEFRAC is a sharp, bounded [0,1] field. An SHT roundtrip rings (Gibbs) near the
ice edge -- it leaves a band-limited "spectral imprint" that leaks ~2x further
into the far field and needs clipping (which biases the mean). A real-space
Gaussian de-blocks MORE, rings far less, and stays in [0,1] by construction (it
is a convex combination of inputs -> no overshoot, no clip, no clip-bias). The
smoother is AREA-AWARE: the zonal kernel widens like 1/cos(lat) (capped), because
the conservative-remap blockiness widens toward the pole (one ne30 source cell
spans several 1-deg lon cells there). It is also MASK-AWARE (normalized by the
smoothed validity mask) so the ice region does not bleed toward the masked-out
exterior.

WHAT IT DOES (per realm)
------------------------
The single physical quantity smoothed is the sea-ice CONCENTRATION
    sic = ocean_sea_ice_fraction   (fraction of the *ocean* area that is ice)
Everything else is derived from it so the surface partition stays consistent:
    ssf      = 1 - LANDFRAC                       # sea-surface fraction (== sfrac_mod)
    sic_sm   = masked_gaussian(sic)               # de-blocked concentration in [0,1]
    ICEFRAC  = sic_sm        * ssf   (inside ice mask)
    OCNFRAC  = (1 - sic_sm)  * ssf   (inside ice mask; original preserved outside)
  =>  ICEFRAC + OCNFRAC + LANDFRAC == 1   (exactly, inside the mask)

This mirrors compute_coupled_sea_ice() in coupled_dataset_utils.py
(ifrac_mod = sic * sfrac_mod ; ofrac_mod = (1 - sic) * sfrac_mod) with
sfrac_mod == 1 - LANDFRAC. OCNFRAC is only updated INSIDE the sea-ice mask
(where ICEFRAC is defined); outside it the original OCNFRAC < ssf because the
time-mean-SST mask drops transient ice, so those originals are preserved.

For the ocean realm we re-apply the sea-ice/iceVolumeTotal consistency check from
fme.core.corrector.ocean.SeaIceFractionConfig:
    iceVolumeTotal := iceVolumeTotal * (sic_sm > 0)     # zero volume where ice-free

NaN handling: ocean ice fields and atmosphere ICEFRAC are masked (NaN where
time-mean SST > threshold), and the atmosphere has a few leading all-NaN steps.
The masked Gaussian smooths only over valid cells; each output field then reuses
its ORIGINAL NaN pattern via `.where(original.notnull())`, so masks and
leading-NaN steps are preserved byte-for-byte.

OUTPUTS
-------
    {OUT_DIR}/2026-06-25-E3SMv3-piControl-105yr-coupled-ocean.zarr
    {OUT_DIR}/2026-06-25-E3SMv3-piControl-105yr-coupled-atmosphere.zarr

By default the script runs a VERIFICATION pass on a sample of timesteps and
prints diagnostics WITHOUT writing. Pass --write to actually create the zarrs.

NOTE: this is RETRAINING-data prep. Only do a *coupled fine-tune* with these.
"""

import argparse
import logging
import os
import time

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ----------------------------------------------------------------------------
NLAT, NLON = 180, 360
INPUT_VERSION = "2026-06-02"
OUTPUT_VERSION = "2026-06-25"
FAMILY = "E3SMv3-piControl-105yr-coupled"
DATA_DIR = "/climate-default"

# xpartition chunking, matching scripts/data_process/writer_utils.py
INNER_CHUNKS = {"time": 1, "lon": -1, "lat": -1}
OUTER_CHUNKS = {"time": 360, "lon": -1, "lat": -1}

SIC = "ocean_sea_ice_fraction"
ICEFRAC = "ICEFRAC"
OCNFRAC = "OCNFRAC"
LANDFRAC = "LANDFRAC"
ICEVOL = "iceVolumeTotal"

# Gaussian smoother params (set from CLI). _SIGMA_LON is the per-latitude zonal
# sigma array (area-aware), computed once per run from the grid latitudes.
SIGMA_LAT = 1.0
SIGMA_LON_BASE = 1.0
SIGMA_LON_CAP = 6.0
_SIGMA_LON = None  # np.ndarray, length NLAT


def _init_sigma_lon(lat_deg: np.ndarray):
    """Zonal sigma per latitude row: widen like 1/cos(lat), capped."""
    global _SIGMA_LON
    coslat = np.clip(np.cos(np.deg2rad(lat_deg)), 0.05, 1.0)
    _SIGMA_LON = np.minimum(SIGMA_LON_BASE / coslat, SIGMA_LON_CAP).astype("float32")


def _gsmooth_np(arr: np.ndarray) -> np.ndarray:
    """Separable, area-aware Gaussian smooth of a (..., lat, lon) array.

    Latitude: fixed sigma (reflect at poles). Longitude: per-row sigma
    (area-aware, periodic). No NaNs expected here (caller pre-fills).
    """
    a = gaussian_filter1d(arr.astype("float32"), SIGMA_LAT, axis=-2, mode="nearest")
    out = np.empty_like(a)
    for j in range(a.shape[-2]):
        out[..., j, :] = gaussian_filter1d(
            a[..., j, :], float(_SIGMA_LON[j]), axis=-1, mode="wrap"
        )
    return out.astype("float32")


def smooth_da(da: xr.DataArray) -> xr.DataArray:
    """Lazily Gaussian-smooth a (time, lat, lon) DataArray (NaNs pre-filled)."""
    return xr.apply_ufunc(
        _gsmooth_np,
        da,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "lon"]],
        dask="parallelized",
        output_dtypes=[np.float32],
    )


# ----------------------------------------------------------------------------
def _smoothed_sic(ds: xr.Dataset, realm: str, ssf: xr.DataArray) -> xr.DataArray:
    """De-blocked sea-ice concentration sic_sm in [0, 1] (lazy), mask-normalized.

    ocean: smooth the stored ocean_sea_ice_fraction (the Samudra target).
    atmosphere: recover sic = ICEFRAC / ssf, then smooth.
    """
    if realm == "ocean":
        sic0 = ds[SIC]
    else:
        sic0 = (ds[ICEFRAC] / ssf.where(ssf > 0)).clip(0.0, 1.0)

    # static validity mask (the sea-ice mask); robust to leading all-NaN steps
    valid2d = sic0.notnull().any("time") if "time" in sic0.dims else sic0.notnull()
    den = _gsmooth_np(valid2d.values.astype("float32"))  # smoothed validity (2D)
    den_da = xr.DataArray(den, coords=valid2d.coords, dims=valid2d.dims)

    num = smooth_da(sic0.fillna(0.0))  # smoothed (sic * valid)
    sic_sm = (num / den_da.where(den_da > 1e-6)).clip(0.0, 1.0)
    return sic_sm


def build_modified_dataset(ds: xr.Dataset, realm: str) -> xr.Dataset:
    """Copy of `ds` with smoothed, consistent sea-ice fractions (lazy)."""
    ssf = (1.0 - ds[LANDFRAC]).clip(0.0, 1.0)  # sea-surface fraction (sfrac_mod)
    sic_sm = _smoothed_sic(ds, realm, ssf)

    # Only modify fractions INSIDE the sea-ice mask (where ICEFRAC is defined).
    in_ice_mask = ds[ICEFRAC].notnull()
    icefrac_new = (sic_sm * ssf).where(in_ice_mask)
    ocnfrac_new = xr.where(in_ice_mask, (1.0 - sic_sm) * ssf, ds[OCNFRAC]).where(
        ds[OCNFRAC].notnull()
    )

    updates = {
        ICEFRAC: icefrac_new.astype("float32"),
        OCNFRAC: ocnfrac_new.astype("float32"),
    }
    updates[ICEFRAC].attrs = ds[ICEFRAC].attrs
    updates[OCNFRAC].attrs = ds[OCNFRAC].attrs

    if realm == "ocean":
        sic_new = sic_sm.where(ds[SIC].notnull())
        updates[SIC] = sic_new.astype("float32")
        updates[SIC].attrs = ds[SIC].attrs
        if ICEVOL in ds:
            icevol_new = xr.where(sic_sm > 0.0, ds[ICEVOL], 0.0).where(
                ds[ICEVOL].notnull()
            )
            updates[ICEVOL] = icevol_new.astype("float32")
            updates[ICEVOL].attrs = ds[ICEVOL].attrs

    out = ds.assign(updates)
    out.attrs = dict(ds.attrs)
    out.attrs["history"] = (
        out.attrs.get("history", "")
        + f" | sea-ice fractions area-aware-Gaussian smoothed "
        f"(sigma_lat={SIGMA_LAT}, sigma_lon_base={SIGMA_LON_BASE}, cap={SIGMA_LON_CAP}) "
        "by roundtrip_sea_ice_fractions.py"
    )
    return out


# ----------------------------------------------------------------------------
def _curv(field: np.ndarray) -> np.ndarray:
    return np.abs(np.roll(field, 1, -1) - 2 * field + np.roll(field, -1, -1))


def verify(realm: str, ds: xr.Dataset, n_samples: int):
    logging.info(f"--- VERIFY {realm}: {n_samples} sampled timesteps ---")
    nt = ds.sizes["time"]
    idx = np.linspace(0, nt - 1, n_samples).astype(int)
    sub = ds.isel(time=idx).load()
    mod = build_modified_dataset(sub, realm).load()

    # 1. partition sums to 1 INSIDE the ice mask
    in_mask = ~np.isnan(sub[ICEFRAC].values)
    s = (
        mod[ICEFRAC].values
        + mod[OCNFRAC].values
        + np.broadcast_to(sub[LANDFRAC].values, mod[ICEFRAC].shape)
    )
    max_sum_err = float(np.nanmax(np.abs(s[in_mask] - 1.0))) if in_mask.any() else 0.0

    # 1b. OCNFRAC unchanged OUTSIDE the ice mask
    out_mask = (~in_mask) & (~np.isnan(sub[OCNFRAC].values))
    ocn_outside = (
        float(np.nanmax(np.abs((mod[OCNFRAC].values - sub[OCNFRAC].values)[out_mask])))
        if out_mask.any()
        else 0.0
    )

    icef = mod[ICEFRAC].values
    bounds_ok = np.nanmin(icef) >= -1e-6 and np.nanmax(icef) <= 1.0 + 1e-6
    nan_ok = bool(
        np.array_equal(np.isnan(sub[ICEFRAC].values), np.isnan(mod[ICEFRAC].values))
    )

    # blockiness over the marginal ice zone (partial ice in the ORIGINAL), same cells
    miz = (sub[ICEFRAC].values > 0.05) & (sub[ICEFRAC].values < 0.95)
    cb = float(np.nanmean(_curv(np.nan_to_num(sub[ICEFRAC].values))[miz]))
    ca = float(np.nanmean(_curv(np.nan_to_num(mod[ICEFRAC].values))[miz]))

    logging.info(f"  partition sum-to-1 (in mask) : {max_sum_err:.2e}")
    logging.info(f"  OCNFRAC unchanged out-of-mask: {ocn_outside:.2e}")
    logging.info(f"  ICEFRAC within [0,1]         : {bounds_ok} "
                 f"(min={np.nanmin(icef):.3f} max={np.nanmax(icef):.3f})")
    logging.info(f"  ICEFRAC NaN pattern preserved: {nan_ok}")
    logging.info(f"  MIZ blockiness ICEFRAC       : {cb:.4f} -> {ca:.4f} "
                 f"({100*(ca-cb)/cb:+.0f}%)")
    if realm == "ocean" and ICEVOL in mod:
        sic = mod[SIC].values
        bad = int(np.nansum((mod[ICEVOL].values > 0) & (sic == 0)))
        vol_ok = np.array_equal(
            np.isnan(sub[ICEVOL].values), np.isnan(mod[ICEVOL].values)
        )
        logging.info(f"  iceVolumeTotal>0 where sic==0 : {bad} cells (want 0)")
        logging.info(f"  iceVolumeTotal NaN preserved : {vol_ok}")


# ----------------------------------------------------------------------------
def write_zarr(ds: xr.Dataset, output_store: str, n_dask_workers):
    import xpartition  # noqa: F401

    if os.path.isdir(output_store):
        raise ValueError(
            f"Output store {output_store} already exists. Delete it to rewrite."
        )

    client = None
    if n_dask_workers:
        import dask
        dask.config.set({"logging.distributed": "error"})
        from dask.distributed import Client
        client = Client(n_workers=n_dask_workers)
        logging.info(client.dashboard_link)

    try:
        ds = ds.chunk(OUTER_CHUNKS)
        nt = ds.sizes["time"]
        n_split = max(1, -(-nt // OUTER_CHUNKS["time"]))
        logging.info(f"Initializing store {output_store} (n_split={n_split})")
        ds.partition.initialize_store(output_store, inner_chunks=INNER_CHUNKS)
        for i in range(n_split):
            t0 = time.time()
            n_retries, delay = 0, 1.0
            while True:
                try:
                    logging.info(f"  writing segment {i + 1}/{n_split}")
                    ds.partition.write(
                        output_store, n_split, ["time"], i,
                        collect_variable_writes=True,
                    )
                    logging.info(f"  segment {i + 1} done in {time.time()-t0:.1f}s")
                    break
                except RuntimeError:
                    if n_retries > 10:
                        raise
                    logging.info("  RuntimeError, retrying...")
                    time.sleep(delay)
                    n_retries += 1
                    delay *= 1.1
        logging.info(f"Completed {output_store}")
    finally:
        if client is not None:
            client.close()


def process_realm(realm: str, args):
    in_path = os.path.join(DATA_DIR, f"{INPUT_VERSION}-{FAMILY}-{realm}.zarr")
    out_path = os.path.join(args.out_dir, f"{OUTPUT_VERSION}-{FAMILY}-{realm}.zarr")
    logging.info("=" * 70)
    logging.info(f"REALM={realm}")
    logging.info(f"  input : {in_path}")
    logging.info(f"  output: {out_path}")

    ds = xr.open_zarr(in_path)
    for req in [ICEFRAC, OCNFRAC, LANDFRAC]:
        if req not in ds:
            raise KeyError(f"{req} missing from {in_path}")
    if realm == "ocean" and SIC not in ds:
        raise KeyError(f"{SIC} missing from ocean zarr {in_path}")

    _init_sigma_lon(ds["lat"].values)

    if not args.write:
        verify(realm, ds, args.sample)
        logging.info("  (verification only; pass --write to create the zarr)")
        return

    mod = build_modified_dataset(ds, realm)
    write_zarr(mod, out_path, args.n_dask_workers)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--realm", choices=["ocean", "atmosphere", "both"], default="both")
    p.add_argument("--sigma-lat", type=float, default=1.0,
                   help="meridional Gaussian sigma (grid cells)")
    p.add_argument("--sigma-lon-base", type=float, default=1.0,
                   help="zonal Gaussian sigma at the equator (grid cells); "
                        "scaled by 1/cos(lat) toward the poles")
    p.add_argument("--sigma-lon-cap", type=float, default=6.0,
                   help="cap on the zonal sigma near the poles")
    p.add_argument("--out-dir", default=DATA_DIR)
    p.add_argument("--write", action="store_true",
                   help="actually write zarrs (default: verify on a sample only)")
    p.add_argument("--sample", type=int, default=24)
    p.add_argument("--n-dask-workers", type=int, default=None)
    args = p.parse_args()

    global SIGMA_LAT, SIGMA_LON_BASE, SIGMA_LON_CAP
    SIGMA_LAT = args.sigma_lat
    SIGMA_LON_BASE = args.sigma_lon_base
    SIGMA_LON_CAP = args.sigma_lon_cap
    logging.info(f"Area-aware Gaussian smoother: sigma_lat={SIGMA_LAT} "
                 f"sigma_lon_base={SIGMA_LON_BASE} (cap {SIGMA_LON_CAP})")

    realms = ["ocean", "atmosphere"] if args.realm == "both" else [args.realm]
    for realm in realms:
        process_realm(realm, args)


if __name__ == "__main__":
    main()
