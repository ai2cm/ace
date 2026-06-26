"""Roundtrip-smooth the sea-ice surface fractions in the *existing* coupled
E3SMv3 zarrs, writing new versioned zarrs for coupled fine-tuning.

WHY
---
The atmosphere fields were remapped online by EAM (ne30pg2 -> shifted gaussian
180x360) with a *first-order conservative* scheme ("blocky" piecewise-constant
reconstruction). ACE's offline pipeline applies an SHT roundtrip to the
prognostic atmosphere variables to de-block them, but ICEFRAC / OCNFRAC are NOT
in that list, so the sea-ice fraction (and the derived ocean_sea_ice_fraction
that Samudra predicts) keep the blocky lon=0 structure. This script applies the
same SHT roundtrip to the sea-ice fractions in the already-processed coupled
zarrs, *post hoc*, so we can do a short coupled fine-tune without re-running the
whole data pipeline.

WHAT IT DOES (per realm)
------------------------
The single physical quantity that is smoothed is the sea-ice CONCENTRATION
    sic = ocean_sea_ice_fraction   (fraction of the *ocean* area that is ice)
Everything else is derived from it so the surface partition stays consistent:
    ssf      = 1 - LANDFRAC                      # sea-surface fraction (sfrac_mod)
    sic_rt   = clip( SHT_roundtrip(sic), 0, 1 )  # de-blocked concentration
    ICEFRAC  = sic_rt        * ssf
    OCNFRAC  = (1 - sic_rt)  * ssf
  =>  ICEFRAC + OCNFRAC + LANDFRAC == 1   (exactly)

This mirrors compute_coupled_sea_ice() in coupled_dataset_utils.py
(ifrac_mod = sic * sfrac_mod ; ofrac_mod = (1 - sic) * sfrac_mod) with
sfrac_mod == 1 - LANDFRAC (== lfrac_mod is what is stored as LANDFRAC).

For the ocean realm we additionally re-apply the sea-ice/iceVolumeTotal
consistency check from fme.core.corrector.ocean.SeaIceFractionConfig:
    iceVolumeTotal := iceVolumeTotal * (sic_rt > 0)     # zero volume where ice-free

NaN handling: the ocean ice fields and the atmosphere ICEFRAC are masked
(NaN where time-mean SST > threshold) and the atmosphere has a few leading
all-NaN steps from the reindex/ffill. We fill NaN with 0 for the (global) SHT,
then restore each output field's ORIGINAL NaN pattern via
`.where(original.notnull())`, so masks and leading-NaN steps are preserved
byte-for-byte.

Roundtrip == forward RealSHT then inverse, on the gaussian grid
(grid='legendre-gauss'), keeping `--fraction` of the spherical-harmonic degrees
(default 1.0, matching the pipeline's roundtrip_fraction_kept). 1.0 is the
validated sweet spot: it removes the blockiness (instantaneous ICEFRAC carries
energy above the grid SH limit, so even full-mode roundtrip smooths it) with the
least Gibbs ringing and least physical edge smearing. Gibbs is clipped to [0,1].

OUTPUTS
-------
    {OUT_DIR}/2026-06-25-E3SMv3-piControl-105yr-coupled-ocean.zarr
    {OUT_DIR}/2026-06-25-E3SMv3-piControl-105yr-coupled-atmosphere.zarr

By default the script runs a VERIFICATION pass on a sample of timesteps and
prints diagnostics WITHOUT writing. Pass --write to actually create the zarrs.

NOTE: this is RETRAINING-data prep. Only do a *coupled fine-tune* with these;
evaluating the old (blocky-trained) weights against smoothed targets would just
raise the measured error.
"""

import argparse
import logging
import os
import time

import numpy as np
import torch
import torch_harmonics as th
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ----------------------------------------------------------------------------
# config / constants
# ----------------------------------------------------------------------------
NLAT, NLON = 180, 360
GRID = "legendre-gauss"  # the data is on a gaussian latitude grid (lat[0]~-89.24)

INPUT_VERSION = "2026-06-02"
OUTPUT_VERSION = "2026-06-25"
FAMILY = "E3SMv3-piControl-105yr-coupled"
DATA_DIR = "/climate-default"

# xpartition chunking, matching scripts/data_process/writer_utils.py
INNER_CHUNKS = {"time": 1, "lon": -1, "lat": -1}
OUTER_CHUNKS = {"time": 360, "lon": -1, "lat": -1}

# fields we touch (others are copied through unchanged)
SIC = "ocean_sea_ice_fraction"
ICEFRAC = "ICEFRAC"
OCNFRAC = "OCNFRAC"
LANDFRAC = "LANDFRAC"
ICEVOL = "iceVolumeTotal"

# Modules are created lazily (once) and shared across dask threads.
_SHT = None
_ISHT = None
_FRACTION = 1.0


def _init_transforms(fraction: float):
    global _SHT, _ISHT, _FRACTION
    _FRACTION = fraction
    _SHT = th.RealSHT(NLAT, NLON, grid=GRID)
    _ISHT = th.InverseRealSHT(NLAT, NLON, grid=GRID)


def _roundtrip_np(arr: np.ndarray) -> np.ndarray:
    """SHT roundtrip on a (..., lat, lon) numpy array. NaNs must be pre-filled.

    Keeps `_FRACTION` of the spherical-harmonic degrees (triangular truncation).
    Returns float32 with the same shape.
    """
    shp = arr.shape
    flat = np.ascontiguousarray(arr.reshape(-1, shp[-2], shp[-1]), dtype=np.float32)
    with torch.no_grad():
        t = torch.from_numpy(flat)
        coeffs = _SHT(t)
        if _FRACTION < 1.0:
            lmax = coeffs.shape[-2]
            lkeep = int(round(_FRACTION * lmax))
            coeffs[..., lkeep:, :] = 0.0
        out = _ISHT(coeffs)
    return out.numpy().reshape(shp).astype(np.float32)


def roundtrip_da(da: xr.DataArray) -> xr.DataArray:
    """Lazily SHT-roundtrip a (time, lat, lon) DataArray (NaNs pre-filled)."""
    return xr.apply_ufunc(
        _roundtrip_np,
        da,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "lon"]],
        dask="parallelized",
        output_dtypes=[np.float32],
    )


# ----------------------------------------------------------------------------
# core: build the smoothed concentration and the consistent fraction set
# ----------------------------------------------------------------------------
def _smoothed_sic(ds: xr.Dataset, realm: str, ssf: xr.DataArray) -> xr.DataArray:
    """Return the de-blocked sea-ice concentration sic_rt in [0, 1] (lazy).

    ocean realm: roundtrip the stored ocean_sea_ice_fraction (the Samudra target).
    atmosphere realm: recover sic = ICEFRAC / ssf, then roundtrip.
    """
    if realm == "ocean":
        sic0 = ds[SIC]
    else:
        # ssf == 1 - LANDFRAC == sfrac_mod; recover concentration from ICEFRAC
        sic0 = (ds[ICEFRAC] / ssf.where(ssf > 0)).clip(0.0, 1.0)
    sic_filled = sic0.fillna(0.0)
    sic_rt = roundtrip_da(sic_filled).clip(0.0, 1.0)
    return sic_rt


def build_modified_dataset(ds: xr.Dataset, realm: str) -> xr.Dataset:
    """Return a copy of `ds` with smoothed, consistent sea-ice fractions (lazy).

    Each output field reuses its ORIGINAL NaN pattern (preserving the sea-ice
    mask and any leading all-NaN steps).
    """
    ssf = (1.0 - ds[LANDFRAC]).clip(0.0, 1.0)  # sea-surface fraction (sfrac_mod)
    sic_rt = _smoothed_sic(ds, realm, ssf)

    # Only modify the fractions INSIDE the sea-ice mask (where ICEFRAC is defined).
    # Outside it, the time-mean-SST mask already drops transient ice (so the
    # original OCNFRAC < sea_surface_fraction there); preserve those originals.
    in_ice_mask = ds[ICEFRAC].notnull()
    icefrac_new = (sic_rt * ssf).where(in_ice_mask)
    ocnfrac_new = xr.where(in_ice_mask, (1.0 - sic_rt) * ssf, ds[OCNFRAC]).where(
        ds[OCNFRAC].notnull()
    )

    updates = {
        ICEFRAC: icefrac_new.astype("float32"),
        OCNFRAC: ocnfrac_new.astype("float32"),
    }
    updates[ICEFRAC].attrs = ds[ICEFRAC].attrs
    updates[OCNFRAC].attrs = ds[OCNFRAC].attrs

    if realm == "ocean":
        sic_new = sic_rt.where(ds[SIC].notnull())
        updates[SIC] = sic_new.astype("float32")
        updates[SIC].attrs = ds[SIC].attrs
        # consistency: zero ice volume where there is no ice (sic_rt == 0),
        # then restore the original mask. Mirrors SeaIceFractionConfig.
        if ICEVOL in ds:
            icevol_new = xr.where(sic_rt > 0.0, ds[ICEVOL], 0.0).where(
                ds[ICEVOL].notnull()
            )
            updates[ICEVOL] = icevol_new.astype("float32")
            updates[ICEVOL].attrs = ds[ICEVOL].attrs

    out = ds.assign(updates)
    out.attrs = dict(ds.attrs)
    out.attrs["history"] = (
        out.attrs.get("history", "")
        + f" | sea-ice fractions SHT-roundtripped (fraction={_FRACTION}) by "
        "roundtrip_sea_ice_fractions.py"
    )
    return out


# ----------------------------------------------------------------------------
# verification (sample of timesteps, eager, no writing)
# ----------------------------------------------------------------------------
def _seam_curvature(field2d: np.ndarray) -> float:
    """Mean |lon 2nd-difference| at the lon=0 seam columns, 60-85N band."""
    band = field2d[150:175, :]  # lat index ~60..85N (lat ascending)
    cv = np.abs(np.roll(band, 1, axis=1) - 2 * band + np.roll(band, -1, axis=1))
    seam = np.nanmean(cv[:, [0, -1]])
    return float(seam)


def verify(realm: str, ds: xr.Dataset, n_samples: int):
    logging.info(f"--- VERIFY {realm}: {n_samples} sampled timesteps ---")
    # pick evenly spaced timesteps with valid (non-all-NaN) ICEFRAC
    nt = ds.sizes["time"]
    idx = np.linspace(0, nt - 1, n_samples).astype(int)
    sub = ds.isel(time=idx).load()
    mod = build_modified_dataset(sub, realm).load()

    # 1. partition sums to 1 INSIDE the ice mask (where ICEFRAC is defined)
    in_mask = ~np.isnan(sub[ICEFRAC].values)
    s = (
        mod[ICEFRAC].values
        + mod[OCNFRAC].values
        + np.broadcast_to(sub[LANDFRAC].values, mod[ICEFRAC].shape)
    )
    max_sum_err = float(np.nanmax(np.abs(s[in_mask] - 1.0))) if in_mask.any() else 0.0

    # 1b. OCNFRAC unchanged OUTSIDE the ice mask (preserve original elsewhere)
    out_mask = (~in_mask) & (~np.isnan(sub[OCNFRAC].values))
    ocn_outside_change = (
        float(np.nanmax(np.abs((mod[OCNFRAC].values - sub[OCNFRAC].values)[out_mask])))
        if out_mask.any()
        else 0.0
    )

    # 2. concentration / fraction bounds + Gibbs
    icef = mod[ICEFRAC].values
    bounds_ok = np.nanmin(icef) >= -1e-6 and np.nanmax(icef) <= 1.0 + 1e-6

    # 3. seam curvature before/after (ICEFRAC), averaged over samples
    cb = np.nanmean([_seam_curvature(x) for x in sub[ICEFRAC].fillna(0).values])
    ca = np.nanmean([_seam_curvature(x) for x in mod[ICEFRAC].fillna(0).values])

    # 4. NaN pattern preserved
    nan_preserved = bool(
        np.array_equal(np.isnan(sub[ICEFRAC].values), np.isnan(mod[ICEFRAC].values))
    )

    logging.info(f"  partition sum-to-1 (in mask) : {max_sum_err:.2e}")
    logging.info(f"  OCNFRAC unchanged out-of-mask: {ocn_outside_change:.2e}")
    logging.info(f"  ICEFRAC within [0,1]         : {bounds_ok}")
    logging.info(f"  ICEFRAC NaN pattern preserved: {nan_preserved}")
    logging.info(f"  seam curvature ICEFRAC       : {cb:.4f} -> {ca:.4f} "
                 f"({100*(ca-cb)/cb:+.0f}%)")

    if realm == "ocean" and ICEVOL in mod:
        sic = mod[SIC].values
        vol = mod[ICEVOL].values
        # no ice volume where ice-free (within mask)
        bad = np.nansum((vol > 0) & (sic == 0))
        logging.info(f"  iceVolumeTotal>0 where sic==0 : {int(bad)} cells (want 0)")
        vol_nan_ok = np.array_equal(
            np.isnan(sub[ICEVOL].values), np.isnan(mod[ICEVOL].values)
        )
        logging.info(f"  iceVolumeTotal NaN preserved : {vol_nan_ok}")


# ----------------------------------------------------------------------------
# writing (xpartition, mirroring writer_utils.OutputWriterConfig.write)
# ----------------------------------------------------------------------------
def write_zarr(ds: xr.Dataset, output_store: str, n_dask_workers: int | None):
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
        n_split = max(1, -(-nt // OUTER_CHUNKS["time"]))  # ceil
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


# ----------------------------------------------------------------------------
def process_realm(realm: str, args):
    in_path = os.path.join(
        DATA_DIR, f"{INPUT_VERSION}-{FAMILY}-{realm}.zarr"
    )
    out_path = os.path.join(
        args.out_dir, f"{OUTPUT_VERSION}-{FAMILY}-{realm}.zarr"
    )
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

    if not args.write:
        verify(realm, ds, args.sample)
        logging.info("  (verification only; pass --write to create the zarr)")
        return

    mod = build_modified_dataset(ds, realm)
    write_zarr(mod, out_path, args.n_dask_workers)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--realm", choices=["ocean", "atmosphere", "both"], default="both")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="fraction of SH degrees kept (1.0 = full roundtrip)")
    p.add_argument("--out-dir", default=DATA_DIR,
                   help="output directory for the new zarrs")
    p.add_argument("--write", action="store_true",
                   help="actually write zarrs (default: verify on a sample only)")
    p.add_argument("--sample", type=int, default=24,
                   help="# timesteps for the verification pass")
    p.add_argument("--n-dask-workers", type=int, default=None,
                   help="optional dask distributed workers for writing")
    args = p.parse_args()

    _init_transforms(args.fraction)
    logging.info(f"SHT roundtrip: grid={GRID} nlat={NLAT} nlon={NLON} "
                 f"fraction={args.fraction} | torch CUDA={torch.cuda.is_available()}")

    realms = ["ocean", "atmosphere"] if args.realm == "both" else [args.realm]
    for realm in realms:
        process_realm(realm, args)


if __name__ == "__main__":
    main()
