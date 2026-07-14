"""Add Nino3.4 lead-label normalization stats to an existing stats dataset.

Starting from a fetched stats dataset (e.g. the beaker coupled-stats dataset,
which has ``ocean/``, ``coupled_atmosphere/``, ``uncoupled_atmosphere/``
subdirectories each containing ``centering.nc``, ``scaling-full-field.nc``,
``scaling-residual.nc`` and ``time-mean.nc``), this writes a *new* stats dataset
with the ``nino34_lead_*`` variables added to the chosen realm's stats files
(``ocean`` by default, since the leads are Samudra ocean outputs). Other realms
are copied unchanged.

Stats are computed to match ``scripts/data_process/get_stats.py``:
    centering          = mean over (time, lat, lon)
    scaling-full-field = std over (time, lat, lon)
    scaling-residual   = std of the time-difference over (time, lat, lon)
    time-mean          = mean over time (a 2D field)
The lead fields are constant across space, so the reductions collapse to the
per-time series; NaNs (running-mean warmup / record tail) are skipped, matching
xarray's default ``skipna=True`` used by ``get_stats.py``.
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

STAT_FILES = (
    "centering.nc",
    "scaling-full-field.nc",
    "scaling-residual.nc",
    "time-mean.nc",
)


def compute_lead_stats(
    nino: xr.Dataset,
    time_dim: str = "time",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> dict[str, dict[str, float]]:
    """Return ``{lead_name: {"mean", "std", "residual_std"}}``.

    The lead fields are constant across space, so a single grid point gives the
    same reduction as the full ``(time, lat, lon)`` reduction; we use a point
    series for speed. Population std (ddof=0) matches xarray / ``get_stats.py``.
    """
    names = sorted(v for v in nino.data_vars if str(v).startswith("nino34_lead_"))
    if not names:
        raise ValueError("No nino34_lead_* variables found in the nino dataset.")
    stats: dict[str, dict[str, float]] = {}
    for name in names:
        series = nino[name].isel({lat_dim: 0, lon_dim: 0})
        stats[name] = {
            "mean": float(series.mean(skipna=True)),
            "std": float(series.std(skipna=True)),
            "residual_std": float(series.diff(time_dim).std(skipna=True)),
        }
    return stats


def _add_scalar_vars(path: Path, values: dict[str, float], units: str) -> None:
    ds = xr.open_dataset(path).load()
    ds.close()
    for name, value in values.items():
        ds[name] = xr.DataArray(np.float32(value), attrs={"units": units})
    ds.to_netcdf(path)


def _add_time_mean_fields(path: Path, means: dict[str, float]) -> None:
    ds = xr.open_dataset(path).load()
    ds.close()
    # Use an existing 2D variable to get the horizontal dims/coords.
    template = next(v for v in ds.data_vars.values() if v.ndim == 2)
    dims = template.dims
    shape = template.shape
    coords = {d: ds[d] for d in dims if d in ds.coords}
    for name, mean_value in means.items():
        field = np.full(shape, np.float32(mean_value), dtype=np.float32)
        ds[name] = xr.DataArray(field, dims=dims, coords=coords, attrs={"units": "K"})
    ds.to_netcdf(path)


def augment_stats(
    src_dir: Path,
    nino: xr.Dataset,
    out_dir: Path,
    realm: str,
    time_dim: str = "time",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> dict[str, dict[str, float]]:
    """Copy ``src_dir`` to ``out_dir`` and add nino stats to ``out_dir/realm``."""
    stats = compute_lead_stats(nino, time_dim, lat_dim, lon_dim)
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    shutil.copytree(src_dir, out_dir)

    realm_dir = out_dir / realm
    if not realm_dir.is_dir():
        raise ValueError(
            f"Realm subdirectory '{realm}' not found in {src_dir} "
            f"(found: {[p.name for p in src_dir.iterdir() if p.is_dir()]})."
        )

    _add_scalar_vars(
        realm_dir / "centering.nc", {n: s["mean"] for n, s in stats.items()}, "K"
    )
    _add_scalar_vars(
        realm_dir / "scaling-full-field.nc",
        {n: s["std"] for n, s in stats.items()},
        "K",
    )
    _add_scalar_vars(
        realm_dir / "scaling-residual.nc",
        {n: s["residual_std"] for n, s in stats.items()},
        "K",
    )
    _add_time_mean_fields(
        realm_dir / "time-mean.nc", {n: s["mean"] for n, s in stats.items()}
    )
    return stats


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stats-dir", required=True, help="Fetched source stats directory."
    )
    parser.add_argument(
        "--nino-dataset", required=True, help="Nino lead-label zarr/netCDF path."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output stats directory (must not exist)."
    )
    parser.add_argument(
        "--realm",
        default="ocean",
        help="Subdirectory whose stats files receive the nino variables.",
    )
    parser.add_argument("--time-dim", default="time")
    parser.add_argument("--lat-dim", default="lat")
    parser.add_argument("--lon-dim", default="lon")
    args = parser.parse_args()

    nino_path = args.nino_dataset
    if nino_path.endswith(".zarr") or ".zarr/" in nino_path:
        nino = xr.open_zarr(nino_path)
    else:
        nino = xr.open_dataset(nino_path)

    stats = augment_stats(
        src_dir=Path(args.stats_dir),
        nino=nino,
        out_dir=Path(args.output_dir),
        realm=args.realm,
        time_dim=args.time_dim,
        lat_dim=args.lat_dim,
        lon_dim=args.lon_dim,
    )
    logging.info(f"Added {len(stats)} nino variables to {args.output_dir}/{args.realm}")
    for name, s in stats.items():
        logging.info(
            f"  {name}: mean={s['mean']:.4f} std={s['std']:.4f} "
            f"residual_std={s['residual_std']:.4f}"
        )


if __name__ == "__main__":
    main()
