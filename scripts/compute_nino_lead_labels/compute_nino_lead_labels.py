"""Compute a "next N-month Nino3.4" label dataset aligned to a host dataset.

Given a host ocean dataset with a sea-surface-temperature field (e.g. the UFS
replay ocean or a CM4 run), this produces a *separate* label dataset whose
``time`` (and lat/lon) coordinates exactly match the host's, containing
variables ``nino34_lead_01 .. nino34_lead_NN``. For a host time ``t``,
``nino34_lead_k(t)`` is the Nino3.4 index for the calendar month ``k`` months
after ``t`` (leads +1..+N months by default). The index is computed to match
FME's ENSO aggregator: area-weighted box-mean SST -> monthly-climatology
anomaly -> trailing 5-month running mean (configurable via
``--running-mean-months``; use 1 for the raw monthly anomaly).

The label dataset is intended to be merged with its host via ACE's
``MergeDatasetConfig`` (same time coordinate, disjoint variable names) and used
as a direct multi-lead forecast target for an auxiliary head.

Design notes
------------
* ACE merges require the label dataset to share the host's timestep and
  *identical* lat/lon coordinates, and grid detection scans ``data_vars`` for a
  variable with ``ndim >= 3``. A scalar-per-time variable therefore cannot
  stand alone (unlike ``global_mean_co2``, which lives inside a full dataset),
  so each lead is materialized as a ``(time, lat, lon)`` field that is
  *constant across space*. These constant fields compress to almost nothing in
  zarr, matching the "scalars are treated as 2D images" convention at load
  time.
* Month arithmetic uses an integer ``ym = year * 12 + (month - 1)`` key so it is
  independent of the calendar (works for both proleptic_gregorian UFS times and
  a model calendar such as noleap used by CM4).
* Anomalies are relative to a monthly climatology computed from the host's own
  SST over a (configurable) reference period, so UFS and CM4 each get their own
  base state. FME's dynamic index applies no detrending, so for a forced run
  like CM4 1pctCO2 the raw index carries the forced warming trend. Two removal
  options (see ``compute_nino_lead_labels``): ``--linear-detrend`` (subtract a
  linear trend; preserves full ENSO amplitude; recommended for CM4) and
  ``--relative-to-tropical`` (subtract tropical-mean SST; also damps event
  amplitude). Both match variants in
  ``scripts/compute_enso_index/compute_enso_index.py``.
* The trailing running mean leaves the first ``running_mean_months - 1`` months
  of the record with no index; leads landing there (or within ``N`` months of
  the record end) are written as NaN. The consuming auxiliary loss must mask
  NaN targets.

The Nino3.4 box (lat (-5, 5), lon (190, 240)) matches
``fme/ace/aggregator/inference/enso/dynamic_index.py``.
"""

import argparse
import logging

import numpy as np
import xarray as xr

NINO34_LAT_BOUNDS = (-5.0, 5.0)
NINO34_LON_BOUNDS = (190.0, 240.0)  # degrees east, 0-360 convention

# Tropical band used for the "relative Nino3.4" trend removal (subtract the
# tropical-mean SST from the box mean), matching
# ``scripts/compute_enso_index/compute_enso_index.py``.
TROPICAL_LAT_BOUNDS = (-5.0, 5.0)
TROPICAL_LON_BOUNDS = (0.0, 360.0)


def advance_year_month(year: int, month: int, k: int) -> tuple[int, int]:
    """Return the (year, month) that is ``k`` calendar months after (year, month).

    ``month`` is 1-indexed. ``k`` may be negative.
    """
    total = year * 12 + (month - 1) + k
    return total // 12, total % 12 + 1


def nino_box_weighted_mean(
    sst: xr.DataArray,
    lat_dim: str,
    lon_dim: str,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
) -> xr.DataArray:
    """Area-weighted mean SST over the Nino box, returned as a 1D time series.

    ``lon_bounds`` are interpreted in the 0-360 convention; the host longitude
    coordinate is wrapped with ``% 360`` before masking so both 0-360 and
    -180-180 grids (and boxes that wrap the antimeridian) are handled.
    """
    lat = sst[lat_dim]
    lon = sst[lon_dim]
    lon360 = lon % 360

    lat_in = lat.where((lat >= lat_bounds[0]) & (lat <= lat_bounds[1]), drop=True)
    if lon_bounds[0] <= lon_bounds[1]:
        lon_mask = (lon360 >= lon_bounds[0]) & (lon360 <= lon_bounds[1])
    else:  # box wraps the antimeridian, e.g. (350, 20)
        lon_mask = (lon360 >= lon_bounds[0]) | (lon360 <= lon_bounds[1])
    lon_in = lon.where(lon_mask, drop=True)

    if lat_in.size == 0 or lon_in.size == 0:
        raise ValueError(
            "No grid cells fall within the Nino box "
            f"lat={lat_bounds}, lon={lon_bounds}. Check the lat/lon dim names "
            "and coordinate ranges."
        )

    sst_box = sst.sel({lat_dim: lat_in, lon_dim: lon_in})
    weights = np.cos(np.deg2rad(sst_box[lat_dim]))
    return sst_box.weighted(weights).mean(dim=[lat_dim, lon_dim], skipna=True)


def subtract_linear_trend(series: xr.DataArray, time_dim: str) -> xr.DataArray:
    """Subtract a least-squares linear trend (fit over the time index).

    Removes only a slow forced trend (e.g. CM4 1pctCO2) while preserving the
    full amplitude of ENSO events, unlike the tropical-mean subtraction which
    also removes each event's basin-wide warming component. Matches
    ``get_time_trendline`` in ``scripts/compute_enso_index/compute_enso_index.py``
    (fit over the integer time index, not the calendar values). NaNs are ignored
    in the fit and preserved in the output.
    """
    values = np.asarray(series.values, dtype=np.float64)
    x = np.arange(values.shape[0], dtype=np.float64)
    finite = np.isfinite(values)
    if int(finite.sum()) < 2:
        return series
    slope, intercept = np.polyfit(x[finite], values[finite], deg=1)
    trend = slope * x + intercept
    return series.copy(data=values - trend)


def fme_monthly_index_lookup(
    box_mean: xr.DataArray,
    time_dim: str,
    n_running_months: int = 5,
    clim_start_year_month: tuple[int, int] | None = None,
    clim_stop_year_month: tuple[int, int] | None = None,
) -> dict[int, float]:
    """Map ``ym = year*12 + (month-1)`` -> Nino3.4 index [K], matching FME.

    Replicates the FME aggregator's dynamic ENSO index
    (``fme/ace/aggregator/inference/utils.py``):

    1. Native-cadence anomalies from a per-calendar-month climatology
       (``anomalies_from_monthly_climo``), computed over the reference period
       (full record if bounds are ``None``).
    2. Monthly-average those anomalies, then take a trailing ``n_running_months``
       running mean (``running_monthly_mean`` with ``n_months=n_running_months``,
       default 5). The first ``n_running_months - 1`` months have no value and
       are omitted (looked up as NaN downstream).

    Set ``n_running_months=1`` to get the raw monthly anomaly with no smoothing.
    """
    values = np.asarray(box_mean.values, dtype=np.float64)
    years = np.asarray(box_mean[f"{time_dim}.year"].values, dtype=np.int64)
    months = np.asarray(box_mean[f"{time_dim}.month"].values, dtype=np.int64)
    ym = years * 12 + (months - 1)

    clim_lo = (
        clim_start_year_month[0] * 12 + (clim_start_year_month[1] - 1)
        if clim_start_year_month is not None
        else None
    )
    clim_hi = (
        clim_stop_year_month[0] * 12 + (clim_stop_year_month[1] - 1)
        if clim_stop_year_month is not None
        else None
    )
    in_ref = np.ones_like(ym, dtype=bool)
    if clim_lo is not None:
        in_ref &= ym >= clim_lo
    if clim_hi is not None:
        in_ref &= ym <= clim_hi

    # (1) Per-calendar-month climatology over the reference period, at native
    # cadence, then native anomalies.
    climatology: dict[int, float] = {}
    for month in range(1, 13):
        sel = values[(months == month) & in_ref]
        sel = sel[~np.isnan(sel)]
        if sel.size > 0:
            climatology[month] = float(sel.mean())
    climo_per_sample = np.array(
        [climatology.get(int(m), np.nan) for m in months], dtype=np.float64
    )
    anom_native = values - climo_per_sample

    # (2a) Monthly-average the native anomalies over sorted unique months.
    unique_ym = np.array(sorted(np.unique(ym).tolist()), dtype=np.int64)
    monthly = np.full(unique_ym.shape, np.nan, dtype=np.float64)
    for i, key in enumerate(unique_ym):
        sel = anom_native[ym == key]
        sel = sel[~np.isnan(sel)]
        if sel.size > 0:
            monthly[i] = float(sel.mean())

    # (2b) Trailing n-month running mean (matches FME's running_monthly_mean).
    index: dict[int, float] = {}
    for i in range(len(unique_ym)):
        if i >= n_running_months - 1:
            window = monthly[i - n_running_months + 1 : i + 1]
            if np.any(~np.isnan(window)):
                index[int(unique_ym[i])] = float(np.nanmean(window))
    return index


def build_lead_values(
    host_years: np.ndarray,
    host_months: np.ndarray,
    anomaly: dict[int, float],
    n_leads: int,
    first_lead: int = 1,
) -> np.ndarray:
    """Build the (n_time, n_leads) array of lead anomalies for the host times.

    ``lead index j`` (0-based) corresponds to ``first_lead + j`` months after
    the host time. Missing target months are filled with NaN.
    """
    n_time = host_years.shape[0]
    out = np.full((n_time, n_leads), np.nan, dtype=np.float64)
    for i in range(n_time):
        base = int(host_years[i]) * 12 + (int(host_months[i]) - 1)
        for j in range(n_leads):
            key = base + first_lead + j
            if key in anomaly:
                out[i, j] = anomaly[key]
    return out


def broadcast_to_grid(
    lead_values: np.ndarray,
    time: xr.DataArray,
    lat: xr.DataArray,
    lon: xr.DataArray,
    lat_dim: str,
    lon_dim: str,
    n_leads: int,
    first_lead: int = 1,
) -> xr.Dataset:
    """Materialize each lead as a (time, lat, lon) field constant across space."""
    nlat = lat.sizes[lat_dim]
    nlon = lon.sizes[lon_dim]
    ones = np.ones((nlat, nlon), dtype=np.float32)
    data_vars = {}
    for j in range(n_leads):
        lead = first_lead + j
        # (time, 1, 1) * (lat, lon) -> (time, lat, lon), constant in space.
        field = lead_values[:, j].astype(np.float32)[:, None, None] * ones[None, :, :]
        data_vars[f"nino34_lead_{lead:02d}"] = xr.DataArray(
            field,
            dims=["time", lat_dim, lon_dim],
            attrs={
                "long_name": (
                    f"Nino3.4 SST monthly anomaly, +{lead} month lead "
                    "(constant across space)"
                ),
                "units": "K",
                "lead_months": lead,
            },
        )
    coords = {"time": time, lat_dim: lat, lon_dim: lon}
    return xr.Dataset(data_vars, coords=coords)


def compute_nino_lead_labels(
    ds: xr.Dataset,
    sst_var: str,
    lat_dim: str,
    lon_dim: str,
    time_dim: str,
    n_leads: int,
    first_lead: int,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    clim_start_year_month: tuple[int, int] | None,
    clim_stop_year_month: tuple[int, int] | None,
    n_running_months: int = 5,
    relative_to_tropical: bool = False,
    linear_detrend: bool = False,
) -> xr.Dataset:
    """End-to-end: host dataset -> Nino3.4 lead-label dataset.

    Two mutually-compatible ways to remove a forced warming trend (e.g. CM4
    1pctCO2) before computing anomalies:

    * ``relative_to_tropical``: subtract the tropical-mean SST (5S-5N, all
      longitudes) from the box mean, as in
      ``scripts/compute_enso_index/compute_enso_index.py``. Removes the trend
      *and* each event's basin-wide warming, so it damps event amplitude.
    * ``linear_detrend``: subtract a least-squares linear trend from the box
      mean. Removes only the slow forced trend and preserves full ENSO
      amplitude (recommended for a forced run when amplitude matters).

    Both may be enabled together (matches ``compute_enso_index.py --detrend``).
    """
    box_mean = nino_box_weighted_mean(
        ds[sst_var], lat_dim, lon_dim, lat_bounds, lon_bounds
    ).compute()
    if relative_to_tropical:
        tropical_mean = nino_box_weighted_mean(
            ds[sst_var], lat_dim, lon_dim, TROPICAL_LAT_BOUNDS, TROPICAL_LON_BOUNDS
        ).compute()
        box_mean = box_mean - tropical_mean
    if linear_detrend:
        box_mean = subtract_linear_trend(box_mean, time_dim)
    index = fme_monthly_index_lookup(
        box_mean,
        time_dim,
        n_running_months=n_running_months,
        clim_start_year_month=clim_start_year_month,
        clim_stop_year_month=clim_stop_year_month,
    )
    host_years = np.asarray(ds[f"{time_dim}.year"].values, dtype=np.int64)
    host_months = np.asarray(ds[f"{time_dim}.month"].values, dtype=np.int64)
    lead_values = build_lead_values(host_years, host_months, index, n_leads, first_lead)
    out = broadcast_to_grid(
        lead_values,
        time=ds[time_dim],
        lat=ds[lat_dim],
        lon=ds[lon_dim],
        lat_dim=lat_dim,
        lon_dim=lon_dim,
        n_leads=n_leads,
        first_lead=first_lead,
    )
    # Preserve the host time encoding (units/calendar) so the written zarr's
    # time coordinate round-trips to values identical to the host's.
    out["time"].encoding = dict(ds[time_dim].encoding)
    out.attrs.update(
        {
            "description": (
                "Next-N-month Nino3.4 index labels (FME-consistent: box "
                "area-weighted mean SST -> monthly-climatology anomaly -> "
                f"{n_running_months}-month trailing running mean), aligned to "
                "the host time coordinate. Each nino34_lead_kk field is constant "
                "across lat/lon."
            ),
            "source_sst_var": sst_var,
            "nino_lat_bounds": list(lat_bounds),
            "nino_lon_bounds": list(lon_bounds),
            "n_leads": n_leads,
            "first_lead": first_lead,
            "running_mean_months": n_running_months,
            "relative_to_tropical": int(relative_to_tropical),
            "linear_detrend": int(linear_detrend),
        }
    )
    return out


def _parse_year_month(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    parts = value.split("-")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            f"Expected YYYY-MM for climatology bound, got '{value}'."
        )
    return int(parts[0]), int(parts[1])


def _open_dataset(path: str) -> xr.Dataset:
    if path.endswith(".zarr") or ".zarr/" in path:
        return xr.open_zarr(path, use_cftime=True)
    return xr.open_dataset(path, use_cftime=True)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-dataset", required=True, help="Host zarr/netCDF path.")
    parser.add_argument("--output-zarr", required=True, help="Output zarr path.")
    parser.add_argument("--sst-var", default="sst")
    parser.add_argument("--lat-dim", default="lat")
    parser.add_argument("--lon-dim", default="lon")
    parser.add_argument("--time-dim", default="time")
    parser.add_argument("--n-leads", type=int, default=12)
    parser.add_argument(
        "--first-lead",
        type=int,
        default=1,
        help="Lead in months of the first output channel (1 = next month).",
    )
    parser.add_argument(
        "--clim-start", default=None, help="Climatology reference start, YYYY-MM."
    )
    parser.add_argument(
        "--clim-stop", default=None, help="Climatology reference stop, YYYY-MM."
    )
    parser.add_argument(
        "--running-mean-months",
        type=int,
        default=5,
        help=(
            "Trailing running-mean window in months for the Nino3.4 index "
            "(FME default is 5; use 1 for raw monthly anomaly)."
        ),
    )
    parser.add_argument(
        "--relative-to-tropical",
        action="store_true",
        help=(
            "Subtract the tropical-mean SST (5S-5N, all longitudes) from the box "
            "mean before computing anomalies. Removes the forced warming trend "
            "but also damps event amplitude. Matches compute_enso_index.py."
        ),
    )
    parser.add_argument(
        "--linear-detrend",
        action="store_true",
        help=(
            "Subtract a least-squares linear trend from the box-mean series "
            "before computing anomalies. Removes the forced warming trend while "
            "preserving full ENSO amplitude (recommended for CM4 1pctCO2)."
        ),
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=365,
        help="Chunk size along time for the output zarr.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the resulting dataset instead of writing it.",
    )
    args = parser.parse_args()

    ds = _open_dataset(args.host_dataset)
    if args.sst_var not in ds:
        raise ValueError(
            f"SST variable '{args.sst_var}' not found in host dataset "
            f"(available: {list(ds.data_vars)})."
        )

    out = compute_nino_lead_labels(
        ds,
        sst_var=args.sst_var,
        lat_dim=args.lat_dim,
        lon_dim=args.lon_dim,
        time_dim=args.time_dim,
        n_leads=args.n_leads,
        first_lead=args.first_lead,
        lat_bounds=NINO34_LAT_BOUNDS,
        lon_bounds=NINO34_LON_BOUNDS,
        clim_start_year_month=_parse_year_month(args.clim_start),
        clim_stop_year_month=_parse_year_month(args.clim_stop),
        n_running_months=args.running_mean_months,
        relative_to_tropical=args.relative_to_tropical,
        linear_detrend=args.linear_detrend,
    )

    n_valid = int(
        np.isfinite(
            out["nino34_lead_01"].isel({args.lat_dim: 0, args.lon_dim: 0}).values
        ).sum()
    )
    logging.info(
        f"Computed {args.n_leads} leads for {out.sizes['time']} host times "
        f"({n_valid} with a valid +{args.first_lead}-month target)."
    )

    if args.debug:
        with xr.set_options(display_max_rows=50):
            print(out)
        return

    chunks = {"time": min(args.time_chunk, out.sizes["time"])}
    out.chunk(chunks).to_zarr(args.output_zarr, mode="w", consolidated=True)
    logging.info(f"Wrote {args.output_zarr}")


if __name__ == "__main__":
    main()
