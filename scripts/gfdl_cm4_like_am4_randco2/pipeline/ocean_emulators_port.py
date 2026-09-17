"""Wetmask-normalized conservative regridding, ported from the ai2cm fork of
m2lines/ocean_emulators (github.com/ai2cm/ocean_emulators), so this pipeline
has no dependency on that repository.

Deliberate departures from the ported code (and from the conventions of the
datasets produced by scripts/data_process/compute_ocean_dataset.py, which
called it):

- Wetmask normalization of the conservative regrid is explicit
  (mask -> regrid field and mask -> divide) rather than xESMF's
  ``skipna=True, na_thres=1`` weight-application option, which is unavailable
  when weights are precomputed. The two are equivalent at
  ``OCEAN_FRACTION_THRESHOLD = 0``: a target cell stays ocean if it overlaps
  any sliver of ocean source area.
- The regridded ocean-fraction field is returned so it can be stored;
  compute_ocean_dataset.py computed it but never exposed it for weighting
  coastal cells.
"""

import xarray as xr

# A regridded target cell is kept (treated as ocean) where its ocean
# fraction exceeds this threshold. Zero reproduces the mask convention of
# xESMF ``skipna=True, na_thres=1`` used for earlier ocean datasets: any
# overlap with ocean source area keeps the cell.
OCEAN_FRACTION_THRESHOLD = 0.0


def regrid_normalized(
    ds: xr.Dataset, regridder, wetmask: xr.DataArray
) -> tuple[xr.Dataset, xr.DataArray]:
    """Wetmask-normalized conservative regrid of every variable in ``ds``.

    Fields are zeroed over land, regridded with the raw conservative
    ``regridder``, and divided by the regridded ``wetmask`` (the target-cell
    ocean fraction), so each target value is an average over ocean source
    area only. Target cells with ocean fraction <= OCEAN_FRACTION_THRESHOLD
    are NaN.

    ``wetmask`` is the 2D tracer-cell ocean mask, broadcastable against each
    variable. Returns the regridded dataset and the regridded ocean fraction.
    """
    ocean_fraction = regridder(wetmask.astype("float64"), keep_attrs=False).fillna(0.0)
    divisor = ocean_fraction.where(ocean_fraction > OCEAN_FRACTION_THRESHOLD)
    out = xr.Dataset(attrs=ds.attrs)
    for name, da in ds.data_vars.items():
        masked = da.where(wetmask).fillna(0.0)
        out[name] = (regridder(masked, keep_attrs=True) / divisor).astype(da.dtype)
        out[name].attrs = da.attrs
    ocean_fraction.attrs = {
        "long_name": "fraction of target cell overlapping ocean source cells",
        "units": "0-1",
    }
    return out, ocean_fraction
