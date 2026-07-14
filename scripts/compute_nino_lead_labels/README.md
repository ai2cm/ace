# compute_nino_lead_labels

Generates a "next N-month Nino3.4" label dataset aligned to a host ocean
dataset (UFS replay ocean, CM4, etc.), for use as a direct multi-lead forecast
target for an auxiliary head on Samudra.

## What it produces

A separate zarr whose `time`, `lat`, and `lon` coordinates exactly match the
host's, containing variables `nino34_lead_01 .. nino34_lead_NN`. For a host
time `t`, `nino34_lead_k(t)` is the Nino3.4 index for the calendar month `k`
months after `t` (leads +1..+N by default).

Each lead is stored as a `(time, lat, lon)` field that is **constant across
space**. This is required because ACE detects the grid from a variable with
`ndim >= 3` and merges assert identical lat/lon coordinates; the constant
fields compress to almost nothing in zarr. At load time these behave like the
2D-broadcast scalar forcings (e.g. `global_mean_co2`).

## Definition (matches FME's ENSO aggregator)

Replicates `fme/ace/aggregator/inference/{enso/dynamic_index,utils}.py`:

- Box: lat (-5, 5), lon (190, 240) in the 0-360 convention.
- Area-weighted (cos-lat) mean SST over the box. (FME's `regional_area_weighted_mean`
  applies cos-lat twice; over the +-5 deg box this differs from single cos-lat by
  <0.1%, i.e. within rounding.)
- Native-cadence anomalies from a per-calendar-month climatology, then a monthly
  mean, then a trailing 5-month running mean (`--running-mean-months`, default 5;
  use 1 for the raw monthly anomaly).
- Climatology is computed from the host's own SST over a configurable reference
  period (UFS and CM4 each get their own base state). FME's dynamic index applies
  no detrending, so for a forced run like CM4 1pctCO2 the raw index carries the
  forced warming trend. Two removal options (both match variants in
  `scripts/compute_enso_index/compute_enso_index.py`):
  - `--linear-detrend` (the `cm4_nino_labels` target uses this): subtract a
    least-squares linear trend from the box-mean series. Removes the forced trend
    while **preserving full ENSO event amplitude** -- recommended for CM4.
  - `--relative-to-tropical`: subtract the tropical-mean SST (5S-5N, all
    longitudes). Removes the trend but also each event's basin-wide warming, so
    it damps event amplitude.

  There is no trailing 30-year climatology anywhere in ACE.
- Month arithmetic uses `ym = year*12 + (month-1)`, so it is calendar-agnostic
  (works for UFS real times and CM4 model calendars such as noleap).
- The trailing running mean leaves the first `running_mean_months - 1` months of
  the record with no index; leads landing there, or within N months of the
  record end, are written as NaN. The consuming auxiliary loss must mask NaN
  targets.

## Environment

```bash
conda activate fme
```

## Usage

```bash
make ufs_nino_labels    # UFS replay ocean host
make cm4_nino_labels     # CM4 1pctCO2 ocean host
```

Override paths/params on the command line, e.g.:

```bash
make ufs_nino_labels HOST=gs://bucket/host.zarr OUTPUT=gs://bucket/out.zarr N_LEADS=12
```

Or call the script directly (see `--help`):

```bash
python compute_nino_lead_labels.py \
  --host-dataset /path/host.zarr \
  --output-zarr /path/host-nino-leads.zarr \
  --n-leads 12 --first-lead 1 \
  --clim-start 1994-01 --clim-stop 2015-12 \
  --debug   # print instead of writing
```

## Merging into training

Add the label dataset as an extra source in the training config's `merge` list
alongside its host (same `time` coordinate, disjoint variable names), then list
`nino34_lead_01 .. nino34_lead_12` as targets for the auxiliary head.
