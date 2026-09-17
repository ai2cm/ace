# Time stride on coupled input datasets; 5-daily fractions for the CM4-like-AM4 random-CO2 coupled sea ice ensemble

`CoupledInputDatasetConfig` gains a `time_stride`, so an input store can be
subsampled to a coarser cadence before any coupling arithmetic runs. Putting the
1-degree sea-surface stores in the `ocean` slot at a 5-day stride makes
`compute_coupled_sea_ice`'s existing ocean-sourced branch emit 6-hourly coupled
sea ice whose sea ice and ocean fractions vary every 5 days and stay coherent
with the land fraction. No coupling code changes.

---

## `scripts/data_process/create_coupled_datasets.py` (modified)

```python
@dataclasses.dataclass
class CoupledInputDatasetConfig:
    zarr_path: str
    time_chunk_size: int
    extra_fields: ExtraFieldsConfig = dataclasses.field(default_factory=ExtraFieldsConfig)
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    time_stride: str | None = None  # NEW — e.g. "5D": keep one snapshot per stride

    def get_dataset(self) -> xr.Dataset:  # CHANGED — apply the stride after the time slice
        ...

    def log_info(self, label) -> None:  # CHANGED — log the stride when configured
        ...


def _stride_steps(time: xr.DataArray, time_stride: str) -> int:  # NEW
    """Number of input timesteps spanning time_stride; raises on a non-uniform
    axis or a stride that is not a whole multiple of the input cadence."""
    ...
```

### Critical detail — where the stride applies and where the windows are anchored

```python
# get_dataset, in order:
ds = xr.open_zarr(self.zarr_path, chunks={"time": self.time_chunk_size})
ds = ds.sel(time=slice(self.first_timestamp, self.last_timestamp))   # unchanged
if self.time_stride is not None:                                     # NEW
    ds = ds.isel(time=slice(None, None, _stride_steps(ds.time, self.time_stride)))
```

- **Composition.** The timestamp slice runs first, so the stride is anchored at
  the first timestamp *of the selected range* — `first_timestamp` is the window
  origin when it is set, and the record's first timestamp otherwise.
- **This config sets no `first_timestamp` on the strided input**, so the anchor
  is the record's first timestamp. It has to be: the ocean-sourced branch does
  `fractions.reindex({time: atmos.time}, method="ffill")`, and any atmosphere
  timestamp earlier than the first selected snapshot would forward-fill from
  nothing and come out NaN. The window origin the config being replaced used for
  `window_avg` is later than the record start and cannot be reused here.
- **Trailing remainder.** The record length need not be a whole multiple of the
  stride; the last selected snapshot forward-fills through the end of the
  atmosphere record, so the final interval is shorter than the others rather
  than truncated.
- **`isel`, not `resample`.** The sea-surface fields are snapshots, so the
  subsample selects instants and does not average.
- **Not `CoupledSeaIceConfig.timedelta`.** That field feeds
  `_make_serializable_time_coord`, which asserts the reconstructed coordinate
  matches the *atmosphere* time axis length; a 5-day value there fails the
  assertion. It stays at the output cadence.

### What runs unchanged

`compute_coupled_sea_ice` reaches its ocean-sourced branch when no `sea_ice`
dataset is configured, no `window_avg` is configured, and the `ocean` dataset
carries sea ice fields. The strided sea-surface store satisfies all three, and
carries `sea_surface_fraction`, `sea_ice_fraction`, and `ocean_sea_ice_fraction`
under the names `OceanInputFieldsConfig` already defaults to. So
`_compute_fractions_from_ocean` computes coherent fractions at the strided
cadence and they are forward-filled onto the atmosphere's 6-hourly index.
`compute_coupled_ocean` runs only when `coupled_sea_surface` is configured, so
the `ocean` input alone produces no coupled ocean output.

Under `include_ts` the emitted `surface_temperature` is `_interpolate_sst`'s
blend, `(1 - ofrac) * ts + ofrac * ts_strided`. It is piecewise constant on the
stride interval only where `ocean_fraction` is 1, and stays raw 6-hourly over
land and under full sea ice. That is intended.

## `scripts/data_process/configs/CM4-like-AM4-random-CO2-ensemble-coupled.yaml` (modified)

Rewritten in place; nine ensemble members, one atmosphere / sea-surface store
pair each.

```yaml
version: "2026-08-18"                     # CHANGED
family_name: CM4-like-AM4-random-CO2-coupled
output_directory: gs://vcm-ml-intermediate  # CHANGED — was /climate-default (Weka)

coupled_datasets:
  coupled_sea_ice:
    include_ts: true
    # window_avg:                         # REMOVED — snapshots, not interval means;
    #   window_timedelta: 120h            #   a window_avg also routes off the
    #   first_timestamp: ...              #   ocean-sourced branch entirely
    #   shift_timestamps_to_avg_interval_midpoint: true
  output_writer:
    n_dask_workers: 32                    # unchanged

stats:
  start_date: "0153-01-01T06:00:00"       # unchanged — same spin-up skip as before

input_datasets:
  runs:
    random-CO2-1xCO2-ic_0001:             # × 9: {1,2,4}xCO2 × ic_000{1,2,3}
      atmosphere:
        zarr_path: gs://vcm-ml-intermediate/2026-06-19-CM4-like-AM4-random-CO2/random-CO2-1xCO2-ic_0001.zarr  # CHANGED — was 2025-12-02
        time_chunk_size: 360              # CHANGED — was 500; the store's time shard
        extra_fields:
          names_and_prefixes: ["ak_", "bk_"]
      ocean:                              # NEW — replaces the sea_ice slot
        zarr_path: gs://vcm-ml-intermediate/2026-08-17-cm4-like-am4-random-co2-sea-surface-1deg/random-CO2-1xCO2-ic_0001.zarr
        time_chunk_size: 365              # the sea-surface store's time shard
        time_stride: "5D"
      # sea_ice:                          # REMOVED — the ocean slot is the sea ice source now
  climate_data_type: "CM4"
  stats:
    atmosphere_dir: gs://vcm-ml-intermediate/2026-06-19-CM4-like-AM4-random-CO2-stats/combined  # CHANGED
```

- The two store generations named here cover the same time record, so the
  merged `uncoupled_atmosphere` stats stay comparable to the previous ensemble's.
- `time_chunk_size` is set to each store's own time shard length rather than the
  previous uniform value, so a dask chunk is one shard.
- Outputs land at
  `gs://vcm-ml-intermediate/2026-08-18-CM4-like-AM4-random-CO2-coupled/<run>-sea_ice.zarr`
  with per-run stats under the sibling `-stats/` tree and the ensemble-combined
  stats under its `combined/`.

---

## Tests

## `scripts/data_process/test_create_coupled_datasets.py` (modified)

```python
def test_time_stride_selects_every_nth_instant():
    # GOAL: get_dataset on a synthetic uniform-cadence store returns the instants
    # a whole-multiple stride selects, anchored at the record's first timestamp,
    # with the trailing remainder left unselected rather than truncating.
    # PARAMETERIZE: stride ∈ {one input step (no-op), several steps}.

def test_time_stride_composes_with_timestamp_range():
    # GOAL: with first_timestamp/last_timestamp set, the range is applied first and
    # the stride anchors on the first in-range instant.

def test_time_stride_none_is_todays_behavior():
    # GOAL: regression guard — no time_stride returns the same time axis as before.

def test_time_stride_invalid_raises():
    # GOAL: a clear error, not a silent wrong cadence.
    # PARAMETERIZE: cause ∈ {non-uniform time axis, stride not a whole multiple of
    #                        the input cadence, stride shorter than one step}.
```

## `scripts/data_process/test_coupled_dataset_utils.py` (modified)

```python
def test_ocean_sourced_strided_fractions_are_piecewise_constant():
    # GOAL: end-to-end over a strided ocean input — the emitted land/ocean/sea-ice
    # fractions are constant within each stride interval, change at its boundaries,
    # partition to 1 at every cell and instant, and the output keeps the atmosphere's
    # own time axis.

def test_ocean_sourced_strided_include_ts_blend():
    # GOAL: the emitted surface_temperature is piecewise constant on the stride
    # interval where ocean_fraction == 1 and equals the raw atmosphere field where
    # ocean_fraction == 0.
```

## `scripts/data_process/test_config.py` (unmodified)

`test_valid_create_coupled_datasets_config` already sweeps every `*-coupled.yaml`
through dacite in strict mode, so the rewritten config's schema is covered as
soon as it lands.

---

## Open Questions

- `output_directory` moves from `/climate-default` to `gs://vcm-ml-intermediate`,
  which changes where this shipped config writes for anyone running it against
  Weka. Objections welcome; the run producing these stores is a GCS run.
- `time_stride: "5D"` spells the subsample as a physical span, matching
  `window_timedelta`. An integer count of input timesteps would need no cadence
  inference — is the timedelta spelling worth the validation it requires?
