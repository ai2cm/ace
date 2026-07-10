# TC radial-structure analysis — status & handoff

Running notes for the radial-based structural verification of downscaled (25 km)
tropical-cyclone forecasts. Companion spec: `eval.md`. Branch:
`scripts/tempest-extreme-3h-tracks`.

## Goal

Evaluate ensembles of *generated* storm snapshots against a target by mapping
storm-centered 2D fields (10 m wind, SLP, precip) into 1D azimuthal-radial
profiles, then scoring structural/dynamical errors (radius of max wind, wind
radii, precip morphology, gradient-wind balance, azimuthal symmetry) — see
`eval.md` for the full metric definitions. This sidesteps the double-penalty of
grid-to-grid RMSE.

## Deliverables in this directory

- `tc_radial_metrics.py` — the metrics library (numpy/xarray, no torch).
  Standardized inputs: `gen` = `[time,] ensemble, lat, lon`; `target` =
  `[time,] lat, lon`. Functions vectorize over leading dims; **time is looped
  externally** because the storm center moves each step.
- `test_tc_radial_metrics.py` — one verified unit test per calculation
  (checked against analytic / brute-force references, not just "runs").
- `tc_radial_eval_demo.py` — end-to-end demo on real storms from the rectified
  25 km tracks; fabricates a synthetic ensemble and writes per-storm figures plus
  a **single compressed-NetCDF statistics summary** (`tc_radial_stats.nc`) to
  `scratch/tc25_eval_demo/`. The NetCDF holds all profiles, ensemble mean/std,
  CRPS(r)/RMSE(r), ring counts and the per-(storm, member) scorecard, with
  provenance in the global attrs (`command`, `source_zarr`, `tracks_source`,
  `crps_estimator`, `rmse_definition`, params). Dims: `storm`, `radius`, `member`.

## How to run (this VM)

```bash
# tests
~/miniconda3/bin/conda run -n fme python -m pytest \
  scripts/tropical_cyclones/test_tc_radial_metrics.py -q
# demo
~/miniconda3/bin/conda run -n fme python \
  scripts/tropical_cyclones/tc_radial_eval_demo.py
```

## Gotchas

- **conda is flaky on this GCP VM (for now).** `conda`/`python` are not on the
  non-login shell PATH, so the plain `conda run -n fme ...` fails with
  `command not found`. Use the **full path** `~/miniconda3/bin/conda run -n fme
  python ...`. Confirmed deps in env `fme`: scipy 1.17.1, xarray, gcsfs,
  matplotlib. On other machines the plain `conda run -n fme` should be fine —
  this is a VM-specific quirk.
- **The 3h zarrs store the whole globe as one chunk per timestep** (zarr v3,
  sharded 128 timesteps). Spatial windowing saves **zero** I/O — any access
  fetches the global chunk (~100–160 MB/s). Scope work to a handful of timesteps;
  don't loop all 45k track points.
- **`PRMSL` is stored in millibar** in these stores — multiply by 100 to get Pa
  before computing PGF / wind-pressure imbalance.
- **Patch = 16° box (~1600 km wide), defined in DEGREES** (±8° lat/lon around the
  center), matching a downscaling-model output tile. Half-width ~800 km, so cap
  the radial range at **r_max ≈ 500–600 km** to keep every ring well-sampled
  azimuthally.
- **Inner radial bins have few azimuthal samples** → noisier means/variances.
  The per-bin `count` array surfaces this; expect larger error at small r.

## Data

- 25 km 3h zarr: `gs://vcm-ml-scratch/andrep/2025-08-12-X-SHiELD-AMIP-downscaling-3h.zarr/`
  (720×1440, 0.25°). Vars: `PRMSL` (mb), `eastward_wind_at_ten_meters`,
  `northward_wind_at_ten_meters`, `PRATEsfc`; coords `latitude`/`longitude`.
- Rectified filtered tracks (TC centers per time):
  `scratch/tc25_rectified_filt/rectified_tracks.csv`
  (`track_id,time,lat,lon,slp,wind,point_type,source_fine_track_id,dist_deg`).
  These are ~4 MB (25 km) / ~3.5 MB (100 km) — too large for the 250 KB
  `check-added-large-files` hook, so **not committed**; they will be hosted in a
  GCS bucket and the demo's `--tracks` pointed at the `gs://` path.
  **Provenance — how the rectified tracks were generated** (record when moving):
  ```bash
  # SLP-only 3h detection on the 25 km store:
  python scripts/tropical_cyclones/detect_tc_tracks.py \
    gs://vcm-ml-scratch/andrep/2025-08-12-X-SHiELD-AMIP-downscaling-3h.zarr/ scratch/tc25_full/ \
    --no-warm-core --timefilter 3hr --chunk-size 128 --workers 6 \
    --u-var eastward_wind_at_ten_meters --v-var northward_wind_at_ten_meters --no-write-sel-args
  # rectify fine SLP-only tracks against the coarse 6h warm-core tracks:
  python scripts/tropical_cyclones/rectify_tc_tracks.py \
    scratch/tc_full/tracks.csv scratch/tc25_full/tracks.csv scratch/tc25_rectified_filt/
  ```
  The demo's own provenance (its exact command) is stored in the output NetCDF's
  `command` global attribute and logged at runtime.

## Status

- [x] `tc_radial_metrics.py` -- 4 layers (transform/binning, structural metrics,
  scorecard, ensemble CRPS/RMSE). CRPS uses the fair estimator to match
  `fme/downscaling/metrics_and_maths.py`.
- [x] `test_tc_radial_metrics.py` -- 24 tests, all passing; each checked against
  a closed-form / hand-computed / independent-implementation reference (incl. a
  scorecard sign/wiring test).
- [x] `tc_radial_eval_demo.py` + figures generated & eyeballed. Ran on 4 mature
  storms (tracks 519/201/515/489) from `tc25_rectified_filt`; outputs in
  `scratch/tc25_eval_demo/`: `tc_radial_stats.nc` (single summary) plus
  `raw_profiles_*.png` and `error_curves_*.png` per storm.
- Sanity confirmed: target wind peaks at RMW ~37 km then decays; pressure rises
  monotonically outward; precip peaks in the core; per-bin `count` rises 1->100
  (inner-core under-sampling). CRPS(r)/RMSE(r) both peak at the eyewall where the
  smoothed synthetic ensemble misses the sharp core, and decay to ~0 in the
  environment.
- RMSE definition (decided): `ensemble_rmse` = `|mean_m(X_m) - x|`, the error of
  the ensemble **mean** (the conventional CRPS companion; both reduce to `|error|`
  for a zero-spread forecast). No forced ordering vs CRPS(r).

### Extra gotcha found while running

- The 25 km zarr time axis is a **julian `CFTimeIndex`** (object dtype), so a
  pandas `Timestamp` fails an exact `.sel`. Convert with
  `cftime.DatetimeJulian(...)` and use `method="nearest"` (see
  `tc_radial_eval_demo.read_patch`).
- Track SLP minima can be extreme in this model data (e.g. track 519 ~868 hPa);
  the "most mature" selection just takes the lowest-SLP point per track.

### Not yet done / next

- The demo ensemble is synthetic (kernel smoothing) -- swap for a real
  downscaling-model ensemble by replacing `make_synthetic_ensemble`.
- Consider a multi-storm / multi-time aggregation of the scorecard once real
  ensembles exist; current summary is member-mean per storm.

## Notes for later / handoff

- 100 km storms are under-resolved; these metrics are marginal there (per eval.md).
- The API is ensemble-native; swapping the synthetic demo ensemble for a real
  downscaling-model ensemble only touches the demo's ensemble generator.
- Self-contained under `scripts/`; no changes to `fme/` core.
