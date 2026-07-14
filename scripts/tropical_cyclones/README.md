# Tropical cyclone track analysis

Tooling to detect, filter, plot, and structurally evaluate tropical-cyclone
(TC) tracks in ACE / FME output zarrs, using
[TempestExtremes](https://github.com/ClimateGlobalChange/tempestextremes)
(`DetectNodes` → `StitchNodes`).

This README is the entry point: **install → detect → plot → (optional) rectify
→ (optional) radial structure eval**. For running status, provenance, and the
detailed rationale behind each choice, see the companion docs:

- `detect_tc_tracks_HANDOFF.md` — detection/rectification session notes, verified
  dataset facts, gotchas.
- `TC_ANALYSIS.md` + `eval.md` — the radial-structure verification framework and
  its status.

## The pieces

| Script | What it does |
| --- | --- |
| `detect_tc_tracks.py` | Detect TC tracks in a zarr via TempestExtremes; writes `tracks.csv`. |
| `plot_tc_tracks.py` | Plot a `tracks.csv` on a global map, colored by peak 10 m wind. |
| `rectify_tc_tracks.py` | Filter fine SLP-only tracks to genuine TCs by matching them against coarse warm-core tracks. |
| `tc_radial_metrics.py` | Radial/azimuthal structural verification metrics (RMW, wind radii, precip morphology, CRPS(r), …). |
| `tc_radial_eval_demo.py` | End-to-end demo of the radial metrics on real storms. |
| `Makefile` | `tc_deps` target: build TempestExtremes from source. |

## Install TempestExtremes

`detect_tc_tracks.py` needs the `DetectNodes` and `StitchNodes` binaries. Two routes:

**conda-forge (recommended — no compiler/apt deps):**

```bash
conda create -y -n tempest -c conda-forge tempest-extremes
# binaries land at ~/miniconda3/envs/tempest/bin/{DetectNodes,StitchNodes}
conda run -n tempest which DetectNodes   # confirm location
```

**Source build (matches the script's default `--detect-exe`/`--stitch-exe` path,
`~/.local/tempestextremes/bin`):**

```bash
make -C scripts/tropical_cyclones tc_deps
# needs cmake + NetCDF/HDF5 dev headers, e.g. on Debian/Ubuntu:
#   sudo apt install cmake libnetcdf-dev libnetcdf-c++4-dev libhdf5-dev
```

> **Two environments.** Run the *Python* script from the `fme` conda env (it has
> `xarray`/`gcsfs` and GCS auth). The TempestExtremes *binaries* live wherever you
> installed them above — if you used the conda-forge `tempest` env, you **must**
> pass `--detect-exe`/`--stitch-exe` explicitly (the script's defaults point at the
> source-build location, not the conda env). The commands below do this.

## Detect tracks on a 100 km generated zarr

The generated 100 km store is 3-hourly, 1° (180×360), and has **no
upper-tropospheric temperature**, so the warm-core criterion cannot run — use the
SLP-only recipe (`--no-warm-core`) at the data's 3-hourly cadence
(`--timefilter 3hr`). `HGTsfc` is present, so add the surface-elevation filter
(`--topo-var HGTsfc --max-topo 500`) to drop spurious lows over high terrain
(monsoon/heat lows) while keeping low-lying landfalling TCs. `PRMSL` is stored in
mb and is auto-converted to Pa; the `latitude`/`longitude` coords are auto-detected;
the Julian calendar is handled internally.

```bash
conda activate fme && cd ~/repos/ace
python scripts/tropical_cyclones/detect_tc_tracks.py \
  gs://vcm-ml-intermediate/2026-07-14-X-SHiELD-AMIP-FME-3h-100km.zarr/ scratch/tc100_gen/ \
  --no-warm-core --timefilter 3hr --chunk-size 128 --workers 6 \
  --u-var eastward_wind_at_ten_meters --v-var northward_wind_at_ten_meters \
  --topo-var HGTsfc --max-topo 500 \
  --detect-exe ~/miniconda3/envs/tempest/bin/DetectNodes \
  --stitch-exe ~/miniconda3/envs/tempest/bin/StitchNodes \
  --no-write-sel-args
```

Output in `scratch/tc100_gen/`: `tracks.csv` (columns `track_id, time, lon, lat,
slp` [Pa], `wind` [m/s]), plus `tracks.dat` (raw StitchNodes) and per-bundle
intermediates under `nodes/`. Peak disk stays ~`workers × bundle` because each
bundle's NetCDF is deleted after its `DetectNodes` finishes.

- **`--no-write-sel-args`** is kept because this store's time axis is Julian
  cftime; the pickled pandas-`datetime64` `.sel` kwargs won't `.sel(method=
  "nearest")` against a cftime-Julian source index. Drop it only for a
  `datetime64`-axis store.
- **Smoke test** a single month first to validate the setup, e.g. add
  `--time-start 2013-08-01 --time-end 2013-08-31`. (Verified: yields 31 tracks /
  1247 points on the 100 km store.)
- **`--workers`** is RAM-bound: each holds ~one bundle in memory. 4–6 is fine for
  the 100 km (180×360) store; the reads fetch a whole global chunk per timestep
  regardless of spatial extent, so I/O — not the flag — dominates.

### 25 km variant

The 25 km generated store (`...-3h-25km.zarr/`, 720×1440) has the same variable
names and cadence but **no `HGTsfc`** — omit the two terrain-filter flags:

```bash
python scripts/tropical_cyclones/detect_tc_tracks.py \
  gs://vcm-ml-intermediate/2026-07-14-X-SHiELD-AMIP-FME-3h-25km.zarr/ scratch/tc25_gen/ \
  --no-warm-core --timefilter 3hr --chunk-size 128 --workers 6 \
  --u-var eastward_wind_at_ten_meters --v-var northward_wind_at_ten_meters \
  --detect-exe ~/miniconda3/envs/tempest/bin/DetectNodes \
  --stitch-exe ~/miniconda3/envs/tempest/bin/StitchNodes \
  --no-write-sel-args
```

A 720×1440 bundle is ~1.6 GB in memory, so keep `--workers` modest and expect a
longer run.

## Filter to warm-core TCs (rectification)

SLP-only detection picks up *every* closed low — extratropical storms, polar
lows, monsoon lows, terrain/equator artifacts — not just TCs. `rectify_tc_tracks.py`
keeps only genuine TCs by corroborating the fine SLP-only tracks against a coarse
**warm-core** track set (which does apply the warm-core criterion): it anchors
each fine track to a warm-core track in space and time, and bounds it to the
warm-core lifetime.

**Warm-core source of truth:** the canonical coarse 6-hourly warm-core track set
(11 yr, terrain-filtered with `HGTsfc ≤ 500 m`, 838 TCs) lives at
`gs://vcm-ml-experiments/2026-temporal-downscaling/tc_6h_coarse_filt`. Use its
`tracks.csv` as the coarse input:

```bash
python scripts/tropical_cyclones/rectify_tc_tracks.py \
  gs://vcm-ml-experiments/2026-temporal-downscaling/tc_6h_coarse_filt/tracks.csv \
  scratch/tc100_gen/tracks.csv scratch/tc100_gen_rectified/
```

> **Requirement — read before using on generated output.** Rectification needs a
> warm-core track set that is **time-aligned with the same run** (the anchor step
> matches within ~1° great-circle at the *same timestamp*). This holds for the
> X-SHiELD reference pipeline, where the coarse 6-hourly warm-core store and the
> fine SLP-only store describe the same simulation. It does **not** automatically
> hold for a free-running / emulated generated store, whose individual TCs need
> not be phase-aligned with the reference simulation — in that case the anchor
> step will mostly miss or match spuriously. If you cannot confirm the generated
> run is time-aligned with an available warm-core track set, you probably cannot
> use this recipe to filter.

## Optional: radial structural verification

`tc_radial_metrics.py` maps storm-centered 2D fields into 1D azimuthal-radial
profiles and scores structural/dynamical errors (RMW, wind radii, precip
morphology, gradient-wind balance, symmetry, ensemble CRPS(r)/RMSE(r)) — see
`eval.md` for the metric definitions and `TC_ANALYSIS.md` for status. Run the
tests and the demo (from the `fme` env) with:

```bash
python -m pytest scripts/tropical_cyclones/test_tc_radial_metrics.py -q
python scripts/tropical_cyclones/tc_radial_eval_demo.py
```

> At 100 km the storm is under-resolved, so these radial metrics are marginal
> there; they are more meaningful at 25 km, especially for larger storms (see the
> dev notes in `eval.md`).

## Gotchas

- **conda PATH on some GCP VMs.** If `conda`/`python` aren't on a non-login
  shell's PATH, source it first (`source ~/miniconda3/etc/profile.d/conda.sh`) or
  use the full path `~/miniconda3/bin/conda run -n fme python ...`.
- **PRMSL in mb** is auto-converted to Pa by the script (TempestExtremes'
  contour thresholds are in Pa). Custom pressure vars must carry a `units` attr.
- **Whole-globe chunking.** The 3-hourly stores chunk the whole globe into one
  chunk per timestep, so spatial windowing saves zero I/O — any read fetches the
  global chunk. Bound memory by processing time-bundles, not by narrowing space.
- **StitchNodes threshold counts are in timesteps, not hours.** The recipe's
  `wind,>=,10.0,10` means 10 steps = 30 h at 3-hourly (vs 60 h at 6-hourly), which
  shapes the fine track set.
