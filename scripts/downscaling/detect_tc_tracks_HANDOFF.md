# TC track detection — session handoff

Working notes for running `scripts/downscaling/detect_tc_tracks.py` on the
X-SHiELD 11-year zarr. Branch: `scripts/tempest-extreme-wrapper`.

## Goal

Produce a CSV of tropical-cyclone tracks from
`gs://vcm-ml-intermediate/2025-09-11-X-SHiELD-AMIP-1deg-8layer-11yr.zarr/`,
then fill in the 3h in-between points from a **separate 3-hourly dataset**
(this zarr is 6-hourly) and derive cyclone characteristics for comparison.

## 3h / high-res detection (SLP-only) — the current work

The 3h downscaling datasets have SLP, 10m winds, sfc-T, rain but **no upper-air
temperature**, so the warm-core criterion can't run on them. Plan: detect TCs on
the coarse 6h store (warm-core, done), and independently run an **SLP-only**
detection on the 3h data, then match those tracks to the 6h warm-core tracks to
keep only real TCs (StitchNodes handles genesis/death by construction).

3h datasets:
- 25km: `gs://vcm-ml-scratch/andrep/2025-08-12-X-SHiELD-AMIP-downscaling-3h.zarr/` (720×1440)
- 100km: `gs://vcm-ml-scratch/andrep/2025-07-25-X-SHiELD-AMIP-FME-3h.zarr/` (180×360)
- vars: `PRMSL` (mb), `eastward_wind_at_ten_meters`, `northward_wind_at_ten_meters`,
  `PRATEsfc`, `air_temperature_at_two_meters`; coords `latitude`/`longitude`; 3-hourly.

**Key I/O finding:** both 3h stores chunk the **whole globe into one chunk per
timestep** (zarr v3, sharded 128 timesteps). So spatial windowing saves *zero*
I/O — any read fetches the whole global chunk. TCs are active 77% of timesteps,
so temporal subsetting only trims ~23%. Measured read ~107–167 MB/s (8 vCPU, not
network-bound — <10% of NIC; scales ~1.5× with concurrency). Full 25km 3-var
pass ≈ 40 min; DetectNodes compute is comparable (~30 min) → parallelism helps.

**Pipeline upgrade (this session):** `detect_tc_tracks.py` now processes
shard-sized time-bundles in parallel (`--workers`, spawn processes) with
read→NetCDF→DetectNodes→**delete** per bundle, so peak disk ≈ `workers × bundle`
(~1.6 GB each) instead of the whole dataset. New flags: `--no-warm-core`
(SLP-only recipe), `--timefilter 3hr`, `--keep-netcdf`. Example 25km run:

```bash
python scripts/downscaling/detect_tc_tracks.py \
  gs://vcm-ml-scratch/andrep/2025-08-12-X-SHiELD-AMIP-downscaling-3h.zarr/ scratch/tc25/ \
  --no-warm-core --timefilter 3hr --chunk-size 128 --workers 6 \
  --u-var eastward_wind_at_ten_meters --v-var northward_wind_at_ten_meters \
  --detect-exe ~/miniconda3/envs/tempest/bin/DetectNodes \
  --stitch-exe ~/miniconda3/envs/tempest/bin/StitchNodes --no-write-sel-args
```

**Validated** on a 2-month 25km subset (Aug–Oct 2013, 4 shard bundles, 6 workers):
67 s wall, peak RSS ~4 GB, NetCDF deleted per bundle (bounded disk), exit 0 →
150 tracks / 8368 points, SLP to 920 mb, wind to 52 m/s. The 150 includes many
**extratropical** systems (lat up to 82, Southern Ocean) since SLP-only catches
all closed lows — the 6h warm-core match is what filters to real TCs. Full 11yr
25km SLP-only ≈ ~75 min at 6 workers (extrapolated). Note: needed `spawn` start
method for the process pool (fork corrupts gcsfs' asyncio state → BrokenProcessPool).

TODO: (1) match 3h SLP-only tracks to 6h warm-core tracks; (2) revisit StitchNodes
threshold *count* — `wind,>=,10.0,10` = 30h at 3h vs 60h at 6h (shapes the 3h set).

## Environment / install (done)

### Installing TempestExtremes (conda-forge — the method used)

The script needs the `DetectNodes` and `StitchNodes` binaries. Install them into
a dedicated conda env from conda-forge (no compiler, cmake, or apt deps needed):

```bash
conda create -y -n tempest -c conda-forge tempest-extremes
```

Installed here: `tempest-extremes 2.4.2` (build `nompi_h8553f88_102`, linux-64),
which pulled `libnetcdf 4.10.0` and `hdf5 2.1.0` (all conda-forge). To reproduce
the exact build on another machine, pin it:
`conda create -y -n tempest -c conda-forge tempest-extremes=2.4.2`.

Binaries land at `~/miniconda3/envs/tempest/bin/{DetectNodes,StitchNodes}` — pass
these via the script's `--detect-exe`/`--stitch-exe` (the script's *default*
exe path is the source-build location below, which we are **not** using). On a
new machine the `~/miniconda3` prefix may differ; use
`conda run -n tempest which DetectNodes` to find them.

Verify the install: `~/miniconda3/envs/tempest/bin/DetectNodes` with no args
should print `Arguments:` then error that no input file was given — that means
the binary runs.

### Source-build fallback (repo's documented path — NOT used here)

`make -C scripts/downscaling tc_deps` builds TempestExtremes from source via
CMake into `~/.local/tempestextremes/bin` (matches the script's default exe
path). On this machine `cmake` and the NetCDF/HDF5 dev headers were missing, so
it needs, on Debian/Ubuntu with sudo:
`sudo apt install cmake libnetcdf-dev libnetcdf-c++4-dev libhdf5-dev`.
Prefer conda-forge above unless you specifically want the source build.

### Running the script

- Run the python script from the `fme` conda env (has xarray/gcsfs; GCS ADC auth works).
- `python` only resolves after `conda activate`; source `~/miniconda3/etc/profile.d/conda.sh` first.

## Dataset facts (verified)

6-hourly, 16080 steps, 2013-01 → 2024-01, 1° (180×360), no `sample` dim.
Script defaults all match this data: `PRMSL` (mb → auto-converted to Pa),
`air_temperature_3` ≈ 341 hPa (valid for the T3 warm-core substitution),
`UGRD10m`/`VGRD10m`, coords `grid_xt`/`grid_yt`. cftime **Julian** calendar
(CSV timestamps still correct across 2013–2024 — no century leap years).

## Bug found & fixed (committed at `6546aed29` "update detect nodes")

DetectNodes with `--in_data_list` auto-suffixes `--out` into one file per chunk
(`candidate_nodes.dat000000.dat` …), so StitchNodes' old single
`--in candidate_nodes.dat` failed with "Unable to open input file". Fix:
`run_detect_nodes` globs the per-chunk files, writes `candidate_files.txt`, and
`run_stitch_nodes` reads them via `--in_list` (also lets tracks span chunk
boundaries). Note: `--out_file_list` does **not** work in this conda-forge build
(exits 0, writes nothing) — the `--out` + glob path is the proven one.

## Validated (test season)

`--time-start 2013-08-01 --time-end 2013-10-31 --no-write-sel-args`
→ `scratch/tc_test_2013/tracks.csv`: 701 points / 28 tracks, SLP down to 952 mb,
wind to 30 m/s. Correct.

## Full 11-year run (DONE)

Command used (~16.7 GB NetCDF intermediates, `--cleanup` removes them after
DetectNodes; ~3 min wall time on this machine):

```bash
conda activate fme && cd ~/repos/ace
python scripts/downscaling/detect_tc_tracks.py \
  gs://vcm-ml-intermediate/2025-09-11-X-SHiELD-AMIP-1deg-8layer-11yr.zarr/ scratch/tc_full/ \
  --detect-exe ~/miniconda3/envs/tempest/bin/DetectNodes \
  --stitch-exe ~/miniconda3/envs/tempest/bin/StitchNodes \
  --cleanup --no-write-sel-args
```

Result: `scratch/tc_full/tracks.csv` — **853 tracks / 21,954 points**, 2013→2024,
SLP to 948 mb, peak wind 33 m/s. Using `--no-write-sel-args` because the pickled
`.sel` kwargs use pandas `datetime64` values that won't `.sel(method="nearest")`
against the cftime-Julian source index (the 3h fill-in is done separately anyway).

## Plotting

`scratch/plot_tc_tracks.py` draws tracks on a global cartopy map colored by peak
10 m wind (genesis = dot): `python scratch/plot_tc_tracks.py <tracks.csv> <out.png>`.
Maps rendered: `scratch/tc_2013/tracks_map.png` (85 tracks) and
`scratch/tc_full/tracks_map.png` (853 tracks). NOTE for cartopy: `bbox_inches="tight"`
collapses the GeoAxes when a colorbar is present — use `constrained_layout=True`
and `fig.canvas.draw()` before `savefig` instead.

## Known false positives

A few tracks sit over land / high latitudes (e.g. a knot near the Caspian ~55°E/42°N,
some extratropical transitions poleward of 50°). Inherent to the recipe at 1° coarse
resolution — StitchNodes requires ≥10 steps within ±50° lat but lets tracks drift
outside afterward. Tighten lat/duration filters or add a land/ocean mask if a cleaner
set is needed for the comparison.

## Everything lives in scratch/ (git-ignored)

`scratch/tc_2013/`, `scratch/tc_full/`, and `scratch/plot_tc_tracks.py` are under the
git-ignored `scratch/` dir, so they don't pollute the repo. Only the script fix and
this handoff are tracked.

## Cleanup

Stray `log000000.txt` … `log000003.txt` in the repo root (DetectNodes logs,
untracked) — safe to delete.
