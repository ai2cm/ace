# TC track detection — session handoff

Working notes for running `scripts/downscaling/detect_tc_tracks.py` on the
X-SHiELD 11-year zarr. Branch: `scripts/tempest-extreme-wrapper`.

## Goal

Produce a CSV of tropical-cyclone tracks from
`gs://vcm-ml-intermediate/2025-09-11-X-SHiELD-AMIP-1deg-8layer-11yr.zarr/`,
then fill in the 3h in-between points from a **separate 3-hourly dataset**
(this zarr is 6-hourly) and derive cyclone characteristics for comparison.

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

## NEXT STEP — full 11-year run (not yet done)

Run in background; ~16.7 GB of NetCDF intermediates (`--cleanup` removes them
after DetectNodes consumes them):

```bash
conda activate fme && cd ~/repos/ace
python scripts/downscaling/detect_tc_tracks.py \
  gs://vcm-ml-intermediate/2025-09-11-X-SHiELD-AMIP-1deg-8layer-11yr.zarr/ scratch/tc_full/ \
  --detect-exe ~/miniconda3/envs/tempest/bin/DetectNodes \
  --stitch-exe ~/miniconda3/envs/tempest/bin/StitchNodes \
  --cleanup --no-write-sel-args
```

Deliverable: `scratch/tc_full/tracks.csv`. Using `--no-write-sel-args` because
the pickled `.sel` kwargs use pandas `datetime64` values that won't
`.sel(method="nearest")` against the cftime-Julian source index (the 3h fill-in
is done separately anyway).

## Cleanup

Stray `log000000.txt` … `log000003.txt` in the repo root (DetectNodes logs,
untracked) — safe to delete.
