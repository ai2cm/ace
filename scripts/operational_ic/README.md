# Operational initial conditions for coupled SamudrACE (GLO12 + ERA5)

One command turns an issue day into everything `fme.coupled.inference` needs
to start a coupled SamudrACE forecast from the latest analysed ocean and
atmosphere: the ocean initial condition from the GLO12 operational analysis,
the atmosphere initial condition from ERA5 (final or ERA5T), the exogenous
forcing for the horizon, and the inference config.

```bash
conda env create -f environment.yaml          # once: samudrace-oper
conda activate samudrace-oper
python make_ic.py --date 2026-10-16 --out /data/ic/2026-10-16
IC_DIR=/data/ic/2026-10-16 DATE=2026-10-16 ./launch-operational-forecast.sh
```

Nothing here re-implements a data recipe. The ocean states are produced by
the per-chunk functions of [`scripts/glorys`](../glorys/) (the pipeline that
built the GLO12 training store) and the atmosphere sample by the functions of
[`scripts/era5`](../era5/) (the pipeline that built the ERA5 store), called
directly on one day of data without Apache Beam. `verify_ic.py` checks that a
day inside the training record reproduces the training stores.

## What `make_ic.py` does for issue day D

1. **Validates D** (`validation.json`). The GLO12 daily means D-4..D must be
   analysed days: on the store's time axis, no later than yesterday (the
   analysis-forecast store also carries the 10-day forecast, which is
   refused), and actually populated (a surface tile is read). ERA5 must cover
   the instant D+1 00Z: the analysis-ready model-level store (final ERA5,
   about three months behind) or the cloud-optimized ERA5T stores (about a
   week behind), plus `full_37` for the surface fields and TOA insolation.
   `--atmos-source auto` takes final ERA5 when it covers D, else ERA5T. Days
   within the last week carry a warning (GLO12's weekly best analysis may
   still revise them); an ERA5T day carries a warning that it is preliminary.
   `--dry-run` stops here.
2. **Ocean.** Downloads the five GLO12 daily means D-4..D (`thetao`, `so`,
   `uo`, `vo` on 50 levels at 1/12 degree, plus `zos`, `siconc`, `sithick`,
   `usi`, `vsi`), about 3 GB per day from CloudFerro, and processes each day
   exactly as the training store was built: fractional 50 -> 19 level remap,
   conservative regrid to F22.5 with the precomputed weights, masking,
   coastal fill, the sea-ice conventions. The static geometry (masks,
   fractions, `deptho`, `hfgeou`, `idepth`) is copied from the training store
   instead of rebuilt, so the initial condition sits on the checkpoint's own
   land-sea mask (a different mask leaves unfilled NaN cells and the first
   ocean step goes NaN). The placeholder channels of the fine-tune's derived
   store are added (`sfdsi`, `UI`, `VI` zero over ocean; `hfds_total_area`,
   `tauuo`, `tauvo`, `wfo`, `unaccounted_heating` zero-filled). The five
   states go to `glo12_daily_4deg_*.zarr`; the initial condition
   `ocean_ic_D.nc` is the day-D state (`--ocean-ic day`, the training
   convention: a state stamped `D 00:00` is the daily mean of day D) or the
   mean of the five (`--ocean-ic mean5`, smoother than any training state;
   for experiments).
3. **Atmosphere.** The ACE2S atmosphere steps daily on a time coarsening of
   the 6-hourly ERA5 store: snapshot fields are the 6-hourly sample at the
   window end, window fields the mean over the day, and the sample for day D
   is stamped `D 00:00` in the coupled store (-24 h relabel). So one ERA5
   instant, D+1 00Z, is downloaded (the eight model-level 3-D fields and
   the surface fields), plus the 24 hourly TOA insolation fields of day D.
   The instant goes through `scripts/era5`'s `_process_model_level_data`
   (total water, pressure-weighted 137 -> 8 layers at 0.25 degree,
   conservative regrid). With ERA5T the model levels come from the
   cloud-optimized stores in their native spectral (T639) and reduced
   Gaussian (N320) forms and are put on the 0.25-degree grid by
   `era5t_transform.py` (pure Python, torch_harmonics; validated against the
   analysis-ready store). `global_mean_co2` comes from the ERA5 CO2 dataset
   (extrapolated linearly beyond its end); `HGTsfc` and `ak/bk` from the
   reference atmosphere store; `land_fraction`, `sea_surface_fraction`,
   `sea_ice_fraction` and `ocean_fraction` from the GLO12 ocean, as in the
   coupled training store. `--atmos-full` also builds the diagnostic
   (output-only) variables of the daily store, about 3 GB more; the default
   builds what the model reads plus the free by-products.
4. **Forcing and config.** During the rollout the model reads only exogenous
   forcing from disk: `DSWRFtoa`, `global_mean_co2`, `HGTsfc` and
   `land_fraction` for the atmosphere; the static `sea_surface_fraction`,
   `deptho`, `hfgeou` for the ocean. `atmos_forcing_D.zarr` (daily) holds
   the TOA insolation calendar-day climatology of the reference store
   (2010-2024; day 0 is the real value), the CO2 series and the statics;
   `ocean_forcing_D.zarr` (5-daily) holds the statics. `inference_config.yaml`
   points `fme.coupled.inference` at the files with local paths;
   `launch-operational-forecast.sh` uploads the directory as a Beaker dataset,
   rewrites the paths to `/ic`, mounts the deployed checkpoint and launches
   through gantry with `--min-runtime`.

`inference_config.yaml` carries two details the loader insists on: the
ocean forcing store has a zero `ocean_forcing_placeholder` (time, lat, lon)
variable because a store of static fields alone cannot be opened, and
`forcing_loader.num_data_workers` is 1 because the loader rejects 0. A
two-step CPU run of `fme.coupled.inference` from a 2025-12-24 build with the
epoch-7 checkpoint ran end to end (finite predictions on every ocean cell).

## Checking a build

```bash
python make_ic.py --date 2025-12-24 --out /tmp/ic-check --forcing-steps 0
python verify_ic.py --date 2025-12-24 --ic-dir /tmp/ic-check
```

compares the ocean state with the GLO12 training store and the atmosphere
sample with the reference daily ERA5 store. Measured 2026-10-24 for
2025-12-24: the ocean matches on all 127 variables to float32 noise (largest
relative RMSE 6e-6, salinity); the final-ERA5 atmosphere matches bitwise on
every 3-D field (TOA insolation 1e-7 relative); the ERA5T atmosphere matches
the final-ERA5 sample to within 0.5% relative RMSE on every variable (worst:
top-layer winds, RMSE 0.02 m/s, max 0.2 m/s; the spectral/N320 transform
error). Run it after any change to the pipelines this one reuses.

Cost on an 8-CPU VM: about 7 min per GLO12 day (download bound), 1.5-3.5 min
for the atmosphere (ERA5T path; the first run builds the Legendre tables),
1.5 min for the forcing; the ERA5T atmosphere stage peaks at 8.5 GB of RSS
once the tables are cached (the first build of the vector table peaks near
14 GB).

## Conventions the files follow (do not change without retraining)

- Ocean state stamped `D 00:00` = GLO12 daily mean of day D; the ocean steps
  5 days. `sea_ice_volume` = `HI` (thickness over the ice-covered part; the
  UFS replay convention the checkpoint reads). `sst` in K, `thetao` in degC,
  `zos` not demeaned.
- Atmosphere sample stamped `D 00:00` = ERA5 instant D+1 00Z (snapshots) and
  means over day D (window fields). Five daily atmosphere steps span one
  ocean step.
- Fractions in the atmosphere file are the ocean's: `sea_ice_fraction` =
  GLO12 concentration x `sea_surface_fraction` (NaN where no ocean),
  `ocean_fraction` = 1 - land - ice.
- Time is encoded `proleptic_gregorian`, dims are `(time, lat, lon)`.

## Latency

GLO12 analyses lag about one day; ERA5T about six days and the final
analysis-ready model levels about three months, so the issue day is bounded
by ERA5T. The cloud-optimized stores' attributes can run ahead of their data;
`validation.json` records what was actually populated.

## Files

| file | role |
|---|---|
| `make_ic.py` | the command: validate, download, build, write |
| `ocean_ic.py` | GLO12 day -> training-store state (reuses `scripts/glorys`) |
| `atmos_ic.py` | ERA5 instant -> ACE2S daily sample (reuses `scripts/era5`) |
| `era5t_transform.py` | spectral / N320 ERA5T -> 0.25-degree analysis-ready layout |
| `make_forcing.py` | forcing datasets for the horizon |
| `verify_ic.py` | regression against the training stores |
| `launch-operational-forecast.sh` | Beaker launch of `fme.coupled.inference` |
| `environment.yaml` | conda env (`scripts/glorys` deps + CPU torch + netCDF) |

One-off inputs (the F22.5 regrid weights, the N320 coordinates, the TOA
climatology, the atmosphere statics) are cached under
`~/.cache/samudrace-oper` (`SAMUDRACE_OPER_CACHE`).
