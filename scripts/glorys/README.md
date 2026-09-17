# GLORYS12 + ERA5 ocean training-set pipeline

xarray-beam pipeline that turns the Copernicus Marine GLORYS12V1 global ocean
reanalysis (1/12°, 50 levels, daily means) plus the 1° ERA5 forcing store into
a SamudrACE ocean training zarr on the F90 (1°) Gaussian grid with the 19
CM4-matched levels at 5-day cadence — the same layout as the UFS-replay ocean
set produced by [`scripts/ufs-replay/`](../ufs-replay/), so training configs
carry over with the variable-list changes noted below.

Why this dataset: the UFS replay ends in October 2023 and has no real-time
continuation. GLORYS12 runs to within a few months of present, and its
operational twin — the GLO12 analysis/forecast (`GLOBAL_ANALYSISFORECAST_PHY_001_024`,
same grid, same 50 levels, daily from 2022-06-01, ~1-day latency) — is the
initial-condition source for an operational model.

## Data sources

| Component | Source | Notes |
|---|---|---|
| Ocean state | `GLOBAL_MULTIYEAR_PHY_001_030` / `cmems_mod_glo_phy_my_0.083deg_P1D-m` | ARCO zarr v2 on CloudFerro S3, anonymously readable; 1993-01-01 → present. **Reanalysis proper ends 2021-06-30; later dates are the interim extension**, re-initialised when the reanalysis is extended. |
| Geometry | `..._static_202311--ext--coords` (`e1t`,`e2t`,`e3t`), `--ext--bathy` (`deptho`, `deptho_lev`, 3-D `mask`) | |
| Forcing | `gs://vcm-ml-intermediate/2026-08-13-era5-1deg-8layer-1940-2025.zarr` | Built by `scripts/era5/`; already F90; 6-hourly; ends 2025-12-31T18. |

GLORYS publishes **no surface fluxes or stress**. ERA5 drove the reanalysis
(via NEMO's bulk formulae), so the ten forcing fields are taken from the ERA5
store; they are the inputs to the flux computation, not the fluxes the ocean
saw, and the assimilation increments are an additional, unpublished term in
every budget.

## Differences from the UFS-replay pipeline

| | UFS replay | GLORYS12 |
|---|---|---|
| Native → output resolution | 0.25° → 1° (4×) | **1/12° → 1° (12×)**; xESMF conservative weights for 8.8 M source cells take minutes and several GB per worker — pass `--regrid_weights` with a precomputed file for production |
| Vertical | 75 MOM6 layers → 19 by integer index groups, weighted by native `ho` | **50 z-levels → 19 by a fractional overlap matrix** built from static `e3t` (levels do not nest); partial bottom cell treated as full |
| Cadence | 6-hourly snapshots, coarsened downstream | **daily means**, every 5th day read (`--time_stride 5`); daily output would cost 5× the egress (~45 TB) |
| Forcing | FV3 replay fields, same stream | ERA5 1° store, window mean over the 5 days **ending** at each state (training-set convention); two renames: `eastward/northward_surface_stress` → `*_wind_stress` |
| Domain | global | **stops at 80°S**: the 10 southernmost F90 rows have zero source overlap and are written as land (`mask_* = 0`, `land_fraction = 1`) |
| Diagnostic outputs | `hfds`, `hfds_total_area`, `wfo`, `tauuo`, `tauvo` | **none** — not published; drop these from `out_names` and keep the heat-content and salt correctors off |
| `sst` | MOM6 top layer + 273.15 | `thetao` at 0.494 m + 273.15 |
| Sea ice | FV3 `icec`/`icetk` | `siconc` → `ocean_sea_ice_fraction`, `sithick` → `HI`, `siconc·sithick` → `sea_ice_volume` |
| `zos` | MOM6 SSH | SSH above geoid tied to an assimilated mean dynamic topography — check the mean offset before use |

## Access gotchas (both verified 2026-09-16)

- The S3 bucket answers **403, not 404**, for keys that do not exist. Two
  consequences handled in `open_cmems`: zarr-python 3 probes for `zarr.json`
  and would fail on the 403 unless `zarr_format=2` is pinned; and all-fill
  chunks (e.g. the deepest level of a 3-D field) come back 403, which the
  custom HTTP filesystem maps to `FileNotFoundError` so zarr fills with NaN.
- Throughput from one connection is ~0.6 MB/s; with ~32–48 concurrent
  connections ~16 MB/s per VM. Dataflow workers each add their own.

## Cost

Reading all 50 levels of the four 3-D fields at 5-day stride for 1993–2025 is
≈ 8.6 TB of int16 (plus 0.3 TB of 2-D fields); the output is ≈ 55 GB.

## Usage

```bash
make create_environment          # conda env glorys-ingest (xesmf + dataflow deps)
make glorys_local_debug          # 2 states, DirectRunner
make build_dataflow push_dataflow
make glorys_dataflow_test_run    # 2 months on Dataflow
make glorys_dataflow             # production: 1993-01-01 .. 2025-12-27
make compute_stats               # normalization stats via scripts/data_process
```

Stats config: [`../data_process/configs/glorys12-ocean-1deg-19level-5day.yaml`](../data_process/configs/glorys12-ocean-1deg-19level-5day.yaml).
