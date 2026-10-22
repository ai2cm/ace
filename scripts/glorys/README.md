# GLORYS12 + ERA5 ocean training-set pipeline

xarray-beam pipeline that turns the Copernicus Marine GLORYS12V1 global ocean
reanalysis (1/12°, 50 levels, daily means) — or, with `--product glo12`, the
GLO12 operational analysis on the same grid — plus ERA5 forcing into a
SamudrACE ocean training zarr with the 19 CM4-matched levels at 5-day cadence,
on the F90 (1°) or F22.5 (4°) Gaussian grid — the same layout as the UFS-replay
ocean set produced by [`scripts/ufs-replay/`](../ufs-replay/), so training
configs carry over with the variable-list changes noted below. Like
[`scripts/era5/`](../era5/), each resolution is built from the native source,
not by coarsening another resolution's output.

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
| Ocean state (`--product glo12`) | `GLOBAL_ANALYSISFORECAST_PHY_001_024`, per-variable `cmems_mod_glo_phy*_anfc_0.083deg_P1D-m_202406` stores | Same ORCA12 grid and 50 levels; 2022-06-01 → present. Only its time-varying fields are used: static geometry is GLORYS12's, intersected with GLO12's mask, so both products share one geometry. |
| Forcing | `gs://vcm-ml-intermediate/2026-08-13-era5-{1deg,4deg}-8layer-1940-2025.zarr` | Built by `scripts/era5/` on the output grid (`GRID_INPUTS`); 6-hourly; ends 2025-12-31T18. |

GLORYS publishes **no surface fluxes or stress**. ERA5 drove the reanalysis
(via NEMO's bulk formulae), so the ten forcing fields are taken from the ERA5
store; they are the inputs to the flux computation, not the fluxes the ocean
saw, and the assimilation increments are an additional, unpublished term in
every budget.

## Differences from the UFS-replay pipeline

| | UFS replay | GLORYS12 |
|---|---|---|
| Native → output resolution | 0.25° → 1° (4×) | **1/12° → 1° (12×) or 4° (48×)**; xESMF conservative weights for 8.8 M source cells take minutes and several GB per worker — build them once per grid with `make_regrid_weights.py` and pass `--regrid_weights` |
| Vertical | 75 MOM6 layers → 19 by integer index groups, weighted by native `ho` | **50 z-levels → 19 by a fractional overlap matrix** built from static `e3t` (levels do not nest); partial bottom cell treated as full |
| Cadence | 6-hourly snapshots, coarsened downstream | **daily means**, every 5th day read (`--time_stride 5`); daily output would cost 5× the egress (~45 TB) |
| Forcing | FV3 replay fields, same stream | ERA5 1° store, window mean over the 5 days **ending** at each state (training-set convention); two renames: `eastward/northward_surface_stress` → `*_wind_stress` |
| Domain | global | **stops at 80°S**: output rows wholly south of it have zero source overlap and are written as land (`mask_* = 0`, `land_fraction = 1`) |
| Diagnostic outputs | `hfds`, `hfds_total_area`, `wfo`, `tauuo`, `tauvo` | **none** — not published; drop these from `out_names` and keep the heat-content and salt correctors off |
| `sst` | MOM6 top layer + 273.15 | `thetao` at 0.494 m + 273.15 |
| Sea ice | FV3 `icec`/`icetk` | `siconc` → `ocean_sea_ice_fraction`, `sithick` → `HI`, `siconc·sithick` → `sea_ice_volume`, `usi`/`vsi` → `UI`/`VI` |
| `hfgeou` | MOM6 static field | not published; copied from the CM4 (1°) or OM4 (4°) static field so configs that list it as an input carry over |
| `zos` | MOM6 SSH | SSH above geoid tied to an assimilated mean dynamic topography — check the mean offset before use |

## Masks

A cell is ocean at a level only if all of these hold:

- some native wet cell contributes to it (the UFS convention);
- it is ocean in the grid's **reference mask** (`GRID_INPUTS`): the dataset
  the fine-tune checkpoint was pretrained on — the UFS replay at 1°, OM4 at 4°
  (no 4° UFS replay exists). `--no_mask_intersection` disables this;
- it is not in the grid's **river-plume mask**: cells whose climatological
  surface salinity for some calendar month (the multi-year mean of the 15th
  of that month) falls below 5 psu in GLORYS12
  (2011–2020) or GLO12 (2023–2025). Runoff is not a model input, so these cells
  are unpredictable by construction. Build with `make_plume_mask.py`
  (`--product`, `--union`, `--output_grid`);
- every 3-D field (`thetao`, `so`, `uo`, `vo`) has data there, in GLORYS12 and
  in the product being processed. Velocities are NaN at some coastal and
  bottom cells the bathymetry mask calls ocean, so without this NaN would sit
  inside the mask. Land is fixed in time, so one reference state per product
  finds every such cell.

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
≈ 8.6 TB of int16 (plus 0.3 TB of 2-D fields); the 1° output is ≈ 30 GB, the
4° output ≈ 2 GB. Reads from CloudFerro are free; Dataflow compute is
≈ $0.02 per output state at list price on the default `n2d-custom-2-49152`
workers (measured on the 1° GLO12 run), about the same at 4° because reading
and vertically remapping the native fields dominates. So ≈ $50 for 1993–2025
and ≈ $6 for GLO12 2022–2025, per grid.

## Usage

```bash
make create_environment          # conda env glorys-ingest (xesmf + dataflow deps)
python make_regrid_weights.py <file> --output_grid F90   # once per grid
make glorys_local_debug          # 2 states, DirectRunner (OUTPUT_GRID_LOCAL=F22.5 for 4°)
make glo12_local_debug           # same for GLO12
make build_dataflow push_dataflow
make glorys_dataflow_test_run    # 2 months on Dataflow
make glorys_dataflow             # 1° production: 1993-01-01 .. 2025-12-27
make glo12_dataflow              # 1° GLO12: 2022-06-01 .. 2025-12-27
make glorys_dataflow_four_degree glo12_dataflow_four_degree   # 4° sets
make compute_stats               # normalization stats via scripts/data_process
```

The 1° targets take the weights through `DATAFLOW_EXTRA_FLAGS="--regrid_weights …"`;
the 4° targets pass `REGRID_WEIGHTS_FOUR_DEGREE`. Stats configs:
`../data_process/configs/{glorys12,glo12}-ocean-{1deg,4deg}-19level-5day.yaml`.
