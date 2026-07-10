# ACE2S land-forcing short-lead evaluation — diurnal-sampling variant

Re-run of `2026-07-09-ace2s-land-forcing-inference` with **one change**: the initial conditions are
spread uniformly across the diurnal cycle. Everything else (checkpoints, variables, 40-step
rollout, land/daytime analysis, 3×100-IC chunking, memory settings, gantry setup) is identical —
see that experiment's README for the full recipe.

## Why

In the 07-09 run every IC sat at a fixed UTC hour (ERA5 00 UTC, CM4 06 UTC), because the IC
`interval` was a multiple of 4 six-hourly steps. So the **first predicted step** was always valid
at a single local solar time (ERA5 06 UTC, CM4 12 UTC). That biased shortwave-dependent quantities:

- `implied_surface_albedo = USWRF/DSWRF` is numerically unstable and physically extreme near the
  day/night terminator / low sun, producing **longitudinal banding** and inflated albedo RMSE
  (e.g. the Sahara at ERA5's 06 UTC low-morning-sun).
- The **different init hour between ERA5 and CM4** made their step-1 solar geometry differ, so the
  apparent "CM4 better / ERA5 worse" dataset split was largely a sampling artifact, not physics.

## The fix

Choose an IC `interval` that is **not** a multiple of 4 steps, so the init hour rotates through
00/06/12/18 UTC. Verified: **75 ICs at each of the four hours** (300 total) for both datasets,
months balanced, all within the holdout. Averaged over the 300 ICs, the first step then samples all
local solar times → diurnally unbiased, no fixed terminator, ERA5/CM4 comparable.

| dataset | dense stride (chunk offset) | per-chunk `interval` | chunk `first`s | holdout |
|---|---|---|---|---|
| ERA5 | 63 steps | 189 (mod 4 = 1) | 84738 / 84801 / 84864 | 1998–2010 (last IC 2010-11-23) |
| CM4  | 193 steps | 579 (mod 4 = 3) | 233600 / 233793 / 233986 | 0311–0351 (last IC 0350-07-12) |

Both strides are odd (init hour rotates) and non-commensurate with the 365-day year (no seasonal
aliasing). Each chunk of 100 ICs internally cycles all four init hours.

## Runs

Same 6 checkpoints × 3 chunks = **18 jobs**. Job names mirror the 07-09 runs with a **`-diurnal`
suffix** (e.g. `lf-eval-era5-snow-diurnal`), plus the `-c{0,1,2}` chunk suffix — parallel to the
originals but distinct for beaker. Outputs go to a
**separate GCS prefix** `gs://vcm-ml-intermediate/2026-07-10-ace2s-land-forcing-inference-diurnal/`
so the 07-09 outputs are preserved for comparison. Treatment checkpoint IDs and controls are the
same as 07-09 (already filled into `run-inference.sh`).

```bash
./run-inference.sh              # submit all 18
./run-inference.sh era5         # filter by substring
./run-inference.sh -c0          # only chunk 0 across all checkpoints
```

Downstream, point the analysis (`explore2 …/land-forcing-eval/report_land_forcing.py`, its `PREFIX`
and job names) at this prefix / the `lf-eval-*-diurnal` jobs.
