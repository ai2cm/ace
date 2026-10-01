# ACE2S snow scenario rollouts (piControl vs 1pctCO2)

The fine-tuned CM4 masked-naive checkpoint held up on the piControl holdout
(`../2026-09-25-ace2s-snow-ft-rollouts/`); this round runs it, with its
fine-tuned control as companion on both scenarios, through the full 140-year
CM4 1pctCO2 ramp against a continuous 40-year piControl baseline.

## Arms and runs

| run | checkpoint (Beaker) | IC | steps |
|---|---|---|---|
| cm4-masked-naive-picontrol | `01M38TTH492G0WT71YFAEH0V58` (epoch 13) | 0311-01-01 | 14,600 (40 yr, the whole untouched 0311-0351 tail) |
| cm4-masked-naive-1pctco2 | `01M38TTH492G0WT71YFAEH0V58` | 0001-01-02 | 51,050 (full ramp) |
| cm4-control-picontrol | `01M33SZ16PFWP822C663RN42Z7` | 0311-01-01 | 14,600 |
| cm4-control-1pctco2 | `01M33SZ16PFWP822C663RN42Z7` | 0001-01-02 | 51,050 |

One IC per scenario (each full-length member exactly saturates its store);
`n_ensemble_per_ic: 4` supplies the stochastic members, at identical valid
times and full run length. `seed: 0` for reproducibility.

## Hard constraints and caveats

- **No CO2 input, by construction**: trained on piControl, CO2 is a constant,
  so the ramp is felt only through prescribed SST / sea-ice / ocean fraction.
  The missing direct-CO2 pathway is bounded by three emulator-target
  differences in `means_50d`: ULWRFtoa vs ln(CO2) (expect the emulator's OLR
  high by roughly the direct forcing), air_temperature_0/1 (the target's
  stratospheric cooling should be largely absent) and land-vs-ocean T2m
  (the integrated land-warming deficit). Expect emulator snow retreat biased
  weak vs target for this reason.
- **`anomaly_memory` is disabled on the 1pctCO2 runs**: its climatology has no
  trend term, so ramp anomalies retain the warming trend and lag correlations
  are biased high. Ramp memory analysis is offline (windowed, detrended) on
  `daily_fields`.
- **`trend` is enabled on both scenarios**: the piControl trend map is the
  null distribution the 1pctCO2 trend is read against.
- Do not start piControl members before 0311 (training data) or extend past
  0351 (end of store).

## Data provenance

1pctCO2 daily store and sidecar (weka copies of `gs://vcm-ml-intermediate/...140yr-daily/`):
coarsened from the 2026-06-19 6-hourly store by
`scripts/data_process/configs/CM4-1pctCO2-atmosphere-1deg-8layer-140yr-daily.yaml`
(branch `scripts/cm4-1pctco2-1deg-daily-dataset`), 12 soil masks appended from
the piControl lsm-masks store (`CM4-1pctCO2-land-mask-append.yaml`), snow
sidecar from `scripts/data_process/snow_masked_channels/` key `cm4-1pctco2`
(branch `scripts/snow-masked-channels-per-land-area`). piControl stores as in
the holdout round.

## Outputs

`gs://vcm-ml-intermediate/2026-09-29-ace2s-snow-scenario-rollouts/<arm>-<scenario>/`:
`daily_fields` (12/10 vars, 56S-90N, daily, all members, predictions only:
the daily targets are a direct slice of the GCS stores) and `means_50d`
(global 50-day means, target copy on all runs).
Sharded zarr v3: offline gcsfs reads need the suffix-range patch
(`scripts/data_process/snow_masked_channels/masked_snow.patch_gcsfs_suffix_ranges`)
until gcsfs is fixed.

## 1pctCO2 W&B summaries

The two 1pctCO2 runs completed and wrote all outputs, but crashed while building
their end-of-run W&B summary: this branch predates the year-0001 date-axis fix
for the ENSO/IPO index plots (#1439). Their summaries come from writer-free
reruns (`-summary` job names and experiment dirs) with `enso_index` and
`ipo_index` disabled; those indices are the prescribed SST read back in these
runs (predicted Nino3.4 equals the target bitwise). The rerun trajectories are
checked bitwise against the original runs via `restart.nc`.

## Launch

The piControl jobs run at normal priority and finish inside the 8h
min-runtime cap (the maximum protection beaker grants). The 1pctCO2 jobs
(~11 GPU-h plus writes) outlive that window, so they run at urgent priority
to make preemption unlikely; 1 GPU each. The evaluator is kept for the ramp
rather than segmented inference because the inference-only W&B metrics are
too sparse; a preempted ramp job restarts from zero.

Dry-run gate first, via the launcher's OVERRIDE hook (no config edit; the
evaluator takes `--override` dotlists), to confirm the compatibility check
passes, measure the write cadence per 50-step window (GCS writes from the
pre-#1534 python codec path are the risk; fall back to weka + copy if they
throttle), and check the cost of the means_50d reference: whether the target
is duplicated across the four members, and how much wall clock its small
one-step objects add (estimated minutes; drop save_reference there too if it
surprises). Then `./run-ace-evaluator.sh`.
