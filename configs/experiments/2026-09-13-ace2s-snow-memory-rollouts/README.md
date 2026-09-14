# ACE2S snow-memory rollouts, 1-deg daily

One-time evaluator runs that save daily fields, to prototype anomaly-memory and snow→temperature
coupling metrics before implementing them as inline aggregators.

Motivation: time-mean bias is saturated by climatology and one-step (24 h) error is already won by
the snow arms over their controls, so neither can rank snow *formulations*. The deficit that
matters is documented in the `ace-eval` snow-albedo report: snow-free ACE has a diagnosed-albedo
memory of 1.8–4.1 d against data's 12.2–24.1 d (ERA5) / 4.0–11.7 d (CM4), and its albedo adds
essentially nothing beyond thermal autocorrelation to future temperature. Measuring that needs
per-gridcell daily time series, which no existing artifact provides — the zonal-mean diagnostic is
averaged over the sample dimension and time-coarsened, and training runs save no fields.

## Arms

| arm | checkpoint dataset | best-inference epoch |
|---|---|---|
| cm4-control | `01KZC1J3R3EW9YVM6HPNSNNNCY` | 43 |
| cm4-masked-naive | `01KZVBJZ8KHR9E84CEF0NF95ES` | 46 |
| cm4-masked-log1p | `01KZVBJZJ1E81EAF5BQJEVN8AH` | 48 |
| era5-control | `01KYX6AQTSXD3N23HP128TJYTC` | 34 |
| era5-masked-naive | `01KZVBA39HPP7ZNZ8FXD2HG9DR` | 39 |
| era5-masked-log1p | `01KZWD52T8QEQW3V08F9ENBE06` | 30 |

masked-naive is the best arm from the 2026-08-12 round; masked-log1p is the known-pathological one
(ERA5 rollout SWE RMSE 599 vs masked-naive's 26.8) and serves as a positive control — a memory
metric worth inlining has to flag it. Same best-inference epochs as the time-mean and one-step
reports.

## Initial conditions

Four ICs per arm, 1825 forward steps (~5 yr) each, non-overlapping:

- **cm4**: 0311, 0321, 0331, 0341 (Jan 1). All held out — training stops at 0306, validation covers
  0306–0311, and the store runs to 0351. Independent: separate stochastic realizations over
  non-overlapping segments of a trend-free piControl run.
- **era5**: 1998, 2003, 2008, 2013 (Jan 1). Deliberately spread across years, unlike the inline
  configs whose eight ICs all start within 1996 and therefore share SST forcing and a single target
  realization. ERA5 training used ≤1995, 2011–2019 and ≥2021, so the 2008 and 2013 ICs draw forcing
  from training years; mild for a free-running climate statistic, but worth stating.

## Output

Daily fields, written as zarr under
`gs://vcm-ml-experiments/2026-09-13-ace2s-snow-memory-rollouts/<arm>/` and cropped to `lat_extent
[-40, 90]`, which covers every snow region in the analysis including the Andes:

- `USWRFsfc`, `DSWRFsfc` — diagnosed albedo = USWRF/DSWRF, the control's only snow proxy and the
  predictor the eval report used.
- `DLWRFsfc`, `ULWRFsfc`, `SHTFLsfc`, `LHTFLsfc` — with the shortwave pair these close the surface
  energy budget, so melt-edge errors can be attributed to shortwave, longwave or turbulent exchange
  rather than assumed to be albedo. Latent flux over snow is also the sublimation mass sink, and
  sensible flux carries the insulation signature behind the rectified snow–temperature coupling.
- `total_frozen_precipitation_rate` — the snow source. Without it a too-short snow memory cannot be
  attributed between melting too fast and failing to accumulate, which imply opposite fixes.
- `TMP2m`, `surface_temperature` — the coupling target; their difference is the snow-insulation
  decoupling diagnostic.
- `surface_snow_amount_masked`, `surface_snow_area_fraction_masked` — treatment arms only.

The target is identical across arms of a dataset, so only the masked-naive arm sets
`save_reference: true` (it carries all six variables) and the other four write predictions only.
The ERA5 control runs against the 2026-07-24 store it was trained on rather than the 2026-08-07
store the treatments use; the two are bit-identical over these years for the variables saved here,
so the shared target applies to it as well. Total output ≈ 115 GB.

Writing to GCS requires the zarr file-writer and `log_to_file: false`. The expensive inference
sub-aggregators (power spectrum, zonal mean, annual, ENSO, IPO) are disabled so the jobs spend
their time writing data; the remaining scalar and time-mean metrics are kept as a sanity check that
each rollout behaves as expected.

All six configs live together here, including the controls, because only this branch has both the
per-field transforms needed to load the log1p checkpoints and the zarr file-writer — running the
controls from their own training branches would score them with different code than the treatments.

## Launch

```bash
./run-ace-evaluator.sh                    # all six, ai2/jupiter
CLUSTER=ai2/titan ./run-ace-evaluator.sh  # elsewhere
./run-ace-evaluator.sh cm4                # substring filter on the job name
```

Each invocation validates its config with `fme.ace.validate_config` before submitting.
