# ACE2S CM4 snow: training on piControl + 1pctCO2 (+ random CO2) with CO2 as input

The CM4 snow arms of `2026-09-17-ace2s-snow-inline-metrics-train` were trained on piControl
only and without `carbon_dioxide`, so under forcing they feel CO2 only through the prescribed
ocean surface, and their CO2 and SST sensitivities cannot be separated. These arms follow the
team's CM4 scenario recipe (`configs/experiments/cm4_randco2_ic1and2_1pct_46to125_piC_156to235/atmos_ace2s_origLR/train-config.yaml`
on `exper/2026-07-16-jamesd`), transposed to the daily step and to the masked snow channels:
train on a concatenation of 1pctCO2, piControl and optionally the CM4-like-AM4 random-CO2
ensemble, with `carbon_dioxide` as an input and next-step forcing, and validate on held-out
windows of each scenario. Architecture, loss and optimizer are those of the 2026-09-17 arms.

## Arms

| config prefix | source set | years | snow channels |
|---|---|---|---|
| `cm4-control-pic-1pct` | piControl + 1pctCO2 | 160 | no |
| `cm4-masked-naive-pic-1pct` | piControl + 1pctCO2 | 160 | masked SWE prognostic, cover diagnostic |
| `cm4-control-pic-1pct-randco2` | + random-CO2 ensemble | 190 | no |
| `cm4-masked-naive-pic-1pct-randco2` | + random-CO2 ensemble | 190 | masked SWE prognostic, cover diagnostic |

Each has a `-1-step-pretrain-daily.yaml` (stage 1; 50 epochs at 160 years, 45 at 190, both
about 370k iterations) and a `-multi-step-finetune-daily.yaml` (stage 2: steps 1-5, last step
optimized, stepper from the stage-1 checkpoint). One seed each.

What the random-CO2 ensemble adds: in piControl and 1pctCO2 CO2 and SST move together, so the
model can attribute the warming to either. In the ensemble (AM4 with prescribed SST, nine members
of about five years) the SST ramps by about 1 K per year identically in every member while CO2
jumps to a new random value every two weeks or so (1x members 75-1101 ppmv, 2x 142-2205,
4x 305-4471), the only data that separates the two sensitivities.

## Data

All daily one-degree stores come from the 2026-06-19 processing of the CM4 runs, coarsened from
6-hourly with the configs on branch `scripts/cm4-scenario-1deg-daily-datasets`. The masked snow
channels are in companion stores attached with `merge`, built on branch
`scripts/snow-masked-channels-per-land-area` (`snow_masked_channels/`) with the shared
`snow_mask.nc`.

| member | weka (`/climate-default`) | used for |
|---|---|---|
| 1pctCO2, years 0001-0141 | `2026-06-19-CM4-1pctCO2-atmosphere-land-1deg-8layer-140yr-daily.zarr` (+ `-land-snow-masked.zarr`) | training 0046-0125; validation 0041-0045 and 0126-0130; holdout 0131-0140 |
| piControl, years 0151-0351 | `2026-06-19-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily.zarr` (+ `-land-snow-masked.zarr`) | training 0156-0235; validation 0236-0240; holdout 0241 on |
| random CO2, nine members | `2026-06-19-CM4-like-AM4-random-CO2-1deg-daily/random-CO2-<1x,2x,4x>CO2-ic_<0001-0003>.zarr` (+ `-land-snow-masked.zarr`) | training ic_0001 and ic_0002 from 0153; validation ic_0003 year 0155 (weight 1/3 per CO2 level); rest of ic_0003 held out |

Normalization stats are pooled over exactly the training windows of the source set
(`snow_masked_channels/pool_daily_stats.py`, the masked snow entries from
`fit_masked_snow_stats.py --pool`), which is what gives `carbon_dioxide` a usable standard
deviation. Each Beaker dataset also holds `time-mean-piControl.nc` and `time-mean-1pctCO2.nc`,
the per-scenario daily time means used as inline-inference references:

| arms | stats dataset (mounted at `/statsdata`) |
|---|---|
| control, `pic-1pct` | `brianhenn/2026-10-07-cm4-pic-1pct-daily-stats` |
| treatment, `pic-1pct` | `brianhenn/2026-10-07-cm4-pic-1pct-daily-land-snow-masked-stats` |
| control, `pic-1pct-randco2` | `brianhenn/2026-10-07-cm4-pic-1pct-randco2-daily-stats` |
| treatment, `pic-1pct-randco2` | `brianhenn/2026-10-07-cm4-pic-1pct-randco2-daily-land-snow-masked-stats` |

## Inline inference

Two named five-year entries, eight initial conditions each, with the `anomaly_memory` (and for
the treatment `snow_season`) aggregators of the 2026-09-17 arms:

- `inference`, piControl, from 0151-01-02, 0152-0154 and 0236-0239 (either side of the training
  years; keeps the W&B keys of the earlier arms);
- `inference_1pct`, 1pctCO2, from 0041-0044 and 0123-0126 (never entering the 0131-0140 holdout).

## Launch

```
./run-ace-train.sh cm4-control-pic-1pct cm4-masked-naive-pic-1pct
./run-ace-train.sh cm4-control-pic-1pct-randco2 cm4-masked-naive-pic-1pct-randco2
./run-ace-finetune.sh ARM STAGE1_RESULT_DATASET [...]
```

8 GPUs on jupiter (4 on titan with `CLUSTER=ai2/titan`), W&B group `ace2s-snow-scenario-train`,
job names `ace2s-snowscen-<arm>-daily-{1-step-pretrain,multi-step-finetune}-rs0`.

## Launches

| job | Beaker experiment | stage-1 checkpoint dataset |
|---|---|---|
| | | |
