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

## Evaluation

Each fine-tuned arm is evaluated from `best_inference_ckpt.tar` of its stage-2 result dataset,
one GPU per run, three runs per arm (`run-ace-evaluator.sh`; the configs are shared by the two
source sets of an arm):

| scenario | what | IC | steps |
|---|---|---|---|
| `1pctco2` | the full 1pctCO2 ramp; years 0046-0125 are training years, 0131-0140 held out | 0001-01-02 | 51,050, four members |
| `picontrol` | the piControl tail of the 2026-09-29 round | 0311-01-01 | 14,600, four members |
| `randco2-4xco2` | the held-out random-CO2 member ic_0003 at 4xCO2 (only its year 0155 was seen, as validation) | 0152-10-02 | 1,916, four members |

The `1pctco2` and `picontrol` runs carry the 2026-09-29 round's writers (`daily_fields`,
`means_50d`) and aggregators and write to
`gs://vcm-ml-intermediate/2026-10-06-ace2s-snow-scenario-train/<arm>-<scenario>/`; the
ENSO and IPO indices are off on the ramp, which crashes their summary plots at year 0001 on
this branch (#1439).

The random-CO2 member is the one test with truth where CO2 moves independently of SST (305 to
4,471 ppmv here): a model using CO2 as a stand-in for the warming, which training on piControl
and 1pctCO2 alone allows, is right on the ramp but not there. It logs to W&B only.

## Launch

```
./run-ace-train.sh cm4-control-pic-1pct cm4-masked-naive-pic-1pct
./run-ace-train.sh cm4-control-pic-1pct-randco2 cm4-masked-naive-pic-1pct-randco2
./run-ace-finetune.sh ARM STAGE1_RESULT_DATASET [...]
./run-ace-evaluator.sh [JOB_NAME_FILTER]
```

8 GPUs on jupiter (4 on titan with `CLUSTER=ai2/titan`), W&B group `ace2s-snow-scenario-train`,
job names `ace2s-snowscen-<arm>-daily-{1-step-pretrain,multi-step-finetune}-rs0`.

## Launches

Stage 1, launched 2026-10-07 on jupiter. Three were preempted and resumed from checkpoint in the
same experiment, so the checkpoint dataset is the result of the experiment's last job.

| job | Beaker experiment | stage-1 checkpoint dataset |
|---|---|---|
| `ace2s-snowscen-cm4-control-pic-1pct-daily-1-step-pretrain-rs0` | `01M4CABX1QTY7AFC0GHZR8MQHC` | `01M4CABX213EN330KTW7RDK5TE` |
| `ace2s-snowscen-cm4-masked-naive-pic-1pct-daily-1-step-pretrain-rs0` | `01M4CAC40BC1KFBXBYEPKE0F92` | `01M4GZPB8EDF7PHJWF1M26RA3S` |
| `ace2s-snowscen-cm4-control-pic-1pct-randco2-daily-1-step-pretrain-rs0` | `01M4CACBZGWTA20WT3R77MM99M` | `01M4GZPB4D79PCPNR95YDQAA7N` |
| `ace2s-snowscen-cm4-masked-naive-pic-1pct-randco2-daily-1-step-pretrain-rs0` | `01M4CACHVD9Z4HEJQS6HNQ6C63` | `01M4KRHD8JVPCJ8PEN6A5GGVD8` |

Stage 2, launched 2026-10-09 to 2026-10-11 on titan (`CLUSTER=ai2/titan`, 4 GPUs), each from the
`best_ckpt.tar` of its stage-1 checkpoint dataset above:

| job | Beaker experiment | stage-2 checkpoint dataset | `best_inference_ckpt` epoch |
|---|---|---|---|
| `ace2s-snowscen-cm4-control-pic-1pct-daily-multi-step-finetune-rs0` | `01M4HBABBR9ZKPWHSNFJ2VV290` | `01M4RE0QHT2DE8V0QRCEWM43VA` | 1 of 50 |
| `ace2s-snowscen-cm4-masked-naive-pic-1pct-daily-multi-step-finetune-rs0` | `01M4J32ED22Q6DXTHF62F2CPR4` | `01M4J32EDGBPFNG7W8PSANBFQN` | 36 of 50 |
| `ace2s-snowscen-cm4-control-pic-1pct-randco2-daily-multi-step-finetune-rs0` | `01M4J32RB5KPZTNSV6NS1MPEX3` | `01M4J32RBKK133V679PS5X2CWC` | 4 of 45 |
| `ace2s-snowscen-cm4-masked-naive-pic-1pct-randco2-daily-multi-step-finetune-rs0` | `01M4PP9PA6ZG34HTHPQ3YANRKQ` | `01M4RZQ6E9Z8TZ8DF9BVG88E5K` | 21, stopped at 21 of 45 |

The control pic-1pct and masked-naive randco2 fine-tunes were preempted once and resumed; their
checkpoint datasets are the last job's. The masked-naive randco2 fine-tune was stopped after its
inline error had been flat for 15 epochs. The control randco2 inline rollouts drifted at epoch 6
(both scenarios, upper troposphere and stratosphere) and never recovered, so its
`best_inference_ckpt` is from before that.
