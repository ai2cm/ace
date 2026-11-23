# ACE2S CM4 snow: random-CO2 arms retrained on the corrected random-CO2 data

The random-CO2 ensemble used by `2026-10-06-ace2s-snow-scenario-train`
(`2026-06-19-CM4-like-AM4-random-CO2`) had misprescribed CO2 timing. The ensemble was re-run
(`gs://vcm-ml-intermediate/2026-10-11-CM4-like-AM4-random-CO2`): same nine members, time axis and
variables, CO2 now stepping cleanly every 30 days, and new weather. This round retrains the two
`pic-1pct-randco2` arms of the 2026-10-06 round on it with the identical recipe (architecture,
loss, optimizer, epochs, splits; see that round's README), now with two seeds each. It also trains
a second seed of the two `pic-1pct` arms with that round's configs unchanged, so both source sets
have a seed spread. All of them, and the two `pic-1pct` fine-tunes of that round, are evaluated on
all three held-out ic_0003 members.

The last ten years of 1pctCO2 (0131-0140) are the project's final holdout and nothing here reads
them. Relative to the 2026-10-06 configs, the `pic-1pct-randco2` configs stop `val_1pct_late` at
0130-12-31 and use `inference_1pct` initial conditions 0041-0044 and 0122-0125 (the last rollout
ends 0130-01-01), and the ramp evaluation stops at 0130-11-13. The `pic-1pct` configs are copied
unchanged so that seed 1 matches the 2026-10-06 seed 0; their validation and inline rollouts read
0131-01-01 only.

## Runs

| run | source set | snow channels | seed |
|---|---|---|---|
| `cm4-control-pic-1pct-randco2-rs0`, `-rs1` | piControl + 1pctCO2 + random CO2 (190 years) | none | 0, 1 |
| `cm4-masked-naive-pic-1pct-randco2-rs0`, `-rs1` | piControl + 1pctCO2 + random CO2 (190 years) | masked SWE prognostic, cover diagnostic | 0, 1 |
| `cm4-control-pic-1pct-rs1` | piControl + 1pctCO2 (160 years) | none | 1 |
| `cm4-masked-naive-pic-1pct-rs1` | piControl + 1pctCO2 (160 years) | masked SWE prognostic, cover diagnostic | 1 |

Seed 0 of the two `pic-1pct` arms is the 2026-10-06 round's.

Each arm has a `-1-step-pretrain-daily.yaml` (stage 1; 45 epochs on 190 years, 50 on 160) and a
`-multi-step-finetune-daily.yaml` (stage 2 from the stage-1 `best_ckpt.tar`), shared by its two
seeds; the launch scripts pass `seed=<seed>` as an override.

## Data

| member | weka (`/climate-default`) | used for |
|---|---|---|
| 1pctCO2, years 0001-0141 | `2026-06-19-CM4-1pctCO2-atmosphere-land-1deg-8layer-140yr-daily.zarr` (+ `-land-snow-masked.zarr`) | training 0046-0125; validation 0041-0045 and 0126-0130; holdout 0131-0140 |
| piControl, years 0151-0351 | `2026-06-19-CM4-piControl-atmosphere-land-1deg-8layer-200yr-daily.zarr` (+ `-land-snow-masked.zarr`) | training 0156-0235; validation 0236-0240 |
| random CO2, nine members | `2026-10-11-CM4-like-AM4-random-CO2-1deg-daily/random-CO2-<1x,2x,4x>CO2-ic_<0001-0003>.zarr` (+ `-land-snow-masked.zarr`) | training ic_0001 and ic_0002 from 0153; validation ic_0003 year 0155; rest of ic_0003 held out |

The daily random-CO2 stores are coarsened from the corrected 6-hourly members with the
`time_coarsen` block of `scripts/data_process/configs/CM4-like-AM4-random-CO2-ensemble-atmosphere-1deg-8layer.yaml`
on branch `scripts/randco2-2026-10-11-1deg-daily`; the snow sidecars and pooled stats come from
`snow_masked_channels/run_scenario_pipeline.sh` on `scripts/snow-masked-channels-per-land-area`.

| arms | stats dataset (mounted at `/statsdata`) |
|---|---|
| control, `pic-1pct-randco2` | `brianhenn/2026-11-22-cm4-pic-1pct-randco2-daily-stats` |
| masked-naive, `pic-1pct-randco2` | `brianhenn/2026-11-22-cm4-pic-1pct-randco2-daily-land-snow-masked-stats` |
| control, `pic-1pct` | `brianhenn/2026-10-07-cm4-pic-1pct-daily-stats` |
| masked-naive, `pic-1pct` | `brianhenn/2026-10-07-cm4-pic-1pct-daily-land-snow-masked-stats` |

## Evaluation

From `best_inference_ckpt.tar` of each stage-2 result dataset, one GPU per run
(`run-ace-evaluator.sh`):

| scenario | IC | steps | runs |
|---|---|---|---|
| `1pctco2` | 0001-01-02 | 47,400 (to 0130-11-13), four members | runs trained here |
| `picontrol` | 0311-01-01 | 14,600, four members | runs trained here |
| `randco2-{1x,2x,4x}co2` | 0152-10-02 | 1,916 (the whole ic_0003 member), four members | runs trained here and the 2026-10-06 `pic-1pct` fine-tunes (`01M4RE0QHT2DE8V0QRCEWM43VA` control, `01M4J32EDGBPFNG7W8PSANBFQN` masked-naive) |

The `1pctco2` and `picontrol` runs write to
`gs://vcm-ml-intermediate/2026-11-23-ace2s-snow-randco2-retrain/<run>-<scenario>/`; the random-CO2
runs log to W&B only.

## Launch

```
./run-ace-train.sh [RUN ...]
./run-ace-finetune.sh RUN STAGE1_RESULT_DATASET [...]
./run-ace-evaluator.sh [JOB_NAME_FILTER]
```

8 GPUs on jupiter (4 on titan with `CLUSTER=ai2/titan`), W&B group `ace2s-snow-randco2-retrain`,
job names `ace2s-snowfix-<arm>-daily-{1-step-pretrain,multi-step-finetune}-rs<seed>` and
`ace2s-snowfix-<run>-<scenario>-eval`.

## Launches

| job | Beaker experiment | checkpoint dataset |
|---|---|---|
