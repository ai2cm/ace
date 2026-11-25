# ACE2S CM4 snow: random-CO2 arms retrained on the corrected random-CO2 data

The random-CO2 ensemble used by `2026-10-06-ace2s-snow-scenario-train`
(`2026-06-19-CM4-like-AM4-random-CO2`) had misprescribed CO2 timing. The ensemble was re-run
(`gs://vcm-ml-intermediate/2026-10-11-CM4-like-AM4-random-CO2`): same nine members, time axis and
variables, CO2 now stepping cleanly every 30 days, and new weather. This round retrains the two
`pic-1pct-randco2` arms of the 2026-10-06 round on it with the identical recipe (architecture,
loss, optimizer, epochs, splits; see that round's README), now with two seeds each. It also trains
a second seed of the two `pic-1pct` arms with that round's configs, so both source sets have a
seed spread. All of them, and the two `pic-1pct` fine-tunes of that round, are evaluated on
all three held-out ic_0003 members.

The last ten years of 1pctCO2 (0131-0140) are the project's final holdout and nothing here reads
them. Relative to the 2026-10-06 configs, the `pic-1pct-randco2` configs stop `val_1pct_late` at
0130-12-31 and use `inference_1pct` initial conditions 0041-0044 and 0122-0125 (the last rollout
ends 0130-01-01), and the ramp evaluation stops at 0130-11-13. The `pic-1pct` configs keep the
2026-10-06 windows, so seed 1 matches seed 0 there; their validation and inline rollouts read
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
seeds; the launch scripts pass `seed=<seed>` as an override. All stage-1 configs set
`evaluate_before_training: true` (not set in the 2026-10-06 round), so every validation set and
inline rollout runs before the first iteration and a data problem fails a job within its first
hour; it does not change training beyond shifting the random-number stream.

Stage 2 runs to `max_epochs` (45 or 50), as in the 2026-10-06 round, whose masked-naive `pic-1pct`
fine-tune selected its `best_inference_ckpt.tar` at epoch 36. Early stopping on a plateau (stop
once the inline inference error, the metric that selects `best_inference_ckpt.tar`, has gone 5
epochs without a new best; `stop_plateaued_finetunes.py`) was planned and stopped the two
`pic-1pct` seed-1 fine-tunes on 2026-11-25 after epochs 8-9 (best inline error at epochs 3 and
4); it was then dropped so that the new fine-tunes reach the old ones' maturity, and those two
were resumed from their latest checkpoints.

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

Stage 1 and the first evaluations launched 2026-11-22 from commit `434357756`; each later
experiment records its commit. The checkpoint dataset is a stage-1 or stage-2 job's result
(filled in as jobs finish) or, for an evaluation, the checkpoint it reads.

| job | Beaker experiment | checkpoint dataset |
|---|---|---|
| `ace2s-snowfix-cm4-control-pic-1pct-randco2-daily-1-step-pretrain-rs0` | `01M83GR3FP6PY4SVF720DEKBZC` | `01M83GR3FYDRWYR3J4XG4T8M2V` |
| `ace2s-snowfix-cm4-control-pic-1pct-randco2-daily-1-step-pretrain-rs1` | `01M83GR95BQV7VZBD1GXW1DDZP` | `01M88KYMJGCM5NPKDGW9XW22QM` (1) |
| `ace2s-snowfix-cm4-masked-naive-pic-1pct-randco2-daily-1-step-pretrain-rs0` | `01M83GREVV9WVY7WV8K37GC2DC` | `01M83GREW1BB1QR1YEGR867XK4` |
| `ace2s-snowfix-cm4-masked-naive-pic-1pct-randco2-daily-1-step-pretrain-rs1` | `01M83GRMGE1Y2BK66XR4X7J0B0` | `01M83GRMGHFDCV6ZKTNK75FK33` |
| `ace2s-snowfix-cm4-control-pic-1pct-daily-1-step-pretrain-rs1` | `01M83GRTDJ767PEWMA5X2395YD` | `01M83GRTDSWVMXEJM0F10J8QN4` |
| `ace2s-snowfix-cm4-masked-naive-pic-1pct-daily-1-step-pretrain-rs1` | `01M83GS09Y2RKK1SW40Z6W7Z70` | `01M83GS0A7NFC52V59M6XDGZQN` |
| `ace2s-snowfix-cm4-control-pic-1pct-randco2-daily-multi-step-finetune-rs0` | `01M88FMZWZS7QX7Z64AG1KG4YK` |  |
| `ace2s-snowfix-cm4-control-pic-1pct-randco2-daily-multi-step-finetune-rs1` | `01M8933WVRQWGJKKEZKV6RG4RT` |  |
| `ace2s-snowfix-cm4-masked-naive-pic-1pct-randco2-daily-multi-step-finetune-rs0` | `01M88N6DX04BJ89A6MH1ZZCS70` |  (2) |
| `ace2s-snowfix-cm4-masked-naive-pic-1pct-randco2-daily-multi-step-finetune-rs1` | `01M88N7E62PQS4NRY8GGHC1F93` |  |
| `ace2s-snowfix-cm4-control-pic-1pct-daily-multi-step-finetune-rs1` | `01M88BXVZ03EHDCN1DBXYBBHKB` |  (3) |
| `ace2s-snowfix-cm4-masked-naive-pic-1pct-daily-multi-step-finetune-rs1` | `01M88BY243219XCJR63JY2TJ58` |  (3) |
| `ace2s-snowfix-cm4-control-pic-1pct-randco2-1xco2-eval` | `01M83H9SB1R4KXF15PMQTHJWY1` | `01M4RE0QHT2DE8V0QRCEWM43VA` (input) |
| `ace2s-snowfix-cm4-control-pic-1pct-randco2-2xco2-eval` | `01M83H9Z5YXXXHZAXHHX7YZSX3` | `01M4RE0QHT2DE8V0QRCEWM43VA` (input) |
| `ace2s-snowfix-cm4-control-pic-1pct-randco2-4xco2-eval` | `01M83HA4WTYC8QSNM5Q3H53ENR` | `01M4RE0QHT2DE8V0QRCEWM43VA` (input) |
| `ace2s-snowfix-cm4-masked-naive-pic-1pct-randco2-1xco2-eval` | `01M83HAB54XN2NZFSKFGCEW3M8` | `01M4J32EDGBPFNG7W8PSANBFQN` (input) |
| `ace2s-snowfix-cm4-masked-naive-pic-1pct-randco2-2xco2-eval` | `01M83HAHA26Z093VJ5D5P30MQS` | `01M4J32EDGBPFNG7W8PSANBFQN` (input) |
| `ace2s-snowfix-cm4-masked-naive-pic-1pct-randco2-4xco2-eval` | `01M83HAQAH5N2G9XJFTPEBK7WK` | `01M4J32EDGBPFNG7W8PSANBFQN` (input) |

(1) Preempted in epoch 44 on 2026-11-24 (job `01M85H9MB0AY2KHVJS61KXG38Y`; its automatic retry failed
to start) and resumed from its latest checkpoint; the dataset is the resumed job's result.
(2) Crashed in epoch 3 on 2026-11-25 (CUDA/NCCL abort) and resumed (job `01M89KGZ8J80D93C8H5QGC2MZG`).
(3) Stopped by the plateau rule after epochs 8-9 on 2026-11-25 and resumed the same day (see Runs).

### Epoch-1 evaluations (2026-11-24/25)

A quick round before the fine-tunes finish: each fine-tune's `best_inference_ckpt.tar` after its
first epoch was saved as its own dataset and evaluated on all five scenarios with no field writes,
under the run name `<run>-ep1` (so the ramp and piControl runs write to `<GCS root>/<run>-ep1-<scenario>`).
That is `run-ace-evaluator.sh` with `OVERRIDE="data_writer.files=[]" JOB_SUFFIX=""` and
`run_all <run>-ep1 <epoch-1 dataset>` in place of the listed runs. Analysis:
explore2 `brianh/2026-06-09-land-atm-coupling/2026-11-25-snow-excursion-seeds/`.

| run | epoch-1 dataset | `1pctco2` | `picontrol` | `randco2-{1x,2x,4x}co2` |
|---|---|---|---|---|
| `cm4-control-pic-1pct-rs1` | `01M88NNKWKQJXEYMY3YJDY5DB7` | `01M88PJ0H266DWBWMD59YX9WY7` | `01M88PJCX8PR0V7DDZDH1MJ6XH` | `01M88P241N7PZ0Z95PMNMT58WM`, `01M88P2A6RN7GXDTGHPGVME1A5`, `01M88P2G8GFY12DQ3XEC0AJ7RE` |
| `cm4-masked-naive-pic-1pct-rs1` | `01M88NNKYM6B020BRCVN31G0F2` | `01M88PJ6N69V40AKN13SVSM9BW` | `01M88PJJVJ8HX014HZ1VE5BWQB` | `01M88P32C3CQBKAYF8SPPDV90P`, `01M88P38JDHP9H2PSSVTESHE1H`, `01M88P3EAJ64WB2ZPDFFTRM4BM` |
| `cm4-control-pic-1pct-randco2-rs0` | `01M88TCFV7GJW4QR5TMC9J73Z4` | `01M88TK5C32STF9ZXHWG1F3RY8` | `01M88TKB37DTBQP5YKFYN24394` | `01M88TKH9H3DD1V4Z29PV2D8T2`, `01M88TKQ3R616SM5A66GH0847A`, `01M88TKWYP1VYQ3WNCEK293VMN` |
| `cm4-control-pic-1pct-randco2-rs1` | `01M89AZR1T1PFW8VEJZRCQVJQY` | `01M89B6PPFQAZEX2V4P6EMHGGD` | `01M89B6W757VC3NV5ZDT78Q5GC` | `01M89B71SPXJ1Y6X1GS7RNDHW5`, `01M89B787HR1KNKRXDQ7MWBW9Q`, `01M89B7E3RAH3AB16S2NRSFXEE` |
| `cm4-masked-naive-pic-1pct-randco2-rs0` | `01M88YKAEXTD23HBTS1NQACKH9` | `01M88YVXJ8FQZ1BC3P03G4K3FH` | `01M88YW40GA4AJWRGEXPCEDQM4` | `01M88YWA2NH2VTTF8E8MBZTY2R`, `01M88YWG5VJWYMSZV9JHNZ9BXT`, `01M88YWP8SC41T9RGRFZN4JSG4` |
| `cm4-masked-naive-pic-1pct-randco2-rs1` | `01M88YK5JMF8T8THKYXS3VK8QV` | `01M88YTD7QDQDEWQA5GTP5S6EN` | `01M88YTKTDT7A16ASBH5P6SPDM` | `01M88YTSZFK3TWG3ZSRZ2T930Z`, `01M88YV09ET4SD4QDR714T2KW8`, `01M88YV76DTTBH370SJ4X1PAYX` |

The first `1pctco2` and `picontrol` runs of the two `pic-1pct` seed-1 models wrote fields and were
stopped and deleted in favour of these.

A mid-ramp check of the same six checkpoints (30 years of 1pctCO2 from the CM4 state of 0041-01-01,
no field writes) is on branch `exp/ace2s-snow-midramp-check` (commit `d2671edac`,
`run-ace-midramp-check.sh`): experiments `01M8ABJQPZ9GEVYGFR50QS2927` (masked-naive `pic-1pct`
rs1), `01M8ABJX8XWY2A3KDZX4NSM2G7` (control `pic-1pct` rs1), `01M8ABK2RFCJTN5ZG67V5DVJV2`
(masked-naive randco2 rs0), `01M8ABK80JEZ3YX3CFZV2HC36F` (control randco2 rs0),
`01M8ABKD5Y1ZRBVKDS4CMPQEGV` (masked-naive randco2 rs1), `01M8ABKJWMFSC43CTHEANXB6HY` (control
randco2 rs1).
