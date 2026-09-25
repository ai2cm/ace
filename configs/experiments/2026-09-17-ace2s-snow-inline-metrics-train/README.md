# ACE2S snow inline-metrics training, 1-deg daily

1-step pre-training of the control and masked-naive snow-prognostic arms on CM4 and ERA5, with
the `anomaly_memory` and `snow_season` aggregators in the inline inference, so that the
snowpack metrics developed in the snow-memory prototype are logged from the first epoch and their
training evolution can be seen.

Four runs, otherwise matching the originals they re-create (2026-08-04 CM4 control, ERA5 daily
baseline, 2026-08-12 masked-naive arms): same architecture, loss, epochs, learning rate, data
splits and seed. The arms are test material for the evaluation methods, not final formulations.

| run | data | arm | recreates |
|---|---|---|---|
| cm4-control | CM4 piControl daily | no prognostic snow | `exp/ace2s-cm4-piControl-train` daily control |
| cm4-masked-naive | CM4 piControl daily + `-land-snow-masked` channels (merged) | prognostic snow, mean/std scaling | `2026-08-12-ace2s-snow-masked-daily` cm4 masked-naive |
| era5-control | ERA5 daily (2026-08-07 store) | no prognostic snow | `config/ace2s-era5-daily-baseline` daily control |
| era5-masked-naive | ERA5 daily + `-land-snow-masked` channels (merged) | prognostic snow, mean/std scaling | `2026-08-12-ace2s-snow-masked-daily` era5 masked-naive |
| cm4-masked-naive-soil-temperature | as cm4-masked-naive | prognostic snow plus prognostic top-layer soil temperature (`temperature_of_soil_layer_0`, masked over ocean by the store's `mask_` variable; stats already in the masked-snow stats dataset) | new (2026-09-25) |

## What differs from the original runs

- **Inline metrics.** `anomaly_memory` on `USWRFsfc`, `TMP2m`, `surface_temperature` and the two
  masked snow channels (treatments only), lags 0/1/3/7/14/30, scalars at 7 and 14, extended cold
  season per hemisphere, over the seven snow boxes plus everything north of 40S. `snow_season` on
  snow amount over the seven boxes (treatments only; the controls have no snow channel).
- **ERA5 initial conditions** for the inline rollout are 1 January of 1998 through 2005, one per
  year, instead of eight starts within 1996. Training uses 1940-1995, 2011-2019 and 2021 onward,
  validation 1996-1997, so all eight 5-year rollouts stay inside the held-out 1998-2010 block and
  each sees a different sequence of observed SSTs.
- **ERA5 control store and stats.** The control runs on the 2026-08-07 store the treatments use
  (bit-identical to the 2026-07-24 store for its variables) and takes its normalization from the
  masked-naive stats dataset, whose files carry the control's variables as well, so the two ERA5
  arms share normalization for every common variable. The original control used the 2026-07-24
  store with its own stats.
- `save_per_epoch_diagnostics: true` on the controls, as the treatments already had.

## Branch

`exp/ace2s-snow-inline-metrics-train`, created from `exp/ace2s-snow-memory-inline-check`: the
`exp/ace2s-snow-prognostic-daily` code base (main as of 2026-07-31 plus the snow work) with the
two aggregators cherry-picked on. The controls are trained from the same code as the treatments.

## Corrected snow channels (2026-09-21)

The first masked-naive runs (`ace2s-snowmetrics-{cm4,era5}-daily-masked-naive-1-step-pretrain-rs0`,
finished 2026-09-21) used the 2026-08-12 masked stores, in which ERA5 snow amount and cover are
per unit **cell** area (diluted by land fraction in coastal cells) while CM4's are per unit
**land** area, and CM4 cover is in percent. Those runs stay as the record of that definition and
are superseded. The masked-naive configs now read the `-land-snow-masked` stores and stats
(`scripts/data_process/snow_masked_channels/` on `scripts/snow-masked-channels-per-land-area`):
both datasets per unit land area, cover as a fraction in [0, 1]; ERA5 divided by land fraction
with cover clipped at 1, CM4 cover divided by 100. The relaunched runs carry `-land-snow` in
their names. The controls are unaffected (no snow channels; their normalization entries are
identical in the old and new stats datasets) and are not relaunched.

## Multi-step fine-tuning

`{cm4,era5}-control-multi-step-finetune-daily.yaml` fine-tune the finished controls with the
ERA5 daily baseline's recipe (`configs/baselines/era5/ace-train-config-multi-step-finetuning-daily.yaml`
on `exp/ace2s-era5-daily-finetune`): weights and stepper from `training_checkpoints/best_ckpt.tar`
of the 1-step run, mounted at `/weights`; forward steps drawn from {1: 0.6, 2: 0.2, 3: 0.1,
4: 0.05, 5: 0.05} days with the loss on the last step only; validation batch 32; same epochs,
learning rate, loss, data splits and inline metrics as the 1-step configs. The checkpoint dataset
is not in the config: `run-ace-finetune.sh` takes it as an argument and mounts it, so the
committed files describe every launch and the Beaker job records which checkpoint it used.

| fine-tune launch | arm | stage-1 result dataset (mounted at /weights) | job | status |
|---|---|---|---|---|
| 2026-09-21 | cm4-control | `01M2VVJ5A75WKXQXJVS4XEMT4Y` (resumed job of `01M2TZC6YYH9TVRTQ7BHPH250K`) | `ace2s-snowmetrics-cm4-daily-control-multi-step-finetune-rs0` | stopped 2026-09-23 after epoch 24 of 50; best-inference epoch 16 |
| 2026-09-21 | era5-control | `01M2TZCJAT224Z4KBKJFJB8TGQ` | `ace2s-snowmetrics-era5-daily-control-multi-step-finetune-rs0` | stopped 2026-09-23 after epoch 31 of 40; best-inference epoch 13 |
| 2026-09-23 | era5-masked-naive | `01M33R5QR1GJSNNRVRJ1A8MW36` (relaunched 1-step run, per-land-area channels) | `ace2s-snowmetrics-era5-daily-masked-naive-land-snow-multi-step-finetune-rs0` | finished 2026-09-24, 40 epochs |
| 2026-09-23 | cm4-masked-naive | `01M33R5HC7EP6N8KAWJNAZ8XMA` (relaunched 1-step run `01M33R5HBHR5XQWPFZ7JSHF4TW`, per-land-area channels) | `ace2s-snowmetrics-cm4-daily-masked-naive-land-snow-multi-step-finetune-rs0` | preempted 2026-09-25 after epoch 16 (workspace-group allocation balance), auto-resumed from the epoch-16 checkpoint as job `01M3BZWMXR4A6QFCM6NRXCPBSP` |

The control fine-tunes were stopped deliberately to free cluster slots for the treatment
fine-tunes: their best-inference metric (`inference/time_mean_norm/rmse/channel_mean`) had not
improved for 8 (CM4) and 18 (ERA5) epochs, so the best-inference checkpoints, which downstream
work uses, were already set. Multi-step fine-tuning left every memory metric unchanged from the
1-step endpoints. Either run can be continued later from the `ckpt.tar` in its result dataset.

`{cm4,era5}-masked-naive-multi-step-finetune-daily.yaml` apply the same recipe to the masked-naive
1-step configs (per-land-area channels merged from the `-land-snow-masked` stores, their stats
mount, the same inline metrics); the stepper, including input masking, comes from the checkpoint.

## Soil-temperature arm (2026-09-25)

The temperature-persistence and slow-mode rounds (explore2 `2026-09-23-snow-temperature-persistence`,
`2026-09-24-snow-slow-mode-carrier`) found that under snow the target's 2 m temperature keeps a
slow, multi-day memory that the treatment lacks, and that in CM4 the carrier is the thermal
state of the top soil layer under the pack, which ACE does not carry. `cm4-masked-naive-soil-temperature`
is the cm4-masked-naive config with `temperature_of_soil_layer_0` added as a prognostic channel
(in and out names) and to the `anomaly_memory` variables. It tests whether ACE can hold and use a
slow reservoir: judged on the soil state's own anomaly memory and time-mean drift, the
within-regime conditional persistence of T2m in the Great Plains (offline, from a rollout), the
cold-season T2m bias and variability under snow, and the one-step skill of everything else.
Launch: `bash run-ace-train.sh cm4-masked-naive-soil-temperature`.

## Launch

Stage 1, all four arms or a subset (`launch-common.sh` holds the shared gantry call):

```bash
./run-ace-train.sh                                   # all four, ai2/jupiter (8 GPUs)
./run-ace-train.sh era5-masked-naive cm4-masked-naive
CLUSTER=ai2/titan ./run-ace-train.sh                 # 4 GPUs
```

Stage 2, from the result dataset of each finished stage-1 job:

```bash
./run-ace-finetune.sh cm4-control 01M2VVJ5A75WKXQXJVS4XEMT4Y era5-control 01M2TZCJAT224Z4KBKJFJB8TGQ
```

Both scripts validate each config with `fme.ace.validate_config` before submitting, and print the
gantry command instead of running it when `DRY_RUN=1`. Jobs are non-preemptible for their first
8 hours (`--min-runtime 8h`) and resume from their checkpoints if interrupted after that.
Launched so far: stage 1 of all four arms (2026-09-18, treatments on the per-cell-area stores),
stage 1 of both treatments on the corrected stores (2026-09-21), and the two control fine-tunes
(2026-09-21).
