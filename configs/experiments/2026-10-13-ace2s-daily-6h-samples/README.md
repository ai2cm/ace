# ACE2S daily step trained on 6-hourly samples (2026-10-13)

Tests one of two hypotheses for why the daily-step ACE2S models trail the
6-hourly ACE2-ERA5 and ACE2S-6h models in weather skill
([reports#126](https://github.com/ai2cm/reports/pull/126)): the daily models
train on a quarter of the samples. The daily training store is the 6-hourly
store coarsened offline, so it holds one sample per day, all starting at 00Z;
the 6-hourly models see four samples per day.

The other hypothesis, that the daily model is capacity-limited, is covered by
Jeremy's `embed_dim: 1024` arm (`1deg-daily-ace2s-embed1024-pretrain-rs0`,
beaker 01M4H1Z49PK978RR8F50MGEMYQ, started 2026-10-13) and `embed_dim: 384`
arm (`1deg-daily-ace2s-n384-pretrain-rs0`, 01M4RK65Z94Z6D6XW6ZSZKW0QB) in
`configs/baselines/era5-ace2s-ablations/`. His `6hourly-quarter-data` arm
(`1deg-6h-ace2s-quarter-data-pretrain-rs0`, 01M4V2JMV2MX246GY1TJD27ZHK) tests
the sample-count hypothesis from the other side, a 6-hourly model on 18 of
the 70 training years.

## The arm

A daily-step model whose training samples start at every 6-hourly time. The
6-hourly 1° store `2026-08-13-era5-1deg-8layer-1940-2025.zarr` is read
through `TimeCoarsenDatasetConfig` (`fme.core.dataset.time_coarsen`, added
for this arm) with `factor: 4`: each sample is 4 x (1 + n_forward_steps)
consecutive 6-hourly steps; snapshot variables are taken at the end of each
24-hour window and window variables (fluxes, `*_mean`, CO2) are averaged over
it, with the coarse time coordinate the window's end. These are the
conventions `scripts/data_process/time_coarsen.py` used to build the daily
stores from this very store, so the quarter of the samples whose windows end
at 00Z are the daily arms' samples, and the other three quarters start at
06Z, 12Z and 18Z. Verified on 2000-06-15 and on a 1996-03-02 sample loaded
through the wrapper: all 67 input/output fields equal the daily stores'
values exactly (the 2026-08-13 daily store for 64 fields, the 2026-07-24
daily store for PRMSL and the surface stresses; CO2 to float32 rounding).

Everything else is the frozen seed-0 paper recipe of the ablation suite
(`ace2s-pretrain-rs1.yaml` with `seed: 0`, as Jeremy's one-knob arms):
validation on the daily stores (1996–1997, 00Z), 5-year inline inference and
the 81-year rollout on the daily stores, the 2026-08-21 merged daily stats
for normalization, `EnsembleLoss` 0.9/0.1 on 2 members, 40 epochs, 8 GPUs.
`diff <(grep -v '^#' ../../baselines/era5-ace2s-ablations/ace2s-pretrain-rs1.yaml) <(grep -v '^#' ace2s-pretrain-daily-6hsamples.yaml)`
shows the seed and the `train_loader.dataset` block as the only changes.

| config | stage | job name |
|---|---|---|
| `ace2s-pretrain-daily-6hsamples.yaml` | 40-epoch 1-step pretrain | `1deg-daily-ace2s-6hsamples-pretrain-rs0` |
| `ace2s-finetune-daily-6hsamples.yaml` | 10-epoch 3-step detached fine-tune (staged: fill `PRETRAIN_RESULT_DATASET`) | `1deg-daily-ace2s-6hsamples-ft3-detached-rs0` |

Per epoch the arm sees 4x the daily arms' samples (about 102k instead of
25.6k at batch size 8, the same number of optimizer steps per epoch as the
6-hourly arm), so the 40-epoch pretrain costs about 4x a daily pretrain. If
the extra samples are what the daily models lack, this arm's weather skill
should close on ACE2S-6h's; if not, the gap is in the daily step itself.

## Data

The 6-hourly 2026-08-13 store was only on GCS; it was copied to
`/climate-default/2026-08-13-era5-1deg-8layer-1940-2025.zarr` on 2026-10-13
with `scripts/data_process/gcs_to_weka.sh` (beaker
01M4VRK7BGWJGW9CYAHRB5WNFB, 3.4 TB).

## Launch

`./run-train.sh [pretrain|finetune]` from this directory: a copy of the
ablation suite's launcher with the wandb identity set to `troya` (same gantry
flags: workspace `ai2/ace`, priority normal, 8 GPUs on jupiter,
`--min-runtime 8h`).
