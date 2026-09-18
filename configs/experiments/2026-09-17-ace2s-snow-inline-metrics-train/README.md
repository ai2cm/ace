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
| cm4-masked-naive | CM4 piControl daily + masked snow sidecar | prognostic snow, mean/std scaling | `2026-08-12-ace2s-snow-masked-daily` cm4 masked-naive |
| era5-control | ERA5 daily (2026-08-07 store) | no prognostic snow | `config/ace2s-era5-daily-baseline` daily control |
| era5-masked-naive | ERA5 daily + masked snow sidecar | prognostic snow, mean/std scaling | `2026-08-12-ace2s-snow-masked-daily` era5 masked-naive |

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

## Launch

```bash
./run-ace-train.sh                    # all four, ai2/jupiter (8 GPUs)
CLUSTER=ai2/titan ./run-ace-train.sh  # 4 GPUs
```

Each invocation validates its config with `fme.ace.validate_config` before submitting. Jobs are
non-preemptible for their first 8 hours (`--min-runtime 8h`) and resume from their checkpoints
if interrupted after that.
