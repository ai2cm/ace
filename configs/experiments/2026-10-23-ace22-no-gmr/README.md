# ACE2.2-ERA5 without global mean removal

ACE2.2-ERA5's training stages 1 and 2, retrained with global mean removal (GMR) off, to test what
GMR contributes in ACE2.2's own setup. The model is compared with ACE2.2-ERA5's four stage-2 seeds.

The only change from ACE2.2-ERA5 (ai2cm/ACE2.1-ERA5-AIMIP `ace2.2-era5` at `8e8d1af`) is
the deleted `global_mean_removal` block in `train.yaml`. Stage 2 takes its stepper config from the
stage-1 checkpoint. `clip_latent_global_means` stays on. It only acts at inference, so the trained
weights are the same with or without it. The stage-2 checkpoint is evaluated both as trained and
with the clip disabled, so the one run separates GMR's effect from the clip's.

| stage | config | notes |
|---|---|---|
| 1 | `train.yaml` | 40 epochs, 1-step |
| 2 | `multi-step-ft.yaml` | 40 epochs, multi-step, from stage 1's `best_ckpt.tar` |

Constraints:

- Nothing may train on, validate on, or be selected using 2015 onward (AIMIP holdout). Stage 1
  keeps ACE2.2-ERA5's three zero-weight inline rollouts that reach the holdout (`10year` from
  2015, `long_46year` through 2024, `weather_2024`); they are ignored, and the long rollout is
  read only through 2014. Stage 2 has none. Checkpoints and the
  comparison with ACE2.2-ERA5 are decided on 1979-2014. The 2015-2024 response is examined only
  after that.
- Both stages run at ace `394e41b5e` with image `oliverwm/fme-deps-only-c82095065`, as ACE2.2-ERA5
  did.

`run-long36.sh` scores a stage-2 checkpoint on the 1979-2014 rollout used for ACE2.2-ERA5's
seeds (`long36-evaluator-config.yaml`, copied unchanged from
`../2026-08-28-ace22-variants/long36-backfill/`). While stage 2 is still running, first copy
the checkpoint out with `snapshot-ckpt.sh`, since Beaker will not mount a running job's results.

Launch with `bash run.sh`. It is safe to rerun: each run launches whatever stage has become ready.
