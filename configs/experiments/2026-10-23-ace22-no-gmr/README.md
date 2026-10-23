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

- Nothing may train on, validate on, or be selected using 2015 onward (AIMIP holdout). Stage 1's
  zero-weight 2015 inline rollout is look-only, as in ACE2.2-ERA5. Checkpoints and the
  comparison with ACE2.2-ERA5 are decided on 1979-2014. The 2015-2024 response is examined only
  after that.
- Both stages run at ace `394e41b5e` with image `oliverwm/fme-deps-only-c82095065`, as ACE2.2-ERA5
  did.

Launch with `bash run.sh`. It is safe to rerun: each run launches whatever stage has become ready.
