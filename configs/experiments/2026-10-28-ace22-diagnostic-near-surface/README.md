# ACE2.2-ERA5 with diagnostic near-surface fields

ACE2.2-ERA5's training stages 1 and 2 with TMP2m, Q2m, UGRD10m and VGRD10m diagnostic instead of
prognostic, as in the ACE2.5 recipe: outputs of the main network, not inputs. Two arms, seeds 0-3
each, are compared with ACE2.2-ERA5's four stage-2 seeds:

| arm | stage 1 | change from ACE2.2-ERA5 |
|---|---|---|
| `ace22-diagnostic-near-surface` | `train.yaml` | near-surface fields diagnostic |
| `ace22-no-gmr-diagnostic-near-surface` | `train-no-gmr.yaml` | near-surface fields diagnostic, no global mean removal |

ACE2.2-ERA5 is ai2cm/ACE2.1-ERA5-AIMIP `ace2.2-era5` at `8e8d1af`. Stage 2 (`multi-step-ft.yaml`,
unchanged) fine-tunes from stage 1's `best_ckpt.tar` and takes its stepper config from that
checkpoint. Each seed's stage-2 `best_inference_ckpt.tar` is scored as ACE2.2-ERA5's seeds were:

- the 1979-2014 rollout (`long36-evaluator-config.yaml`, copied unchanged from
  `../2026-10-23-ace22-no-gmr/`);
- the control and +4 K SST pair (`inference-{control,p4k}.yaml`, the same configs with only the
  file labels' model token changed), initial condition 1978-09-30 and inference seed 1.

`clip_latent_global_means` stays on. It acts only at inference, so each scoring job also runs on a
copy of the checkpoint with the clip disabled (`clip_off.py`).

Constraints:

- Nothing may train on, validate on, or be selected using 2015 onward (AIMIP holdout). Stage 1
  keeps ACE2.2-ERA5's three zero-weight inline rollouts that reach the holdout (`10year` from
  2015, `long_46year` through 2024, `weather_2024`); they are ignored. The +4 K pair runs through
  2024 like ACE2.2-ERA5's, and is scored over 1979-2014 for any decision.
- All jobs run at ace `394e41b5e` with image `oliverwm/fme-deps-only-c82095065`, as ACE2.2-ERA5
  did.

Launch with `bash run.sh`. It is safe to rerun: each run launches whatever stage has become ready.
