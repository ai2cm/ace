# Seed replicates of im4ecamc

`im4ecamc` (wandb) is Jeremy's 4-degree no-CO2 model with ACE2.2's recipe: shared global mean
removal, latent global-mean clipping, prognostic near-surface fields, AIMIP training split,
120 epochs in one stage. These are seeds 1-3 of it, for comparison with the ACE2S recipe's
seeds in `../2026-10-23-ace25-aimip-4deg/`.

`train.yaml` is `configs/baselines/aimip-like/train-4deg-daily-v2-era5-only-no-residual-no-co2.yaml`
from ai2cm/ace `d9faa9185`, unchanged; `run.sh` overrides only the seed. Each run uses that
commit and `im4ecamc`'s image. The in-sample trend is read from the epoch-120 `long_46year`
rollout restricted to 1979-2014, as for `im4ecamc`.

Constraints:

- Nothing may train on, validate on, or be selected using 2015 onward (AIMIP holdout). The
  config's inline rollouts starting in 2015 and 2024 have weight 0, as in `im4ecamc`, and are not
  used.

Launch with `bash run.sh`. It is safe to rerun.
