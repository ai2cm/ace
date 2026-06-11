# Code Changes on experiment/2026-06-05-aimip-like

Tracking commits made on this experimental branch for later extraction into PRs.

| Commit | Description | Files |
|--------|-------------|-------|
| 7c39253c5 | Add `ensemble_metrics` option to `TrainAggregatorConfig` — logs CRPS, SSR bias, and ensemble mean RMSE during training when enabled. Off by default. | `fme/ace/aggregator/train.py`, `fme/ace/aggregator/test_train.py` |
| ca802e837 | Add combined labels+residual+lr-tuning config with `ensemble_metrics: true`. | `configs/baselines/aimip-like/train-4deg-daily-v1-labels-residual-lr-tuning.yaml` |
| e2bde9577 | Add labels+lr-tuning config (no residual prediction). | `configs/baselines/aimip-like/train-4deg-daily-v1-labels-lr-tuning.yaml` |
| 015c8e9e7 | Add 10 experiment configs: 384/256 embed_dim variants, seed replicates. | `configs/baselines/aimip-like/train-4deg-daily-v1-{era5-only,labels}-{384,256,rs1}*.yaml` |
