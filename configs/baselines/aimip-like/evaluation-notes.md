# Evaluation Notes and Experiment Plan

## Operational constraints

- Workspace `ai2/climate-titan` is capped at 32 experiments. Monitor usage with `beaker workspace experiments ai2/climate-titan`.
- Default active job target: 4 at urgent priority.
- If no high/urgent jobs are queued and waiting to run, increase active job target to 8.
- Unpromising runs can be lowered to `low` priority instead of stopped — low priority jobs don't count toward the active job target.

## Metrics that matter

**46-year annual mean temperature evolution** (air_temperature_7, surface_temperature): The most important climate metric. We want the model to track observed warming trends without drifting. The reference run (ww3nugpp) drifts cold by ~0.5 K in air_temperature_7 and ~0.3 K in surface_temperature over 46 years, with the bias growing in the last decade. Most of the 46-year period is in-sample (training covers 1979-2013 minus 1994), so poor trend-tracking in the early period is a model quality issue, not an extrapolation problem. Out-of-sample performance (2015-2024) is less tractable since the model may not generalize to higher CO2/temperatures.

**Inference mean_abs_norm_bias**: Measures percentage error in spectral power averaged across spatial scales. This is a harder target than time-mean RMSE and catches problems like missing small-scale precipitation structure. High PRATEsfc values here indicate the model isn't generating fine-scale rain patterns.

**Short-range weather skill (day 5 CRPS/RMSE)**: Good semi-out-of-sample metric since training uses 1-step targets but weather evaluation runs 5 steps forward. Comparable skill between 1994 and 2024 periods indicates the model generalizes across time.

**Train/val loss gap**: Proxy for overfitting severity. The reference run's gap grew from 0.004 to 0.038 over 60 epochs, with val loss nearly plateauing after epoch 45. Overfitting likely degrades trend-tracking: some models show better climate trends early in training when the model hasn't yet memorized the training distribution. Reducing overfitting is probably a prerequisite for better trend performance.

**SSR bias**: The reference run has negative SSR bias on 162/183 variables, meaning the ensemble is too spread (underconfident). This is likely because the model overfits to ERA5 — it becomes too confident on training samples, making the noise injection produce excessive spread relative to actual forecast error. Adding SSR bias to the train aggregator would confirm this hypothesis (train SSR bias should stay near 0 or go positive even as val SSR goes negative).

**46-year annual mean evolution across epochs**: Looking at the air_temperature_7 (and surface_temperature) annual curves at different training epochs reveals whether drift is a structural bias or an overfitting artifact. If the trend degrades with more training, that implicates overfitting. If it's bad from the start, the issue is architectural or data-related. Also reveals spin-up behavior: how many simulated years the model needs before it settles into the correct interannual variability amplitude. A model that takes 10 years to spin up is less useful than one that locks in after 1-2 years.

**10-year inference error**: Based on out-of-sample time-mean RMSEs. This is a relatively easy target and the post-training period makes it partly an extrapolation problem, so it's less diagnostic than the 46-year in-sample metrics. Still useful as a quick summary statistic.

## Key issues in the reference run

1. **Temperature drift**: The model doesn't warm fast enough, especially after ~2010. This is the top priority to fix. It may be partly intrinsic (the model acts like training climatology) and partly related to overfitting (overfit models anchor harder to training-period statistics).

2. **Overfitting**: The monotonically growing train/val gap suggests the model has capacity to spare. This may be the root cause of the drift problem — a model that generalizes better may respond more faithfully to CO2 forcing.

3. **Surface pressure bias**: -23.6 hPa time-mean bias in the 46-year run is the largest absolute bias.

4. **Spectral structure**: mean_abs_norm_bias for precipitation and other fields indicates the model smooths out small-scale features.

## Experiment perturbations (prioritized)

### High priority

**1. LR tuning + residual prediction**: The lr-tuning run already shows dramatically better 46-year drift than the residual run (near-zero vs -0.8 K in air_temperature_7). Residual prediction is a natural inductive bias for a system where day-to-day changes are small relative to the full field, but the residual run's poor drift suggests it needs help converging. Combining both perturbations tests whether residual prediction works when the optimizer is better tuned.

**2. Labels (AMIP data) + lr-tuning**: The labels run adds c96-shield synthetic data to reduce ERA5 overfitting. If overfitting is the root cause of poor trends, this is the most direct attack on it. Combining with lr-tuning (the best-performing optimizer so far) gives it the best chance.

**3. Explicit LR schedule from lr-tuning results**: Once the lr-tuning run finishes, examine the LR reduction history (which epochs triggered reductions, what final LR was reached). Use this to set a fixed cosine or step schedule that front-loads learning at a high rate and decays predictably, rather than relying on the reactive reduce-on-plateau. This avoids wasting the initial high-LR epochs on plateau detection.

### Medium priority

**4. Smaller embed_dim (e.g. 384 or 256)**: If overfitting is the core problem, reducing model capacity is the most direct intervention. A smaller model that can't memorize the training set as thoroughly should generalize better and may track trends more faithfully. The trade-off is worse short-range weather skill — worth testing whether the climate stability gains outweigh this.

**5. All three perturbations combined (labels + lr-tuning + residual)**: If individual perturbations each show benefits, the combination is worth trying. Risk is that it becomes hard to attribute improvements.

**6. Labels + residual**: Tests whether the extra training data compensates for residual prediction's drift issues. Lower priority than options 1-2 since we don't yet know if labels alone helps with drift.

### Lower priority / exploratory

**7. Larger embed_dim (e.g. 768)**: If we solve overfitting through data augmentation (labels) or regularization, a larger model might improve both weather skill and climate trends by having more capacity to represent the CO2 response. Only worth trying if overfitting is under control.

**8. Longer training with early stopping on inference metrics**: Train for more epochs but select the checkpoint based on 46-year drift rather than val loss, since val loss keeps improving (due to overfitting) after drift performance peaks. This requires running long inference evaluations frequently, which is expensive.

**9. Multi-step training (n_forward_steps > 1)**: Training with 2-3 step rollouts penalizes error accumulation directly. This could reduce drift by forcing the model to stay on-manifold over longer horizons. Significantly more expensive per epoch.

**10. Adding SSR bias to train aggregator**: Not an experiment perturbation, but a diagnostic change. Would confirm the overfitting hypothesis for ensemble spread. Low effort, should be done on one of the existing configs.
