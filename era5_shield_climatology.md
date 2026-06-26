# Training ACE on ERA5 + rSHIELD for CO2 Robustness

## Goal

ACE trained on ERA5 overfits to CO2 and generalizes poorly on out-of-distribution
CO2. Training on randomly-perturbed-CO2 SHIELD data (rSHIELD) has been shown to
restore out-of-distribution CO2 generalization. The objective here is to train a
single ACE model on **both ERA5 and rSHIELD** so that rSHIELD's CO2 robustness
transfers to ERA5 inference.

Key requirement: **the model must not know which dataset a sample came from.** Any
mechanism that lets the network partition behavior by source (a dataset-id input
channel, per-dataset normalization stats, separate heads) lets it learn "CO2
matters in SHIELD-world, ignore it in ERA5-world" and the robustness fails to
transfer. This rules out per-dataset normalization and source conditioning.

## How ACE normalizes (background)

- Normalization stats are per-variable scalar mean/std, loaded from netCDF,
  frozen at build time (`fme/core/normalizer.py`, `StandardNormalizer`).
- The network sees normalized space; the loss is measured in normalized space.
  The std a variable is divided by sets how much a given physical error counts
  toward the loss (`effective_loss_scaling`).
- `train_loader.dataset.concat` already pools multiple datasets into one loader.
  Pooling + a single stats file is the "treat both as one dataset" approach.
- `NetworkAndLossNormalizationConfig` supports a `residual` override: the loss
  normalizer = network normalizer, except prognostic ("residual-scaled")
  variables get a separate std (typically tendency-scale) via a dict merge in
  `_combine_normalizers`.

## Recommended configuration

### 1. Pool both datasets, no source label

Add rSHIELD as an additional `train_loader.dataset.concat` entry alongside the
existing ERA5 subsets. Variable names (`in_names` / `out_names`) and grid must
match exactly. No dataset-id channel.

### 2. Shared stats recomputed over the pooled union (the decisive part)

Replace ERA5-only `centering.nc` / `scaling-full-field.nc` /
`scaling-residual.nc` with versions computed over **ERA5 ∪ rSHIELD**:

```yaml
normalization:
  network:
    global_means_path: <pooled>/centering.nc
    global_stds_path:  <pooled>/scaling-full-field.nc
  residual:
    global_means_path: <pooled>/centering.nc
    global_stds_path:  <pooled>/scaling-residual.nc
```

**Non-negotiable:** `global_mean_co2` std must span the rSHIELD perturbation
range. Pooling gives this automatically (rSHIELD dominates CO2 variance).
If CO2 were normalized with ERA5-only (tiny) std, ERA5 CO2 → ~0 normalized and
rSHIELD CO2 → huge normalized values, putting the two datasets in disjoint
regions of CO2 input space — no transfer plus numerical blowup. Verify the
pooled CO2 std reflects the perturbation range before any long run.

Cheaper alternative: keep ERA5 stats for everything and override only the
`global_mean_co2` std (and any other perturbed channel) to cover the rSHIELD
range. Clean pooled stats is the safe default.

### 3. Keep what is already correct

- `global_mean_removal: {kind: shared, append_as_input: true}` — keep. Removes
  per-sample spatial mean and re-feeds it as an input channel, damping
  absolute-level offset between products without losing level info.
- `residual_prediction: false` — fine, orthogonal to normalization choice.

### 4. No RevIN

- The learnable affine (γ, β) on top of fixed global stats is a redundant
  reparameterization — absorbed by the encoder/decoder linear layers, zero
  expressivity gain. SFNO already has `affine_norms: true` internally.
- The genuinely-new part of RevIN, per-sample variance normalization, is
  dangerous here: the CO2 field is ~uniform, so per-sample std ≈ 0 → divide
  blows up. RevIN also buys invariance to absolute level/scale, but CO2 forcing
  acts *through* absolute level — exactly the signal the model must stay
  sensitive to.
- Note: the per-sample *mean*-removal half of instance norm already exists via
  `global_mean_removal` (and dodges the level-loss problem via `append_as_input`).
  Only the std half would be new, and it is the harmful part.

## Limiting SHIELD bias bleed into ERA5

With no source label, SHIELD's GCM climatology bias can drift ERA5 predictions.
You cannot get "CO2 response transfers but bias does not" purely via
normalization — that needs either a label (rejected) or bias-correcting rSHIELD
to ERA5 climatology before pooling.

Options, by appetite:

- **Easiest:** weight sampling/loss toward ERA5 so it anchors climatology while
  rSHIELD supplies the CO2 range.
- **Best:** bias-correct / anomaly-frame rSHIELD to ERA5 climatology before
  pooling (see below).
- **Always:** validate ERA5 climatology metrics (`time_mean`, `annual`, zonal
  mean) during training to catch drift early.

## Bias correction: trend-preserving additive delta method

Use an additive delta ("delta method"), not full quantile mapping (overkill and
risks eating the CO2 signal).

```
δ(var, cell, month) = clim_rSHIELD_ref − clim_ERA5
corrected_rSHIELD   = rSHIELD − δ
```

Removes SHIELD's mean-state bias; rSHIELD anomalies (including the CO2 response)
ride on ERA5's climatology and are preserved.

### Make-or-break subtlety: the reference climatology

`clim_rSHIELD_ref` must be computed at a CO2 regime **matching ERA5's** (SHIELD's
control / near-historical-CO2 subset), **not** averaged over the perturbations.
Averaging over perturbed CO2 makes the "bias" absorb the CO2 response, so
subtracting it deletes the signal. Bias is defined at fixed reference CO2; the
signal is the departure from it.

### Per-variable treatment

- **Additive** for unbounded fields: temperatures, winds, geopotential, pressure.
- **Multiplicative ratio** (or log-space) for positive-definite vars
  (`specific_total_water_*`, `Q2m`, `PRATEsfc`, fluxes — the `force_positive_names`
  set); additive δ can push them negative:
  `corrected = rSHIELD · (clim_ERA5 / clim_rSHIELD_ref)`.

### Granularity

- Per gridcell (bias is spatially structured).
- Per calendar month (12 climatologies) to avoid smearing the seasonal cycle.

### Do not correct

- `global_mean_co2` — leave raw; it is the augmentation range.
- Other pure forcings you want as-is (e.g. `DSWRFtoa`).

### Scope

- Mean only — no variance/quantile correction unless residual distribution-shape
  bias clearly hurts ERA5 metrics. Escalate to trend-preserving (ISIMIP-style)
  quantile mapping only then.
- Correct state fields, not tendencies (mean-state bias dominates).
- Run offline as preprocessing → write a corrected rSHIELD zarr → point the new
  `concat` entry at it.

## Priority order

1. Shared pooled stats + correct CO2 std (largest effect — ~80% of success).
2. Pool via `concat` (mechanical).
3. Bias-correct rSHIELD (quality lever once it trains).
4. Keep `global_mean_removal`; skip RevIN / per-dataset stats / source label.

## Pre-run sanity checks

- rSHIELD has all `in_names` / `out_names` channels including `global_mean_co2`,
  `DSWRFtoa`, masks.
- Pooled CO2 std covers the perturbation range (the single make-or-break number).
- Short smoke run; confirm ERA5 validation does not blow up (a sign of bad CO2
  scaling or a missing variable).
