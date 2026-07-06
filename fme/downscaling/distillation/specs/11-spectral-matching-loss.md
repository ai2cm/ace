# 11 — Per-variable spectral matching loss for the distilled student

## Goal

Give the few-step distilled student a direct, gentle gradient that restores the
**high-wavenumber energy** it loses relative to the many-step teacher, so we stop
leaning on the adversarial (GAN) term as the *only* source of small-scale
sharpness. Concretely: add an optional, per-variable, band-weighted
**spectral-matching loss** to the generator objective that penalizes the
difference between the student's and the teacher's power spectra.

Start on the single-variable **PRATEsfc** downscaling student (cheapest, cleanest
testbed), then generalize to the two multi-variable distillation runs via
per-variable weights.

## Why this exists

The distillation generator loss in both methods is score-based distribution
matching plus GAN:

- DMD2: `loss = vsd_loss + gan_loss_weight_gen * gan_loss_gen`
  (`FastGen/fastgen/methods/distribution_matching/dmd2.py:238`)
- f-distill: `loss = f_distill_loss + gan_loss_weight_gen * gan_loss_gen`
  (`FastGen/fastgen/methods/distribution_matching/f_distill.py:169`)

The variational-score-distillation term matches the teacher's *score*, which is
smooth and provides a **weak gradient at high wavenumbers** — few-step sampling
systematically blurs the small-scale tail. Today the GAN is the only term
pushing that tail back up. Re-tuning the GAN (tap depth, weight, R1) is a blunt,
unstable instrument, especially now that the residual/validation bug (spec 10) is
fixed and we can trust the spectra we measure.

**Key division of labor (why this is gentle, not another hammer):** a
power-spectrum loss constrains *amplitude per wavenumber only* — it says nothing
about phase / spatial placement. VSD already pins spatial structure by matching
`teacher_x0` pointwise. So the spectral term supplies the one thing VSD is weak
at (tail amplitude) without the freedom to hallucinate texture that a GAN has.
The two are complementary; the spectral term should let us *reduce* the GAN
weight, and for DMD2 (where `gan_loss_weight_gen` may be 0) potentially remove it.

## Design decisions (agreed 2026-07-06)

1. **Placement:** a generic optional auxiliary-loss hook in fastgen, with the
   atmosphere-specific spectral op implemented on the ACE side. Keeps the
   FFT/grid code in `fme/`, keeps fastgen domain-agnostic, and covers both DMD2
   and f-distill through one hook.
2. **Target:** match the student spectrum to the **teacher** spectrum
   (`teacher_x0`), not ground truth. Distillation-consistent (same target as
   VSD), recovers exactly what few-step sampling loses, and cannot fight the
   distribution-matching objective. Teacher spectrum is the ceiling for this
   term — pushing past it (toward real 3km data) is deliberately out of scope
   for v1.
3. **Transform + formulation:** zonal `rfft` power spectrum
   (`fme/downscaling/metrics_and_maths.py:223 compute_zonal_power_spectrum`),
   band-weighted **L1 on log-power**. Cheap enough to run every step, reuses the
   existing diagnostic op, and the band weighting is where the "smart weighting"
   lives — we can up-weight the degraded high-k tail.

## Verified mechanics (do not re-derive)

- Both generator steps expose, at the `loss =` line, `gen_data` (student output,
  data space) and `teacher_x0` (teacher x0 prediction, data space). In DMD2
  `teacher_x0` is already `.detach()`ed (`dmd2.py:156`); in f-distill it comes
  from the frozen `self.teacher`. Target is constant → gradient flows only into
  `gen_data`. Good.
- Downscaling student output shape is `(B, C_out, H_fine, W_fine)` with
  `channel_axis = -3` (`models.py`, `DiffusionModel.__init__`); for PRATEsfc
  `C_out = 1`.
- `compute_zonal_power_spectrum(tensor)` (`metrics_and_maths.py:223`):
  `torch.fft.rfft` along the last dim (lon), `power = Re(uhat * conj(uhat))`,
  doubles negative-wavenumber power, then `mean` over dim `-2` (lat). Returns
  per-wavenumber power. It is grid-agnostic (needs only `(…, lat, lon)` layout)
  and differentiable. **Verify** it is differentiable end-to-end before relying
  on it (rfft is; the conj-multiply is) and add a `.mean` over batch.
- f-distill **asserts** `gan_loss_weight_gen > 0` (`f_distill.py:41`). Additive
  spectral loss is fine there; fully removing the GAN in f-distill is a separate
  change (relax the assert) and not required for v1.
- The paired zonal power-spectrum aggregator already runs at train/val time as a
  diagnostic (`fme/downscaling/aggregators/main.py:432
  ZonalPowerSpectrumAggregator` / `ZonalPowerSpectrumComparison`), so we get a
  validation signal for the effect for free — no new diagnostic needed for v1.

## Proposed implementation

### A. ACE side — `SpectralMatchingLoss` module (`fme/downscaling/`)

A small `nn.Module` (config-built) computing:

```
P_g = zonal_power_spectrum(gen_data)      # (B, C, K)
P_t = zonal_power_spectrum(teacher_x0)    # (B, C, K), no grad
d   = |log(P_g + eps) - log(P_t + eps)|   # (B, C, K)
loss = mean_over(B,C,K) [ band_weight[K] * var_weight[C] * d ]
```

Config (`SpectralMatchingLossConfig`, per naming rules → `…Config`):
- `weight: float` — overall `w_spec`.
- `band_gamma: float = 0.0` — high-k emphasis; `band_weight_k ∝ (k/k_max)**gamma`
  (0 = flat, matches all wavenumbers equally; >0 emphasizes the tail).
- `min_wavenumber: int = 0` — optionally ignore the large-scale modes VSD already
  gets right, so the term spends its budget on the tail.
- `variable_weights: dict[str, float]` — per-variable weights (defaults 1.0),
  same pattern as `StepLossConfig.weights`; trivial for single-var PRATEsfc,
  needed for the multi-var runs.
- `eps: float = 1e-12`, `log: bool = True`.

Validate in `__post_init__`. Unit tests: differentiability, correct shape,
matches a numpy reference spectrum, band weighting monotonic in `gamma`, zero
loss when `gen == teacher`, per-variable weighting selects the right channel.

### B. fastgen side — generic auxiliary-loss hook

- Add optional `config.model.auxiliary_loss` (lazy `L(...)` module, default
  `None`) and `config.model.auxiliary_loss_weight: float = 0.0` to the shared
  method config.
- In `dmd2.py._student_update_step` and `f_distill.py` generator step, after
  `gen_data`/`teacher_x0` are available:
  ```
  aux_loss = torch.tensor(0.0, device=..., dtype=...)
  if self.auxiliary_loss is not None:
      aux_loss = self.auxiliary_loss(gen_data, teacher_x0)
  loss = ... + self.config.auxiliary_loss_weight * aux_loss
  loss_map["aux_loss"] = aux_loss
  ```
- Keep the hook signature `(gen_data, teacher_x0) -> scalar` domain-agnostic
  (no atmosphere terms in fastgen). fastgen unit test: with a stub module the
  term is added and weighted; with `None` the loss is unchanged.

### C. ACE adapter — build + inject

- In `fastgen_train.py`, next to the discriminator auto-wiring (~`:606-627`),
  build the `SpectralMatchingLoss` from CLI/config and set
  `config.model.auxiliary_loss = L(SpectralMatchingLoss)(...)` and
  `config.model.auxiliary_loss_weight`.
- New CLI args mirroring the existing style: `--spectral-loss-weight`,
  `--spectral-band-gamma`, `--spectral-min-wavenumber` (+ env-var fallbacks like
  the others).

## Rollout / experiments

1. Land A + B + C behind a default-off weight (0.0) — no behavior change when
   unset; existing runs unaffected.
2. **PRATEsfc DMD2 testbed** (single variable, `C_OUT=1`): resume/retrain with
   two arms — (a) spectral on + reduced GAN weight, (b) spectral on + GAN off
   (DMD2 allows weight 0). Baseline = current GAN-only student. Compare via the
   existing zonal power-spectrum diagnostic + CRPS/tail from
   `BestStudentCheckpointCallback`.
3. Sweep `weight`, `band_gamma`, `min_wavenumber` on PRATEsfc.
4. Port the tuned config to the two multi-variable runs with per-variable
   `variable_weights` (up-weight the fields whose spectra degrade most).

## Open questions / risks

- **Teacher ceiling:** teacher-matching cannot fix a teacher that is itself
  under-powered vs data. If PRATEsfc spectra still fall short after matching,
  that's the signal to consider ground-truth matching (deferred decision 2).
- **Per-sample vs batch spectra:** gen and teacher share condition + noise, so
  per-sample spectra are directly comparable; batch-mean log-power is the
  estimator. Confirm variance is acceptable at the training batch size.
- **PRATEsfc positivity / heavy tail:** precip is non-negative and highly
  intermittent; its spectrum is dominated by rare intense cells. Check whether
  log-power L1 on the raw field is well-behaved or whether the loss should act in
  the model's (already residual/normalized) space — align with where `gen_data`
  lives at the `loss=` line.
- **f-distill GAN coupling:** v1 keeps the GAN in f-distill (assert). Only
  revisit removing it if the DMD2 GAN-off arm shows the spectral term is
  sufficient on its own.
