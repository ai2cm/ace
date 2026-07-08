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

1. **Placement:** FastGen is a *pinned upstream submodule* (`NVlabs/FastGen`),
   not a vendored copy, so we do **not** edit its source. Instead the ACE side
   subclasses the method (`AceFdistillModel(FdistillModel)`) and adds the
   spectral term by overriding the generator step; the subclass is selected by
   pointing `config.model_class` at it from `fastgen_train.py`. Scoped to
   **f-distill only** for v1 (the run being tuned); DMD2 is a trivial parallel
   subclass if needed later.
2. **Target:** match the student sample's spectrum to the teacher's clean EDM
   **sample** (`data["real"]`), not ground truth. **Corrected 2026-07-07** — v1
   wrongly used `teacher_x0` (the teacher's x0 *prediction* `E[x0|x_t]`, a
   conditional mean), which is smoother than a sample and drove the student to
   *over-smooth* (see Revision below). The clean sample is distillation-
   consistent (it is the distillation target) and carries the correct ensemble
   high-k power. Teacher spectrum is the ceiling; pushing past it (toward real
   3km data) is out of scope for v1.
3. **Transform + formulation:** zonal `rfft` power spectrum
   (`fme/downscaling/metrics_and_maths.py:223 compute_zonal_power_spectrum`),
   band-weighted **L1 on log-power**, computed **spectrum-then-average**: per-
   sample power spectra are averaged over the batch *before* the log-L1, so the
   term matches the *ensemble* power `E[|FFT|^2]` (preserving incoherent high-k
   energy) and mirrors how the eval `spec_mae` aggregator computes spectra.
   Cheap enough to run every step, reuses the existing diagnostic op, and the
   band weighting is where the "smart weighting"
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

### B. ACE method subclass — `spectral_method.AceFdistillModel`

`AceFdistillModel(FdistillModel)` (`fme/downscaling/distillation/spectral_method.py`):
- `set_spectral_loss(module, weight)` stores `(module.to(device), weight)` in a
  **tuple** attribute so `nn.Module.__setattr__` does not register it as a
  submodule — the param-less loss must stay out of the optimizer, DDP parameter
  list, and state_dict.
- Overrides `_student_update_step`: delegates verbatim to `super()` when no
  spectral loss is set (or weight 0), otherwise mirrors the parent body (FastGen
  pinned at `123e6a2`) and adds `weight * spectral_loss(gen_data, teacher_x0)`
  to `loss`, logging `spectral_loss` / `spectral_loss_weighted` in `loss_map`.
  Manual mirror because upstream exposes no hook at the `loss=` line; the guard
  test `test_overrides_student_update_step` flags upstream drift.

### C. ACE entry point — build + inject

- In `fastgen_train.py`, when `--spectral-loss-weight > 0`: assert
  `config.model_class._target_` is the f-distill method, build
  `SpectralMatchingLoss` from `SpectralMatchingLossConfig` and the teacher's
  `out_packer.names`, swap `config.model_class = L(AceFdistillModel)(config=None)`,
  and after `instantiate` call `model.set_spectral_loss(...)`.
- CLI args (env-var fallbacks like the others): `--spectral-loss-weight`
  (`$ACE_SPECTRAL_LOSS_WEIGHT`, the single overall weight), `--spectral-band-gamma`
  (`$ACE_SPECTRAL_BAND_GAMMA`), `--spectral-min-wavenumber`
  (`$ACE_SPECTRAL_MIN_WAVENUMBER`). Default weight 0 → method unchanged.

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
- **Field space is resolved:** the loss compares `gen_data` to `teacher_x0`,
  both of which are the network's output in the *same* (residual/normalized)
  space at the `loss=` line, so no residual add-back / denorm is needed — it
  compares like with like. (Consequence: it matches the teacher's spectrum *in
  that space*, which is exactly the distillation-consistent target.)
- **log-power near-zero sensitivity (confirmed in tests):** `log(power + eps)`
  amplifies the rfft roundoff floor at near-empty wavenumbers, so `eps` must be
  set relative to the field's power scale, not left at the tiny default, or the
  loss will chase FP noise at the highest wavenumbers. Revisit `eps` (or use
  `log=False`) once we see real PRATEsfc spectra.
- **PRATEsfc positivity / heavy tail:** precip is non-negative and highly
  intermittent; its spectrum is dominated by rare intense cells. Confirm the
  log-power L1 tail behaves on real fields.
- **f-distill GAN coupling:** v1 keeps the GAN in f-distill (assert). Only
  revisit removing it if the DMD2 GAN-off arm shows the spectral term is
  sufficient on its own.

## Revision — wrong target diagnosed and fixed (2026-07-07)

**Symptom.** The first spectral arm (`W=1e-2`, wandb `s4abc6ba`) did not improve
spectra and hurt distillation. Vs the baseline (`f7z93y0a`) over ~18k steps:
`train/spectral_loss` fell 0.667→0.432 (so the term *was* responding), but
`train/f_distill_loss` rose to ~0.22 (3× the baseline's ~0.066) and
`val/spec_mae_PRATEsfc` stalled at ~0.96 while the baseline improved to ~0.65.
Tails under-produced (`tail_99.99` ~0.83). So the weighting was not too weak —
the loss was optimizing toward the wrong thing.

**Root cause (two coupled errors).**
1. **Wrong target.** v1 matched `gen_data` (a student *sample*) to `teacher_x0`,
   the teacher's x0 *prediction* `E[x0 | x_t]` at a random perturbation σ. A
   conditional mean is systematically **smoother** than a sample (less high-k
   power), so matching its spectrum pushed the student to over-smooth — the
   opposite of the goal — and fought VSD. VSD tolerates the same `teacher_x0`
   only because it uses the *difference* `(fake_score − teacher)`, cancelling
   the shared σ-dependent smoothing bias; the standalone spectral term inherits
   it fully.
2. **Wrong reduction order.** `spectrum(mean(fields))` (average fields, then
   spectrum) cancels incoherent high-k energy; the correct estimate of ensemble
   power is `mean(spectrum(fields))` (spectrum per sample, then average).

**Fix.** (a) Target = the teacher's clean EDM **sample** `data["real"]` (a real
sample with correct high-k power; already in the batch, no extra teacher
forward; obtained via the deterministic `_prepare_training_data`). (b)
`SpectralMatchingLoss.forward` now averages per-sample power spectra over the
batch **before** the log-L1 (spectrum-then-average), matching `E[|FFT|^2]` and
the eval `spec_mae` aggregator. Guarded by
`test_uses_ensemble_power_not_field_mean`.

Supersedes the "Field space is resolved" and "Per-sample vs batch spectra" notes
above (they described the `teacher_x0`, per-sample design).
