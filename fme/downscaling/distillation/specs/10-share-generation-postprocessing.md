# 10 — Share residual/generation post-processing between DiffusionModel and distillation validation

## Goal

Make the distillation **validation** sampler go through the *same* residual and
denormalization logic as the real `DiffusionModel` generation path, so the two
can never silently diverge. Concretely: give the residual-base computation a
single source of truth and have the validation callback reuse the model's
`postprocess_generated` instead of reimplementing it.

## Why this exists (history, diagnosed 2026-07-02)

The teachers are **residual** models (`predict_residual=True`): the network
predicts `fine − interpolate(coarse)`, and the full field only exists after the
interpolated-coarse **base is added back**. That base formula was written **three
times** with no shared owner:

- `DiffusionModel.train_on_batch` — *subtracts* the base from the target
  (`models.py`, ~line 512).
- `DiffusionModel.postprocess_generated` — *adds* it back at generation
  (`models.py`, ~line 601).
- `BestStudentCheckpointCallback._base_prediction_norm` — a **copy** added by the
  bug fix (`best_student_callback.py`, commit `b2a47628b`).

Before that fix there was a **fourth** path that had *no* copy: the validation
callback sampled the student via `fastgen_sampler` on the raw module and **never
added the base**, so it compared a *residual* student against a *full-field*
teacher val zarr. For PRMSL (signal ≈ entirely the smooth large-scale base) this
looked like a ~100× low-k power deficit, a collapsed-inward / negative deep-low
tail, and confounded every PRMSL metric (spectra / tail / histogram / CRPS →
checkpoint selection). Training and deployment were correct; the bug was
validation-only. Root cause: the generation logic was **duplicated, not shared**,
so the validation copy could go missing undetected.

Fixed functionally in `b2a47628b` (validation now adds the base back for
`from_noise` and `lo_renoise`, and subtracts it from the `lo_renoise` re-noise
target), and guarded by
`test_best_student_callback.py::test_validation_base_addback_matches_model_generation`
(asserts the callback's residual→full processing equals `postprocess_generated`).
This task removes the underlying **duplication** so the guard can't be end-run.

## Verified mechanics (do not re-derive)

- The base formula, identical in all copies:
  `interpolate(out_packer.pack(normalizer.coarse.normalize({k: coarse[k] for k in
  out_packer.names}), axis=channel_axis), downscale_factor)` → `(B, C_out,
  H_fine, W_fine)` in normalized space. `interpolate` is
  `fme/downscaling/metrics_and_maths.py:255` (bicubic, `align_corners=True`).
  `channel_axis = -3` (`models.py`, `DiffusionModel.__init__`).
- `postprocess_generated(generated_norm, coarse_data, n_samples)`
  (`models.py:594-620`): adds `_repeat_batch_by_samples(base, n_samples)` when
  `predict_residual`, then `_separate_interleaved_samples` → `(B, n, C, H, W)`,
  then `normalizer.fine.denormalize(out_packer.unpack(..., axis=-3))`. Returns
  `(denorm_dict, generated_norm)`. It does **not** call the net.
- The validation callback holds a real `DiffusionModel` as `self._teacher_model`
  (the MoE `_primary` for a bundle), which shares the student's config
  (`predict_residual`, normalizers, `downscale_factor`, `out_names`) — so its
  `residual_base` / `postprocess_generated` are correct for the student output.
- The callback's sampler is **intentionally not** `DiffusionModel.generate`: it
  needs distillation-specific modes — `lo_renoise` (re-noise the target), the
  segment student's own `sigma_min/sigma_max` (Lo runs [0.005, 200], not the
  full config range), and few-step `student_sample_steps`. Only the deterministic
  pre/post transforms are shared; the sampler call stays in the callback.
- Backward-compat (AGENTS.md): this is a pure refactor — **no behavior change**,
  no config/checkpoint format change. The existing parity test plus
  `python -m pytest fme/downscaling/ -q` pin behavior.

## Design

Owner of `predict_residual` semantics is `DiffusionModel`; consolidate there.

1. **Extract `DiffusionModel.residual_base(coarse_data) -> torch.Tensor | None`**
   (`models.py`). Returns the normalized interpolated-coarse base `(B, C_out,
   H_fine, W_fine)`, or `None` when `not self.config.predict_residual`. This is
   the *single source* of the formula above.

2. **Refactor the existing copies to call it** (no behavior change):
   - `train_on_batch`: `base = self.residual_base(coarse)`; subtract when not
     `None`.
   - `postprocess_generated`: `base = self.residual_base(coarse_data)`; add
     `_repeat_batch_by_samples(base, n_samples)` when not `None`.

3. **Callback reuses the model, not a copy** (`best_student_callback.py`):
   - `_base_prediction_norm` becomes a thin wrapper: subset the coarse batch by
     `keep_mask`, then return `teacher_model.residual_base(coarse_kept)` (drop the
     duplicated formula). Keep the `keep_mask` subsetting here — it's
     validation-specific.
   - Replace the callback's hand-rolled add-back + denormalize with
     `teacher_model.postprocess_generated`. The callback holds `(B, n, C, H, W)`;
     reshape to interleaved `(B*n, C, H, W)` (residual), call
     `postprocess_generated(..., n_samples=n)`, get back the denormalized
     `{var: (B, n, H, W)}` dict — identical layout to what the metric loop already
     consumes. `lo_renoise`'s input-side base subtraction uses the same
     `residual_base`.

4. **Do not unify the sampler.** Leave `generate` / the callback's
   `_sample_student_output` sampler dispatch separate (see verified mechanics for
   why). Only steps 1–3 are shared.

## Tests

- **Formula single-source:** unit-test `DiffusionModel.residual_base` returns the
  expected `interpolate(pack(coarse_norm.normalize(...)))` for a small residual
  model, and `None` for a non-residual model. (Build via `_build_small_model`
  pattern; `test_best_student_callback._build_residual_model` already exists.)
- **Parity preserved:** the existing
  `test_validation_base_addback_matches_model_generation` must still pass against
  the refactored code (now it exercises the *shared* `postprocess_generated`, so
  it also proves the callback and model agree by construction).
- **Three call sites agree:** assert `train_on_batch`'s subtract, and
  `postprocess_generated`'s add, use the same base as `residual_base` — e.g. a
  round-trip `postprocess_generated(residual, coarse) == residual_denorm + base`
  for a known base.
- **No regressions:** `python -m pytest fme/downscaling/ -q` clean; the
  distillation callback tests (`test_best_student_callback.py`, plain `fme` env)
  and model tests (`test_models.py`) pass.

## Acceptance criteria

- The residual-base formula appears **once** in the codebase
  (`DiffusionModel.residual_base`); `train_on_batch`, `postprocess_generated`, and
  the validation callback all call it.
- The validation callback's residual→full-field output is produced by
  `DiffusionModel.postprocess_generated` (no duplicated add-back/denormalize).
- No behavior change: pre/post-refactor validation metrics and generated fields
  are bit-identical for a fixed input; all existing tests pass.
- A future change to the residual/denorm scheme is impossible to apply to
  generation without also applying it to validation.

## Out of scope

- Unifying the **sampler** / `generate` wrapper with the callback's sampling
  (the callback's `lo_renoise` / segment-σ / few-step modes are distillation-
  specific; forcing them through `generate` would over-couple).
- Running the in-training student through a real `DiffusionModel.generate` by
  swapping `.module` (stateful/unsafe mid-training; the shared-helper approach
  captures the safety benefit without it).
- The `hi_cascade` validation mode (not yet implemented; when added it must use
  the same `residual_base` / `postprocess_generated`).
- Any change to `predict_residual` semantics itself — this is a pure
  consolidation.
