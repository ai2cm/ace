# Multivariate MoE Distillation — Status & Handoff

Living notes for the multivariate MoE FastGen distillation effort on branch
`experiment/fastgen-distill`. Pick up here from any clone.

_Last updated: 2026-07-07._

---

## ★ RESULT (2026-07-07): Student-Hi run — trained fine, but hi_cascade validation can't see it

**Run** `4mez4kmn` / `…-hi-1step-moe-teacher-expert1` (beaker
`01KWTXGADFPB4GKVZ33C7ZGJP4`), the first per-expert **Student-Hi** (expert 1,
σ∈[200,2000], `student_sample_steps=1`, GAN weight 1e-3, R1 off, `hi_cascade`
validation through the frozen Lo). Launched 2026-07-06, **stopped 2026-07-07 at
~step 20.3k** (max_iter was 100k) because the metrics looked flat. Post
residual-fix era, so the absolute numbers are trustworthy.

**Stopping was justified — training had converged.** `f_distill_loss` did the
real work: 1.40 → ~0.72 by step ~9k, then oscillates flat 0.63–0.79 through 20k.
The last ~10k steps added nothing. So "flat converged" is correct on the
*training* side.

**★ Two things the flat metrics actually reveal:**

1. **The GAN never engaged.** `gan_loss_gen ≈ 0.693 (=ln2)` and
   `gan_loss_disc ≈ 1.386 (=2·ln2)` are pinned at their chance-level values the
   *entire* run (gen 0.6924→0.6931, disc 1.387→1.386). The discriminator stays at
   coin-flip — it never learns to tell teacher-x0 from student-x0. This is exactly
   the design-note prediction: a **coarse/bottleneck critic at high σ (200–2000,
   near-pure-noise) is uninformative** — there's no coherent structure at that σ
   for a coarse critic to grade. So **Student-Hi is effectively f-distill-only**;
   the GAN is inert. Good news: **no GAN collapse here** (unlike every Lo run, where
   the critic tipped and dragged spectra to 0.8+). `fake_score_loss` shows the mild
   familiar late rise (0.001→~0.02) but nothing tips.

2. **★ The `hi_cascade` end-to-end validation is nearly INSENSITIVE to Hi
   training.** Every val metric is flat **to 4 significant figures** across 18k
   steps while f_distill halved: `crps_mean` 0.082016→0.082034 (<0.02% drift),
   `crps_PRMSL` 0.17378→0.17382, `spec_mae_*` and tails all pinned. That is far too
   flat to be "the model converged instantly" — it means the validated final sample
   **barely depends on Hi's weights**. Mechanism: Hi does a single σ=2000→boundary
   step, then the **frozen Lo** finishes; at the σ=200 handoff the `200·ε` term
   dominates the O(1) clean part ~200×, washing out *which* x0 Hi produced. The very
   "washout" argument the plan used to justify the Lo train/deploy mismatch (§
   "Re-noise the target zarr") **cuts both ways** — it also means the cascade metric
   cannot *see* Hi. ⇒ **`hi_cascade` CRPS/spectra/tails are a poor checkpoint-
   selection signal for Student-Hi**; `best_student.ckpt` selection is ~arbitrary
   here. Select a mid/late Hi ckpt by the f_distill plateau (~step 10k+), not by
   val CRPS.

**Absolute val numbers (last ckpt, all trustworthy post-fix):**
- **PRMSL spectra excellent, no collapse:** `spec_mae` hi **0.018** / mid 0.053 /
  lo 0.077. (Contrast the Lo runs that collapsed to 0.8+.)
- **Tails near 1.0:** `tail_99.99_mean` 0.957 (PRMSL 0.978, precip 0.905, winds
  ~0.97); `tail_99.9999_mean` 1.09.
- **Winds moderate:** `spec_mae` ~0.14–0.20 overall, mid-band worse (eastward 0.31,
  northward 0.23).
- **Precip under-powered at high-k:** `spec_mae_hi_PRATEsfc` 0.35 — the familiar
  too-smooth fine-scale deficit. `crps_mean` 0.082.

**Next / actionable:**
- **The discriminating test is the assembled bundle, not `hi_cascade`.** Bundle
  the chosen Hi ckpt (~step 10k+) with the selected Lo into the 2-step MoE student
  and evaluate end-to-end vs the full-teacher zarr (the "Bundle sampler" +
  "Eval" checklist items). Only there will Hi's coarse contribution be visible.
- Don't spend more compute tuning the Hi GAN (tap depth, weight, R1): at high σ the
  coarse critic is structurally blind, so those levers are moot for Hi. If the
  bundle eval shows a coarse/deep-low deficit, that's the Hi lever to revisit —
  otherwise leave Hi as f-distill-only.
- **Open question for the checklist:** whether `hi_cascade` needs a more
  Hi-sensitive variant (e.g. validate Hi's boundary x0 directly at σ=200 against
  the teacher's σ=200 state, rather than end-to-end through Lo) — otherwise Hi runs
  have no usable early-stopping/selection signal.

#### Loss-mechanics trace (2026-07-07): Hi is a COARSE-ONLY distillation by construction

Traced the f-distill/DMD2 training path (`f_distill.py`, `dmd2.py`,
`fastgen_teacher.py`, `fastgen_loader.py`) to answer "how does training respect
the teacher only being defined on [200,2000]?" **Every teacher query is in-domain,
and the objective is structurally blind to any structure finer than what survives
re-noising to σ≥200.** Three legs, all confirmed:

1. **VSD/f-distill score term** (`dmd2.py:_student_update_step`): student emits an
   x0 estimate `gen_data`; it is **re-noised to a random `t`** via
   `forward_process(gen_data, eps, t)`, and the frozen teacher is queried at
   `(perturbed_data, t)` — never at σ=0. `t ~ sample_t`, **clamped to
   `[min_t=200, max_t=2000]`** (`fastgen_teacher.py:73-74`, scheduler built with
   `min_t=sigma_min=200, max_t=sigma_max=2000`). It is *score matching in the band*,
   not L2 to a clean target ⇒ zero gradient for any x0 content that washes out under
   `t·ε`, i.e. Hi is supervised **only on the ≥200-surviving coarse modes.**
2. **GAN real + fake branches** (`_compute_real_feat`, `dmd2.py:250`): the real data
   is re-noised (`forward_process(real_data, eps_real, t_real)`) with `t_real` also
   from `sample_t` clamped to `[200,2000]` before the teacher-encoder feature tap.
   Both real and fake features are extracted in-domain. (R1 same; off here anyway.)
3. **The "real" target itself** (`data["real"] = teacher.sample()`,
   `fastgen_loader.py:91`): for the Hi teacher `expert_index=1` sets
   **`moe_experts=None` (no dispatch)** (`fastgen_teacher.py:169`), so `sample()`
   runs **expert-1 alone** via `stochastic_sampler(sigma_min=200, sigma_max=2000)`.
   The EDM grid runs 2000→200 and the final Euler step goes 200→0 **without a net
   eval at 0**, so expert 1 is queried only at grid points ≥200 (lone exception: a
   negligible `S_churn` overshoot slightly above 2000 at the pure-noise top).

**Consequence for "is Hi worth keeping":** the target, the score-matching band, and
the surviving-renoise content **all** agree Hi's job is the coarse (≥200) band and
nothing finer. Note `data["real"]` is **expert-1's own solo sample** (a coarse
field), not ground-truth fine data nor the full-MoE output — so the discriminator
compares student-coarse vs expert1-coarse. This is why the flat `hi_cascade`
metric, the σ=200 handoff washout, and the loss design tell one story: Hi's
marginal budget is small by construction. The Lo-from-noise@200 ablation remains
the way to decide whether that small coarse contribution is worth an expert + an
NFE, or whether Lo-only suffices.

---

## SPEC (2026-07-07): distilled 2-step bundle sampler + end-to-end eval  ✅ DONE

> Status: **implemented** — `DenoisingMoEStudentPredictor` +
> `DenoisingMoEStudentConfig` / `DenoisingMoEStudentBundledConfig` land the
> predict-x0-renoise cascade as a deployable predictor. See "Implementation" below.
> Closes the ⏳ "Bundle sampler" checklist item.

### Goal

Assemble the per-expert students (Student-Hi + Student-Lo) into a single
**deployable 2-step MoE predictor** that generates via the **fastgen
predict-x0-then-renoise cascade** over a boundary-aligned `t_list`, and can be run
end-to-end through the standard evaluator (`generate_on_batch` vs the full-teacher
val zarr). This is the discriminating test that `hi_cascade` validation cannot be
(see the 2026-07-07 RESULT) — the only place Hi's coarse contribution is measured
against a fixed Lo in the real deployment path.

### Why a new predictor variant (not a flag on the existing one)

`DenoisingMoEPredictor.generate` (`predictors/serial_denoising.py:234`) runs
`edm_sampler` (Heun/EDM stochastic sampler) over a **continuous Karras grid**
`[sigma_schedule_max → min]`. Correct for the **teacher** (each expert is a full
multi-step denoiser, valid at any Karras σ). **Wrong for distilled one-shot
students:** it queries each student at continuous σ it was never distilled against
and never places a node at the σ=200 boundary, so there is no renoise handoff.

Distilled students need the **fastgen predict-x0-renoise cascade** instead: each
student predicts x0 at its own query σ, then x0 is re-noised down to the next
student's start σ (the boundary handoff). This is the same "renoise between steps"
mechanism as the 2-step Lo, generalized across experts.

A **sibling predictor class** (not a `sampler_type` flag inside `generate`) is
chosen because: (a) **teacher-checkpoint backward-compat is critical** (AGENTS.md)
— the existing EDM path and bundle format must stay byte-for-byte untouched; a new
config type + a `sampler_type` marker in the student state leaves it alone;
(b) it avoids an `isinstance`/flag branch inside `generate` (AGENTS.md flags those
as a refactor smell); the two sampling semantics are genuinely different objects.

### Verified mechanics (do not re-derive)

- **The cascade sampler already exists and is tested** as
  `sample_student_hi_cascade` (`distillation/student_sampling.py:122`): it builds a
  `_SigmaDispatchModule(sigma_ranges, [lo_net, hi_net])` + `boundary_aligned_t_list`
  and calls `fastgen_sampler(dispatch, noise, cond, t_list=...)`. The new predictor
  reuses these three primitives directly — the deployable path == the validated path.
- **`boundary_aligned_t_list(sigma_ranges, steps_per_range)`**
  (`samplers.py:211`): descending schedule visiting each student's own query nodes
  (its `_fastgen_t_list` minus the trailing 0) then a single final 0. `steps_per_range`
  aligns with `sigma_ranges` (ascending) and sets each segment's step count
  (e.g. `[2, 1]` = 2-step Lo + 1-step Hi for `[(0.005,200),(200,2000)]`).
- **`fastgen_sampler(net, latents, img_lr, t_list=...)`** (`samplers.py:264`):
  when `t_list` is given, `num_steps`/`sigma_min`/`sigma_max` are ignored; with
  default `skip_noise_scale=False` the initial state is `latents * t_list[0]`
  (unit-variance `latents` → noised to the top σ). Returns `(x_pred, latent_steps)`.
- **`_SigmaDispatchModule`** (`serial_denoising.py:64`) routes each `t_list` step to
  the expert whose inclusive range contains it (boundary → lower-σ expert). The
  predictor already builds `self._dispatch_module` from its experts in `__init__` —
  reuse it as-is.
- **Residual base is handled by `_primary.postprocess_generated`** (unchanged from
  the teacher path): the students are `predict_residual=True`; the cascade runs
  entirely in residual space and the interpolated-coarse base is added back **once**
  at the end (consistent with the `b2a47628b` residual-bug fix and spec 10).
- **`prepare_generation_inputs`** (`models.py`) returns `(inputs, latents)` with
  unit-variance `latents` — exactly what `fastgen_sampler` expects.
- **Sigma ranges are authoritative from the bundle config**, not the expert
  modules: dispatch and `boundary_aligned_t_list` both read the explicit
  `sigma_ranges`, so the Lo checkpoint's recorded σ range (which may inherit the MoE
  `_primary`) is irrelevant here (resolves the open checklist caveat).

### Design / implementation steps

1. **`DenoisingMoEStudentPredictor(DenoisingMoEPredictor)`**
   (`predictors/serial_denoising.py`). `__init__(experts, sigma_ranges,
   steps_per_range, expert_renames=None)` calls `super().__init__(...,
   num_diffusion_generation_steps=sum(steps_per_range), churn=0.0, ...)` (parent
   builds `_dispatch_module`), validates `len(steps_per_range) == len(experts)` in
   `__post_init__`-style guard, stores `self._steps_per_range`. Override:
   - `generate()`: `prepare_generation_inputs` → `boundary_aligned_t_list(
     self._sigma_ranges, self._steps_per_range, device=…)` →
     `fastgen_sampler(self._dispatch_module, latents, inputs, t_list=t_list)` →
     `self._primary.postprocess_generated(...)`. Returns
     `(generated, generated_norm, latent_steps)` (same signature; inherited
     `generate_on_batch` / `_no_target` work unchanged).
   - `get_state()`: experts + sigma_ranges + `steps_per_range` + expert_renames +
     `sampler_type="fastgen_cascade"`.
   - `from_state()`: reads the above.
2. **`DenoisingMoEStudentConfig`** — sibling of `DenoisingMoEConfig` for assembling
   from per-expert `DenoisingExpertCheckpointConfig`s, with `steps_per_range:
   list[int]` replacing `num_diffusion_generation_steps` (no `churn`). `build()` →
   `DenoisingMoEStudentPredictor`.
3. **`DenoisingMoEStudentBundledConfig`** — sibling of `DenoisingMoEBundledConfig`
   to load a saved student bundle; `build()` → `DenoisingMoEStudentPredictor
   .from_state`. (Teacher `DenoisingMoEBundledConfig` untouched.)
4. **Reuse, don't fork, the cascade math** — the predictor calls
   `boundary_aligned_t_list` + `fastgen_sampler` + `self._dispatch_module` directly;
   no new sampler code. `sample_student_hi_cascade` stays as the validation helper.

### Tests (`predictors/test_serial_denoising.py`; reuse `_mock_expert` /
`_build_two_expert_predictor`)

- **Dispatch/step order:** `generate_on_batch_no_target` on a 2-expert student
  predictor (steps `[1,1]`) queries the hi expert only above the boundary and the
  lo expert at/below — mirror `test_sample_student_hi_cascade_routes_high_then_low`
  using a tracked module.
- **Uses the fastgen cascade, not EDM:** assert the student predictor's per-step σ
  sequence equals `boundary_aligned_t_list(...)` (the boundary node is present),
  distinguishing it from the teacher's continuous Karras grid.
- **`steps_per_range` honored:** `[2,1]` yields the expected number of net calls per
  expert.
- **Save/load round-trip:** `save` → `DenoisingMoEStudentBundledConfig.build`
  preserves `steps_per_range` + `sampler_type` and produces identical generation
  (seeded) — parallel to `test_save_and_load_roundtrip_preserves_predictor`.
- **Residual add-back:** generated full field = residual cascade output +
  interpolated-coarse base (via `postprocess_generated`), matching the teacher
  predictor's post-processing for a known input.
- **No regressions:** `python -m pytest fme/downscaling/predictors/ fme/downscaling/distillation/ -q` clean.

### Acceptance criteria

- A `[Student-Hi, Student-Lo]` bundle assembles + saves via
  `DenoisingMoEStudentConfig` and reloads via `DenoisingMoEStudentBundledConfig`.
- `generate` runs the boundary-aligned predict-x0-renoise cascade (hi @≥200 →
  renoise to 200 → lo @≤200 → 0), never invoking the EDM Heun grid.
- The teacher `DenoisingMoEPredictor` / `DenoisingMoEBundledConfig` / bundle format
  are unchanged (existing tests pass byte-for-byte).
- Runs end-to-end through the evaluator against the full-teacher val zarr.

### Out of scope

- Choosing the Lo checkpoint / Lo step count (a runtime eval decision; base case
  `steps_per_range=[2,1]` per the 2-step-Lo finding).
- The Lo-only (from-noise@200) ablation — separate eval config, decided after this.
- YAML wiring for the student bundle in the production config union (spec 03 area).

### Implementation (done 2026-07-07)

`predictors/serial_denoising.py` — three new symbols; teacher path untouched:
- **`DenoisingMoEStudentPredictor(DenoisingMoEPredictor)`** — overrides `generate`
  to run `boundary_aligned_t_list(self._sigma_ranges, self._steps_per_range)` →
  `fastgen_sampler(self._dispatch_module, latents, inputs, t_list=…)` →
  `_primary.postprocess_generated`. `__init__` takes `steps_per_range` (validates
  length + ≥1), calls `super().__init__` with `num_diffusion_generation_steps=
  sum(steps_per_range)`, `churn=0.0`. `get_state`/`from_state` carry
  `steps_per_range` + `sampler_type="fastgen_cascade"`. `generate_on_batch` /
  `_no_target` inherited unchanged.
- **`DenoisingMoEStudentConfig`** — assembles from per-expert
  `DenoisingExpertCheckpointConfig`s; `__post_init__` sorts experts *and*
  `steps_per_range` together by ascending `sigma_min`; `build()` →
  `DenoisingMoEStudentPredictor`.
- **Loading a student bundle: reuse `DenoisingMoEBundledConfig`.** `build()`
  dispatches on the persisted `sampler_type` tag — `"fastgen_cascade"` →
  `DenoisingMoEStudentPredictor.from_state`, else the teacher
  `DenoisingMoEPredictor.from_state` (backward-compatible; old teacher bundles
  lack the key). This is a factory dispatch on a format tag, chosen over a
  separate `DenoisingMoEStudentBundledConfig` because two bundle configs sharing
  only `mixture_of_experts_path` are **ambiguous in the evaluator's dacite
  `model:` union** (a student bundle would silently match the teacher config
  first). One config, one union member, no ambiguity.

Tests (`predictors/test_serial_denoising.py`, 8 new, all pass; reuse
`_get_diffusion_model` + forward-pre-hooks to record dispatch σ): cascade routes
hi@>boundary / lo@≤boundary; per-step σ == `boundary_aligned_t_list` (not the EDM
grid); `steps_per_range=[2,1]` → lo×2/hi×1; length mismatch raises; save/load
round-trip preserves `steps_per_range` + seeded generation; config sorts steps
with ranges + bundle carries the `fastgen_cascade` marker; `predict_residual=True`
output is full-field via `postprocess_generated`. Clean on ruff/ruff-format/mypy.

**YAML-loadable (2026-07-07):** a student bundle loads from an evaluator/inference
YAML via the existing `model: {mixture_of_experts_path: …}` union member —
`DenoisingMoEBundledConfig.build()` now dispatches teacher vs student on the
bundle's `sampler_type` tag, so no new union member was needed.

**CONUS comparison eval — wired, ready to launch** (`configs/experiments/2026-07-07-distilled-moe-eval/`):
- **Dataset:** CONUS 2023, **100km→3km** (not 25km). Coarse merge `100km.zarr` +
  `prmsl_100km.zarr`; fine merge `3km.zarr` +
  `instantaneous_surface_and_sea_level_pressure_3km.zarr`.
- **Bundle assembly:** `./run.sh bundle` — gantry job mounts *only* the
  `student_checkpoints` subpath of each source dataset (Lo=expert0 baseline-fixed
  `01KWJAFM694MAE55M2JMZSE89M`, Hi=expert1 hi-1step `01KWTXGAM1CCGDH29JWDSN9KPF`),
  runs `scripts/downscaling/bundle_denoising_moe_checkpoint.py` on
  `distilled-bundle.yaml` (`best_student_tail.ckpt` each, `steps_per_range [2,1]`),
  writes `distilled_moe_bundle.ckpt` to weka
  `/climate-default/2026-07-07-distilled-moe-bundle/`.
- **Eval:** `./run.sh all` — two `EvaluatorConfig`s (teacher bundle via beaker
  dataset, distilled bundle via weka), identical data/patch/`n_samples=4`, logged
  to `andrep-downscaling`. Run `bundle` first and let it finish.
- **Verify before trusting the comparison:** teacher and students must share the
  same output vars (status says 4: u10/v10/PRMSL/PRATEsfc; `ARCHITECTURE.md` notes
  a 5-var T2m teacher variant — if the bundle is 5-var the fine merge needs T2m).

---

## Goal

Distill the multivariate MoE downscaling teacher (CONUS 100km→25km) into a
fast 2-step student via FastGen f-distill (forward KL), preserving validation
CRPS / spectra / extreme-event tails.

## Teacher

Bundled MoE checkpoint `scratch/moe/bundled_moe_multivariate.ckpt`
(Beaker dataset `01KTCHVDHY0SATWH9E0AW2PDS6`, file
`bundled_moe_multivariate.ckpt`).

- **2 experts, split by sigma:**
  - Expert 0 — sigma `[0.005, 200]`, 32.75M, `channel_mult [1,2,2,2,1,1]`
    (deepest feature 128ch). Low-noise specialist.
  - Expert 1 — sigma `[200, 2000]`, 45.87M, `channel_mult [1,2,2,2,2,2]`
    (deepest feature 256ch). High-noise specialist.
- 4 output vars: `eastward_wind_at_ten_meters`, `northward_wind_at_ten_meters`,
  `PRMSL`, `PRATEsfc`. 18 generation steps. Full schedule sigma `[0.005, 2000]`.
- Single-var teacher (earlier baseline): Beaker `01KNM6H3JB1ZNS76HX17AAZRF7`,
  file `best_histogram_tail.ckpt`.

---

## Current direction (2026-06-24): distill each expert separately

**The plan below supersedes the single-student approach.** Everything after the
"Active runs" section documents the single-student line of work (distill the whole
MoE into one 2-step student). That line produced a usable model (dispatch-v2 ≈
step 7k) but hit a structural wall: **one student net can only host one
expert's backbone** for the in-loss score term *and* the discriminator feature
extractor, so it is in-domain at one end of the σ range and out-of-domain at the
other. The PRMSL coarse-spectral collapse (see 2026-06-24 update) is the symptom;
Run 3 and the discriminator-backbone design note are both workarounds for it.

### New approach: per-expert single-step distillation → bundled 2-step MoE student

Distill **each teacher expert independently** into its own single-step student
over **that expert's own σ range**, then bundle the two students into an MoE that
does **2-step denoising** (one step per student, dispatched by σ — reusing the
`_dispatch` infra already built).

| Student | Distilled from | σ range | Steps | Target capacity | Discriminator `in_channels` |
|---|---|---|---|---|---|
| **Student-Hi** | Expert 1 (high-noise) | [200, 2000] | 1 | ~45.87M, `channel_mult [1,2,2,2,2,2]` (deepest 256) | 256 |
| **Student-Lo** | Expert 0 (low-noise) | [0.005, 200] | 1 | ~32.75M, `channel_mult [1,2,2,2,1,1]` (deepest 128) | 128 |

Inference (2-step deterministic sampler): start at σ≈2000 → **Student-Hi** →
re-noise the x0 estimate to the boundary σ≈200 → **Student-Lo** → final x0. Bundle
= sigma-routed dispatch over the two students (boundary→lower, same convention as
the teacher `_dispatch`). **Match each student's capacity to its source expert**
(table above) so neither is bottlenecked — the single-student run was forced to one
capacity for the whole range.

### Why this fixes the documented problems (all of them, structurally)

- **Score term is always in-domain.** Each student's frozen teacher = its own
  expert, used only over the σ range that expert was trained on. No
  expert-0-extrapolation, no out-of-domain low-σ score. **Run 3 (dispatched score
  term) becomes unnecessary** — dispatch is now structural, not in-loss.
- **Discriminator backbone is always in-domain.** Each student's critic taps its
  own expert's bottleneck over that expert's σ range. This is exactly what the
  "which expert should the discriminator use?" design note was wrestling with —
  the answer becomes "each student uses its own," and the question dissolves. The
  PRMSL coarse-critic collapse should not recur (a single coarse backbone policing
  a field across all σ was the cause).
- **Capacity matched per expert** — no shared-net bottleneck.

### What carries over

- **The σ-dispatch infra** (commits `3a2dced4e`, `19d5f0467`, dtype fix) is reused
  at *inference* to route between the two students — and at bundling time. The
  per-sample routing logic is already tested.
- **The GAN-fix levers** (R1 reg `gan_r1_reg_weight`, lower `gan_loss_weight_gen`,
  LR decay — see the 2026-06-24 "config knobs") still apply **per student**,
  especially Student-Hi. They should matter less now that each backbone is
  in-domain, but keep them in the toolkit.
- **Training is simpler than Run 3:** two independent FastGen runs, each with
  `config.teacher = config.net =` a single expert (no MoE in the loss at all).
  Each student auto-derives its discriminator from its own expert
  (`in_channels` 256 / 128 respectively).

### Structural findings from the code (2026-06-24)

Traced `student_checkpoint.py`, `serial_denoising.py`
(`DenoisingMoEPredictor` / `_SigmaDispatchModule`), `fastgen_teacher.py`, and
FastGen `model.py::generator_fn`. Three concrete answers:

1. **A FastGen single-step student is a one-shot map from its own `sigma_max` → x0**,
   not a σ-conditioned denoiser usable at arbitrary intermediate σ. `generator_fn`
   inits latents at `t_list[0] = sigma_max` and asserts `t_list[-1] == 0`; with
   `student_sample_steps=1` the student only ever sees its `sigma_max`. So
   "single step across its range" = one-shot `sigma_max → x0`, refined by the
   *next* student. A 2-step student is trained at `get_t_list(2) = [sigma_max,
   σ_mid, 0]`, i.e. it does learn to denoise at two σ within its range.

2. **The handoff at the boundary is automatic and exact** — because each student's
   `sigma_max` *is* the boundary it starts from. Low-student `sigma_max = 200` =
   the shared boundary. The bundle sampler runs a **fastgen predict-x0-then-renoise
   trajectory with the boundaries as explicit `t_list` nodes**: e.g.
   `t_list = [2000, 200, 0]` → step 1 at σ=2000 dispatches to Student-Hi → x0 →
   renoise to 200; step 2 at σ=200 dispatches (boundary→lower) to Student-Lo → x0.
   No Karras-grid node needs to "happen" to land on 200; it is placed there by
   construction. (2-step low student → `t_list = [2000, 200, σ_mid, 0]`.)

3. **Bundling: container reuses the teacher path; the sampler does NOT.**
   - ✅ **Container is identical.** `save_student_checkpoint` writes each student
     in `DiffusionModel` format (`sampler_type="fastgen"` baked in), so each loads
     via `CheckpointModelConfig`, drops into a `DenoisingExpertCheckpointConfig`
     (checkpoint + σ range), and bundles via `DenoisingMoEConfig.build()` →
     `DenoisingMoEPredictor.save()` → `DenoisingMoEBundledConfig`. Exactly the
     teacher flow. ✅
   - ⚠️ **Sampler differs.** `DenoisingMoEPredictor.generate` hardcodes the
     Heun/EDM **stochastic** `edm_sampler` over a continuous Karras grid of
     `num_diffusion_generation_steps`, dispatching each grid step by σ. That is
     correct for teacher experts (full multi-step denoisers) but **wrong for
     one-shot students** — it would invoke a student at Karras σ it never trained
     on and would not place a node at the boundary. The student bundle needs the
     **fastgen predict-x0-renoise sampler driven by an explicit boundary-aligned
     `t_list`** instead. → **Structural code change: generalize
     `DenoisingMoEPredictor.generate` (or add a student-bundle variant) to accept
     an explicit `t_list` + fastgen sampler.** The σ-dispatch module itself is
     reused unchanged.

### 1 vs 2 steps for the low-noise expert (config, not architecture)

The 1-vs-2-step choice is **`student_sample_steps`** (env `ACE_STUDENT_STEPS`,
`fdistill_kl_spike.py:37`) — no architectural/structural difference, just the
`t_list` length the student is distilled against. So it's a cheap config A/B:
- **Student-Hi: 1 step** (coarse-from-noise; little fine structure to resolve).
- **Student-Lo: 1 vs 2 steps** — the low-σ decade is where the interesting
  denoising happens, so 2 steps may help. Total NFE: 2 (1+1) vs 3 (1+2). Run both
  and compare on the val suite.

### Validation & checkpoint selection for per-expert students (open design)

**The current validation does not transfer to per-expert students.**
`BestStudentCheckpointCallback` (`best_student_callback.py`) selects on CRPS (+
tail/spectra) of the **end-to-end student sample vs a pre-saved full-teacher zarr**
(`generate_val_dataset.py`, the full 18-step MoE denoising → final clean
ensemble). It samples the student via `fastgen_sampler(student._ace_module, …,
sigma_min=student._sigma_min, sigma_max=student._sigma_max)`, and
`fastgen_sampler` **forces `t_list[-1]=0`** (`samplers.py:244`) regardless of
`sigma_min`. So for Student-Hi (σ∈[200,2000]) it denoises straight to a clean x0
and compares to the full teacher — asking a segment specialist to do the whole
job. **This selection signal is wrong for both students and must change before
per-expert runs.**

**Decision (2026-06-24): train Lo first, then Hi — both validated end-to-end
against the *existing* full-teacher zarr; the teacher expert is never run live at
validation.** Validate each student *in the cascade* paired with a fixed partner,
so what's selected is its marginal contribution to the final sample. The metric
that matters is the final bundled sample and the existing zarr is already the
right final target — so we do **not** build per-segment *target* datasets.

**Why Lo-first.** Whichever student trains first has no student-partner yet and
must lean on a teacher-derived proxy; sequential training only moves the awkward
case to whoever's first. Lo-first makes *both* validations cheap and end-to-end:

- **Step 1 — Student-Lo (σ∈[0,200]), trained first.** Input is built on the fly
  by **re-noising the existing target zarr to σ=200** (no new dataset — see
  below); run Student-Lo once → x0; compare to the same target zarr. No live
  teacher.
- **Step 2 — Student-Hi (σ∈[200,2000]), trained second.** Validate end-to-end
  through the now-**frozen 1-step Student-Lo**: fresh noise@2000 → Student-Hi →
  renoise to 200 → frozen Student-Lo → x0; compare to the zarr. Cheap (Lo is one
  forward pass, not the multi-step teacher) and it's the real deployment path.
- **Final bundle selection** — assemble the real `[Student-Hi, Student-Lo]` bundle
  and validate end-to-end vs the zarr. Needs no new data.

The only train/deploy mismatch (Lo trains on the *teacher's* σ=200 state, deploys
on *Hi's*) is negligible: at σ=200 the input is `x0 + 200·ε` with `sigma_data≈1`,
so the noise dominates the clean part ~200× and washes out which upstream produced
the x0.

#### Re-noise the target zarr on the fly — no new dataset

Earlier drafts proposed dumping the teacher's intermediate latent at σ=200 to a
new store. **Not needed.** Construct Lo's validation input from the target zarr
the callback already loads: `x_200 = normalize(teacher_target_member) + 200·ε` with
fresh `ε ~ N(0, I)`. This is just standard diffusion validation — take clean data
(the teacher target), noise it to σ=200, ask Lo to denoise.

Why this stands in for the "true" teacher σ=200 latent: that latent's clean part
is the teacher's *running* x0 at the 200 step; this uses the teacher's *final* x0.
At σ=200 the `200·ε` term dominates the `O(1)` clean part ~200×, so the difference
is washed out (same argument as the negligible Lo train/deploy mismatch above).

No sampler change strictly required. `fastgen_sampler` does
`x = latents * t_list[0]` with `t_list[0] = sigma_max = 200` for Lo
(`samplers.py:246`), so passing `latents = ε + normalize(target)/200` yields
`x = 200·ε + normalize(target) = x_200` exactly. (Cleaner alternative: add an
explicit `x_init` path to `fastgen_sampler` that skips the `* t_list[0]` scaling
and takes the pre-built `x_200`; preferred if `t_list[0]` might not equal
`sigma_max` under some `max_step_percent` — confirm before relying on the fold
trick.)

Leakage is a non-issue: input member *i* is derived from target member *i*, but
because `x_200` is noise-dominated it carries almost no recoverable information
about that specific target, so Lo cannot "cheat" it back. Use fresh independent
`ε`; the energy-score CRPS pairs all student×teacher members regardless.

Ensemble semantics (the key subtlety): **Student-Lo is deterministic given its
input** (a 1-step map adds no fresh noise; a 2-step Lo's interior renoise is the
deterministic ODE step). The n-member output ensemble comes from re-noising the
**n teacher target members** (one `ε` draw each) → one Lo output per member
("ensemble-size-1 per member") — not from the current callback's fresh-noise draws.
For Student-Hi the ensemble *does* come from fresh noise@2000 (Hi is deterministic
per noise draw, so n draws → n members), each finished by the same frozen Lo.

Plumbing in `BestStudentCheckpointCallback`:
1. **Lo path:** for each batch, take the target members already selected from the
   zarr (the callback's existing `.sel(time=…).sel(lat/lon, method="nearest")`),
   normalize, form `latents = ε + x0_norm/200` (or pass `x_200` via the `x_init`
   path), run Lo, denormalize, compare — all other metric code unchanged.
2. **Hi path:** fresh-noise init chained through the frozen Lo — i.e. the
   boundary-aligned bundle sampler with `[Hi(training), Lo(frozen)]`.

Net new data: **none.** The expensive full-teacher target zarr is reused as both
the re-noise source and the comparison target. The cascade/bundle sampler (Hi
path) is the same one required for inference (structural finding 3 above).

### Remaining work / launch checklist

**Student-Lo is now runnable** (commit `7a48a342a`). Launch:
```bash
conda run -n fme bash configs/experiments/2026-05-18-distillation-with-val/run.sh \
    fdistill --moe-teacher --expert 0 --student-steps 1 --suffix lo-1step
# (also run --student-steps 2 for the 1-vs-2-step Lo A/B)
```

- ✅ **Single-expert teacher.** `AceDiffusionTeacher(model, expert_index=…)`
  selects one expert and restricts the teacher to its own σ range with no
  dispatch (`config.teacher = config.net =` that expert). Exposed via
  `--expert-index` / `$ACE_EXPERT_INDEX`; `run.sh --expert <0|1>` sets the σ
  range + index + val mode. Each student auto-derives its discriminator from its
  own in-domain expert (`in_channels` 128 / 256).
- ✅ **lo_renoise validation** (no new dataset). `BestStudentCheckpointCallback`
  `validation_mode="lo_renoise"` re-noises the target zarr members to the
  student's `sigma_max` and denoises from there (uses the
  `student_sampling.sample_student_*` helpers + `skip_noise_scale`). Wired via
  `--val-mode` / `$ACE_VAL_MODE` (set to `lo_renoise` automatically for
  `--expert 0`). GAN-fix levers (R1 reg, weight) still apply per-student.
- ⏳ **Then Student-Hi** from expert 1 (`--expert 1`, `min_t=200`, `max_t=2000`,
  `student_sample_steps=1`). **Blocked on `hi_cascade` validation**: `--expert 1`
  currently falls back to `from_noise` (mis-specified — denoises straight to 0).
  Needs: a `frozen_lo_checkpoint` arg on the callback + a `"hi_cascade"`
  `validation_mode` that runs `student_sampling.sample_student_hi_cascade`
  through the frozen Lo from the step above.
- ✅ **Bundle sampler** (structural finding 3) — **DONE**, see
  "SPEC (2026-07-07): distilled 2-step bundle sampler + end-to-end eval" above.
  `DenoisingMoEStudentPredictor` runs the fastgen predict-x0-renoise cascade over a
  boundary-aligned `t_list`; the explicit `sigma_ranges` are authoritative (the Lo
  `_primary` σ-range caveat is moot). Teacher EDM path untouched.
- ⏳ **Eval**: bundled 2-/3-step student on the same full-teacher val zarr
  (CRPS/spectra/tail). Per-student selection via the fixed-partner cascade.

### Design note: teacher input — bundle+`expert_index` vs. individual checkpoints

The validation **noising type is its own config** (`--val-mode` /
`validation_mode`), independent of how the teacher is loaded. For the *teacher
input* there are two possible mechanisms:

1. **Bundle + `--expert-index N`** (chosen, implemented). Load the bundled MoE
   and select expert N in-memory; the teacher's σ range is pulled from the
   bundle's **routing config** (`state["sigma_ranges"]` = `[0.005,200]`,
   `[200,2000]`).
2. **Individual component teacher via `--teacher-checkpoint`** (pre-existing
   single-`DiffusionModel` path; needs no `expert_index`). Fully decouples
   per-expert distillation from the MoE machinery.

**Why (1) was chosen and (2) was *not* extended:** the segment σ ranges live at
the **MoE level**, stored separately from each expert's own
`config.sigma_min/sigma_max` (see `DenoisingExpertCheckpointConfig` and
`serial_denoising.py:324`). A standalone checkpoint loaded via
`--teacher-checkpoint` takes its range from its **own** config
(`fastgen_teacher.py:139-140`) — the expert's *training* range, **not guaranteed
to equal the segment boundary**. That directly breaks `lo_renoise`, which
re-noises to `student._sigma_max` (must be exactly 200 for Lo). And
`ACE_SIGMA_MIN/MAX` only drives the *training noise distribution*
(`sample_t_cfg`), not the single-model teacher's σ range — so there is no
existing knob to correct it. Approach (1) gets the authoritative segment range
for free and needs only the bundle (which is what we have on hand; the
pre-bundle individual checkpoints are not referenced anywhere in this doc).

**Decision (2026-06-27): do not add a teacher σ-range override.** Making (2)
correct would require an explicit teacher-σ-range override (e.g.
`--teacher-sigma-min/max`, or feeding `ACE_SIGMA_MIN/MAX` into the single-model
teacher branch). We deliberately skipped it — bundle+`expert_index` is
sufficient and avoids the mismatch entirely. Revisit only if we ever need to
distill from a standalone expert checkpoint that is *not* in a bundle.

---

## ★★ ROOT-CAUSE BUG (2026-07-02): PRMSL "collapse" was a residual-model validation bug (fixed `b2a47628b`)

**Much of the PRMSL story below is confounded by a validation bug — read this first.**

The teachers are **residual models** (`predict_residual=True`): the net predicts
`fine − interpolate(coarse)`; the full field is recovered only by **adding the
interpolated-coarse base back** (`models.py::postprocess_generated`). The audit found:

- **Teacher val zarr = full field** (built by the standard `fme.downscaling.inference`
  path → base added).
- **Student validation output = residual only** — `BestStudentCheckpointCallback`
  sampled via `fastgen_sampler` on the raw module and **never added the base**
  (no `predict_residual` handling anywhere in `samplers.py` / `student_sampling.py`
  / the callback).
- ⇒ validation compared a **residual student to a full-field teacher**. For **PRMSL**
  (signal ≈ entirely the smooth large-scale base) the residual is nearly empty →
  the ~**100× low-k PSD deficit**, the **collapsed-inward / negative deep-low tail**,
  and the PRMSL-worst/precip-mildest pattern. It also explains why `spec_mae_lo_PRMSL`
  was ~constant-bad across all runs (the base bug) while `spec_mae_hi_PRMSL` varied
  with tap/GAN (the real fine-scale residual). `lo_renoise` had a second bug: it built
  the re-noise state from the **full** target, but the student expects `residual+noise`.

**Why it went undetected:** the `train_media` panels show the student's *normalized
residual* output (not the full field), per-channel min-max-stretched — and the paired
`data/real` is *also* the residual (the `teacher.sample()` target is residual). So
student-vs-real media were apples-to-apples (both residuals) and looked fine; the only
place a residual meets a *full field* is the callback vs the val zarr — the one buggy
comparison. The media looking fine actually *confirms* the student produces a correct
residual; the bug is purely validation-side.

**Training and deployment were already correct** — training is residual-consistent
(teacher targets, VSD, GAN all residual), and `save_student_checkpoint` preserves
`predict_residual=True`, so `DiffusionModel.generate` adds the base at inference. **The
bug was validation-only** — but it confounded every PRMSL metric (CRPS/spectra/tail →
**checkpoint selection**).

**Fix (`b2a47628b`, validation only):** `_base_prediction_norm` computes the
interpolated-coarse base (guarded by `predict_residual`), added back to the student
output for both modes; `lo_renoise` also subtracts it from the target before re-noising.

**Reset runs (submitted 2026-07-02, commit `184fa298b`) — CHECK THESE.** Correct-
validation baseline + tap2, 2-step expert-0:
- **baseline-fixed** (offset 0, original bottleneck critic): `01KWJAFKZ96YBR73F0TETBKC0Q`
- **tap2-fixed** (offset 2, 64²): `01KWJAFQW38E8AQ70YK7JYHCAK`

(The old confounded runs — tap4/tap6/tap2-gan3e3 — are canceled.)

**What's still valid vs. bunk** (the bug is *common-mode* — same missing base every
run — and *low-k only* — the base is smooth, ~no high-k):
- **Valid:** all **training metrics** (GAN losses / `fake_score` / `f_distill` /
  `ar1`) → the **GAN-collapse dynamics are real**; **all hi-band spectra** (base-free);
  **precip** (small base fraction); **relative run-to-run comparisons on the hi-band**
  (common base cancels) → the **non-monotone tap finding (64² sweet spot) and 2-step >
  1-step both stand**.
- **Bunk/confounded:** absolute **PRMSL lo/mid spectra**, the **PRMSL deep-low tail**,
  the **PRMSL histogram / full distribution shape** (the tail metric *is* the
  histogram — the student's PRMSL histogram was a narrow spike near the ~1013 mean:
  `denorm(residual) ≈ fine − coarse + mean`, no synoptic spread/deep lows),
  **CRPS_PRMSL**, and PRMSL-driven **checkpoint selection**; the "PRMSL large-scale
  collapse" as a real phenomenon. (Evaluator histograms via `model.generate_on_batch`
  add the base → not affected, but were never run on the students.)
- **GAN-reg runs (r1-instr/allfix): not bunk** — their conclusion (R1/allfix did NOT
  stop the GAN tipping / hi-band degradation) rests on training dynamics + hi-band,
  both valid. What was inflated is the *motivation* — the PRMSL "catastrophe" was
  largely the bug. Reset will show whether the real (milder, hi-band) collapse is even
  worth chasing.

**Consequences for the record below:**
- The **"PRMSL spectral collapse"**, the **negative deep-low tail**, and much of the
  **lo/mid-PRMSL tap comparison** are largely this bug, NOT a model/GAN failure. The
  **hi-band PRMSL** and precip/wind findings, and the GAN-collapse dynamics, still
  stand (those are the real residual/fine-scale signal).
- **Re-run validation on the fixed code** before trusting any PRMSL number or
  re-deciding tap depth / GAN weight. The tap "64² sweet spot / finer collapses" read
  is on the hi-band (real), but re-confirm once PRMSL low-k is measured correctly.

---

## ★ RESULT (2026-06-29): per-expert pivot did NOT fix the spectral collapse — GAN-fix runs launched

> ⚠️ **Residual-bug caveat** (see "ROOT-CAUSE BUG (2026-07-02)" above): the "PRMSL
> spectral collapse" central to this section is **largely the residual-base
> validation bug** (residual student vs full-field zarr; fixed `b2a47628b`), not a
> real large-scale failure. Hi-band spectra, GAN/training dynamics, and precip
> stand; the PRMSL lo/mid/tail claims are confounded — re-confirm on the reset runs.

**Both Lo runs completed** (~30–32k steps, ~44 h; wandb `6xt93hci` = Lo-1step,
`3vk3or7v` = Lo-2step). Wiring confirmed correct in logs (`expert_index=0,
sigma range [0.005, 200.0] (no dispatch)`, `validation_mode=lo_renoise`). Two
findings:

1. **2-step beat 1-step on every axis** — better PRMSL (`crps_PRMSL` 6.89 vs
   7.06), and far better precip spectra (`spec_mae_mid_PRATEsfc` best 0.017 vs
   0.077; `spec_mae_hi_PRATEsfc` 0.15 vs 0.43). **2-step is the base going forward.**
2. **★ The PRMSL coarse-spectral collapse RECURRED in both runs — falsifying the
   structural hypothesis.** The pivot's key claim was that an in-domain expert-0
   discriminator would prevent the collapse. It did not. Lo-2step
   `spec_mae_mid_PRMSL` 0.16→**0.82**, `spec_mae_hi_PRMSL` best 0.046@78%→**0.88**;
   tails blow up (`tail_99.9999_PRATEsfc` 0.54@1%→**0.96**). Same disc-winning
   signature as dispatch-v2: `fake_score_loss` spikes late (0.017→0.032),
   `gan_loss_disc`↓ / `gan_loss_gen`↑.
   - **Precip is good at lo/mid** (`spec_mae_lo/mid_PRATEsfc` → 0.02–0.04) — the
     pivot fixed the *domain* problem. **[Corrected 2026-06-30: precip is NOT fully
     healthy — it is under-powered at high-k (`spec_mae_hi_PRATEsfc` ~0.15–0.3 →
     ~0.4–0.5× teacher), i.e. too smooth at fine scales too; see "Corrected
     diagnosis".]**
   - **Checkpoint-selection trap (again):** the good model is mid-training
     (~78–88% for 2-step); `best_student.ckpt` (CRPS-selected) and the last
     checkpoint are spectrally collapsed. Pull a mid-training ckpt before judging.
   - The GAN-fix levers the doc filed as "should matter less now" are in fact the
     **primary** lever — none were enabled in these runs (`gan_r1_reg_weight`
     unset, `gan_loss_weight_gen` 1e-3, no LR decay).

### GAN-fix runs (submitted 2026-06-29, 2-step expert-0 base) — CHECK THESE

Commit `e39c6b237` added env knobs (`ACE_GAN_R1_REG_WEIGHT`,
`ACE_GAN_LOSS_WEIGHT_GEN`, `ACE_LR_DECAY_STEPS`/`ACE_LR_F_MIN`) surfaced via
`run.sh --gan-r1 / --gan-weight / --lr-decay-steps`. Both on `ai2/climate-titan`,
wandb `ai2cm/fastgen`.

| Run | GAN fixes | Beaker experiment | wandb name |
|---|---|---|---|
| ~~**Lo-2step-r1**~~ (cancelled — superseded by `-instr` below) | R1 reg `0.1` only | `01KWAKEV8SS9RACQH8AA6CGY2Y` | `…-lo-2step-r1-moe-teacher-expert0` |
| **Lo-2step-allfix** (everything) | R1 `0.1` + `gan_loss_weight_gen 3e-4` + LR decay→5% over 20k steps (max_iter capped 20k) | `01KWAKF2PV0G3E9V418207R7R0` | `…-lo-2step-allfix-moe-teacher-expert0` |
| **Lo-2step-r1-instr** (R1-only, fully instrumented) | R1 reg `0.1` only, on commit `8ce57479d` | `01KWBDJ5CH98Q0HMWS05BHFA8Z` | `…-lo-2step-r1-instr-moe-teacher-expert0` |

**`Lo-2step-r1-instr` (submitted 2026-06-29) is the one to read going forward** —
it's the only run with the full new instrumentation: normalized cross-var CRPS
selection, direction-aware multi-var tail selection, per-var `val/psd_<var>` raw
PSD curves, and all-4-variable train-media panels. The two runs above predate
those commits (old channel-0-only media, old selection, no PSD curves). Caveats:
`val/crps_mean`/`crps_best`/`tail_best_score` are **not comparable** to the
earlier runs (selection meaning changed); compare on per-var spectra, the PSD
curves, and the GAN-stability signals instead.

**What to check:** (1) jobs healthy + R1 active — expect a `gan_loss_ar1` metric
in wandb (logs only when both GAN weights >0); (2) **does `spec_mae_mid/hi_PRMSL`
stay flat/declining late** instead of climbing to ~0.8 (the whole point); (3)
`fake_score_loss` / `gan_loss_disc`-vs-`gan_loss_gen` stay balanced (no late
disc-winning tip); (4) precip spectra stay healthy (they already were); (5) tails
don't blow up. If R1-only holds the spectra, prefer it (simplest). If only allfix
works, the LR-decay/weight cut were load-bearing. Compare both at matched steps.

#### Check-in (2026-06-30): both GAN fixes FAILED to stop the collapse

> ⚠️ **Residual-bug caveat:** the PRMSL-`spec_mae` table below is confounded — the
> lo/mid rows are mostly the residual-base bug (constant common-mode); the hi rows
> and the GAN-balance rows are real. "Neither fix stopped the tipping" holds (rests
> on training dynamics + hi-band), but the PRMSL-collapse framing is inflated.

Read both live runs (`rzisfp5c` r1-instr @6.4k, `3lrjuahv` allfix @13k/20k). The
canceled non-instr r1 (`p1jxfj9x`) is superseded. **Neither fix prevented the
PRMSL collapse**; both reproduce the disc-winning signature.

| metric | allfix (13k) | r1-instr (6.4k) | read |
|---|---|---|---|
| `spec_mae_mid_PRMSL` | 0.13→**0.87** | 0.18→**0.94** | REAL deficit (student too smooth; not an artifact — no teacher floor) |
| `spec_mae_hi_PRMSL` | 0.075→**0.93** | 0.14→**0.88** | REAL deficit (too smooth) |
| `spec_mae_lo_PRMSL` | 0.59→0.67 | 0.62→**0.80** | real degradation |
| `crps_PRMSL` | 6.91→7.10 | 7.03→7.12 | ~flat |
| `gan_loss_gen`↑ / `gan_loss_disc`↓ | 0.74→1.01 / 1.34→1.17 | 0.73→0.78 / 1.37→1.33 | gen losing |
| `fake_score_loss` (late spike) | 0.016→0.031 | 0.016→0.022 | regime change |
| `gan_loss_ar1` | 0.02→**0.165** (8×) | 0.015→0.078 (5×) | R1 active but overpowered |
| precip `spec_mae_lo/mid_PRATEsfc` best | 0.05 / 0.025 | 0.06 / 0.056 | **healthy** |

- **R1@0.1 is active but losing** — `gan_loss_ar1` grows 5–8×, i.e. the disc's
  real-data gradient keeps rising; R1 at this weight is overpowered, not taming it.
- **allfix ≈ r1-instr** despite R1 + 3× lower gen weight + LR decay → the
  weight-cut and LR-decay are **not** load-bearing; none of the three toolkit
  levers fixed it.
- Precip (the low-noise expert's real job) stays strong throughout.

#### PSD calc verified; the "artifact" caveat was WRONG (retracted 2026-06-30)

Audited `compute_zonal_power_spectrum` + the `spec_mae` band reduction. Findings:

- **The PSD function is correct.** Parseval holds to 0.19% (the residual is the
  known Nyquist double-count seam — `ones_and_twos` multiplies *all* k>0 by 2
  including Nyquist; negligible, not the cause of anything).
- **`spec_mae` is a relative (log-PSD-ratio) error** — so the *raw* `val/psd_*`
  curves (absolute power + sign) are the ground truth, not the scalar.
- **⚠️ RETRACTION: the "mid/hi PRMSL is largely a metric artifact" claim was wrong.**
  It was inferred from the `spec_mae` scalars + a synthetic smooth-field demo, and
  *assumed* the teacher PRMSL/wind PSD craters to ~0 (a floor) at high k. **Reading
  the actual log10 mean PSD curves refutes this: there is NO floor for PRMSL or
  winds — the teacher retains real, finite energy across wavenumber.** So the
  deficits are **genuine, not division-by-near-zero.** Example (PRMSL, real curve):
  student `log10 PSD ≈ 1.0` vs teacher `≈ 1.5` → ratio ≈ 0.32 → **~3× too little
  power (real, ~0.5 dex)**. The artifact only *could* apply in a true teacher-floor
  region, which these fields don't have — so treat mid/hi `spec_mae` here as real.
- **Sign: it's a DEFICIT (student below teacher, too smooth), not excess texture** —
  `spec_mae = |log ratio|` hid the sign; the earlier "injects excess hi-k texture"
  reading was also wrong.
- **Trustworthy PRMSL signals are `spec_mae_lo_PRMSL` + `crps_PRMSL`** (and the new
  raw PSD curves). On those the damage is mild–moderate and coarse-band (lo
  0.59→0.67 / 0.62→0.80).
  → The headline mid/hi "collapse" number overstates the harm; read the **raw PSD
  curves** (absolute power vs wavenumber) and the coarse/lo signals, not mid/hi
  `spec_mae`.

#### Tail-generation verified (2026-06-30) — tails are healthy, not failing

> ⚠️ **Residual-bug caveat:** the **PRMSL** tail row here is confounded (residual
> student vs full-field zarr → narrow histogram near the mean; the later
> depth-based metric then read negative — both the residual bug, not real). Precip
> tails (small base) are approximately valid; winds intermediate. The general "tails
> are generated / improve with maturity" conclusion holds for precip.

Checked per-variable tail ratios (student pXX ÷ target pXX, ideal 1.0) across
runs. **The model is generating the tails.**

| var | 99.99 | 99.9999 | read |
|---|---|---|---|
| PRMSL | 1.01 | 1.02 | **exact, flat all training** |
| eastward_wind | 0.64 | 1.24 | 99.99 mod. under; 99.9999 noisy |
| northward_wind | 0.65 | 1.18 | same |
| PRATEsfc @7k (r1-instr) | 0.59 | 0.69 | under but climbing |
| PRATEsfc @30k (`3vk3or7v`) | **0.83** | **0.96** | recovers to near-target |

- **PRMSL extremes read ~1.0 — but that's the offset-blind metric bug, not real**
  (see "Metric bug found" below: a raw-pressure ratio is ~1.0 regardless of deep-low
  error). PRMSL is genuinely too smooth at high-k too (real deficit, no teacher
  floor), so its extremes are likely under-deepened — we just weren't measuring it.
  Smoothness and tails share one root (the high-k deficit) across all broadband
  fields; see "Corrected diagnosis" below.
- **Precip tails improve monotonically with maturity** (0.37→0.83 / 0.56→0.96 by
  30k). The under-prediction in the *young* GAN-fix runs (~0.6 @7k) is training
  immaturity, not collapse.
- **Winds ~0.64 under at the reliable 99.99**; the 99.99-under / 99.9999-over split
  shows **99.9999 is unreliable** (1-in-1e6 on a finite val set + histogram bins) —
  trust 99.99.
- **Correction:** the 2026-06-29 RESULT note called `tail_99.9999_PRATEsfc`
  0.54→0.96 a "tails blow up" symptom. That was a misread — for a student/target
  ratio, 0.54→0.96 is the tail matched *better*. Precip tails improve, not degrade.

**Metric bug found (2026-06-30, fixed `80db7e7b1`): the PRMSL tail ratio was
offset-blind.** The tail metric was `student_pXX / target_pXX` on *raw* values.
For PRMSL, direction is correctly the **lower** tail (deep lows), but the ratio of
raw pressures sits on a ~1000 hPa DC offset → a several-hPa deep-low error reads
`958/953 ≈ 1.005 ≈ 1.0`. **So the "PRMSL tails ~1.0, exact" readings above are an
offset artifact, not evidence deep lows are captured** — and given the
smoothness/high-k-deficit finding the student almost certainly *under-deepens*
lows, which this metric couldn't see. It also made PRMSL's `|ratio−1|` term ~0, so
it contributed ~nothing to `best_student_tail` selection. Winds (~0-centered) and
precip (≥0) ratios were fine. **Fix:** per-variable `tail_references`
(PRMSL=1000 hPa) + `_tail_magnitude` → PRMSL tail is now the **depth-below-1000
ratio** `(1000 − p_0.01)`; zero-referenced vars unchanged. `val/tail_*_PRMSL`
changes meaning (depth ratio) — not comparable to pre-fix PRMSL tail values.

**Implication for the planned spectral loss** (see also the corrected diagnosis
below, which links spectra and tails): restoring high-k variance plausibly helps
*both* spectra and tails (same root cause), but a *pure mean-PSD match* can be
satisfied by incoherent high-k noise — so it won't reliably fix tails on its own; an
adversarial fine-scale critic is better for coherent extremes, with the spectral
term as a cheaper scaffold.

#### Corrected diagnosis (2026-06-30): the student is too SMOOTH (high-k deficit), not over-textured

> ⚠️ **Superseded in large part by the residual bug (2026-07-02).** The "uniformly
> too smooth / low-k deficit" read below was taken from `val/psd` curves that
> compared a **residual student to a full-field teacher** — so the PRMSL (and some
> wind) *low-k* deficit was mostly the missing base, NOT a real model deficit. What
> survives: the **high-k** deficit (base-free) and precip. Re-derive the smoothness
> picture from the reset runs before trusting the low-k story.

Reading the raw `val/psd_<var>` curves directly (not `spec_mae`): **the student PSD
sits BELOW the teacher across all variables, with a deficit that grows toward high
wavenumber — the student is uniformly too smooth.** This **corrects the sign** of the
earlier interpretation (`spec_mae = |log(student/teacher)|` carries no sign; the
"GAN injects excess hi-k texture" reading was wrong — it's *under*-power).

- **Root cause is the few-step distillation itself, not (primarily) the GAN.** A 1–2
  step student's output ≈ the posterior mean `E[x0|x]` (Tweedie), which averages over
  fine-scale realizations → blurry by construction; the mass-covering forward-KL
  compounds it (hedges across modes instead of committing to a sharp one). "Too
  smooth" is the *default* when nothing restores high-frequency variance.
- **Why the GAN doesn't fix it — both prior hypotheses, reframed:**
  - *Encoder tap = the primary structural culprit, and why the deficit is universal.*
    The critic taps the **coarse bottleneck** → blind to high-k → gives **zero**
    gradient to add fine-scale power for *any* variable. No GAN weight/stabilizer can
    restore a band the critic can't see.
  - *Collapse compounds it.* When the disc wins, the generator gradient saturates →
    even the coarse bands it could shape stop improving.
- **Unifies smoothness and tails** (supersedes the earlier "spectra and extremes are
  decoupled"): too smooth = under-dispersed = high-k variance deficit → both
  **under-powered high-k spectra** AND the **tail** under-prediction (extremes are
  local high-frequency features). Same root cause. **This includes precip** — an
  earlier overstatement that "precip spectra are healthy" was only true at lo/mid;
  precip is **under-powered at high-k** (`spec_mae_hi_PRATEsfc` ~0.15–0.3 → ~0.4–0.5×
  teacher), and it directly drives the precip tail deficit. **PRMSL is NOT an
  exception** (earlier claim retracted): the raw curves show it too has real hi-k
  energy (no floor) and the student is genuinely too smooth there. PRMSL only *looked*
  fine on the tail metric because that metric was offset-blind (now fixed). The high-k
  deficit is **universal** — every broadband field is too smooth at fine scale.

**Relevance to experiments (updated):**
- The **finer / decoder / output-space critic** (tap1/tap2, decoder-tap, per-variable
  critics) is now the **primary lever**, not a side experiment: a critic that can
  *see* high-k is the only thing that can push the generator to produce it. Success
  signal = the student's `val/psd_*` high-k tail rising toward the teacher's.
- A **spectral loss must penalize the DEFICIT** (student PSD below teacher), not the
  excess assumed earlier. Caveat: a pure mean-PSD match can be met with *incoherent*
  high-k noise → prefer the adversarial fine-scale critic for coherent structure; use
  the spectral term as the cheap scaffold.
- The tap "risk" is reframed: we're texture-*starved*, not over-textured — the worry
  isn't unwanted texture but *incoherent* texture; the `val/psd_*` curves + tails are
  the check.

**Misconfiguration assessment of the GAN-fix line.** `r1-instr` and especially
`allfix` were rational under the *old* (wrong) diagnosis (GAN injecting texture /
collapsing) but **misconfigured for the real deficit**: R1 stabilizes, and `allfix`
*lowers*, the one force that could add high-k power — `allfix` (GAN weight 3e-4 + LR
decay) pushed toward **more** smoothness, the wrong direction. The *opposite* move
(higher GAN weight / less stabilization) is the sensible direction, **but it's
second-order to the tap blindness**: through the coarse bottleneck critic, more
weight just pushes harder on coarse structure (most likely worsening the coarse-PRMSL
degradation) and still can't manufacture high-k the critic can't see. So **tap
location is the gate; GAN weight is a multiplier only useful downstream of a
fine-seeing critic** — fix the tap first, then tune weight up. The tap runs
disentangle it: high-k PSD rising at the same ~1e-3 weight ⇒ blindness was binding
(weight then becomes a real knob); no lift ⇒ weight/coherence or a spectral term is
also needed.

#### PSD curves now step-slidable (commit `dc7876eeb`)

`val/psd_<var>` was a `wandb.plot.line_series` (Vega chart, latest-step only).
Converted to a matplotlib `loglog` figure logged as `wandb.Image` per validation
(`step=iteration`) — now a **step-slidable media panel** like the other
downscaling power-spectrum aggregators, with lo/mid/hi band boundaries marked.
Scrub the training axis to watch the high-k tail drift (the decisive real-vs-
artifact check). Takes effect on runs launched from `dc7876eeb`+.

#### Discriminator-tap A/B (submitted 2026-06-30, commit `9aca3c418`) — CHECK THESE

Shift the critic's tap up the encoder (finer band) to relieve the coarse-PRMSL
GAN damage. New knob `ACE_DISC_FEATURE_DEPTH` = offset toward finer resolution
from the bottleneck (`run.sh --disc-feature-depth`); the tapped level's channel
count sets the discriminator `in_channels` automatically, printed at launch.
Expert 0 levels finest→coarsest `[512,256,128,64,32,16]`, channels
`[128,256,256,256,128,128]`; offset 0 = bottleneck (16², 128ch, the historical
critic).

**Clean tap-only A/B (no R1, default gan_weight, 2-step expert 0)** so the tap
level is the *only* variable; the offset-0 reference is the original Lo-2step run
(`3vk3or7v`). Both launched from `9aca3c418`, so they carry the step-slidable PSD
curves + new selection/media.

| Run | tap offset | resolution / in_ch | Beaker | wandb name |
|---|---|---|---|---|
| **tap1** | 1 | 32² / 128ch (cheapest; disc size unchanged) | `01KWCXMJG2STT6VA6G078ZB86K` | `…-tap1-2step-moe-teacher-expert0` |
| **tap2** | 2 | 64² / 256ch (mid; disc doubles to 256ch) | `01KWCXMVRH7G0X4F3XW22XH33Q` | `…-tap2-2step-moe-teacher-expert0` |

**What to check:** (1) jobs healthy + the `DMD2 discriminator: feature_index=… depth
offset … resolution=… in_channels=…` log line confirms the tap moved (tap1→32²/128,
tap2→64²/256); (2) **does the coarse-band damage ease** — `spec_mae_lo_PRMSL` and
`crps_PRMSL` (the trustworthy signals) stay flat/declining vs the offset-0 baseline
(lo 0.62→0.80); (3) **side effect to watch (the open risk):** a finer critic on a
smooth field could *inject* hi-k texture — watch `spec_mae_hi_PRMSL` and especially
the raw `val/psd_PRMSL` curves (now step-slidable) for the student's high-k tail
rising above teacher; (4) precip stays healthy; (5) training not destabilized
(`f_distill_loss`, GAN balance) — tap2's 256ch disc is the higher-risk one. Decide
mid vs cheap tap on whether the coarse gain justifies any hi-k/stability cost.

**Also check the tap runs for tails/variance, not just PRMSL spectra.** Wind tails
plateau early (eastward/northward `tail_99.99` flat ~0.62 over r1-instr's 7k; no
mature run logs per-var wind tails yet). Hypothesis: the current critic taps only
the **coarse bottleneck**, so it's blind to the fine-scale local extremes that
wind/precip tails are made of; a **finer tap could lift wind/precip tails** (and
local variance), since adversarial training is the classical restorer of the
variance/sharpness that the mass-covering forward-KL blurs. So watch
`tail_99.99_{eastward,northward}_wind` and `tail_99.99_PRATEsfc` on tap1/tap2 vs
the offset-0 baseline. Caveats: double-edged (finer GAN = texture/instability
risk), GAN weight is only 1e-3 (stabilizer, not the dominant tail driver), and the
primary tail levers remain the forward-KL + maturity (a direct quantile loss is
more targeted than cranking the GAN). A PSD loss will not move tails (2nd moment).

(deferred: offset 4 / 256² high-res tap, and combining the winning tap with R1.)

##### Check-in (2026-07-01, tap1 @9.6k / tap2 @11.2k) — ★ finer tap WORKS on PRMSL

> ⚠️ **Residual-bug caveat:** these PRMSL `spec_mae` numbers (esp. lo/mid) are
> confounded by the residual base bug, so "finer tap fixes PRMSL" overstates it —
> the **hi_PRMSL** difference (tap2 0.05 vs collapsed 0.8) is base-free and real, so
> the tap-depth *effect on the fine-scale* stands, but the low-k "prevents collapse"
> framing is the bug. Re-confirm on the reset runs (baseline vs tap2, fixed).

**The finer critic largely prevents the PRMSL spectral collapse — confirming
tap-location (fine-scale visibility) was the binding constraint.** PRMSL
`spec_mae` last values (real deficits, no floor):

| band | tap1 (32², 128ch) | **tap2 (64², 256ch)** | rzisfp5c (bottleneck+R1) |
|---|---|---|---|
| lo_PRMSL | 0.66 | **0.51** (improving) | 0.75 |
| mid_PRMSL | 0.83 | **0.35** | 0.86 |
| hi_PRMSL | 0.84 (collapsed) | **0.05** (held to 11k) | 0.72 (collapsed) |

- **tap2 (64²) holds hi_PRMSL at ~0.05 from step ~500 to 11k** (no collapse),
  mid degrades only mildly, lo *improves* — the finer critic adds the missing
  high-k power and keeps it. Every prior run collapsed here.
- **tap1 (32²) collapses like the baseline** → the shallower tap isn't enough; the
  win needs the 64² tap. (Confound: tap2 is also the larger 256ch critic — finer
  *and* higher-capacity; can't fully separate the two yet.)
- **Open gaps:** winds high-k still degrades in *both* (`spec_mae_hi_eastward_wind`
  → 0.60 tap2 / 0.73 tap1) — the finer tap fixed PRMSL, not winds (needs even finer
  or per-variable). Precip slightly traded in tap2 (lo/mid 0.11/0.13 vs tap1's
  0.05/0.03). `fake_score_loss` still drifts up (0.017→0.031) but tap2's PRMSL is
  robust to it.
- **Tails largely unchanged** (precip ~0.63–0.71 @10k, young; winds ~0.6 plateau).
  ⚠️ **PRMSL tail reads 1.014 but these runs predate the depth-based fix
  (`80db7e7b1`) — still the offset-blind metric; ignore it.**

**Takeaway / next:** tap location is confirmed as the primary lever. Candidates:
(a) go finer still (offset 3/4 = 128²/256²) and/or per-variable to fix winds;
(b) now that the critic sees fine scales, GAN weight becomes a real knob (push
*up*); (c) relaunch the winner from a post-`80db7e7b1` commit so the depth-based
PRMSL tail is measured. Confirm tap2's high-k gain is *coherent* (not incoherent
noise) via the raw `val/psd_*` curves + a post-fix PRMSL depth tail.

##### Follow-up runs (submitted 2026-07-01, commit `f00d554bd`) — CHECK THESE

Crank the tap finer + test GAN weight. All from HEAD, so they carry the
depth-based PRMSL tail (`80db7e7b1`) + step-slidable PSD (unlike tap1/tap2).
Expert 0 has 6 levels (finest→coarsest `[512,256,128,64,32,16]`); `level =
5 − offset`; offset ≥5 clamps to the finest 512².

| Run | tap | resolution / in_ch | Beaker |
|---|---|---|---|
| **tap4-2step** | offset 4 | 256² / 256ch | `01KWF8W37GNK20SX8PYXH8ECAA` (running; static-image PSD) |
| **tap6-2step** | offset 6→5 | 512² / 128ch (finest, clamped) | `01KWFCEHZQ25E2TXQ8QJMFVEFN` (relaunched from `9dda320d6`) |
| **tap2-gan3e3** | offset 2 (64²) + `gan_weight 3e-3` | 256ch, 3× GAN | `01KWFCEN0DNASSXG0J5DF6FVB8` (relaunched from `9dda320d6`) |

tap6 & tap2-gan3e3 were relaunched from `9dda320d6` so they log the interactive
`val/power_spectrum/<var>` chart (raw fig, like the evaluator) instead of a static
Media image; their original submissions (`01KWF8WCQ…`, `01KWF8WQ6…`) were canceled
before starting. tap4 kept running (static-image PSD; data present).

What to check: (1) confirm the resolved tap in logs (`DMD2 discriminator:
feature_index=… resolution=… in_channels=…`); (2) does finer-than-64² help PRMSL
*more* or start injecting *incoherent* hi-k (watch raw `val/psd_*`, esp. whether
the student tail overshoots teacher); (3) **do winds high-k finally improve** at
256²/512² (the gap tap2 left); (4) does 3× GAN weight (with the 64² critic) push
high-k up further or destabilize (`fake_score_loss`, `gan_loss_*`); (5) the
**depth-based PRMSL tail** (`tail_99.99_PRMSL`) is now real here — watch whether it
drops below 1.0 (student under-deepening lows) rather than the old offset-blind
~1.0. NB: 512² is a near-pixel critic (expensive; big head) — watch throughput.

##### Check-in (2026-07-02) — ★ tap depth is NON-MONOTONE; 64² is the sweet spot

> ⚠️ **Residual-bug caveat:** the PRMSL `spec_mae` table below is confounded at
> lo/mid by the residual base bug, but the **non-monotone finding rests on hi_PRMSL
> (base-free) collapsing 0.05→0.8+ for 256²/512²** vs 64² holding — so "64² sweet
> spot, finer collapses" is a real (hi-band + training-dynamics) result and stands.

wandb: tap4 `bl59c5c5` @11.7k, tap6 `5x0409tg` @8.2k, tap2-gan3e3 `orzudu08` @4.5k.
PRMSL `spec_mae` (best→last):

| run | tap | mid_PRMSL | hi_PRMSL | verdict |
|---|---|---|---|---|
| tap2 (orig, `kutgg3xo`) | 64² | held | 0.05 @11.4k | ✓ **held** |
| tap4 (`bl59c5c5`) | 256² | 0.17→**0.80** | 0.05→**0.79** | ✗ collapsed |
| tap6 (`5x0409tg`) | 512² | 0.15→**0.87** | 0.03→**0.86** | ✗ collapsed |
| tap2-gan3e3 (`orzudu08`) | 64² + 3× GAN | 0.13→0.17 | 0.06→**0.50** (young) | ✗ degrading faster than orig |

**Finer than 64² does NOT help — 256²/512² collapse, and raising the GAN weight on
64² also collapses.** So it's not "finer/stronger = better"; there's an **optimum
around 64²** at the baseline 1e-3 weight, and going past it (finer critic OR higher
weight) tips the discriminator into winning → collapse (`gan_loss_gen`↑ to 1.1–1.4,
`gan_loss_disc`↓). At over-fine taps the critic likely **injects incoherent hi-k**
(the original texture-injection failure mode, now real). This supersedes the
tap1(32²)-vs-tap2(64²) "finer is better" read — that trend does NOT extrapolate.

**Unfixed by any tap:** winds high-k still degrades (`spec_mae_hi_eastward_wind`
0.58–0.75); the **PRMSL deep-low tail stays negative** (−0.1 to −0.25, *more*
negative in the collapsed 256²/512²) — the large-scale/deep-low deficit is
independent of tap; precip similar (hi→~0.4, tail~0.6).

**Caveats:** single runs (collapse *timing* may carry seed noise, but 64²-held vs
256²-collapsed at matched steps is a clear gap); tap2-gan3e3 young (4.5k). **Next:
64² is the operating point** — don't go finer or raise GAN weight. To fix winds /
deep lows, the tap lever alone is insufficient; per-variable critics or a
deficit-penalizing spectral term are the remaining levers (deep lows also need the
bundle's Hi expert, not testable in lo_renoise-Lo-alone).

### Checkpoint-selection fix (2026-06-29, `best_student_callback.py`)

The selection criteria were rebuilt so the saved checkpoints stop ignoring the
fields that were collapsing:

- **best-CRPS** (`best_student.ckpt`) now selects on a **std-normalized**
  cross-variable mean (`val/crps_mean`): each var's physical CRPS ÷ its
  normalizer std, then averaged. Previously the physical-units mean was
  dominated by PRMSL/winds and ~blind to precip (~1e-5). Per-var `val/crps_<var>`
  stay in physical units.
- **best-tail** (`best_student_tail.ckpt`) now reduces across variables with
  **per-variable tail direction**: PRMSL → **lower** tail (deep lows), precip +
  winds → **upper**. Score = mean over vars of `|student_p/target_p − 1|` at the
  top percentile (no cross-var cancellation). New per-var metrics
  `val/tail_<pct>_{PRMSL,eastward_wind…,northward_wind…}` now log too.
- Knobs: `tail_directions` arg + `_DEFAULT_TAIL_HIST_RANGES`/
  `_DEFAULT_TAIL_DIRECTIONS`. **PRMSL range assumes hPa** (`(900,1080)`) — units
  inferred from `plot_events.py` UNITS + the ~7 CRPS magnitude; revisit if the
  zarr is actually Pa. Pure helpers (`_tail_quantile_level`, `_normalized_mean`,
  `_tail_deviation_score`) are unit-tested in `test_best_student_callback.py`.
- ⚠️ `val/crps_mean`, `val/crps_best`, `val/tail_best_score` **changed meaning** —
  not comparable across this commit. The current GAN-fix runs predate it.

**Raw PSD curves now logged** (`val/psd_<var>`, log10 mean PSD line charts,
student vs teacher, per variable, each validation). Motivation was originally the
suspected smooth-field artifact — **but reading the curves refuted that
(retracted; see the PSD-verified retraction above): PRMSL/winds have NO teacher
floor, so the mid/hi `spec_mae` reflects a REAL high-k deficit (student too
smooth), not a division-by-~0 artifact.** The curves remain the right thing to
read because `spec_mae` is *relative* and unsigned — but here they *confirm* the
deficit rather than explain it away. (Coarse PRMSL structure also lives at high σ —
more the high-noise expert's job than Lo's, which standalone `lo_renoise` can't
exercise.)

### Train-media fix (2026-06-29, `fastgen_train.py`)

`train_media/student/generation` and `train_media/data/real` previously showed
**only output channel 0** (the first variable, ~u10): FastGen's `to_wandb`
asserts C==3, and the ACE patch had been replicating channel 0 to grayscale and
dropping channels 1-3. The "4 panels" were just the 4 per-GPU batch samples
(global batch 16 / 4 GPUs) of that one variable — *not* the 4 variables.

Now `_ace_to_wandb` renders **every output variable as its own panel** via
`_channels_to_grid`: one row per sample, one column per variable, each channel
**independently min-max normalized** so PRMSL and precip are each visible (they
live on very different scales in normalized space). Columns are labeled from
`out_packer.names` (caption `cols: u10 | v10 | PRMSL | PRATEsfc`). Guarded so a
viz failure falls back to the old renderer and never crashes training. Helper
logic (per-channel norm + sample-major ordering) verified; the `torchvision`
`make_grid` path only runs in the distillation Docker image (not the local
`fme` env).

### Design note (2026-06-29): discriminator tap location as a spectral-band lever (candidate experiment)

Hypothesis to revisit once we know whether the GAN even tips in the current R1
runs: the **coarse PRMSL damage may be caused by where the GAN taps the UNet.**

**Grounding (traced in code):**
- `fastgen_train.py:491-499` builds the discriminator with
  `feature_indices={deepest_idx}`, `deepest_idx = len(enc_info)-1`.
- `encoder_feature_info()` (`fastgen_teacher.py:306-334`) is ordered
  **finest→coarsest**, so `deepest_idx` = the **coarsest/bottleneck** level
  (16×16 for the 512² UNet). The critic today is a **pure coarse/global critic**.
- `Discriminator_EDM` (`discriminators.py:62-118`) builds one strided-conv head
  per tapped resolution down to 1×1, sharing a **single `in_channels`**.
- The capture infra `_capture_encoder_features` already accepts a **set** of
  indices, so a single shallower tap is plumbed; a multi-tap pyramid is not
  (shared `in_channels`).

**Why it could explain the damage.** Tap depth ≈ which spectral band the
adversarial gradient acts on. PRMSL's realism *is* its coarse/global structure,
so a bottleneck critic polices exactly the band that collapsed in dispatch-v2;
precip (fine/mid structure) is barely touched by a coarse critic — matching the
observed "precip healthy, coarse PRMSL collapses." **No attention in the UNet
sharpens the lever:** attention (normally at 16/8 levels) is what would give the
bottleneck a *global* receptive field; without it, scale sensitivity is set
purely by which conv level you tap, so tap-depth is a clean, near-monotone knob
(shallow = mid/fine, deep = coarse).

**Two options:**
1. **Single shallower tap — cheap (~3 lines):** `deepest_idx → deepest_idx - k`,
   set `in_channels` from `enc_info[level]`. For expert 0 (`[1,2,2,2,1,1]`) the
   two deepest levels are both 128ch, so one level up likely keeps
   `in_channels=128`. Shifts the policed band up ~1 octave, freeing coarse PRMSL
   for the score term (which is fully conditioned on the smooth coarse input).
   Suggest exposing as `ACE_DISC_FEATURE_DEPTH` (offset from deepest, default 0).
2. **Multi-resolution pyramid (coarse+mid+fine):** the principled projected-GAN /
   LADD design, but needs `Discriminator_EDM` to take a per-level `in_channels`
   list (or 1×1 projections) since heads currently share one channel count.

**Caveats / sequencing:**
- Pre-tip in the current runs — confirm the GAN actually damages coarse PRMSL
  (late window + `val/psd_PRMSL` + `spec_mae_lo_PRMSL`) before re-architecting.
- The mid/hi `spec_mae` is a **real high-k deficit** (retraction above — no teacher
  floor), so a finer tap targeting fine scales is addressing a real problem, not a
  metric artifact.
- Shallow-tap risk (reframed per the 2026-06-30 corrected diagnosis): the student
  is too *smooth* (high-k deficit) across all broadband fields, so a finer critic is
  *wanted* to ADD power — the worry is not unwanted texture but *incoherent* texture.
  Watch the `val/psd_*` curves and tails.
- Separable lever from the backbone-expert choice and the GAN weight. Related to
  the older "which expert should the discriminator use?" design note below.

### Design note (2026-06-30): per-variable / multi-critic discriminator (self-calibrating per-channel emphasis)

Motivated by the precip-good / winds-under asymmetry: precip tails recover to ~0.83
by 30k, wind tails plateau ~0.62 (at 7k) — a single shared objective can't push one
channel without touching the others. Traced the loss to pin down what's actually
adjustable.

**Key mechanic — `ratio_upper` is per-sample, not per-channel.**
`_get_f_div_weighting_h` (`f_distill.py:59-69`) does `fake_logits.mean(dim=1)` →
**one scalar per sample**, then `ratio = exp(·)`, then `clamp(·, ratio_lower,
ratio_upper)`; assert `ratio.shape == t.shape` (i.e. `[B]`). So `ratio_upper` caps
how much a rare/hard *sample* up-weights the VSD loss — the channels are already
collapsed into one critic logit before the clip. **You cannot make `ratio_upper`
per-channel with one critic**, and a *global* bump is doubly blunt here: (1)
per-sample → up-weighting a wind-extreme sample also re-weights its (already-good)
precip; (2) with today's coarse bottleneck critic the ratio tracks *coarse*
mismatch, not wind extremes, so it isn't even aimed at the problem.

**Two orthogonal axes** (don't conflate):
- **Tap depth / scale** = what the critic can *see* (coarse bottleneck → blind to
  fine-scale local extremes; finer tap → sees them). This is the queued tap1/tap2
  A/B. But those are still *single* critics mixing all 4 vars into one logit, so
  they buy visibility, **not** per-channel emphasis.
- **Per-channel** = whether a channel's signal is *weighted* enough vs the others.

**Favored direction (per the 2026-06-30 discussion): per-variable / multiple
critics — self-calibrating.** Each variable getting its own critic → its own
density ratio → its own `h` (and a per-channel `ratio_upper` falls out naturally),
so per-channel emphasis *emerges* from how distinguishable each field currently is
— no subjective static weights. This could also help **tails**: a critic that sees
a variable's local extremes gives that variable tail-relevant gradient (the
under-dispersed channels get pushed harder automatically).

**Architectural reality to solve first.** The critic is a *projected* GAN on the
**teacher's entangled ENCODER features** (`_capture_encoder_features` hooks
`unet.enc[block_key]` only — `fastgen_teacher.py:347`), not on the decoder or the
reconstructed output, and not on separable per-variable channels — so "one critic
per output variable" isn't clean at the feature level (the bottleneck feature
doesn't decompose per-variable, and the encoder mixes channels from layer 1).

**New axis (2026-06-30): encoder vs decoder tap.** The teacher forward runs the
full U (encoder→bottleneck→decoder→x0) in one call, so we can hook `unet.dec`
instead of/in addition to `unet.enc`. Decoder features are **output-aligned** (the
decoder reconstructs toward x0, fusing skips), so a decoder-side critic polices
fine-scale, output-relevant structure rather than compressed input semantics — the
right end for the per-variable / tail goals. Per-variable separability still only
appears at the **final output projection**, so clean per-variable critics still
mean output-space (pixel/spectral) critics; the decoder is the richer *intermediate*
between encoder-projected and raw-pixel. So the lever generalizes from "encoder tap
depth" to **where in the U (encoder vs decoder, how deep)**. Realistic routes:
- **Per-variable critics in output (pixel/spectral) space** — abandon the
  projected/feature design for a per-variable patch or *spectral* discriminator.
  Cleanest per-variable signal; also the natural home for a spectral-shape critic.
- **Multi-resolution pyramid (per-scale, not per-variable)** — the LADD-style
  coarse+mid+fine projected critic. `_capture_encoder_features` already accepts a
  **set** of indices (multi-tap capture is plumbed), but `Discriminator_EDM` shares
  a single `in_channels` across tapped resolutions, so a true pyramid needs
  per-level `in_channels` (or 1×1 projections) — the blocker.
- **Per-variable × per-level grid** (the fullest version raised 2026-06-30):
  combine both axes. Most expressive, most parameters/critics to balance.

**Self-calibrating critics vs manual per-channel weights (the tradeoff raised):**
- *Per-variable critics (self-cal):* no subjective weights, adapts as a channel is
  solved. Cost: more params and **C× the GAN-balance problem** — each critic can
  win/collapse, multiplying the instability we're already fighting; harder to tune,
  more compute.
- *Manual per-channel loss weights (option b):* one `w_c` vector, decoupled from the
  GAN, simple and likely **easier training dynamics**. Cost: subjective and *static*
  (a solved channel keeps its weight; needs retuning).
- *Cheap middle ground:* adaptive per-channel weights without C critics —
  uncertainty-weighting (Kendall multi-task) or GradNorm / running-magnitude
  normalization. Self-calibrating-ish, single critic, far simpler dynamics.

**Sequencing.** (1) Let tap1/tap2 mature and report whether finer *visibility* alone
lifts wind/precip tails (and whether winds climb on their own past 7k, as precip
did). (2) If a channel is visible but still under because the shared objective
favors precip → add per-channel *emphasis*: start with the cheap adaptive weights
or a static `w_c`, escalate to per-variable critics if self-calibration is worth the
C× dynamics cost. Do **not** use global `ratio_upper` for this (per-sample,
channel-collapsed, and coarse-critic-filtered).

### (superseded) Active per-expert runs (submitted 2026-06-27)

**Student-Lo distillation is submitted** (commit `fa6b49e9e`). Both step
variants of the 1-vs-2-step A/B, on Beaker workspace `ai2/climate-titan`, wandb
project `ai2cm/fastgen`.

| Run | What | Beaker experiment | wandb name |
|---|---|---|---|
| **Lo-1step** | expert 0, σ[0.005,200], `student_sample_steps=1`, val=lo_renoise | `01KW5QJ4K6JV2952RFQK9F07JW` | `…-fdistill-with-val-lo-1step-moe-teacher-expert0` |
| **Lo-2step** | expert 0, σ[0.005,200], `student_sample_steps=2`, val=lo_renoise | `01KW5QJJ39JS5GA9T50NH5PQSZ` | `…-fdistill-with-val-lo-2step-moe-teacher-expert0` |

**Capacity (confirmed, no change needed).** Student-Lo is a deep copy of teacher
**expert 0** (`expert_index=0` → `_ace_module = expert_modules[0]`): ~32.75M,
`channel_mult [1,2,2,2,1,1]`, discriminator `in_channels=128`. It already matches
the teacher-expert parameter count by construction — the multivariate experts are
the larger nets, and each student inherits its source expert's size. The old
capacity worry (one student pinned to a single size across the whole σ range) is
removed structurally by per-expert distillation. **Do not bump capacity.**

### What an agent should check (success criteria)

Use the committed helper (pure wandb, runs in the plain `fme` env — does **not**
import fastgen):
```bash
conda run -n fme python -m fme.downscaling.distillation.check_runs --list   # find ids by name
conda run -n fme python -m fme.downscaling.distillation.check_runs <id1> <id2>
conda run -n fme beaker experiment get 01KW5QJ4K6JV2952RFQK9F07JW   # Lo-1step health
conda run -n fme beaker experiment get 01KW5QJJ39JS5GA9T50NH5PQSZ   # Lo-2step health
```

1. **Job is healthy and on the new path.** Beaker state `running` (not crashed —
   the first dispatch-v2 died on a dtype bug, so confirm it survives init). In the
   logs confirm the per-expert wiring fired:
   `Single-expert teacher: expert_index=0, sigma range [0.005, 200.0] (no dispatch)`
   and `BestStudentCheckpointCallback active: … validation_mode=lo_renoise`.
2. **Validation CRPS improving** (`val/crps_mean` and per-var `crps_PRMSL`,
   `crps_PRATEsfc`, `crps_eastward_wind`, `crps_northward_wind`) — should fall from
   init now that the teacher (= expert 0) is in-domain for the whole [0.005,200]
   range the student is trained on.
3. **Spectra healthy, esp. PRATEsfc** (`val/spec_mae_*_PRATEsfc`). Fine-scale
   precip is where the low-noise expert does its work; expect monotone improvement.
4. **★ Key hypothesis — no PRMSL coarse-spectral collapse.** dispatch-v2 collapsed
   `spec_mae_lo/mid_PRMSL` ≈ step 7700 because a coarse, out-of-domain (expert-1)
   critic policed PRMSL across all σ. Lo's discriminator backbone is now **expert 0,
   in-domain at low σ**, so the collapse should **not** recur. If `spec_mae_lo_PRMSL`
   / `spec_mae_mid_PRMSL` stay flat/declining through late training, the structural
   fix is validated. If they climb, the GAN-fix levers still apply per-student
   (`gan_loss_weight_gen`, `gan_r1_reg_weight`; see "config knobs").
5. **GAN/score stable** (`gan_loss_gen`, `gan_loss_disc`, `fake_score_loss`,
   `f_distill_loss`). Watch for the disc-winning signature (`gan_loss_disc`↓ while
   `gan_loss_gen`↑) that preceded the dispatch-v2 collapse; expected to be milder
   now that the backbone is in-domain.
6. **Tail** (`val/tail_*_PRATEsfc` → 1.0). `best_student_tail.ckpt` tracks this
   separately from CRPS.

**Caveats when reading the metrics:**
- **CRPS is NOT comparable to the prior from_noise runs.** lo_renoise validates by
  re-noising the target to σ=200 and denoising back — a different (easier, noise-
  dominated) task than denoising from σ=2000 to 0. Judge Lo against its own
  trajectory and spectra, not against dispatch-v2's CRPS numbers.
- **Checkpoint-selection trap (carry-over watch).** dispatch-v2's CRPS-best
  checkpoint was the spectrally *worst* one. For Lo, confirm `val/crps_best` and the
  spectra agree on the same era; if they diverge, prefer the checkpoint with intact
  spectra and flag it. `best_student.ckpt` = CRPS-selected, `best_student_tail.ckpt`
  = tail-selected.
- A/B outcome: compare Lo-1step vs Lo-2step on the val suite; 2-step adds one NFE
  (total bundle NFE 3 vs 2) and may help the low-σ detail. Pick the better Lo before
  training Student-Hi (Hi is validated through the **frozen** chosen Lo).

**Next once Lo is selected:** implement `hi_cascade` validation (frozen-Lo arg +
mode), then submit Student-Hi (`--expert 1`); then the bundle sampler + eval.

---

## Root causes found

Previous multivar runs plateaued with poor validation: **CRPS was essentially
flat from step 0** while `spec_mae` improved early then degraded (spectral
artifacts). Two distinct bugs:

1. **Noise schedule truncated at the expert boundary.**
   `run.sh` set `ACE_SIGMA_MAX=200` (the expert-0/expert-1 boundary) instead of
   the true schedule max `2000`. The student was never trained on the
   high-noise decade `[200, 2000]`, yet its inference sigma grid spans the full
   range and its first (most important) generation step starts near sigma 2000.
   Also `ACE_C_OUT=5` was wrong (teacher has 4 out vars).

2. **Teacher never used its high-noise expert (bigger bug).**
   `AceDiffusionTeacher.sample()` / `forward()` ran the entire EDM sampler on
   **expert 0 alone**, so the x0 targets the student regresses to were expert 0
   *extrapolating* across `[200, 2000]`. Expert 1 was loaded, frozen, and never
   invoked. The MoE was effectively unused.

## Fixes applied

| Commit | Change |
|---|---|
| `8cdf58a3d` | `run.sh`: `ACE_SIGMA_MAX 200→2000`, `ACE_C_OUT 5→4` (+ comments). |
| `3a2dced4e` | `AceDiffusionTeacher._dispatch` — per-sample sigma routing (inclusive ranges; boundary→lower expert; out-of-range→nearest). Used for target generation (`sample`) and scoring (`forward`) on the original teacher. Student + auto-derived discriminator now initialised from the **high-noise expert 1** (45.87M, discriminator `in_channels=256`). Tests added in `test_teacher.py`. |

---

## Results status (previous, pre-fix runs)

> ⚠️ **Residual-bug caveat:** all `crps_PRMSL` / PRMSL-spectra numbers in this and
> the following legacy sections predate the residual-base fix (`b2a47628b`) and use
> the buggy validation (residual student vs full-field zarr), so **PRMSL values here
> are unreliable**. The flat-CRPS diagnosis for precip/winds and the training/GAN
> observations stand.

wandb project `ai2cm/fastgen`. Per-variable validation (first → best@frac → last):

**`z5usj8so`** (v3: loguniform noise but max_t still 200, MoE):
| metric | first | best | last |
|---|---|---|---|
| crps_PRMSL | 7.10 | 7.01 @76% | 7.08 |
| crps_eastward_wind | 3.23 | 3.04 @80% | 3.07 |
| crps_PRATEsfc | 5.73e-5 | 5.34e-5 @22% | 5.51e-5 |
| spec_mae_PRATEsfc | 1.24 | 0.19 @52% | 0.37 |

**`hvk8i27t`** (original MoE, wrong noise): same flat CRPS pattern.
**`k9irth9z`** (single-var "intended recipe", net 54.59M): trained fine —
reference for what "working" looks like.

Takeaway: CRPS basically didn't move (student stuck near init); spectra
degraded late. Consistent with both root causes above.

Param-count contrast (why capacity was also suspected): multivar student `net`
32.75M / discriminator 0.79M (`in_channels=128`) vs single-var 54.59M /
2.10M (`in_channels=256`). Initialising from expert 1 closes most of this gap
automatically.

---

## Active runs (launched 2026-06-23) — CHECK THESE

Both on Beaker workspace `ai2/climate-titan`, wandb project `ai2cm/fastgen`.

| Run | What | wandb | Beaker experiment | Commit |
|---|---|---|---|---|
| **Run 1** — noise fix only | `ACE_SIGMA_MAX=2000`, `ACE_C_OUT=4` | `syz25njv` | `01KVTPTSTSC9V4Z99TP374WC43` (`…-sigmafix-moe-teacher`) — **running, ~step 15.7k** (negative control, done its job) | `8cdf58a3d` |
| **Run 2** — noise + dispatch + capacity | Run 1 + MoE dispatch + student from expert 1 | `r9lerxok` | `01KVV3RZ8GVTSA2QA8M65AQMJ1` (`…-dispatch-v2-moe-teacher`) — **running, ~step 12.7k** (partial success; PRMSL spectra collapsing late — see 2026-06-24 update) | `19d5f0467` |
| ~~Run 2 (first attempt)~~ | crashed: `_dispatch` index_put dtype mismatch (float64 latents vs float32 module output) | — | ~~`01KVTQV9N2QMCEDGCK035KDSJH`~~ exit 1 | `3a2dced4e` |

Fix for the crash: cast module output to `out.dtype` in `_dispatch` (commit
`19d5f0467`); the EDM sampler runs float64 latents while experts return float32
under autocast, and `index_put` requires an exact dtype match. Regression test
`test_moe_dispatch_handles_dtype_mismatch` added.

### How to check the latest runs

Beaker status:
```bash
conda run -n fme beaker experiment get 01KVTPTSTSC9V4Z99TP374WC43  # run 1 (sigmafix)
conda run -n fme beaker experiment get 01KVV3RZ8GVTSA2QA8M65AQMJ1  # run 2 (dispatch-v2)
# or browse: https://beaker.org/ex/<id>  and  https://beaker.org/ws/ai2/climate-titan
```

wandb progress — use the committed helper
[`check_runs.py`](check_runs.py) (pure wandb; runs in the plain `fme` env, does
**not** import fastgen):
```bash
# list the most recent runs (id | state | step | name)
conda run -n fme python -m fme.downscaling.distillation.check_runs --list

# compare metrics across runs (current two + an old flat baseline)
conda run -n fme python -m fme.downscaling.distillation.check_runs \
    syz25njv r9lerxok z5usj8so
```
Current wandb run ids: **`syz25njv`** = sigmafix (run 1), **`r9lerxok`** =
dispatch-v2 (run 2). Default metric set mirrors the flat-CRPS diagnosis;
override with `--keys`.

Run launcher (re-run / new variant):
```bash
conda run -n fme bash configs/experiments/2026-05-18-distillation-with-val/run.sh \
    fdistill --moe-teacher --suffix <variant>
```
`gantry` lives in the `fme` conda env; it clones the **pushed** commit, so
commit + push before launching.

### Interim observations (2026-06-23, sigmafix ~step 7.4k / dispatch-v2 ~step 4.5k)

> **Superseded by the 2026-06-24 update below.** The 2026-06-23 read ("dispatch-v2
> spec_mae monotonic, no late degradation; PRATEsfc flat → maybe Run 3") was taken
> at step 4.5k, before the late-stage dynamics appeared. Both conclusions reversed:
> PRATEsfc is now the *healthiest* signal (Run-3 trigger averted) and the late
> degradation is real but localized to PRMSL. Kept for history.

Both running; dispatch-v2 is younger (~35% of sigmafix's progress) so its read
is preliminary.

- **sigmafix ≈ the old flat runs.** crps_PRMSL 7.13→best 7.00→7.08, crps_PRATEsfc
  flat, spec_mae improves then degrades — basically identical to `z5usj8so`.
  Confirms the prediction: **the noise-max fix alone does ~nothing**, because it
  only changes the *sampled* noise levels, not the teacher *targets* (still
  expert-0-extrapolated in this run). Useful as a clean negative control.
- **dispatch-v2 is qualitatively different and healthier where expected:**
  - `spec_mae_mean` 0.715 → ~0.54 **monotonic, no late degradation** — unlike
    every prior run. Encouraging.
  - `crps_PRMSL` → best 6.91 (vs the old ~7.00 floor), trending down.
  - `crps_PRATEsfc` ~5.6–5.8e-5, **flat** (≈ or slightly worse than sigmafix).
  - `f_distill_loss` higher (~1.1–1.4 plateau vs sigmafix's ~0.15) — **expected**,
    the dispatched targets are sharper/harder so absolute loss isn't comparable;
    it plateaued, not diverging. `gan_loss_gen` creeping up (0.70→0.84) — watch.
- **Pattern matches the Run-3 limitation exactly:** coarse/large-scale (PRMSL,
  spectra) improves while fine-scale precip (PRATEsfc) lags — consistent with the
  student + in-loss score term anchored to **expert 1** (good at high σ/coarse,
  out-of-domain at low σ/fine). **If PRATEsfc stays flat as dispatch-v2 matures,
  that is the trigger to do Run 3** (separate full-dispatch teacher config).
- **What to watch next:** spec_mae keeps declining without a late blow-up; PRMSL
  keeps dropping; PRATEsfc starts moving (good) or stays flat (→ Run 3);
  f_distill_loss / gan_loss_gen stay bounded.

### Update (2026-06-24, sigmafix ~step 15.7k / dispatch-v2 ~step 12.7k)

Both runs still **running** (`syz25njv` sigmafix, `r9lerxok` dispatch-v2). The
dispatch-v2 picture changed substantially as it matured.

**Dispatch-v2 broke the flat-CRPS plateau on every variable — including PRATEsfc.**
Per-variable CRPS (first → best@frac):

| metric | first | best | vs old floor |
|---|---|---|---|
| crps_PRMSL | 7.036 | 6.206 @99% | was ~7.00 |
| crps_PRATEsfc | 5.79e-5 | **4.75e-5 @81%** | was flat ~5.6–5.8e-5 |
| crps_eastward_wind | 3.244 | 2.718 @88% | was ~3.0 |
| crps_northward_wind | 3.594 | 3.042 @95% | — |

→ **Run-3 trigger is averted.** The doc's trigger was "PRATEsfc stays flat";
instead PRATEsfc dropped ~18% and — see decile analysis — its spectra (lo/mid/hi)
improve **monotonically to the end**. The dispatch fix alone is moving fine-scale
precip; the separate full-dispatch teacher is **no longer the priority**.

**sigmafix remains the negative control.** Step 15.7k, still ≈ old flat runs
(crps_PRMSL best 7.00, PRATEsfc flat 5.34e-5; GAN stable, gen ~0.92 flat). It did
its job; nothing more to learn from it.

#### Late-stage diagnosis (dispatch-v2 decile trajectories)

> ⚠️ **Residual-bug caveat (2026-07-02):** this "PRMSL large-scale spectral
> collapse" — the original observation that motivated the whole per-expert pivot —
> is measured with the buggy validation (residual student vs full-field zarr), so
> the PRMSL lo/mid-band collapse is **substantially the residual base bug**. A
> *late-in-training* worsening beyond the constant base offset may still be real
> (GAN-timed, visible in hi-band + GAN losses), but the large-scale-collapse framing
> and its magnitude are unreliable. The pivot may have been chasing a partly-illusory
> problem — reassess against the reset runs.

The "late degradation" seen in first/best/last is **not generic** — it is a
**PRMSL large-scale spectral collapse** starting ≈decile 6 (**step ~7700**), while
everything else keeps improving. Per-decile means (10 bins, steps ~1300→12900):

```
spec_mae_lo_PRMSL   0.65 0.60 0.52 0.42 0.28 0.18 | 0.77 1.22 1.51 1.70   ← collapse
spec_mae_mid_PRMSL  0.17 0.17 0.17 0.16 0.16 0.18 | 0.52 0.92 1.19 1.38   ← collapse
crps_PRMSL          7.04 7.03 6.98 7.03 7.15 7.49 | 8.17 8.35 7.40 6.52   ← spike, then recovers
spec_mae_lo_PRATEsfc 1.24 1.14 1.08 1.05 0.98 0.85 0.71 0.70 0.64 0.55    ← healthy, monotone
spec_mae_mid_PRATEsfc 1.47 1.42 1.37 1.33 1.26 1.13 0.97 0.94 0.85 0.71   ← healthy, monotone
spec_mae_hi_PRATEsfc 0.74 0.71 0.67 0.64 0.59 0.51 0.42 0.43 0.42 0.44    ← healthy
```

The collapse coincides **exactly** with the GAN/score machinery tipping at the
same decile:

```
gan_loss_gen    0.70 0.72 0.73 0.75 0.78 0.76 | 0.78 0.82 0.84 0.83   ← gen losing
gan_loss_disc   1.38 1.37 1.36 1.35 1.34 1.34 | 1.34 1.32 1.31 1.31   ← disc winning
fake_score_loss 0.05 0.05 0.04 0.03 0.03 0.05 | 0.10 0.09 0.11 0.13   ← regime change
```

sigmafix shows **none** of this (GAN flat, no PRMSL collapse), so it is specific
to dispatch-v2's dynamics and GAN-correlated, not a generic schedule artifact.

**Interpretation.** The damage is concentrated in the **coarse bands (lo/mid) of
the smoothest field (PRMSL)** — precisely the failure mode the discriminator
design note predicts: the GAN here is a **bottleneck-only (coarse/global) critic
on expert 1**. As the discriminator gains the upper hand (~step 7700) it trades
PRMSL large-scale spectral fidelity for pointwise sharpness — note crps_PRMSL
*recovers* to 6.52 at the end while its spectra keep climbing. PRATEsfc (energy at
fine scales, not what a coarse critic governs) is untouched and keeps improving.

**Checkpoint-selection trap.** `val/crps_best` and `crps_mean` both reach their
**minimum at the last decile** (3.055 / 3.079) — i.e. the spectrally-collapsed
model. `tail_best_score` saturates early (0.0015) and is non-discriminating. If
checkpoint selection is CRPS-only, **the saved "best" model is the spectrally
worst one.** The actual sweet spot is **decile 5–6 (≈ step 6000–7700)**: PRMSL
spectra at their best (lo 0.18, mid 0.16), precip already strongly improved, tails
not yet fully blown. Verify the selection criterion and/or keep the ~step-7k
checkpoint explicitly.

#### Is the run salvageable? / next experiments

The run is a **partial success worth keeping** (the ~step-7k checkpoint beats every
prior run on CRPS with intact PRMSL spectra), but left running it degrades PRMSL.
The precip side needs no further intervention. Candidate next experiments, ordered:

1. **Tame the GAN (highest leverage).** The collapse is GAN-timed and GAN-shaped.
   Options, cheapest first: (a) lower `gan_loss_weight_gen` (currently 1e-3) by
   3–10×; (b) **anneal/warmup-then-decay** the GAN weight so it stabilizes early
   training but doesn't take over after step ~7k; (c) gate the GAN loss to low/mid
   `t` (design-note candidate) so it stops policing the coarse band where it does
   the PRMSL damage.
2. **Switch discriminator backbone to expert 0** (design-note candidate, decouple
   from student init). Makes the critic less of a pure coarse critic; directly
   targets the PRMSL-large-scale failure. Pair with (1c) `t`-gating.
3. **Shorten schedule / decay LR.** Useful learning is essentially done by step
   ~7k; an LR cosine-decay to ~step 8–9k may bank the gains before the GAN tips.
   Cheapest "save the recipe" option but treats the symptom, not the cause.
4. **Run 3 (dispatched score term) — deprioritized.** Its motivation (flat
   PRATEsfc) no longer holds. Revisit only if a GAN-fixed run then shows a
   *low-σ / fine-detail* gap.

**Relevant config knobs** (`configs/fdistill_kl_spike.py`):
- `config.model.gan_loss_weight_gen = 1e-3` (line 95) — lever for (1a).
- `config.model.gan_r1_reg_weight` defaults to **0.0** (FastGen
  `config_dmd2.py:49`) — **R1 discriminator regularization is off.** Turning it on
  is the standard fix for a discriminator that's "winning" (our `gan_loss_disc`↓ /
  `gan_loss_gen`↑ signature) and is a cheap additional stabilizer to try alongside
  (1).
- GAN weight **anneal/warmup (1b) and `t`-gating (1c) have no existing knob** —
  they need a code change at `f_distill.py:169` (`loss = f_distill_loss +
  weight * gan_loss_gen`), e.g. a step- or `t`-dependent `weight`.
- All three optimizers at `lr=1e-5` with **no decay schedule** (lines 87–89) —
  relevant to (3).
- Discriminator backbone = expert 1 is fixed in the teacher code
  (`_primary_ace_module`), not this config — relevant to (2).

---

## Known limitation → candidate Run 3

> **Background / superseded** by the per-expert distillation direction (see
> "Current direction (2026-06-24)" above). Run 3 was the single-student workaround
> for the score-term domain mismatch; per-expert distillation removes the mismatch
> structurally. Kept for context.

FastGen builds the frozen `self.teacher` (used inside the f-distill loss) from
the **same factory** as the student (`config.net`), via `_copy_ace_teacher`,
which drops the expert list so the student is a single net. Consequence:

- The **x0 targets** the student regresses to (`data["real"]`, from the
  original teacher's `sample()`) **do** fully dispatch — correct. ✅
- But the in-loss `teacher_x0` **score term** and the discriminator's feature
  extractor run on a **single expert** across all sigmas. With expert 1 as
  primary: correct at high sigma, but **out-of-range at low sigma** (where
  expert 0 belongs).
- The discriminator's real-vs-fake comparison is still *consistent* (same
  extractor for both), so it is not comparing mismatched references — the
  features are just less meaningful at low sigma.

**Proper fix (Run 3 if Run 2 stalls):** FastGen supports a separate teacher
config (`FastGen/fastgen/methods/model.py:179-195`:
`self.teacher = instantiate(config.teacher or config.net)`). Set
`config.model.teacher` to a factory that **keeps all experts** (full dispatch)
while `config.model.net` stays the single-expert student. Then dispatch the
in-loss `teacher_x0` across experts; extract discriminator features from one
consistent expert (a small two-pass tweak in the `forward()` feature branch,
since the discriminator has a fixed `in_channels`).

Trigger for doing this: Run 2 closes the high-noise gap but **low-sigma /
fine-detail metrics lag** (`spec_mae_hi_*`, `PRATEsfc` tails).

---

## Design note: which expert should the discriminator use? (resolved by the pivot)

> **Resolved** by per-expert distillation (see "Current direction (2026-06-24)"):
> each student uses its own expert as the discriminator backbone, over that
> expert's σ range, so the backbone is always in-domain and the question below no
> longer needs an answer for the single-student case. Kept for the analysis.


The GAN here is a **projected / feature-space discriminator**, not a pixel GAN.
A sample (real = teacher x0, fake = student output) is re-noised to a random
`t`, run through the teacher UNet encoder, and the activation at the **single
deepest/bottleneck block** (`feature_indices={deepest_idx}`) is captured via
hook and fed to `Discriminator_EDM` (`fastgen_train.py:447-455`;
`f_distill.py:146-154`; real side `dmd2.py:277`). Notes:

- Real and fake use the **same backbone**, so the comparison is consistent
  (not mismatched) regardless of which expert it is.
- It taps **only the bottleneck**, so it is structurally a **coarse/global**
  critic, not a fine-texture one (skip-connection high-res features never reach
  it). GAN weight is small (`gan_loss_weight_gen=1e-3`) — a stabilizer.

**The concern (raised 2026-06-23):** the backbone is currently **expert 1**
(coarse-structure specialist), but **expert 0** is the detail refiner that
holds the fine information. Whether expert 1 makes sense is **`t`-dependent**:

- **Low `t`** (detail survives the perturbation): this is where detail-realism
  artifacts show and where the GAN adds value — want **expert 0** (detail-
  sensitive, in-domain). Expert 1 here is out-of-domain *and* detail-blind.
- **High `t`** (sample ≈ pure noise): detail is destroyed anyway, only coarse
  structure remains — and **expert 0 is now far out of its training domain**
  (never saw σ=2000), so its features get unreliable. Expert 1 is in-domain.

So neither expert is universally right, same as the score term. Can't simply
dispatch the backbone by `t`: the experts have **different bottleneck channel
counts (128 vs 256)** and the discriminator head has a fixed `in_channels`
(would need two discriminators or a projection).

**Three separable roles** were bundled into `primary = expert 1`; they need not
agree:

| Role | Best expert | Why |
|---|---|---|
| Student init | Expert 1 | bigger net; first 2-step gen step starts at σ≈2000 (coarse-from-noise) |
| In-loss score (`teacher_x0`) | dispatch both | routes by σ cleanly (Run 3 fix) |
| Discriminator backbone | arguably Expert 0 | detail-sensitive where the GAN helps; but out-of-domain at high `t` |

`_primary_ace_module` (feature extractor) does **not** have to be the module the
student is copied from — these can be decoupled.

**Candidate change (not yet decided):** decouple the discriminator backbone from
student init — use **expert 0** as the feature extractor **and gate the GAN loss
to low/mid `t`** (where detail matters and expert 0 is in-domain), while the
dispatched score term (Run 3) handles correctness across all σ. Keep student
init = expert 1. Two per-expert discriminators is the faithful-but-overkill
alternative at this GAN weight.

Status: **noted for decision; no code change yet.** Priority is behind the
Run 3 score-term fix (the score term is the primary driver; the GAN is 1e-3).

---

## Local dev notes

- Use the `fme` conda env. FastGen is the `FastGen/` submodule, **not**
  pip-installed; do **not** install its deps into `fme` — the distillation
  tests (`test_teacher.py`) run in the distillation Docker image / CI.
- The dispatch routing logic was validated locally with a standalone replica;
  `fastgen_teacher.py` / `test_teacher.py` pass ruff + mypy via pre-commit.
- `scratch/` is gitignored (used `scratch/moe/` for the checkpoint and
  throwaway inspection scripts).
