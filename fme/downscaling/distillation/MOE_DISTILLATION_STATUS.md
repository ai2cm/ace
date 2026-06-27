# Multivariate MoE Distillation — Status & Handoff

Living notes for the multivariate MoE FastGen distillation effort on branch
`experiment/fastgen-distill`. Pick up here from any clone.

_Last updated: 2026-06-24._

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
- ⏳ **Bundle sampler** (structural finding 3): generalize
  `DenoisingMoEPredictor.generate` to a fastgen predict-x0-renoise cascade over a
  boundary-aligned `t_list` (the primitives `boundary_aligned_t_list` /
  `sample_student_hi_cascade` already exist). Also: the Lo checkpoint's recorded
  σ range inherits from the MoE `_primary`, not the expert range — confirm
  `DenoisingExpertCheckpointConfig`'s explicit range overrides it at bundle time.
- ⏳ **Eval**: bundled 2-/3-step student on the same full-teacher val zarr
  (CRPS/spectra/tail). Per-student selection via the fixed-partner cascade.

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
