# SESSION HANDOFF — MoE distillation (2026-07-02)

> **This is a transient session-handoff note, not the living record.** It captures
> the *current* state and *immediate* next actions so work can resume in a fresh
> session. The durable, full history is in **`MOE_DISTILLATION_STATUS.md`**; the
> mechanism reference is **`FDISTILL_LEARNING.md`**; the refactor backlog item is
> **`specs/10-share-generation-postprocessing.md`**. When this note goes stale,
> trust those.

Branch: `experiment/fastgen-distill` — HEAD `45c7a0e73`, pushed/synced to origin.
Env: conda `fme`. Cluster: Beaker `ai2/climate-titan`, wandb `ai2cm/fastgen`.

---

## TL;DR — the headline of this session

**Found and fixed a validation bug that confounded all PRMSL results.** Teachers
are **residual** models (`predict_residual=True`): the net predicts
`fine − interpolate(coarse)` and the full field only exists after adding the
interpolated-coarse **base** back. The distillation *validation* callback sampled
the student via `fastgen_sampler` on the raw module and **never added the base**,
so it compared a **residual student** to a **full-field** teacher zarr. For PRMSL
(≈ all large-scale base) that manifested as the "~100× low-k spectral collapse,"
the negative deep-low tail, the narrow histogram — **all of it largely the bug,
not a model failure.** Training and deployment were already correct.

- **Fix:** `b2a47628b` (validation adds the base back for `from_noise` &
  `lo_renoise`; `lo_renoise` also subtracts it from the re-noise target).
- **Guard:** `febd81f37` — parity test that the callback's residual→full
  processing equals `DiffusionModel.postprocess_generated`.
- **Backlog:** `specs/10` — de-duplicate the residual/base logic (single-source
  `DiffusionModel.residual_base` + reuse `postprocess_generated`) so this class of
  bug can't recur.

**What's still valid vs. bunk** (bug is common-mode + low-k):
- **Valid:** all training metrics → GAN-collapse dynamics are real; **all hi-band
  spectra**; **precip** (small base); relative hi-band comparisons → the
  **non-monotone tap finding (64² sweet spot, finer collapses)** and
  **2-step > 1-step** both stand.
- **Bunk/confounded:** absolute PRMSL lo/mid spectra, PRMSL deep-low tail, PRMSL
  histogram, CRPS_PRMSL, and PRMSL-driven checkpoint selection.

---

## Live runs — CHECK THESE (reset on fixed code)

Submitted 2026-07-02 from `184fa298b` (has the residual fix). **Status: queued /
not yet started (waiting on GPUs)** as of this note — re-check.

| Run | Critic | Beaker | wandb |
|---|---|---|---|
| **baseline-fixed** | offset 0 (original bottleneck) | `01KWJAFKZ96YBR73F0TETBKC0Q` | (name `…-baseline-fixed-…-expert0`) |
| **tap2-fixed** | offset 2 (64²) | `01KWJAFQW38E8AQ70YK7JYHCAK` | (name `…-tap2-fixed-…-expert0`) |

All older tap/GAN-fix runs are **canceled** (confounded validation). Both are
2-step, expert-0, `lo_renoise` validation.

**What to read once they train (~7–12k steps before the picture is clear):**
1. **Does PRMSL now validate correctly?** `spec_mae_lo_PRMSL` should track the
   teacher (not the constant ~0.5–0.8 base artifact); the depth-based
   `val/tail_99.99_PRMSL` should be **positive / near 1** (not the old negative);
   the raw `val/power_spectrum/PRMSL` curve's low-k should sit near the teacher.
2. **baseline vs tap2 under correct PRMSL validation** — does the 64²-tap
   advantage persist, or was it mostly the (now-removed) low-k confound?
3. **Is the real (hi-band, GAN-timed) collapse even worth chasing** now that the
   PRMSL "catastrophe" is gone? (`spec_mae_hi_*`, `gan_loss_gen/disc`,
   `fake_score_loss`.)

Check commands:
```bash
conda run -n fme python -m fme.downscaling.distillation.check_runs --list
conda run -n fme python -m fme.downscaling.distillation.check_runs <wandb_id> --keys \
  val/spec_mae_lo_PRMSL val/spec_mae_hi_PRMSL val/tail_99.99_PRMSL \
  val/spec_mae_hi_PRATEsfc train/gan_loss_gen train/gan_loss_disc train/fake_score_loss
conda run -n fme beaker experiment get 01KWJAFKZ96YBR73F0TETBKC0Q
```
Raw PSD curves: wandb run → **Media** section → `val/power_spectrum/<var>` (interactive,
step-slidable; the evaluator-style chart).

---

## Immediate next actions (priority order)

1. **Read the reset runs** once they've trained (see above). This gates every
   downstream decision — most of the prior analysis needs re-confirming on correct
   PRMSL validation.
2. **Operating point is settled from valid results:** 64² tap, 2-step, GAN weight
   1e-3. Do **not** go finer than 64² or raise GAN weight (both confirmed
   counterproductive — non-monotone; see status doc 2026-07-02 tap check-in).
3. **Original next milestone still pending: Student-Hi + the bundle sampler.**
   `hi_cascade` validation isn't wired in (needs a frozen-Lo arg + mode calling
   `student_sampling.sample_student_hi_cascade`); the bundle sampler
   (`DenoisingMoEPredictor.generate` → boundary-aligned fastgen `t_list`) is
   scaffolded (`boundary_aligned_t_list`, `sample_student_hi_cascade`) but not
   wired into the predictor. Deep lows / large-scale PRMSL are genuinely the Hi
   expert's job, testable only in the bundle.
4. **Candidate levers (post-reset, if a real gap remains):** per-variable /
   output-space critics (with per-variable weights — smooth PRMSL is collapse-
   prone, keep it weak; see status "per-variable / multi-critic" design note); a
   deficit-penalizing spectral loss (helps PRMSL low-k + precip fine-scale, but a
   pure PSD match won't fix tails; a critic is better for coherent extremes).
5. **Backlog refactor:** `specs/10` (de-dup residual/postprocess). Low-risk,
   no-behavior-change; do before more validation-path changes.

---

## Code landed this session (all committed + pushed)

- **Residual bug fix** (`b2a47628b`) + **parity test** (`febd81f37`):
  `best_student_callback.py` `_base_prediction_norm` / `_sample_student_output`;
  `test_best_student_callback.py`.
- **PRMSL tail = depth-below-1000** (`80db7e7b1`): raw-pressure ratio was
  offset-blind (~1.0 regardless of deep-low error); now measures anomaly depth.
- **Step-slidable → interactive PSD chart** (`dc7876eeb`, then raw-fig
  `9dda320d6`): `val/power_spectrum/<var>` logged as a raw matplotlib fig (like the
  evaluator's `power_spectrum/<var>`), not a static image.
- **`ACE_DISC_FEATURE_DEPTH` tap-depth knob** (`9aca3c418`): `fastgen_train.py` +
  `run.sh --disc-feature-depth`; the tapped encoder level sets the discriminator
  `in_channels`.
- **`FDISTILL_LEARNING.md`** (`2493b8389`): mechanism explainer (VSD, fake_score,
  h/forward-KL, Tweedie vs reverse-ODE, where the critic taps).
- **`specs/10`** (`45c7a0e73`): the de-dup refactor task.
- Many `MOE_DISTILLATION_STATUS.md` updates, incl. ⚠️ residual-bug caveats on all
  earlier PRMSL analysis sections.

(Earlier in the session: `ACE_EXPERT_INDEX` single-expert teacher + `lo_renoise`
validation to make Student-Lo runnable; GAN-reg knobs `--gan-r1/--gan-weight/
--lr-decay-steps`.)

---

## Watch-outs / gotchas for the next session

- **`test_teacher.py` needs FastGen** (`diffusers` etc.) → runs only in the
  distillation Docker image / CI, not the plain `fme` env. `test_student_sampling.py`
  and `test_best_student_callback.py` **do** run in `fme`.
- **`run.sh` clones the *pushed* commit** — commit + push before launching.
- Expert 0 UNet has 6 levels (finest→coarsest `[512,256,128,64,32,16]`); tap
  `offset = 5 − level`; offset ≥5 clamps to finest 512².
- Validation is `lo_renoise` on the **Lo student alone** from σ=200 — a faithful
  proxy for its bundle role, but deep lows/large-scale ultimately need the Hi
  expert (not testable standalone).
- Don't re-trust any PRMSL number from before `b2a47628b`.
