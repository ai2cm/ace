# SESSION HANDOFF — MoE distillation (2026-07-05)

> **This is a transient session-handoff note, not the living record.** It captures
> the *current* state and *immediate* next actions so work can resume in a fresh
> session. The durable, full history is in **`MOE_DISTILLATION_STATUS.md`**; the
> mechanism reference is **`FDISTILL_LEARNING.md`**; the refactor backlog item is
> **`specs/10-share-generation-postprocessing.md`**. When this note goes stale,
> trust those.

Branch: `experiment/fastgen-distill` — HEAD `e920ca7f4`, pushed/synced to origin.
Env: conda `fme`. Cluster: Beaker `ai2/climate-titan`, wandb `ai2cm/fastgen`.

---

## TL;DR — the headline of this session

**The residual-bug fix is CONFIRMED by the reset runs, and Student-Hi is now
launched.** Two things happened:

1. **Reset runs (expert-0 Student-Lo, fixed validation) confirm the residual bug
   was the cause of the "PRMSL catastrophe."** Both trained ~75 h to 47–51k steps,
   then canceled. PRMSL `spec_mae` lo/mid/hi dropped from the old collapsed
   0.5–0.9 to **0.01–0.05 and held stable** to 50k; `tail_99.99_PRMSL` is now
   **~0.98–1.0** (was negative); winds hi-band healthy (0.04–0.08). **The offset-0
   bottleneck critic no longer collapses** → the whole tap-depth saga (64² sweet
   spot, finer-collapses) was **largely a validation artifact**. baseline-fixed ≈
   tap2-fixed → the simpler offset-0 critic is fine; the 64²/256ch tap is not
   needed.
2. **`hi_cascade` validation is now wired** (`e920ca7f4`) and **Student-Hi is
   launched** (see Live runs). This unblocks the per-expert Hi milestone.

Prior-session context (still true): teachers are **residual** models; the
validation bug compared a residual student to a full-field zarr. Fix `b2a47628b`
adds the base back; guard `febd81f37`. Backlog `specs/10` de-dups the logic.

---

## Live runs — CHECK THESE

**Student-Hi (expert 1), launched 2026-07-05 from `e920ca7f4`** — the first run
with `hi_cascade` validation (cascades the trained Hi through the frozen
baseline-fixed Lo). 1-step Hi, `lo` frozen at 2-step.

| Run | Beaker | Notes |
|---|---|---|
| **hi-1step** | `01KWTXGADFPB4GKVZ33C7ZGJP4` | expert 1, `hi_cascade`, frozen-Lo = baseline-fixed |

Frozen Lo checkpoint: dataset `01KWJAFM694MAE55M2JMZSE89M`, path
`fastgen/…-baseline-fixed-…-expert0/student_checkpoints/best_student.ckpt`
(mounted at `/frozen_lo`).

**Reset Lo runs (expert-0, canceled/finalized — DONE, results committed):**

| Run | Critic | Beaker | wandb |
|---|---|---|---|
| **baseline-fixed** | offset 0 (bottleneck) | `01KWJAFKZ96YBR73F0TETBKC0Q` | `zct08386` (step 46.8k) |
| **tap2-fixed** | offset 2 (64²) | `01KWJAFQW38E8AQ70YK7JYHCAK` | `985c6mia` (step 51.4k) |

**What to read on hi-1step once it trains:**
1. **Does hi_cascade validate cleanly?** Confirm startup logged "Frozen Lo student
   loaded … for hi_cascade validation" and no load error. Then watch
   `val/spec_mae_*_PRMSL`, `val/tail_99.99_PRMSL`, `val/crps_*` — the Hi's job is
   the deep lows / large-scale PRMSL that Lo-alone couldn't reach.
2. **Does the frozen-Lo cascade improve PRMSL deep lows** vs the Lo-alone
   `lo_renoise` numbers (the reset runs)?
3. GAN/training stability as usual (`gan_loss_*`, `fake_score_loss`).

Check commands:
```bash
conda run -n fme python -m fme.downscaling.distillation.check_runs --list
conda run -n fme python -m fme.downscaling.distillation.check_runs <wandb_id> --keys \
  val/spec_mae_lo_PRMSL val/spec_mae_mid_PRMSL val/spec_mae_hi_PRMSL \
  val/tail_99.99_PRMSL val/crps_PRMSL train/gan_loss_gen train/gan_loss_disc \
  train/fake_score_loss
conda run -n fme beaker experiment get 01KWTXGADFPB4GKVZ33C7ZGJP4
```
(NB: pass the **exact** wandb key names — `check_runs --keys` silently drops
non-existent keys. Wind vars are `…_at_ten_meters`.)

---

## Immediate next actions (priority order)

1. **Watch hi-1step start** — verify the frozen-Lo load line appears and the first
   validation logs plausible PRMSL numbers (hi_cascade path is new code, first
   real-env run). If it errors on load/unwrap, that's the place to look.
2. **Read hi-1step once it trains** (see above). Its marginal job is deep
   lows / large-scale PRMSL, only testable in the cascade.
3. **Operating point for the critic is settled:** offset-0 bottleneck is fine
   (tap saga was a validation artifact); 2-step Lo, GAN weight 1e-3. Do **not**
   chase finer taps / higher GAN weight.
4. **Then: bundle sampler + final eval.** Assemble `[Student-Hi, Student-Lo]` via
   `DenoisingMoEConfig.build()`; generalize `DenoisingMoEPredictor.generate` to a
   fastgen predict-x0-renoise cascade over a boundary-aligned `t_list` (primitives
   `boundary_aligned_t_list` / `sample_student_hi_cascade` exist and are now
   exercised by the callback). Then eval the bundle on the full-teacher zarr.
5. **Candidate levers (if a real gap remains):** per-variable / output-space
   critics; a deficit-penalizing spectral loss (scaffold only — a critic is better
   for coherent extremes).
6. **Backlog refactor:** `specs/10` (de-dup residual/postprocess). Low-risk;
   do before more validation-path changes.

---

## Code landed 2026-07-05 (this session)

- **`hi_cascade` validation wired** (`e920ca7f4`): `best_student_callback.py`
  accepts `validation_mode="hi_cascade"` + a pre-loaded `frozen_lo_net`
  (`frozen_lo_sample_steps` / `frozen_lo_sigma_min`), dispatching to the existing
  `student_sampling.sample_student_hi_cascade`. **Boundary σ=200 is taken from the
  trained Hi student's own `_sigma_min`** (not the frozen Lo checkpoint's recorded
  range, which is unreliable — inherits the MoE `_primary` range). `fastgen_train.py`
  adds `--val-mode hi_cascade` + `--frozen-lo-checkpoint` / `--frozen-lo-steps` /
  `--frozen-lo-sigma-min` (env `ACE_FROZEN_LO_*`) and loads the frozen Lo via
  `CheckpointModelConfig` + `_unwrap_denoiser`. `run.sh --expert 1` now uses
  `hi_cascade` and requires `--frozen-lo <dataset>` (+ `--frozen-lo-path`). Tests:
  hi_cascade shape / residual-base / missing-frozen-lo guard.

## Code landed prior session (2026-07-02)

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
- **hi_cascade** validates the Hi student **through the frozen Lo** — the real
  bundle deployment path. The frozen Lo is a beaker-dataset checkpoint mounted at
  `/frozen_lo`; `run.sh --frozen-lo <dataset> [--frozen-lo-path <rel/path.ckpt>]`.
  The boundary comes from the Hi student's `_sigma_min`, so it's exact regardless
  of the Lo checkpoint's recorded range.
- `test_best_student_callback.py` (incl. new hi_cascade tests) runs in the plain
  `fme` env; `test_teacher.py` needs the distillation Docker image.
- Don't re-trust any PRMSL number from before `b2a47628b`.
