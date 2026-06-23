# Multivariate MoE Distillation — Status & Handoff

Living notes for the multivariate MoE FastGen distillation effort on branch
`experiment/fastgen-distill`. Pick up here from any clone.

_Last updated: 2026-06-23._

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

| Run | What | Beaker experiment | Commit |
|---|---|---|---|
| **Run 1** — noise fix only | `ACE_SIGMA_MAX=2000`, `ACE_C_OUT=4` | `01KVTPTSTSC9V4Z99TP374WC43` (`…-sigmafix-moe-teacher`) | `8cdf58a3d` |
| **Run 2** — noise + dispatch + capacity | Run 1 + MoE dispatch + student from expert 1 | `01KVTQV9N2QMCEDGCK035KDSJH` (`…-dispatch-moe-teacher`) | `3a2dced4e` |

### How to check the latest runs

Beaker status:
```bash
conda run -n fme beaker experiment get 01KVTPTSTSC9V4Z99TP374WC43
conda run -n fme beaker experiment get 01KVTQV9N2QMCEDGCK035KDSJH
# or browse: https://beaker.org/ex/<id>  and  https://beaker.org/ws/ai2/climate-titan
```

Latest distillation runs in wandb (newest first):
```bash
conda run -n fme python - <<'PY'
import wandb
api = wandb.Api()
runs = api.runs("ai2cm/fastgen", order="-created_at")
for r in list(runs)[:15]:
    print(r.id, r.state, r.name)
PY
```

Pull validation metrics for a run (compare against the flat baselines above):
```bash
conda run -n fme python - <<'PY'
import wandb
api = wandb.Api()
r = api.run("ai2cm/fastgen/<run_id>")
keys = ["val/crps_PRMSL","val/crps_PRATEsfc","val/crps_eastward_wind_at_ten_meters",
        "val/spec_mae_mean","val/spec_mae_hi_PRATEsfc","val/tail_99.99_PRATEsfc"]
h = r.history(keys=keys, samples=2000, pandas=True)
for k in keys:
    if k in h:
        s = h[k].dropna()
        if len(s): print(f"{k:42s} first={s.iloc[0]:.4g} best={s.min():.4g} last={s.iloc[-1]:.4g}")
PY
```

Run launcher (re-run / new variant):
```bash
conda run -n fme bash configs/experiments/2026-05-18-distillation-with-val/run.sh \
    fdistill --moe-teacher --suffix <variant>
```
`gantry` lives in the `fme` conda env; it clones the **pushed** commit, so
commit + push before launching.

---

## Known limitation → candidate Run 3

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

## Design note: which expert should the discriminator use? (open, undecided)

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
