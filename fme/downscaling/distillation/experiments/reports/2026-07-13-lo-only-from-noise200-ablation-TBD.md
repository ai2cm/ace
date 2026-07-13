<!--
Phase-2 experiment write-up (WORKFLOW.md). ABLATION eval, not a training run.
Hypothesis + config + comparison target filled before launch; metric sections +
verdict filled after the eval run finishes. Rename TBD -> wandb id once known.
-->
# Experiment — Lo-only (from-noise@200) ablation: is Student-Hi worth keeping?

_Hypothesis: **Student-Hi (expert 1, σ∈[200, 2000]) adds no measurable utility over
Student-Lo (expert 0, σ≤200) run from fresh noise at σ=200.** At σ=200 the `200·ε`
noise term dominates the O(1) clean part ~200×, so which upstream produced the σ=200
state washes out. If Lo-from-noise@200 matches the full 2-step `[Hi→Lo]` bundle on the
same held-out eval, then Hi + its NFE can be **dropped** — simplifying the deployed
student to a single-model Lo (fewer parameters, one fewer network eval)._

## Why (the strong prior) — `MOE_DISTILLATION_STATUS.md`

This ablation is the deferred decision the MoE analysis kept pointing to. The 2026-07-07
Student-Hi result + loss-mechanics trace built a structural case that Hi's contribution
is tiny **by construction**:

- **`hi_cascade` validation is nearly insensitive to Hi training** (`:44–57`): all val
  metrics flat to 4 sig figs across 18k steps while `f_distill_loss` halved — the σ=200
  handoff washes out *which* x0 Hi produced (`200·ε` dominates ~200×).
- **Hi is a coarse-only distillation** (`:83–119`): the score-matching band (clamped
  [200,2000]), the GAN feature tap, and the target (`data["real"]` = expert-1's own
  coarse sample) all agree Hi is supervised only on the ≥200 coarse modes.
- **The GAN never engaged for Hi** (`:33–42`): a coarse critic at near-pure-noise σ has
  no structure to grade.
- **Deferred, never run** (`:117–119`, `:254`): _"The Lo-from-noise@200 ablation remains
  the way to decide whether that small coarse contribution is worth an expert + an NFE,
  or whether Lo-only suffices"_; filed out-of-scope in the bundle-sampler spec as a
  _"separate eval config, decided after this."_ No Lo-only config/run existed until this.

## What already exists — the combined Hi/Lo bundle eval (the comparison target)

The full 2-step `[Hi→Lo]` MoE student bundle was evaluated vs the teacher on CONUS 2023
(100km→3km X-SHiELD): **`rmoodemk`** (distilled bundle) vs **`1r1p6djp`** (teacher),
report [`2026-07-08-moe-eval-distilled-vs-teacher.md`](2026-07-08-moe-eval-distilled-vs-teacher.md)
(LOG "Eval-bundle comparisons"). That bundle used `best_student_tail.ckpt` for each
expert with `steps_per_range: [2, 1]`. **This ablation reuses the exact same Lo
checkpoint and the exact same eval data/patch/events**, changing only whether Hi runs in
front of Lo — so the delta isolates Hi's marginal contribution.

## Config — a single-model Lo checkpoint, noise schedule capped at σ=200

**Not a bundle.** The model is the single-model Student-Lo checkpoint alone
(`CheckpointModelConfig`), with `model_updates` forcing the inference noise schedule to
start at σ=200 and keeping the 2-step Lo cadence the bundle used:

```yaml
model:
  checkpoint_path: /lo/best_student_tail.ckpt   # expert-0 baseline-fixed, the SAME ckpt in rmoodemk
  model_updates:
    sigma_min: 0.005
    sigma_max: 200.0                 # ← "noise schedule specified at 200" (fresh noise@200 → x0)
    num_diffusion_generation_steps: 2 # matches the bundle's 2-step Lo (steps_per_range [2,1])
```

- **Config file:** `configs/experiments/2026-07-07-distilled-moe-eval/config-lo-only.yaml`
  — identical to `config-distilled.yaml` except the `model:` section (same data / patch /
  `n_samples=4` / events / CONUS extent / `andrep-downscaling` logging).
- **Lo checkpoint source:** beaker `01KWJAFM694MAE55M2JMZSE89M`
  (`…-baseline-fixed-moe-teacher-expert0`), subpath `fastgen/<exp>/student_checkpoints`,
  `best_student_tail.ckpt` — mount at `/lo`.
- **Launch:** reuse the distilled-moe-eval harness with `config-lo-only.yaml` + the `/lo`
  mount (the bundle mode already references this dataset). Optionally add a
  **maritime-continent** variant (as the baseline/spectral evals had) for a second region.

## Comparison & decision criterion

Compare Lo-only vs the combined bundle `rmoodemk` (and the teacher `1r1p6djp`) on the
same held-out metrics — `metrics/crps/<VAR>`, `power_spectrum/mean_abs_norm_bias/<VAR>`,
tail ratios — via `check_runs.py --compare-eval <rmoodemk> <lo-only> --project
andrep-downscaling`.

- **If Lo-only ≈ bundle (esp. on coarse/synoptic-scale metrics — the low-k spectrum and
  PRMSL/large-scale skill, where Hi's coarse contribution would show up):** ✅ **drop
  Hi.** Deploy single-model Lo from noise@200 — fewer params, one fewer NFE. This is the
  hypothesis's predicted outcome.
- **If the bundle is clearly better on the coarse/deep-low bands:** Hi earns its keep;
  its small marginal budget is nonetheless load-bearing at synoptic scales.
- Watch specifically: `power_spectrum/mean_abs_norm_bias/PRMSL` and low-k, plus PRMSL /
  wind CRPS (the coarse fields Hi would most plausibly help).

## Launched

| | |
|---|---|
| Job | `evaluate-moe-lo-only-from-noise200-xshield-amip-100km-to-3km-conus` |
| wandb run | `p337gcg9` — https://wandb.ai/ai2cm/andrep-downscaling/runs/p337gcg9 |
| Beaker experiment | `01KXEYCC9HAZ7F1G85E3KRPKFD` — https://beaker.org/ex/01KXEYCC9HAZ7F1G85E3KRPKFD |
| Commit | `af4d134` — https://github.com/ai2cm/ace/commit/af4d134 |
| Launcher | `configs/experiments/2026-07-07-distilled-moe-eval/run-lo-only.sh` (CONUS only) |
| State | `running` (checkpoint loaded cleanly under `sigma_max=200` + `/lo` mount; eval logs metrics at completion) |

## Result  <!-- filled after the eval run -->

_Pending eval completion — then `check_runs.py --compare-eval rmoodemk <this-run>
--project andrep-downscaling`._

## Verdict  <!-- HUMAN: fill this in -->

- **Does Hi add utility over Lo-from-noise@200?** TODO.
- **Deployment implication (drop Hi / keep Hi):** TODO.

## Caveats

- Multivariate MoE student (u10/v10/PRMSL/PRATEsfc), unlike the single-var PRATEsfc
  spectral-loss line — the interesting Hi contribution, if any, is on PRMSL/large scales.
- Uses `best_student_tail.ckpt` for Lo (same as the bundle) — fair by construction.
- ⚠️ _Prepend here if a later run invalidates this one._
