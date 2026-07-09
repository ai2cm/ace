# Distillation experiment log

Central planning + outcomes log for distilled downscaling students. Process:
[`WORKFLOW.md`](WORKFLOW.md). Per-run reports: [`reports/`](reports/).

> Pre-2026-07 MoE-distillation history is frozen in
> [`../MOE_DISTILLATION_STATUS.md`](../MOE_DISTILLATION_STATUS.md).

---

## Run registry

Every launched run gets a row. `verdict`: ✅ win · ➖ flat · ❌ degrade · ⏳ running
· ⚠️ invalid. Regenerate a row with `check_runs.py --registry-row <id>`.

| wandb | date | experiment name | beaker | commit | method / knobs | state | verdict | report |
|---|---|---|---|---|---|---|---|---|
| `f7z93y0a` | 2026-07-07 | …-prate-baseline | `01KWX5CVJQ2BP53VH95WKPVPED` | `26868ca` | fdistill, no spectral (reference) | crashed@29510 | ➖ baseline | [report](reports/2026-07-07-prate-baseline-f7z93y0a.md) |
| `i26sidsm` | 2026-07-08 | …-prate-spectral-fix | `01KX00N9SE3ZVQFHQJ54XS0TAP` | `e29f797` | fdistill, spectral W=1e-2, gan=1e-3 | crashed@27820 | ✅ win (mid-train ckpt) | [report](reports/2026-07-08-prate-spectral-fix-i26sidsm.md) |
| `6dotglmg` | 2026-07-09 | …-prate-spectral-lowgan-fix | `01KX4DRYQ0RSQEWRY5F6QBP9BY` | `e29f797` | fdistill, spectral W=1e-2, **gan=3e-4** | ⏳ running | ⏳ | [report](reports/2026-07-09-prate-spectral-lowgan-fix-6dotglmg.md) |
| `s4abc6ba` | 2026-07-07 | …-prate-spectral | — | `ae3979b` | fdistill, spectral W=1e-2 (**pre-fix target**) | stopped | ⚠️ invalid (wrong target) | — |
| `gpx5574t` | 2026-07-07 | …-prate-spectral-lowgan | `01KWYPADNHC7SK58FMA981XTQV` | `ae3979b` | fdistill, spectral+gan=3e-4 (**pre-fix target**) | crashed@3770 | ⚠️ invalid (wrong target) | — |

### Eval-bundle comparisons (project `andrep-downscaling`)

| distilled | teacher | date | region/period | beaker (distilled / teacher) | verdict | report |
|---|---|---|---|---|---|---|
| `rmoodemk` | `1r1p6djp` | 2026-07-08 | CONUS 2023, 100km→3km X-SHiELD | `01KWZD6YMZSD37XZHDMYB8RFC7` / `01KWZD6WFN4TCSMMC48BTFMN8Q` | see report | [report](reports/2026-07-08-moe-eval-distilled-vs-teacher.md) |

---

## Active / planned

- **`6dotglmg` (reduce-GAN arm)** — first *valid* low-GAN test (gan 1e-3→3e-4 on the
  fixed target). Hypothesis: leaning off the GAN cuts the late tail-overshoot / drift
  seen in `i26sidsm` without giving back the spectral gains. Report when it has
  enough history; pick a mid-training checkpoint.
- **Next if it helps:** port the tuned spectral config to the multi-variable MoE runs
  with per-variable `variable_weights` (up-weight the worst-spectrum variables).

---

## Outcomes log

_Reverse-chronological; one line per finding, linking the run report._

- **2026-07-09** — Launched the first valid reduce-GAN arm `6dotglmg` (gan=3e-4);
  the earlier `gpx5574t` low-GAN run was invalid (pre-fix target, crashed early).
- **2026-07-08** — ✅ **Corrected spectral-matching loss is a large win.** `i26sidsm`
  beats the GAN-only baseline 5–20× on `spec_mae` and improves the independent `crps`
  + `tail_99.99`, without fighting distillation (`f_distill_loss` ≈ baseline). Late
  drift persists → best model is mid-training. See
  [report](reports/2026-07-08-prate-spectral-fix-i26sidsm.md).
- **2026-07-07** — ❌ First spectral arms (`s4abc6ba`, `gpx5574t`) were net-harmful
  due to two coupled bugs (matched teacher's x0 *prediction* not a *sample*;
  average-then-spectrum instead of spectrum-then-average). Fixed in `e29f797`.
