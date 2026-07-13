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
| `f7z93y0a` | 2026-07-07 | …-prate-baseline | `01KWX5CVJQ2BP53VH95WKPVPED` | [`26868ca`](https://github.com/ai2cm/ace/commit/26868ca) | fdistill, no spectral (reference) | crashed@29510 | ➖ baseline | [report](reports/2026-07-07-prate-baseline-f7z93y0a.md) |
| `i26sidsm` | 2026-07-08 | …-prate-spectral-fix | `01KX00N9SE3ZVQFHQJ54XS0TAP` | [`e29f797`](https://github.com/ai2cm/ace/commit/e29f797) | fdistill, spectral W=1e-2, gan=1e-3 | crashed@27820 | ✅ win (mid-train ckpt) | [report](reports/2026-07-08-prate-spectral-fix-i26sidsm.md) |
| `6dotglmg` | 2026-07-09 | …-prate-spectral-lowgan-fix | `01KX4DRYQ0RSQEWRY5F6QBP9BY` | [`e29f797`](https://github.com/ai2cm/ace/commit/e29f797) | fdistill, spectral W=1e-2, **gan=3e-4** | ⏳ running | ⏳ | [report](reports/2026-07-09-prate-spectral-lowgan-fix-6dotglmg.md) |
| `xgcaf2rt` | 2026-07-10 | …-prate-spectral-midhi | `01KX6T1BM73VETZF53TWBHSEFE` | [`e7679c0`](https://github.com/ai2cm/ace/commit/e7679c0a9583bc42ee07d7eacf8e8db619c120d0) | fdistill, spectral W=1e-2, **min_wavenumber=85** (drop lo third, flat mid+hi) | canceled@52k | ➖ neutral (tied at best-sustained spectrum; `best_student.ckpt`@2730) | [report](reports/2026-07-10-prate-spectral-midhi-xgcaf2rt.md) |
| `s4abc6ba` | 2026-07-07 | …-prate-spectral | — | [`ae3979b`](https://github.com/ai2cm/ace/commit/ae3979b) | fdistill, spectral W=1e-2 (**pre-fix target**) | stopped | ⚠️ invalid (wrong target) | — |
| `gpx5574t` | 2026-07-07 | …-prate-spectral-lowgan | `01KWYPADNHC7SK58FMA981XTQV` | [`ae3979b`](https://github.com/ai2cm/ace/commit/ae3979b) | fdistill, spectral+gan=3e-4 (**pre-fix target**) | crashed@3770 | ⚠️ invalid (wrong target) | — |

### Eval-bundle comparisons (project `andrep-downscaling`)

| distilled | teacher | date | region/period | commit | beaker (distilled / teacher) | verdict | report |
|---|---|---|---|---|---|---|---|
| `rmoodemk` | `1r1p6djp` | 2026-07-08 | CONUS 2023, 100km→3km X-SHiELD | [`de3e00c`](https://github.com/ai2cm/ace/commit/de3e00ce2bf8215114a818faae11700afd8005f9) | `01KWZD6YMZSD37XZHDMYB8RFC7` / `01KWZD6WFN4TCSMMC48BTFMN8Q` | see report | [report](reports/2026-07-08-moe-eval-distilled-vs-teacher.md) |

### MoE per-expert base models (bundled into `rmoodemk`)

The two per-expert students assembled into the distilled 2-step MoE bundle above.
Full lineage/diagnoses are frozen in
[`../MOE_DISTILLATION_STATUS.md`](../MOE_DISTILLATION_STATUS.md); these rows just
point at the standardized reports.

| wandb | date | expert / role | beaker | commit | verdict | report |
|---|---|---|---|---|---|---|
| `zct08386` | 2026-07-03 | expert 0 · Student-Lo (σ 0.005–200) | `01KWJAFKZ96YBR73F0TETBKC0Q` | [`184fa29`](https://github.com/ai2cm/ace/commit/184fa298b6dadad9ad40252d83e0d697b73d0c84) | ✅ clean (fine-scale carrier) | [report](reports/2026-07-03-baseline-fixed-moe-teacher-expert0-zct08386.md) |
| `4mez4kmn` | 2026-07-06 | expert 1 · Student-Hi (σ 200–2000) | `01KWTXGADFPB4GKVZ33C7ZGJP4` | [`e920ca7`](https://github.com/ai2cm/ace/commit/e920ca7f425be97fbbfbddae7a700b97ac04e536) | ➖ f-distill-only (GAN inert by design) | [report](reports/2026-07-06-hi-1step-moe-teacher-expert1-4mez4kmn.md) |

---

## Active / planned

- **`6dotglmg` (reduce-GAN arm)** — first *valid* low-GAN test (gan 1e-3→3e-4 on the
  fixed target). Hypothesis: leaning off the GAN cuts the late tail-overshoot / drift
  seen in `i26sidsm` without giving back the spectral gains. Report when it has
  enough history; pick a mid-training checkpoint.
- ~~**`xgcaf2rt` (mid+high band arm)**~~ — ➖ **done, neutral** (checked & canceled
  2026-07-13): the `min_wavenumber=85` cut is tied with flat-band `i26sidsm` at the
  best-sustained spectrum (marginally better mid+hi, within noise). See report +
  outcomes bullet.
- **★ TASK — spectral-aware early stopping / checkpoint selection** (motivated by
  `xgcaf2rt`). Two coupled problems this run exposed: (a) **wasted compute** — it ran
  to 52k steps but its useful spectral optimum was ~2.6k; `val/crps_mean` is flat to
  ~1%, so it gives no stop signal, and the run just drifts (late `spec_mae_mean` +691%,
  `tail_99.99` → 2.2). (b) **selection misses the spectral optimum** — `best_student.ckpt`
  (CRPS-min) and `best_student_tail.ckpt` (tail-min) landed at very different, often
  un-converged fracs (CRPS-min noise-determined; tail-min 3% for midhi vs 29% for base).
  **Proposal:** add a spectral-based early-stop + a `best_student_spec.ckpt` selector to
  `BestStudentCheckpointCallback` — track running-min `spec_mae_mean`, save on
  improvement, and stop after N consecutive vals without spectral improvement (patience).
  This both saves compute and gives a checkpoint that actually sits at the spectral
  optimum (the analyses keep hand-picking mid-training ckpts because no selector does).
  Durable pipeline change → write a numbered spec under `../specs/` first. **Would also
  make every future arm's baseline comparison honest** (all runs stop/select at their
  own spectral optimum instead of an arbitrary flat-CRPS argmin).
- **Next (experiments):** port the tuned spectral config (flat all-band, `i26sidsm`) to
  the multi-variable MoE runs with per-variable `variable_weights` (up-weight the
  worst-spectrum variables). If the mid band still lags, the remaining lever is a
  genuine mid *bump* — a **non-monotonic** band weight that adds mid emphasis on top of
  the full flat band (needs a `SpectralMatchingLoss` change; `band_gamma` is monotonic
  today).

---

## Outcomes log

_Reverse-chronological; one line per finding, linking the run report._

- **2026-07-13** — ➖ **The mid+high band-cut arm `xgcaf2rt` is roughly neutral**
  (corrected — an earlier entry called it ❌ degrade; that was a windowing artifact,
  see the report Caveats). Compared **at the selected checkpoints** and at each run's
  **best-sustained spectrum** (both step-controlled), midhi and flat-band `i26sidsm`
  are tied (`spec_mae_mean` 0.044 vs 0.043), midhi marginally better on mid+hi — a
  weak nod to the hypothesis, within noise. No clear win, no clear loss; flat all-band
  weighting stays the default as the simpler config. **The real finding: CRPS/tail
  checkpoint selection is decoupled from spectral quality** — CRPS is flat to ~1% so
  its argmin is noise, and tail-min landed at 3% (midhi) vs 29% (base); the "winner"
  flips by which selector you read. Motivates a spectral-aware early-stop/selection
  criterion (new planned item). See
  [report](reports/2026-07-10-prate-spectral-midhi-xgcaf2rt.md).
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
