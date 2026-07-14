# Distillation experiment log

Central planning + outcomes log for distilled downscaling students. Process:
[`WORKFLOW.md`](WORKFLOW.md). Per-run reports: [`reports/`](reports/).

> Pre-2026-07 MoE-distillation history is frozen in
> [`../MOE_DISTILLATION_STATUS.md`](../MOE_DISTILLATION_STATUS.md).

---

## ⚡ At a glance  <!-- keep this current: the daily check-in view -->

_Last updated: 2026-07-13._

### 🔴 In flight — check for updates, finish write-ups when done

| run | kind | question | on completion |
|---|---|---|---|
| `2yhjonz9` (band_gamma=0.5) | training | does a gentle hi-k spectral tilt beat flat `i26sidsm`? | `check_runs --report`; compare vs `i26sidsm` at best-sustained; write verdict in its stub |
| `34rg7wii` (band_gamma=1) | training | " (stronger tilt) | same as above; fit the 0/0.5/1 `band_gamma` response curve |
| `p337gcg9` (Lo-only ablation, CONUS) | eval | is Student-Hi droppable? | `check_runs --compare-eval rmoodemk p337gcg9`; write verdict (drop-Hi?) in the ablation report |

### 🟢 Next up — likely-good experiments (queued, not launched)

1. **Native step-count sweep** — 1-step (task #3) + 4-step (task #2) f-distill vs the
   2-step `i26sidsm`; find the quality-vs-NFE knee.
   ([write-up](reports/2026-07-13-fdistill-step-count-sweep-TBD.md))
2. **Spectral-aware early stop + `best_student_spec` selector** — spec 13 / task #1;
   also fixes the checkpoint-selection trap that confounds every comparison.
3. **Port the tuned spectral loss → multi-variable MoE** with per-var `variable_weights`
   (up-weight the worst-spectrum vars; winds/PRMSL were the MoE weak spots).
4. **Multi-scale (multi-head) discriminator** — spec 12 flagship E2; the big untested GAN
   texture lever for the winds hi-k gap single taps never closed.
5. **Non-monotonic mid *bump*** (SpectralMatchingLoss code change) — only if the
   `band_gamma` sweep shows the mid band still lags.
6. **Longer reduce-GAN re-run** (`gan=3e-4`) — `6dotglmg` stopped at 14k, before the
   late-drift regime it was meant to test; re-run with checkpointing.

---

## Run registry

Every launched run gets a row. `verdict`: ✅ win · ➖ flat · ❌ degrade · ⏳ running
· ⚠️ invalid. Regenerate a row with `check_runs.py --registry-row <id>`.

> **Note on `state`:** a wandb/beaker state of **`crashed` usually means the run was
> *manually cancelled***, not that it hit a genuine error — these are experiment arms
> we stop once we've seen enough (or that get preempted). Don't over-interpret
> "crashed" as a failure; check the last step and metrics. A real error shows a
> traceback in the beaker logs.

| wandb | date | experiment name | beaker | commit | method / knobs | state | verdict | report |
|---|---|---|---|---|---|---|---|---|
| `f7z93y0a` | 2026-07-07 | …-prate-baseline | `01KWX5CVJQ2BP53VH95WKPVPED` | [`26868ca`](https://github.com/ai2cm/ace/commit/26868ca) | fdistill, no spectral (reference) | crashed@29510 | ➖ baseline | [report](reports/2026-07-07-prate-baseline-f7z93y0a.md) |
| `i26sidsm` | 2026-07-08 | …-prate-spectral-fix | `01KX00N9SE3ZVQFHQJ54XS0TAP` | [`e29f797`](https://github.com/ai2cm/ace/commit/e29f797) | fdistill, spectral W=1e-2, gan=1e-3 | crashed@27820 | ✅ win (mid-train ckpt) | [report](reports/2026-07-08-prate-spectral-fix-i26sidsm.md) |
| `6dotglmg` | 2026-07-09 | …-prate-spectral-lowgan-fix | `01KX4DRYQ0RSQEWRY5F6QBP9BY` | [`e29f797`](https://github.com/ai2cm/ace/commit/e29f797) | fdistill, spectral W=1e-2, **gan=3e-4** | crashed@14040 | ➖ inconclusive (mild tail gain; crashed before late-drift regime) | [report](reports/2026-07-09-prate-spectral-lowgan-fix-6dotglmg.md) |
| `xgcaf2rt` | 2026-07-10 | …-prate-spectral-midhi | `01KX6T1BM73VETZF53TWBHSEFE` | [`e7679c0`](https://github.com/ai2cm/ace/commit/e7679c0a9583bc42ee07d7eacf8e8db619c120d0) | fdistill, spectral W=1e-2, **min_wavenumber=85** (drop lo third, flat mid+hi) | canceled@52k | ➖ neutral (tied at best-sustained spectrum; `best_student.ckpt`@2730) | [report](reports/2026-07-10-prate-spectral-midhi-xgcaf2rt.md) |
| `2yhjonz9` | 2026-07-13 | …-prate-spectral-gamma0p5 | `01KXEN0NJ81G7R1SF1F4ZFZV2R` | [`06aee7f`](https://github.com/ai2cm/ace/commit/06aee7f9c) | fdistill, spectral W=1e-2, **band_gamma=0.5** (gentle hi tilt; lo≈0.61× hi≈1.37×) | running | ⏳ | [report](reports/2026-07-13-prate-spectral-gamma0p5-2yhjonz9.md) |
| `34rg7wii` | 2026-07-13 | …-prate-spectral-gamma1 | `01KXEN0PH05655AQD3FWJRSCXQ` | [`06aee7f`](https://github.com/ai2cm/ace/commit/06aee7f9c) | fdistill, spectral W=1e-2, **band_gamma=1** (linear hi tilt; lo≈0.33× hi≈1.7×) | running | ⏳ | [report](reports/2026-07-13-prate-spectral-gamma1-34rg7wii.md) |
| `s4abc6ba` | 2026-07-07 | …-prate-spectral | — | [`ae3979b`](https://github.com/ai2cm/ace/commit/ae3979b) | fdistill, spectral W=1e-2 (**pre-fix target**) | stopped | ⚠️ invalid (wrong target) | — |
| `gpx5574t` | 2026-07-07 | …-prate-spectral-lowgan | `01KWYPADNHC7SK58FMA981XTQV` | [`ae3979b`](https://github.com/ai2cm/ace/commit/ae3979b) | fdistill, spectral+gan=3e-4 (**pre-fix target**) | crashed@3770 | ⚠️ invalid (wrong target) | — |

### Eval-bundle comparisons (project `andrep-downscaling`)

| distilled | teacher | date | region/period | commit | beaker (distilled / teacher) | verdict | report |
|---|---|---|---|---|---|---|---|
| `rmoodemk` | `1r1p6djp` | 2026-07-08 | CONUS 2023, 100km→3km X-SHiELD | [`de3e00c`](https://github.com/ai2cm/ace/commit/de3e00ce2bf8215114a818faae11700afd8005f9) | `01KWZD6YMZSD37XZHDMYB8RFC7` / `01KWZD6WFN4TCSMMC48BTFMN8Q` | see report | [report](reports/2026-07-08-moe-eval-distilled-vs-teacher.md) |
| `x2nyzmzh` (spectral) | `flzvb6tp` (baseline) | 2026-07-13 | CONUS, 100km→3km X-SHiELD AMIP control | [`d6cd8dd`](https://github.com/ai2cm/ace/commit/d6cd8dd261a45aaa999e58cc551c460ee68dc940) | — | ✅ spectral wins: PSD bias 0.46→0.13 (−71%), CRPS −3.5%, tails ~ideal | [report](reports/2026-07-13-prate-eval-baseline-vs-spectral.md) |
| `l6vv7yx0` (spectral) | `fg9byv9y` (baseline) | 2026-07-13 | maritime continent, 100km→3km X-SHiELD AMIP control | [`d6cd8dd`](https://github.com/ai2cm/ace/commit/d6cd8dd261a45aaa999e58cc551c460ee68dc940) | — | ✅ spectral wins: PSD bias 0.60→0.13 (−78%), CRPS −2.6%, tails closer to 1 | [report](reports/2026-07-13-prate-eval-baseline-vs-spectral.md) |

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

- ~~**`6dotglmg` (reduce-GAN arm)**~~ — ➖ **done, inconclusive** (reported 2026-07-13):
  crashed@14k, mildly better tails at matched steps but **crashed before the late-drift
  regime** it was meant to test (drift at 14k identical to baseline). Needs a **longer
  re-run** (≥28k, with checkpointing) to actually test the drift hypothesis — ideally
  after spec 13 early-stop. See report.
- **⏳ `band_gamma` sweep (running, launched 2026-07-13)** — `2yhjonz9` (gamma=0.5) +
  `34rg7wii` (gamma=1), f-distill PRATEsfc, W=1e-2 / gan=1e-3 / min_wavenumber=0, only
  `band_gamma` varies. Both verified at iteration 1 with the right band_gamma. Fills the
  gentle-tilt regime between the flat win `i26sidsm` (gamma=0) and the low-suppression
  `xgcaf2rt` (~gamma=2). Report when they have history; watch `spec_mae_hi` (should
  improve) vs `spec_mae_lo` (the cost) and the PSD tail for overshoot. Compare at
  best-sustained / matched steps.
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
- **⏳ RUNNING — Lo-only (from-noise@200) ablation: is Student-Hi worth keeping?** The
  deferred MoE decision (`MOE_DISTILLATION_STATUS.md:117–119, 254`): evaluate a
  **single-model Student-Lo** checkpoint (expert 0, `best_student_tail.ckpt`) with its
  noise schedule capped at **σ=200** (sample from fresh noise@200), on the *same*
  held-out eval as the combined `[Hi→Lo]` bundle `rmoodemk`
  ([eval report](reports/2026-07-08-moe-eval-distilled-vs-teacher.md)). If Lo-only ≈ the
  bundle (esp. coarse/PRMSL/low-k), **drop Hi** — fewer params + one fewer NFE. Strong
  prior that Hi's marginal budget is tiny (σ=200 washout; Hi is coarse-only by
  construction). Config: `configs/experiments/2026-07-07-distilled-moe-eval/config-lo-only.yaml`
  (single-model, `sigma_max=200`, 2-step Lo); launcher `run-lo-only.sh` (mounts Lo at
  /lo). **Launched 2026-07-13, CONUS only** — beaker `01KXEYCC9HAZ7F1G85E3KRPKFD`,
  commit `af4d134`. Write-up:
  [report](reports/2026-07-13-lo-only-from-noise200-ablation-TBD.md).
- **★ PLANNED — native f-distill step-count sweep (1 / 2 / 4 step).** Train a native
  **1-step** (task #3) and native **4-step** (task #2) student from scratch
  (`--student-steps 1|4`, spectral W=1e-2), baseline = the 2-step `i26sidsm`; find the
  quality-vs-NFE knee. No warm-start (training is short; a native run at each step count
  is the fair test — a 1-step *eval* of the 2-step model only lower-bounds native-1-step).
  **Mechanism:** f-distill training is *not* step-independent — `student_sample_steps` sets
  the discrete `t_list` nodes `t_student` is drawn from and whether `input_student` is pure
  noise (1-step) or real-data re-noised (N-step interior, teacher-forced → inference
  exposure bias). See [[fdistill-step-coupling]] / `dmd2.py:97–116`. Write-up:
  [report](reports/2026-07-13-fdistill-step-count-sweep-TBD.md).
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
- **2026-07-13** — ✅ **Held-out eval confirms the spectral loss is a real, generalizing
  win.** On X-SHiELD AMIP control (out-of-sample vs the training val period), 100km→3km,
  the spectral student beats the GAN-only baseline on **power-spectrum bias 3.5–4.5×**
  (CONUS 0.46→0.13, maritime continent 0.60→0.13) with CRPS ~3% better and tails
  near-ideal — no regression, both regions, both bundling `best_student_tail.ckpt` (fair).
  Confirms the training-val `i26sidsm` result transfers out-of-sample; de-risks porting
  the loss to the MoE runs. See
  [report](reports/2026-07-13-prate-eval-baseline-vs-spectral.md).
- **2026-07-13** — ➖ **Reduce-GAN arm `6dotglmg` (gan=3e-4) reported: inconclusive.**
  Marginally better spectrum + tails than `i26sidsm` at matched steps (tail 1.10 vs 1.17
  @14k), no downside — but it **crashed@14k, before the late-drift regime** (baseline
  drifts +61%→+632% only after 14k; at 14k both are ~+60%). The headline "does low-GAN
  tame late drift" question is untested; the +92%-vs-+632% gap was a run-length artifact.
  Re-run longer with checkpointing. See
  [report](reports/2026-07-09-prate-spectral-lowgan-fix-6dotglmg.md).
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
