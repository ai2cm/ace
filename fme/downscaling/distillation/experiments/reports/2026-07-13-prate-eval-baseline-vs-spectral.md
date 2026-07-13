<!--
Comparison-eval report: two DISTILLED students against each other (GAN-only
baseline vs spectral-loss), held-out X-SHiELD AMIP control 100km->3km, two
regions. Generated from `check_runs.py --compare-eval <baseline> <spectral>
--project andrep-downscaling` (per region), then relabeled teacher/distilled ->
baseline/spectral (the tool is a generic A-vs-B diff of two eval-run summaries).
-->
# Eval comparison — distilled **baseline** vs distilled **spectral** (held-out X-SHiELD AMIP control, 100km→3km)

_The held-out confirmation of the training-val spectral win `i26sidsm`. Compares the
GAN-only distilled PRATEsfc student (`f7z93y0a`) against the spectral-matching-loss
student (`i26sidsm`) on genuinely out-of-sample data — X-SHiELD **AMIP control** (not
the 2023 val period) — over two regions (CONUS + maritime continent). Both bundle
`best_student_tail.ckpt`; both eval runs share commit `d6cd8dd`._

## Artifacts

| role | region | wandb run | commit | checkpoint |
|---|---|---|---|---|
| baseline (GAN-only, `f7z93y0a`) | CONUS | `flzvb6tp` — https://wandb.ai/ai2cm/andrep-downscaling/runs/flzvb6tp | `d6cd8dd` | `best_student_tail.ckpt` |
| spectral (`i26sidsm`) | CONUS | `x2nyzmzh` — https://wandb.ai/ai2cm/andrep-downscaling/runs/x2nyzmzh | `d6cd8dd` | `best_student_tail.ckpt` |
| baseline (GAN-only, `f7z93y0a`) | maritime continent | `fg9byv9y` — https://wandb.ai/ai2cm/andrep-downscaling/runs/fg9byv9y | `d6cd8dd` | `best_student_tail.ckpt` |
| spectral (`i26sidsm`) | maritime continent | `l6vv7yx0` — https://wandb.ai/ai2cm/andrep-downscaling/runs/l6vv7yx0 | `d6cd8dd` | `best_student_tail.ckpt` |

Single output variable: **PRATEsfc**. Δ = spectral − baseline.

## CONUS

| metric (PRATEsfc) | baseline | spectral | Δ | read |
|---|---|---|---|---|
| CRPS (lower better) | 8.66e-6 | 8.35e-6 | −3.1e-7 (−3.5%) | spectral better |
| tail ratio @99.9999 (~1.0) | 1.009 | 0.995 | — | both ~ideal |
| **power-spectrum bias** (lower better) | **0.464** | **0.134** | **−71%** | **spectral far better** |

## Maritime continent

| metric (PRATEsfc) | baseline | spectral | Δ | read |
|---|---|---|---|---|
| CRPS (lower better) | 4.77e-5 | 4.64e-5 | −1.2e-6 (−2.6%) | spectral better |
| tail ratio @99.9999 (~1.0) | 0.876 | 0.898 | — | both under-produce; spectral closer to 1 |
| **power-spectrum bias** (lower better) | **0.596** | **0.132** | **−78%** | **spectral far better** |

## Figures  <!-- generated separately -->

- Histograms / spectra via `scripts/downscaling/plot_compared_histograms.py` +
  `plot_beaker_histograms.py` on the per-event netCDFs (`fetch_beaker_dataset`).

## Verdict

- **✅ Decisive, generalizing win for the spectral-matching loss.** On genuinely
  held-out data (X-SHiELD AMIP control, a different period than the training val, over
  two regions including the tropical maritime continent), the spectral student cuts the
  **power-spectrum bias 3.5–4.5×** (CONUS 0.46→0.13, maritime 0.60→0.13) — the exact
  textural-fidelity gain the loss targets — while also **improving CRPS ~3%** and
  keeping tails near-ideal (CONUS ~1.0; maritime slightly under-produced ~0.88–0.90,
  spectral marginally closer to 1). No regression on any metric in either region.
- **Why it's trustworthy:** both arms bundle `best_student_tail.ckpt` (same selector →
  fair), which for the spectral run lands mid-training near its spectral optimum (so the
  0.13 bias reflects a good checkpoint, not the drifted late one). Same eval commit
  `d6cd8dd`, identical data/regions. This is the held-out confirmation that the
  training-val `i26sidsm` win is real and transfers out-of-sample.
- **Caveats:** PRATEsfc single-variable only (the winds/PRMSL story is the separate MoE
  eval `rmoodemk`). Absolute `power_spectrum/mean_abs_norm_bias` is a different metric
  from the training-val `spec_mae` — read the 3.5–4.5× *relative* improvement, not the
  absolute value against training numbers.
- **Next action:** this de-risks porting the spectral loss to the multi-variable MoE
  runs (LOG "Next"). The `band_gamma` sweep (`2yhjonz9`/`34rg7wii`, running) tests
  whether tilting the same loss toward the tail improves the spectrum bias further.
