<!--
Comparison-eval report template: distilled student vs teacher on a held-out eval
bundle (e.g. CONUS 2023, project andrep-downscaling). `check_runs.py
--compare-eval <teacher_run> <distilled_run> --project andrep-downscaling`
pre-fills the side-by-side tables from run summaries (eval runs log scalars at
step 0). Figures are generated separately (see WORKFLOW.md, Phase 4).
-->
# Eval comparison — distilled vs teacher (`<eval-region-period>`)

_What is being compared and why (e.g. first apples-to-apples eval of the distilled
2-step MoE bundle vs the bundled teacher on CONUS 2023)._

## Artifacts

| role | wandb run | commit | Beaker experiment | checkpoint / bundle |
|---|---|---|---|---|
| Teacher | `<id>` — <url> | `<short-sha>` — https://github.com/ai2cm/ace/commit/<full-sha> | `<ULID>` | |
| Distilled | `<id>` — <url> | `<short-sha>` — https://github.com/ai2cm/ace/commit/<full-sha> | `<ULID>` | |

## CRPS  (`metrics/crps/<VAR>` — lower better)

| variable | teacher | distilled | Δ (distilled−teacher) |
|---|---|---|---|
| ... | | | |

## Tail ratio  (`histogram/prediction_frac_of_target/<pct>th-percentile/<VAR>` — ~1.0 ideal)

| variable | teacher | distilled |
|---|---|---|
| ... | | |

## Power spectrum bias  (`power_spectrum/mean_abs_norm_bias/<VAR>` — lower better)

| variable | teacher | distilled |
|---|---|---|
| ... | | |

## Figures  <!-- generated separately -->

- Histograms / spectra via `scripts/downscaling/plot_compared_histograms.py`
  + `plot_beaker_histograms.py` on the per-event netCDFs (`fetch_beaker_dataset`).

## Verdict  <!-- HUMAN: fill this in -->

- **Does the distilled student hold up vs teacher?** per-variable summary.
- **Regressions to watch:** _..._
- **Next action:** _..._
