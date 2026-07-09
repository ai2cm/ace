# Eval comparison — distilled vs teacher

_First apples-to-apples eval of the distilled 2-step MoE student bundle vs the
bundled MoE teacher on CONUS 2023 (100km→3km X-SHiELD). Gives the teacher its
first eval baseline too._

## Artifacts

| role | wandb run |
|---|---|
| Teacher | `1r1p6djp` — https://wandb.ai/ai2cm/andrep-downscaling/runs/1r1p6djp |
| Distilled | `rmoodemk` — https://wandb.ai/ai2cm/andrep-downscaling/runs/rmoodemk |

## CRPS  (`metrics/crps/<VAR>` — lower better)

| variable | teacher | distilled | Δ (dist−teach) |
|---|---|---|---|
| PRATEsfc | 8.107e-06 | 8.455e-06 | +3.48e-07 |
| PRMSL | 0.14 | 0.1676 | +0.0276 |
| eastward_wind_at_ten_meters | 0.4222 | 0.4369 | +0.0147 |
| northward_wind_at_ten_meters | 0.4266 | 0.4359 | +0.0093 |

## Tail ratio  (`prediction_frac_of_target` @ 99.9999th pct — ~1.0 ideal)

| variable | teacher | distilled |
|---|---|---|
| PRATEsfc | 1.006 | 1.013 |
| PRMSL | — | — |
| eastward_wind_at_ten_meters | 0.9994 | 1.006 |
| northward_wind_at_ten_meters | 0.9843 | 0.9802 |

## Power spectrum bias  (`power_spectrum/mean_abs_norm_bias/<VAR>`)

| variable | teacher | distilled |
|---|---|---|
| PRATEsfc | 0.2465 | 0.255 |
| PRMSL | 0.08805 | 0.153 |
| eastward_wind_at_ten_meters | 0.2773 | 0.1169 |
| northward_wind_at_ten_meters | 0.1895 | 0.121 |

## Verdict  <!-- HUMAN: fill this in -->

- **Does the distilled student hold up vs teacher?** Broadly yes. CRPS is marginally
  worse on every variable (largest: PRMSL +0.028), tails match teacher (~1.0 both),
  and spectrum bias is mixed — distilled is **better** on both winds
  (eastward 0.28→0.12, northward 0.19→0.12) but **worse** on PRMSL (0.088→0.153) and
  slightly worse on PRATEsfc.
- **Regressions to watch:** PRMSL — both CRPS and spectrum bias degrade most there,
  consistent with the coarse-PRMSL GAN sensitivity seen in the MoE history. Candidate
  for the per-variable spectral weighting once ported to MoE.
