# SamudrACE ENSO rollout interventions

Training experiments testing the intervention matrix in the reports repo
(`troya/2026-08-18-samudrace-enso-rollout-interventions`), which builds on the
ENSO skill diagnosis (`troya/2026-08-12-samudrace-enso-skill-diagnosis`).
Branched clean from `main`; arm configs land here as they are launched.

## Baseline lineage (what every arm is compared against)

Derive arm configs from the **archived run configs** below, not from repo
yamls — the yamls on `feature/mlp-readout-samudra` differ from what actually
ran (e.g. the run's ocean corrector force-positive list is
`[sea_ice_volume]` only, without the salinity levels the yaml shows).

| role | experiment | run config / checkpoint |
|---|---|---|
| ocean pretrain (init for the coupled FT) | `troya/cm4-1pct-samudra-nino-066c88b` | dataset `01KXKZ85HTDSGGXWD2DPW2QRFW` (`config.yaml`, `training_checkpoints/best_inference_ckpt.tar`) |
| coupled FT (the model both reports evaluated) | `troya/cm4-1pct-coupled-ft-atmos-nino-ocean-6595` | dataset `01KY3DATM3CAEA479JQZQDPT9W` |
| atmosphere init used by that FT | — | dataset `01KJ70WK2NH4T2T4AVAAPYFSHA` |

**Known property of this lineage:** it runs with no ocean heat-content or
surface-energy-flux correction — only sea-ice positivity and the sea-ice
fraction fix. Sibling Samudra fine-tunes (below) run with both corrections on;
the nino-channel retrain forked without them.

## The corrected sibling lineage (precedent + candidate init)

| role | experiment | run config / checkpoint |
|---|---|---|
| Samudra FT with `ocean_heat_content_correction: scaled_temperature` **and** `surface_energy_flux_correction: residual_prediction` | `jamesd/1pct_nosmooth_0256to0350-hfds_resid-ft_correctOHC-rs0-train-3f73` (exit 0) | dataset `01KK88F6B2BKGD7KCSNVBNHVZX` |

Usable as (a) the precedent corrector config for corrected arms, and (b) a
candidate ocean init for a corrected coupled FT. Caveats: no `nino34_lead`
output channels (fine — the readout was a probe), and it bundles two
corrections, so attribution needs an OHC-only arm alongside.

## Evaluation

Every arm is scored with the canonical Niño3.4 verification from the diagnosis
report (reports repo, `scripts/`), scouting on 36 ICs (years 0233/0246/0250)
with the free-run baseline and prescribe-probe results as the bars.
