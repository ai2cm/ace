# Learning the unaccounted ocean heating on UFS replay (4°)

Owner: Troy Arcomano. Started 2026-10-08. Branch `troya/2026-08-18-enso-rollout-interventions`.

## Why

The heat-content corrector closes the column budget against the atmosphere-rebuilt surface
flux plus a constant `constant_unaccounted_heating`. On CM4 that constant is 0 and the
budget is closed by construction. On UFS replay the column heat change is not the surface
flux: the ORAS5 nudging removes heat, and the amount varies. Measured on the 4° 5-day store,
1994 to 2023, ocean-area means (scratchpad `ufs_budget/budget_residual.py`):

| term | W/m² |
|---|---|
| dOHC/dt | +1.51 |
| MOM6 `hfds_total_area` | +5.57 |
| dOHC/dt minus MOM6 flux (nudging) | −4.06 mean; annual values −12 (1994) to −0.5 (2021), annual std 2.7 |
| atmosphere-rebuilt flux (what the corrector sees) | +2.37 |
| dOHC/dt minus rebuilt flux | −0.85 mean, annual std 2.6 |

A constant cannot represent this (tested earlier: a constant "did not work well"), and per
cell the residual is dominated by heat-transport convergence (time-mean rms 32 W/m², zonal
structure −18 W/m² in 0 to 30°N, +15 in 30 to 60°N). So the term has to be learned as a
time-varying, state-dependent field, and only its ocean-area mean enters the budget.

## Design

1. **Target field** (dataset step). For each store time k, in the convention the corrector
   uses (per cell area, ocean cells, NaN over land):

   `U_k = (OHC_k − OHC_{k−1}) / Δt − [F_rebuild,k · ssf + hfgeou · ssf]`

   with `F_rebuild` = `net_surface_energy_flux(DLWRF, ULWRF, DSWRF, USWRF, LHTFL, SHTFL,
   frozen) + c_p (PRATE + frozen − LHTFL / L_v)(SST − 273.15)`, exactly the corrector's
   `_compute_ocean_net_surface_energy_flux`, evaluated with the true fields of window k and
   the SST at k−1 (the input SST the corrector would use). OHC is the thickness-weighted
   column integral of `thetao_0..18` with `rho c_p` over the store's `idepth` interfaces.
   The corrector's budget is an ocean-area mean of exactly this bracket plus the unaccounted
   term, so `mean_ocean(U_k)` is the number the constant stood in for.

   Stored in the training zarr as two fields: `unaccounted_heating` = trailing 73-step
   (one-year) mean of `U` per cell, the training target, and `unaccounted_heating_5day` = raw
   `U_k`, diagnostic only. The trailing mean is causal (no future information) and removes
   the 5-day dOHC noise (~270 W/m² per cell) that would otherwise dominate the regression.
   Both NaN over land. Normalization stats (mean, std) for the two fields are appended to
   copies of the 2026-10-06 stats files.

2. **Model**. `unaccounted_heating` becomes an ordinary diagnostic output channel of the
   ocean stepper (in `out_names`, not in `in_names`), scored in the loss like every other
   channel, masked like the surface fields.

3. **Corrector**. `OceanHeatContentBudgetConfig` gains
   `unaccounted_heating_source: "constant" | "generated"` (default `"constant"`, the current
   behaviour) and `unaccounted_heating_name: str = "unaccounted_heating"`. With
   `"generated"`, the per-step unaccounted heating is the ocean-area mean of the stepper's
   own `unaccounted_heating` output plus `constant_unaccounted_heating`, computed with the
   same masked area weights as the flux term. Unit test: a generated field with a known
   ocean mean reproduces the constant case with that constant.

4. **Experiment** (4°, UFS-native variables, 2026-10-06 store + the two new fields,
   2026-10-06 stats + the new entries; train 1994-2001 + 2012-2021, validation 2022-2023,
   inline 10-yr rollouts from 2012 and 2013; 2002-2011 untouched):

   | arm | corrector | unaccounted heating |
   |---|---|---|
   | resid-noohc (exists, rerunning on the clean store) | none | — |
   | resid-cap0005 (exists, rerunning) | scaled, capped 0.5% | constant 0 |
   | **resid-cap0005-learnedheat** (new) | scaled, capped 0.5% | generated field |
   | **ff-ohc-learnedheat** (new) | scaled | generated field |

   Scored on: 10-yr inline drift (SST and column), `unaccounted_heating` channel RMSE and
   its ocean-mean time series against the data's, and later the 2002-2006 development
   window (5-yr rollouts from 2002-01 ICs) for drift and skill.

## Steps and status

- [x] Characterize the target on the 4° store (2026-10-07).
- [x] Regenerate the UFS stores (stress, zos), 5-day products, stats, training dataset
      `01M4BV9103G4C330XQM7R17RPT`, stats `01M4C171K2J619BXZH9CSVM43N` (2026-10-07).
- [x] Rerun the six existing UFS arms on the clean store (launched 2026-10-08; old runs
      renamed `*-dsv3`).
- [x] Dataset step (2026-10-08): `unaccounted_heating{,_5day}` written into a copy of the
      training zarr (148 variables) and their stats appended (`make_uh_dataset.py` in this
      directory). Ocean-area mean of the trailing-year field 1994-2023: −0.06 W/m², annual
      values −6.9 (1994) to +2.8; grid-point stats mean 3.96, std 51.6 (5-day field std 506).
      Stats dataset `01M4E83B1B83CB8DV9YEQTGA4G`; training dataset
      `ufs-replay-ocean-4deg-19level-5day-2026-10-06-cm4vars-uh` = `01M4E83P5GMXQEN2JN8SZFB5BN`.
- [x] Corrector option + tests (ace 719be7d38).
- [x] Configs and launcher (ace ff18a0fd9).
- [x] Launched 2026-10-08 `samudra-ufs4deg-{resid-cap0005,ff-ohc}-learnedheat` (200 epochs,
      as the from-scratch arms; wandb group samudra-ufs4deg-learnedheat) after the
      training-dataset upload committed.
- [x] Both arms train and log the new channel; `resid-cap0005-learnedheat` finished 200
      epochs on 2026-10-09 (inline channel mean 0.041, learned-channel normalized RMSE 0.087).
      Its learned term's ocean mean drifts with training epoch (−1.9 W/m² at epoch 75,
      ~0 at 125, +1.3 at 200 against a data mean of +0.45 over the inline windows) and the
      inline SST bias follows it (−0.08, +0.04, +0.10 K). Checkpoint selection on the
      inline channel mean picks epoch 200.
- [ ] 10-yr ocean-only rollouts with monthly `unaccounted_heating` output
      (`launch-ufs4deg-eval-10yr.sh`; launched 2026-10-09 for the finished residual arm) to
      compare the learned ocean-mean series with the data's; same for the constant-0 twin
      when it finishes.
- [ ] Evaluate: inline drift vs the constant-0 twins; the learned field's ocean mean vs the
      data's annual series; 2002-2006 window rollouts.
- [ ] Write up (reports repo).

## Open questions

- Whether the network should also receive the previous step's unaccounted heating as an
  input (persistence of the nudging). Not in the first pass.
- Whether the per-cell pattern should feed the corrector's deposition rather than only the
  mean. Not in the first pass; the pattern is scored and inspected first.
- The target uses the plain rebuild everywhere; the coastal/ice term the network supplies in
  the `prescribed` correction is a separate, known bias (surface-flux-budget report).
