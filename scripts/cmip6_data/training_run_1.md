# Training Run 1: holdout design

Plan for the first end-to-end CMIP6 emulator training on the v2 cohort
(243 datasets, 38 source models, `historical 1940-2014` + `ssp245
2015-2100` + `ssp585 2015-2100`, schema 0.6.0). Goal of this doc:
state the generalization dimensions we want to measure, propose the
held-out cohorts that probe each, and sketch the matching validation
+ inline inference entries. Numbers below are starting proposals; we
will iterate on them when we build the actual `TrainConfig`.

## Dimensions tested

Four kinds of generalization, each with a holdout designed to isolate
it:

1. **Internal variability** (same model, same scenario, perturbed
   initial condition). Tests whether the embedding + forcing alone
   can reproduce a model's ensemble spread — the "synthetic
   ensemble from a single-member model" Wow goal.
2. **Future scenario for a known model** (model is in train on
   historical+ssp245; its ssp585 is held out). Tests whether the
   model has learned a response to forcing rather than memorising
   trajectories.
3. **Single-realisation training (ERA5-analog)**. *Separate
   training experiment.* Train on **one historical realisation of
   one source model only** (no other variants, no SSPs of that
   source), then test that source's other variants and its SSPs.
   Simulates "what happens when we only have one historical (like
   ERA5)" — can we still produce reasonable scenarios?
4. **Temporal interpolation within historical** (replaces the
   "extrapolation" sketch). Withhold a 20-year window in the middle
   of historical (e.g. 1970-1989) for ~5 training source models.
   Tests whether the model is just memorising timestamp ↔ state vs
   learning dynamics around forcing — interpolation across a gap
   is harder to fake than extrapolation off the end.

Deferred for this run:

- **Unseen-model entirely** (full source held out). Punt to a later
  run; we'll have a clearer picture of where the cohort-wide
  baseline lands first.
- **Variable-set heterogeneity** (deliberately mask some variables
  in training to test the `allow_variable_masking` path). The
  ~33% of v2 datasets that organically lack 3D state already
  exercise this code path; a targeted experiment can wait.
- **Forcing-trajectory extrapolation** (ssp126 / ssp370 / piControl
  / abrupt-4xCO2). Not in v2; flag for a v3 ingest if we want
  these later.

## Held-out cohorts (main training run)

After holdouts, training set ≈ 175–185 datasets across ~30 source
models.

### A. Held-out variants (internal-variability test)

For source models that still have ≥3 variants in a given scenario
after holdouts B/C are applied, hold out their `r2` from training but
keep the same (source, scenario) cell in. Tentative:

- `CNRM-CM6-1/ssp245/r2i1p1f2`
- `CNRM-CM6-1/ssp585/r2i1p1f2`
- `CNRM-ESM2-1/historical/r2i1p1f2`
- `MRI-ESM2-0/historical/r2i1p1f1`
- `EC-Earth3-Veg-LR/historical/r2i1p1f1`

~5–10 datasets. Multiple sources so the result isn't model-specific.

### B. Held-out (model, ssp585)

For sources with rich multi-variant ssp585 coverage so the holdout
signal is statistically meaningful:

- `CanESM5/ssp585/*` (10 variants)
- `MPI-ESM1-2-LR/ssp585/*` (5)
- `UKESM1-0-LL/ssp585/*` (5)

Their `historical` and `ssp245` stay in training, so we're testing
whether the emulator can extrapolate forcing within the same model
family. Total: 20 datasets.

### C. Single-realisation training (separate experiment)

A second training run, otherwise identical to the main one, except
the train set for one specific source model is reduced to:

- exactly one historical variant of that source (e.g. `CanESM5
  historical r1i1p1f1`)
- nothing else from that source

The eval set then includes:

- that source's other historical variants → ensemble-spread under
  data-scarcity
- that source's `ssp245` and `ssp585` → can-we-predict-the-future
  -from-one-historical (the ERA5 analog)

Pick a model with rich variant + scenario coverage (CanESM5 fits;
~30 of its datasets become eval). Compare its eval metrics to the
main-run eval for the same datasets to see how much the multi-variant
+ multi-scenario training contributed.

### D. Temporal interpolation (held-out 1970-1989)

For ~5 training source models, restrict their `historical` training
data to `[1940-01-01, 1969-12-31] ∪ [1990-01-01, 2014-12-31]`, dropping
the 20-year middle. Done by filtering the time axis at the data
loader, not by removing datasets. Eval set is the same source's
historical 1970-1989 window.

The 20-year gap is large enough that the model can't trivially
interpolate from neighbouring days, but short enough that the
forcing trajectory across the gap isn't wildly different from what
the model sees.

## Validation + inline inference (config sketch)

`TrainConfig.validation` is a list of `InlineValidationConfig` —
one-step error, cheap, run every epoch. `TrainConfig.inference` is a
list of `InlineInferenceConfig` — multi-step rollout, expensive,
also run every epoch by default (we should run the long ones every
epoch since training is going to take many epochs to converge
against this dataset size, and the per-epoch cost is small relative
to the train pass).

Held-out cohorts get `weight: 0.0` so they don't contaminate
checkpoint selection (otherwise we'd overfit to the small eval set).
Only in-sample validation drives the checkpoint metric.

```yaml
validation:
  - name: val_in_sample           # training-distribution sanity
    weight: 1.0
    loader:
      dataset: { source_ids: <train>, experiments: [historical, ssp245, ssp585] }
  - name: val_holdout_variants    # internal-variability (A)
    weight: 0.0
    loader:
      dataset: { source_ids: [CNRM-CM6-1, CNRM-ESM2-1, MRI-ESM2-0, EC-Earth3-Veg-LR],
                 realizations: [2] }
  - name: val_holdout_ssp585      # scenario-for-known-model (B)
    weight: 0.0
    loader:
      dataset: { source_ids: [CanESM5, MPI-ESM1-2-LR, UKESM1-0-LL],
                 experiments: [ssp585] }
  - name: val_holdout_years       # temporal interpolation (D)
    weight: 0.0
    loader:
      dataset: { source_ids: <D models>, experiments: [historical] }
      # time_slice (or equivalent) restricts to 1970-1989
```

```yaml
inference:
  # short rollouts at training distribution → catch obvious blowups
  - name: inf_in_sample_5day
    weight: 1.0
    n_forward_steps: 5
    # ... aggregator etc.

  - name: inf_in_sample_seasonal
    weight: 0.5
    n_forward_steps: 365

  # long rollouts — 20-40 years; every epoch is fine, train pass
  # cost dominates and we'll run many epochs anyway
  - name: inf_in_sample_20yr      # in-distribution climate
    weight: 0.5
    n_forward_steps: 7300          # 20 years × 365

  - name: inf_holdout_variants_decadal   # (A) ensemble-spread
    weight: 0.0
    n_forward_steps: 3650          # 10 years
    n_ensemble_per_ic: 5

  - name: inf_holdout_ssp585_30yr        # (B) future-scenario climate
    weight: 0.0
    n_forward_steps: 10950         # 30 years × 365

  - name: inf_holdout_years_20yr         # (D) bridging the 1970-1989 gap
    weight: 0.0
    n_forward_steps: 7300

  # (C) lives in a separate run; same inference set but trained on
  # a single historical realisation, see "Single-realisation training"
  # above.
```

## Open items to iterate on

- Which source(s) for the temporal-interpolation cohort (D) — needs
  ≥1940 coverage and clean data; pick from the strongest in-cohort
  models so the gap-fill isn't masked by data-quality noise.
- Which source for the ERA5-analog separate run (C) — `CanESM5` is
  the default candidate (deepest multi-variant + multi-scenario
  coverage), but a smaller-ensemble model would more honestly
  simulate the data-scarcity setting we'll have with ERA5. The
  trade-off is statistical power of the eval signal.
- Whether to add a `val_in_sample_recent` (last 5 years of training
  data) variant — useful as a smoke check for distribution drift.
- Rollout horizons: 5-day / 365-day / 7300-day are placeholders.
  Confirm 7300-day rollouts fit in time + memory before committing.
