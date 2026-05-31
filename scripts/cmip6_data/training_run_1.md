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

## Model-similarity survey

Before fixing the holdouts, we ran a cross-source clustering pass on
the v2 per-dataset scalar stats (`(source, variable) → cohort
z-score`, Pearson similarity, average-linkage hierarchical
clustering). Two things came out of it that constrain holdout
selection:

**Outlier models (≥3 outlier vars at |z| > 2.5).** Real models with
real biases — keep in *training* (they're part of the cohort spread
we're trying to capture) but **don't put them in eval-only holdouts**,
or the metric will mostly measure their idiosyncratic biases rather
than the emulator's generalization:

| Source | # outlier vars | Most extreme variables |
| --- | --- | --- |
| MIROC-ES2L | 6 | HGTsfc, omon_zos, UGRD10m, psl, ua850 |
| IPSL-CM5A2-INCA | 5 | ua10, sfcWind, DLWRFsfc, sea-ice |
| MIROC6 | 4 | psl/PRESsfc systematically low (~290 Pa below cohort) |
| IITM-ESM | 3 | negative stratospheric humidity (already excluded) |
| BCC-ESM1 | 3 | wind at 100 hPa, ULWRFtoa |
| CanESM5 | 3 | Q2m, ua100 |

**Family pairs (r > 0.85).** If we held out one of these and kept the
other in training, the "held-out" model would effectively be
in-sample. Critical if we ever do unseen-model-entirely; informative
for the current holdouts too:

- CMCC-CM2-SR5 ↔ CMCC-ESM2 (r=0.991 — nearly identical)
- EC-Earth3-CC ↔ EC-Earth3-Veg (r=0.962)
- CNRM-CM6-1 ↔ CNRM-ESM2-1 (r=0.935)
- IPSL-CM6A-LR ↔ IPSL-CM6A-LR-INCA (r=0.876)
- CNRM-CM6-1-HR ↔ CNRM-ESM2-1 (r=0.860)
- CESM2 ↔ CESM2-WACCM (r=0.858)
- HadGEM3-GC31-LL ↔ UKESM1-0-LL (r=0.803)

**Hierarchical clustering (8 clusters).** Eight model families emerge
cleanly. The ERA5-analog single-realisation run (C below) and any
future unseen-model holdout should keep at least one representative
from each family in training:

| Family | Members |
| --- | --- |
| EC-Earth | EC-Earth3, -AerChem, -CC, -Veg, -Veg-LR; + IITM-ESM clusters here |
| BCC / Chinese | BCC-CSM2-MR, BCC-ESM1, CAMS-CSM1-0, CMCC-CM2-HR4, NESM3 |
| IPSL / MIROC | IPSL-CM6A-LR, IPSL-CM6A-LR-INCA, MIROC-ES2L, MIROC6 |
| GFDL / MRI | GFDL-CM4, MRI-ESM2-0 |
| CNRM | CNRM-CM6-1, CNRM-CM6-1-HR, CNRM-ESM2-1 |
| MPI-adjacent | MPI-ESM-1-2-HAM, MPI-ESM1-2-LR, CESM2-FV2, FGOALS-f3-L, INM-CM5-0, IPSL-CM5A2-INCA |
| Hadley / ACCESS | ACCESS-CM2, ACCESS-ESM1-5, HadGEM3-GC31-LL, UKESM1-0-LL |
| CESM-derived | CESM2, CESM2-WACCM, CMCC-CM2-SR5, CMCC-ESM2, CanESM5, NorESM2-LM, NorESM2-MM, TaiESM1 |

## Warming response (ssp585 − historical)

Cohort warming response, computed as area-weighted (ssp585
time_mean_map − historical time_mean_map) per (source, variable),
exposes a clear **climate-sensitivity axis** that aligns with known
CMIP6 ECS rankings:

| Sensitivity tier | Sources | TMP2m Δ (K) |
| --- | --- | --- |
| High (z > +1) | CanESM5, UKESM1-0-LL, HadGEM3-GC31-LL | +3.8, +3.7, +3.5 |
| Median cohort | most models | +2.5 to +3.0 |
| Low (z < −1) | MIROC6 (z=−2.89), IITM-ESM, MPI-ESM1-2-LR | +0.74, +1.76, +1.89 |

The Hadley family (HadGEM3 ↔ UKESM1 ↔ ACCESS) consistently leads
warming response on `TMP2m`, `amon_ts`, `PRATEsfc`, `h500`, `Q2m`;
MIROC6 / IITM-ESM consistently lag. CanESM5 is at the top with
ECS ≈ 5.6 K — among the highest in CMIP6.

Cross-variable warming-response outliers (|z|>2.5 on ≥1 variable):

- **MIROC6 (8 vars)**: consistently *low* — known low-sensitivity.
- **EC-Earth3 (8)**: stratospheric heights *decrease* with warming
  (zg10 Δ=−290 m vs cohort −30) — unusual response shape.
- **CanESM5 (5)**: high Q2m / hus250, very negative psl response.
- **BCC-CSM2-MR (5)**: all `ua` levels — **suspect** (same model
  has unphysical 88,800 m/s wind values that the 1e10 sanity clip
  missed; investigation queued).
- **UKESM1-0-LL (3)**: high stratospheric hus + ua10 response.

Implications for the holdout design:

- **(C) ERA5-analog on CanESM5** picks a high-sensitivity *outlier*.
  Deliberately hard test, but means failure could be either
  sample-size or unusual physics. Note: if (C) results look bad,
  re-run with a cohort-typical source (CMCC-ESM2 or MRI-ESM2-0)
  to disentangle.
- **(B) ssp585 holdout** (CanESM5, MPI-ESM1-2-LR, UKESM1-0-LL)
  happens to span the sensitivity range cleanly (high / low / high).
  Good as-is.
- **MIROC6 and EC-Earth3 ssp585 rollouts** are particularly
  informative as inline-inference targets even though they're in
  training — if the emulator reproduces their unusual sensitivity
  patterns from forcing alone, that's evidence the embedding has
  learned ECS-axis structure.

## Variability outliers

Per-source `std` and `d1_std` scans across all variables surface
several models with anomalously wide variability:

- **BCC-CSM2-MR**: 19 variables at |z|>2 on std, 25 on d1_std.
  Caused by remaining unphysical wind values up to 88,800 m/s
  that the 1e10 clip missed. **Treat with caution until the
  follow-up investigation completes.**
- **ACCESS-ESM1-5**: massively elevated stratospheric `hus` std
  (z≈+4.6 on hus10/50/100). Real model behaviour or another
  publisher quirk — flag for inspection during training, may
  warrant per-model normalization for stratospheric humidity.
- **EC-Earth3**: elevated stratospheric `zg` variability (z=+3.2
  to +3.6 on zg10/50/100). Consistent with EC-Earth's known
  strong stratospheric dynamics; real, not pathological.
- **GFDL-CM4**: very strong `omon_hfds` variability (z=+4.3) —
  noisy ocean heat-flux field.
- **MIROC-ES2L, IITM-ESM**: *suppressed* variability (e.g. psl std
  ~25% below cohort, stratospheric zg std ~20% below). Real model
  behaviour but means these are easy "in-sample" cases — they
  contribute little signal to the loss.

Practical impact for training:

- High-std models dominate shared-normalization loss; the
  stratospheric `hus` problem (training.md issue #11) is
  ACCESS-ESM1-5 and BCC-CSM2-MR amplifying that effect.
- Low-std models (MIROC-ES2L, IITM-ESM) contribute little to the
  loss signal under shared normalization — per-source normalization
  matters most for them.

## Held-out cohorts (main training run)

After holdouts, training set ≈ 175–185 datasets across ~30 source
models.

### A. Held-out variants (internal-variability test)

For source models with ≥3 variants in a given scenario, hold out their
`r2` (or similar) from training but keep the same (source, scenario)
cell in. Picks chosen from non-outlier sources spanning multiple
families (CNRM, GFDL/MRI, EC-Earth, Hadley, MPI), so the result isn't
dominated by a single family or by an outlier model's biases:

- `CNRM-CM6-1/ssp245/r2i1p1f2`
- `CNRM-CM6-1/ssp585/r2i1p1f2`
- `CNRM-ESM2-1/historical/r2i1p1f2`
- `MRI-ESM2-0/historical/r2i1p1f1`
- `EC-Earth3-Veg-LR/historical/r2i1p1f1`
- `MPI-ESM1-2-LR/historical/r2i1p1f1`

~6 datasets across 5 families.

### B. Held-out (model, ssp585)

For sources with rich multi-variant ssp585 coverage so the holdout
signal is statistically meaningful, drawn from non-outlier sources
across distinct families:

- `CanESM5/ssp585/*` (10 variants, CESM-derived family)
- `MPI-ESM1-2-LR/ssp585/*` (5 variants, MPI family)
- `UKESM1-0-LL/ssp585/*` (5 variants, Hadley/ACCESS family — but
  note its sibling HadGEM3-GC31-LL stays in train and has r=0.80
  similarity, so this is on the easier side of the test; the
  CanESM5 + MPI picks are the harder probes)

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

Pick a model with rich variant + scenario coverage. **CanESM5** fits
(10/10/10 historical/ssp245/ssp585 = ~30 eval datasets, sits in the
CESM-derived family but isn't near-identical to any cohort member —
closest relatives are CESM2 at r≈0.6 and NorESM2-LM at r≈0.6, both
much weaker than the r>0.85 family pairs above). The single
realisation kept in train would be `CanESM5/historical/r1i1p1f1`;
everything else `CanESM5/*` moves to eval. Compare its eval metrics
to the main-run eval for the same datasets to see how much the
multi-variant + multi-scenario training contributed.

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

Candidates — drawn from non-outlier sources with ≥3 historical
variants (so we can also see ensemble spread of the gap-fill) and
spanning multiple families to avoid family-specific signals:

- `MRI-ESM2-0/historical/*` (5 variants, GFDL/MRI family)
- `MPI-ESM1-2-LR/historical/*` (5 variants, MPI family — but note
  `r2i1p1f1` is in A; use the others)
- `INM-CM5-0/historical/*` (5 variants, MPI-adjacent)
- `NorESM2-LM/historical/*` (3 variants, CESM-derived)
- `EC-Earth3-Veg-LR/historical/*` (3 variants, EC-Earth — but
  `r2i1p1f1` is in A; use the others)

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
