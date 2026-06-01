# Training Run 1: holdout design

Plan for the first end-to-end CMIP6 emulator training on the v2 cohort
(243 datasets, 38 source models, `historical 1940-2014` + `ssp245
2015-2100` + `ssp585 2015-2100`, schema 0.6.0). Goal of this doc:
state the generalization dimensions we want to measure, propose the
held-out cohorts that probe each, and sketch the matching validation
+ inline inference entries.

This is a **single training configuration**: one model, one train
set, one set of held-out cohorts. Each source model appears in at
most one cohort so each eval signal maps to exactly one
generalization dimension.

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
3. **Single-realisation training (ERA5-analog)**. In the *same*
   training config as the other cohorts, deliberately reduce one
   source model's training data to a single historical realisation
   (no other variants, no SSPs of that source), then test that
   source's other variants and its SSPs. Simulates "what happens
   when we only have one historical (like ERA5)" — can we still
   produce reasonable scenarios?
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

- **(C) ERA5-analog spans two ECS regimes.** CanESM5 picks the
  high-sensitivity outlier (Δ +3.83 K) — a deliberately hard test.
  **GFDL-CM4** joins as the mid-ECS complement (Δ ≈ +2.5 K, sits
  near the cohort median); cleanest pick at this ECS tier because
  no source pairs with it at r>0.85 (closest cluster-mate
  MRI-ESM2-0 is well below threshold). GFDL-CM4 has only one
  historical realisation in v2, so the ERA5-analog setup
  collapses to "train on that one historical, hold out all SSPs"
  — same single-realisation spirit but a thinner eval surface
  (2 SSP datasets vs CanESM5's 29).
- **(B) ssp585 holdout** spans the ECS axis cleanly under the
  one-source-per-cohort rule: UKESM1-0-LL (high) + MPI-ESM1-2-LR
  (low). CanESM5 moved to (C) since it's the strongest (C) target.
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

## Held-out cohorts

**Rule.** Each source appears in at most one cohort. 11 of 38 sources
are assigned to a holdout; the remaining 27 are unconstrained
training models.

After holdouts: train set = **196 datasets** (81% of 243), with 8 of
those (from D) carrying a 1970-1989 time mask.

### A. Held-out variants — internal variability

For source models with ≥3 variants in a given scenario, hold out
their `r2` (or similar) from training. Picks chosen from non-outlier
sources spanning 5 families:

- `CNRM-CM6-1/ssp245/r2i1p1f2` (CNRM family, mid-ECS)
- `CNRM-CM6-1/ssp585/r2i1p1f2` (same source, ssp585 sibling)
- `CNRM-ESM2-1/historical/r2i1p1f2` (CNRM family)
- `MRI-ESM2-0/historical/r2i1p1f1` (GFDL/MRI family)
- `EC-Earth3/ssp585/r3i1p1f1` (EC-Earth family, doubles as the
  unusual-response-shape probe — see (A) inline inference below).
  EC-Earth3/ssp585 has no `r2i1p1f1` variant in v2 (available are
  r1, r3, r4); r1 has a publisher-side full-slab NaN issue (v2
  report §8b), so r3 is the cleanest pick at the planned "second
  realisation" slot.
- `IPSL-CM6A-LR/historical/r2i1p1f1` (IPSL/MIROC family)

= **6 datasets** across 5 sources, 5 families.

EC-Earth3 itself sits in (A) rather than EC-Earth3-Veg-LR so that the
held-out variant rollout doubles as the test of whether the embedding
captured EC-Earth3's unusual stratospheric warming-response shape
(`zg10` Δ=−290 m vs cohort −30; see "Warming response" above). The
model has seen EC-Earth3/historical + ssp245 + the other ssp585
variants (r1, r4) in training, so this asks whether the embedding
*represents* that shape in a transferable way — stronger test than an
in-sample rollout.

### B. Held-out (model, ssp585) — future-scenario extrapolation

Two sources, all ssp585 variants held out; their historical + ssp245
stay in train:

- `UKESM1-0-LL/ssp585/*` (5 variants, **high-ECS** at +3.66 K)
- `MPI-ESM1-2-LR/ssp585/*` (5 variants, **low-ECS** at +1.89 K)

= **10 datasets** removed. Cleanly spans the sensitivity axis.
UKESM1's sibling HadGEM3-GC31-LL stays in train at r=0.80 — slightly
easier than CanESM5 would have been, but CanESM5 is the (C) target
and one source can't be in two cohorts. Acceptable trade.

### C. Single-realisation training — ERA5-analog

Two sources, each with **only** `<source>/historical/r1i1p1f1` in
training; every other variant + every SSP variant of that source
becomes eval.

- **`CanESM5`** (high-ECS, +3.83 K) — 30 v2 datasets, 1 stays in
  train, **29 to eval** (9 historical variants + 10 ssp245 +
  10 ssp585). The deliberately hard test: rich coverage, no
  r>0.85 sibling (closest are CESM2 and NorESM2-LM at r≈0.6).
- **`GFDL-CM4`** (mid-ECS, ≈ +2.5 K) — 3 v2 datasets, 1 stays in
  train, **2 to eval** (ssp245 + ssp585; no other historical
  realisations exist in v2). Thinner eval surface but no
  r>0.85 sibling either (closest is MRI-ESM2-0, well below
  threshold).

= **31 datasets** removed from train (29 CanESM5 + 2 GFDL-CM4).
The two ECS regimes let us tell apart "ERA5-analog failed because
of high-ECS extrapolation" from "ERA5-analog failed because of
single-realisation training" — failure on both points at the
latter, failure on CanESM5 only points at the former.

### D. Temporal interpolation — held-out 1970-1989

Two sources, time-mask the 1970-1989 window of their historical
training data (keep `[1940, 1969] ∪ [1990, 2014]`). Eval set is the
same sources' historical restricted to the 1970-1989 window.

- `INM-CM5-0/historical/*` (5 variants, MPI-adjacent family)
- `NorESM2-LM/historical/*` (3 variants, CESM-derived family)

= **0 datasets** removed (time slice only on 8 dataset-equivalents).
Both sources are non-outlier, multi-variant, and chosen to avoid
overlap with (A)/(B)/(C). Different families so the result isn't
family-specific.

The 20-year gap is large enough that the model can't trivially
interpolate from neighbouring days, but short enough that the
forcing trajectory across the gap isn't wildly different from what
the model sees.

## Variables

**Input-only**: statics + exogenous forcings only — `HGTsfc`,
`land_fraction`, `log_input4mips_co2`, `input4mips_so2`,
`input4mips_bc`, `luh2_forest`. Everything else is in both
`in_names` and `out_names`.

**Excluded**: all CMIP6 monthly-Amon / monthly-Omon / monthly-SImon
variables that the v2 ingest sampled as ``monthly_causal`` (each
day in month M takes month M-1's value, producing stepped daily
fields). They are physically correct but misleading as daily
training targets — the model would learn to be artificially smooth
within each month and then discontinuous on month boundaries.
Specifically dropped from this run: `amon_ts` (universal but
stepped), `omon_zos`, `omon_hfds`, `omon_mlotst`, `omon_tob`,
`simon_sea_ice_fraction`, `simon_sitemptop`, `simon_ocean_fraction`.

**Surface temperature**: the proper daily field is `eday_ts`
(Eday.ts, 40% coverage in v2). v2 keeps it under the prefixed
`eday_ts` name; the schema-0.8.0 update renames it to
`surface_temperature` at the data-processing layer so future cohort
ingests align with the SHIELD/ERA5 baseline convention. No
migration is written for v2 → v3 — we'll reprocess fresh rather
than rename in place.

**In + out** (everything except the input-only six):

- Universal (100%): TMP2m, Q2m, UGRD10m, VGRD10m, PRATEsfc, h500,
  psl, ua/va/hus/zg × 8 plev (32 vars)
- Daily, sub-universal (with allow_variable_masking on the step):
  sfcWind (84%), DSWRFsfc (93%), LHTFLsfc (89%), ULWRFsfc (86%),
  ULWRFtoa (82%), SHTFLsfc (80%), USWRFsfc (65%), oday_tos (56%),
  DLWRFsfc (52%), eday_ts (40%), water_vapor_path (40%),
  siday_sitemptop (38%), siday_sithick (27%).

The 13 sub-universal variables use the per-sample variable-masking
path; datasets without them contribute no loss signal for those
channels but stay in training for everything else.

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

### Validation (one-step error, all datasets in cohort)

Run on *all* datasets in each cohort, every epoch. **Stride to
~1000 samples per dataset** (`subset: { step: N }` on the loader,
with N chosen per dataset so 23k-31k timesteps land at ~1000
samples — N≈25 for historical, N≈30 for SSP) — that's roughly 3
years of daily data per dataset, enough for a tight per-source
one-step error estimate while keeping per-epoch cost bounded.

**No `val_in_sample` entry.** The training loop already runs a
"training" evaluation each epoch on a random subsample of the
training data (`train_evaluation_samples`); a separate
`val_in_sample` entry would duplicate that signal. **Set
`train_evaluation_samples: 2000`** (default is 1000) — with a
train set of 196 datasets × ~30k timesteps each ≈ 6M total, 2000
samples gives a relative SE under 2-3% on the per-epoch
train-distribution loss without materially adding to epoch cost.

`val_holdout_variants` (A) carries `weight: 1.0` and drives
checkpoint selection. Cohorts B/C/D get `weight: 0.0` (eval
signal only; not in the checkpoint metric).

**Separate entries** for the two SSP holdouts (B) and the two
ERA5-analog targets (C) so we can read per-source signal at a
glance. Cohorts A and D combine their multiple sources into a
single entry each.

**6-month gap (D only).** For temporal-interpolation eval, the
training data ends at 1969-12-31 on the early side and starts at
1990-01-01 on the late side; trim the eval window to
[1970-07-01, 1989-06-30] so no eval timestep sits within 6 months
of a training timestep. For A/B/C the train and eval datasets are
different (variant, scenario, or source), so calendar-time
proximity isn't a leak vector and no time slicing is needed.

```yaml
validation:
  - name: val_holdout_variants      # (A) internal-variability — drives checkpoint
    weight: 1.0
    loader:
      dataset:
        source_ids: [CNRM-CM6-1, CNRM-ESM2-1, MRI-ESM2-0,
                     EC-Earth3, IPSL-CM6A-LR]
        realizations: [2]
        subset: { step: 25 }        # ~1000 samples per dataset

  - name: val_holdout_ssp585_ukesm  # (B) high-ECS scenario probe
    weight: 0.0
    loader:
      dataset:
        source_ids: [UKESM1-0-LL]
        experiments: [ssp585]
        subset: { step: 30 }
  - name: val_holdout_ssp585_mpi    # (B) low-ECS scenario probe
    weight: 0.0
    loader:
      dataset:
        source_ids: [MPI-ESM1-2-LR]
        experiments: [ssp585]
        subset: { step: 30 }

  - name: val_era5_analog_canesm5   # (C) high-ECS ERA5-analog
    weight: 0.0
    loader:
      dataset:
        source_ids: [CanESM5]
        # held-out: everything except historical/r1i1p1f1
        subset: { step: 28 }
  - name: val_era5_analog_gfdl      # (C) mid-ECS ERA5-analog
    weight: 0.0
    loader:
      dataset:
        source_ids: [GFDL-CM4]
        experiments: [ssp245, ssp585]
        subset: { step: 30 }

  - name: val_holdout_years         # (D) temporal interpolation, 6mo gap
    weight: 0.0
    loader:
      dataset:
        source_ids: [INM-CM5-0, NorESM2-LM]
        experiments: [historical]
        time_slice: { start: 1970-07-01, end: 1989-06-30 }
        subset: { step: 7 }         # 19y × 365 / 7 ≈ 990 samples
```

### Long inference (multi-step rollout)

**One dataset per inference entry.** The current
`InferenceDataLoaderConfig` accepts a single
`XarrayDataConfig` (or a `MergeNoConcatDatasetConfig` for
side-by-side merge, not concat-over-time across sources), so
"hold out N variants and run rollouts on all" must be expressed
as N separate inference entries. That keeps per-entry metrics
clearly attributable to one source × scenario × variant.

**IC convention.** Unless noted otherwise:

- **4 ICs starting at the first of each season** (Mar 1, Jun 1,
  Sep 1, Dec 1) of the first available year of the dataset.
- **1 ensemble member per IC.**

The one exception: **5-day weather forecasts use 8 ICs × 5
ensemble members per IC** so we can compute CRPS / ensemble
spread-skill / rank histograms. Other rollouts are too long for
ensemble skill measures to add useful signal beyond a single
realisation.

**Rollout horizons.** Calibrated so each rollout fits inside its
dataset's available time window even when the last IC (Dec 1)
needs the full horizon ahead of it:

| Cohort | Horizon | n_forward_steps (noleap) | Notes |
| --- | --- | --- | --- |
| in-sample weather | 5 days | 5 | 8 ICs × 5 members |
| in-sample long | 30 years | 10950 | 4 seasonal ICs, 1 source |
| (A) held-out variants | 10 years | 3650 | 1 ensemble |
| (B) future scenario | 30 years | 10950 | 1 ensemble |
| (C) ERA5-analog | 30 years | 10950 | 1 ensemble |
| (D) temporal interp | 19 years | 6935 | 19y so Dec-1-1970 IC stays inside 1970-1989 window |

**`evaluate_before_training: true`** at the `TrainConfig` level —
inline validation + inference run once before the first training
epoch, so any data-loading / OOM / config issue trips before we
commit to a long training pass.

```yaml
evaluate_before_training: true
train_evaluation_samples: 2000

inference:
  # ----- in-sample sanity -----
  - name: inf_in_sample_5day             # weather-scale, IC-ensemble
    weight: 1.0
    n_forward_steps: 5
    n_ensemble_per_ic: 5
    loader:
      dataset:
        source_ids: [EC-Earth3]          # representative cohort-median source
        experiments: [historical]
        realizations: [1]
      start_indices:
        # 8 ICs spread across the historical period; ensemble of 5 per IC.
        times: [1950-03-01, 1950-09-01, 1970-03-01, 1970-09-01,
                1990-03-01, 1990-09-01, 2010-03-01, 2010-09-01]

  - name: inf_in_sample_30yr             # long climate-scale, single source
    weight: 0.5
    n_forward_steps: 10950
    loader:
      dataset:
        source_ids: [EC-Earth3]
        experiments: [historical]
        realizations: [1]
      start_indices: { times: [1950-03-01, 1950-06-01, 1950-09-01, 1950-12-01] }

  # ----- (A) held-out variants — internal variability -----
  # 3 representative picks across families + scenarios; the EC-Earth3
  # entry doubles as the unusual-stratospheric-warming-shape probe.
  - name: inf_holdout_variants_cnrm
    weight: 0.0
    n_forward_steps: 3650                # 10 years
    loader:
      dataset:
        source_ids: [CNRM-CM6-1]
        experiments: [ssp585]
        realizations: [2]
      start_indices: { times: [2015-03-01, 2015-06-01, 2015-09-01, 2015-12-01] }
  - name: inf_holdout_variants_ipsl
    weight: 0.0
    n_forward_steps: 3650
    loader:
      dataset:
        source_ids: [IPSL-CM6A-LR]
        experiments: [historical]
        realizations: [2]
      start_indices: { times: [1940-03-01, 1940-06-01, 1940-09-01, 1940-12-01] }
  - name: inf_holdout_variants_ecearth3_shape
    weight: 0.0
    n_forward_steps: 3650
    loader:
      dataset:
        source_ids: [EC-Earth3]
        experiments: [ssp585]
        realizations: [3]   # r3i1p1f1 — see "Held-out cohorts §A" note
      start_indices: { times: [2015-03-01, 2015-06-01, 2015-09-01, 2015-12-01] }

  # ----- (B) future scenario — one entry per held-out source -----
  - name: inf_holdout_ssp585_ukesm       # high-ECS, +3.66 K
    weight: 0.0
    n_forward_steps: 10950               # 30 years
    loader:
      dataset:
        source_ids: [UKESM1-0-LL]
        experiments: [ssp585]
        realizations: [1]
      start_indices: { times: [2015-03-01, 2015-06-01, 2015-09-01, 2015-12-01] }
  - name: inf_holdout_ssp585_mpi         # low-ECS, +1.89 K
    weight: 0.0
    n_forward_steps: 10950
    loader:
      dataset:
        source_ids: [MPI-ESM1-2-LR]
        experiments: [ssp585]
        realizations: [1]
      start_indices: { times: [2015-03-01, 2015-06-01, 2015-09-01, 2015-12-01] }

  # ----- (C) ERA5-analog — one entry per held-out source -----
  - name: inf_era5_analog_canesm5        # high-ECS (+3.83 K)
    weight: 0.0
    n_forward_steps: 10950               # 30 years
    loader:
      dataset:
        source_ids: [CanESM5]
        experiments: [ssp585]
        realizations: [1]
      start_indices: { times: [2015-03-01, 2015-06-01, 2015-09-01, 2015-12-01] }
  - name: inf_era5_analog_gfdl           # mid-ECS (+2.5 K)
    weight: 0.0
    n_forward_steps: 10950
    loader:
      dataset:
        source_ids: [GFDL-CM4]
        experiments: [ssp585]
        realizations: [1]
      start_indices: { times: [2015-03-01, 2015-06-01, 2015-09-01, 2015-12-01] }

  # ----- (D) temporal interpolation — one entry per source -----
  # 19y horizon so Dec-1-1970 IC's rollout stays inside 1970-01-01..1989-12-31.
  - name: inf_holdout_years_inm
    weight: 0.0
    n_forward_steps: 6935                # 19 years × 365
    loader:
      dataset:
        source_ids: [INM-CM5-0]
        experiments: [historical]
        realizations: [1]
      start_indices: { times: [1970-03-01, 1970-06-01, 1970-09-01, 1970-12-01] }
  - name: inf_holdout_years_noresm
    weight: 0.0
    n_forward_steps: 6935
    loader:
      dataset:
        source_ids: [NorESM2-LM]
        experiments: [historical]
        realizations: [1]
      start_indices: { times: [1970-03-01, 1970-06-01, 1970-09-01, 1970-12-01] }
```

**Total inference entries: 11.** Per-entry runtime is bounded by
the longest single rollout (30 years × 4 ICs); aggregate cost is
small relative to a single training epoch on the 196-dataset
train set.

## Open items to iterate on

- **Rollout-horizon memory check.** 30-year rollouts (10,950 steps)
  with the configured stepper haven't been benchmarked on the
  target training pod. Confirm peak GPU memory + wall time before
  committing — if the 4-IC × 30y combination doesn't fit, options
  are: drop to 2 ICs (Mar + Sep), or run the long rollouts on a
  subset of epochs via `epochs` slice rather than every epoch.

- **BCC-CSM2-MR**: task #87 is now resolved — the corrupt last
  day was truncated by migration 0.6.0 → 0.7.0 (sidecar
  `n_timesteps`: 23725 → 23724). BCC-CSM2-MR/historical/r1i1p1f1
  stays in train as the Chinese-family representative; no
  further action.

- **Per-entry aggregator override pattern.** Several metric-pruning
  rules (histograms full vs subset, zonal_mean on vs off, annual
  plev subset) would be cleaner with a "base aggregator + per-
  entry override" mechanism in `TrainConfig` rather than repeating
  the full `aggregator: {...}` block per entry. Not blocking — we
  can author the verbose form first and tidy later.

- **Aggregator config additions** for metric pruning have landed
  (`PowerSpectrumMetricConfig.report_directional_bias`,
  `PowerSpectrumMetricConfig.plot_variables`,
  `HistogramMetricConfig.percentile_variables`); see Metric
  pruning section below for the values to set per entry. All
  three default to current behaviour so existing configs aren't
  affected.

## Metric pruning

Default per-inference-entry metric output is wide enough that 11
inference entries × every-epoch reporting will quickly bloat the
W&B run page and the per-epoch artefact size. Trim by tier:

### Cohort-wide (apply to every train / val / inference entry)

- **Power spectrum scalar metrics: drop `positive_norm_bias` and
  `negative_norm_bias`.** Set
  `power_spectrum.report_directional_bias: false` on every
  inference entry's aggregator. `mean_abs_norm_bias` is the
  directional pair's redundant summary and stays.

- **Power spectrum chart-plot: restrict to `ua10, va10, hus10,
  zg10, h500, PRATEsfc` by default.** Set
  `power_spectrum.plot_variables: [ua10, va10, hus10, zg10, h500,
  PRATEsfc]` on every non-reference inference entry. The
  per-variable spectrum-pair PNGs are the expensive part to
  store (one figure per variable per epoch per entry × 11
  entries); the scalar metrics are cheap and stay for all
  variables because `plot_variables` is finer-grained than
  `variables`. Uppermost-level (10 hPa) probes the
  stratosphere where most models have spectrum issues, `h500`
  is the standard mid-troposphere reference, and `PRATEsfc`
  catches precipitation tail behaviour. Reference runs (see
  per-entry rules below) leave `plot_variables: null` for full
  coverage.

- **Histogram percentile reporting: 99.9999th percentile values
  only for precipitation** (`PRATEsfc`, plus `pr` where present).
  Set `histogram.percentile_variables: [PRATEsfc, pr]` on every
  inference entry. Histogram plots are unaffected; only the
  scalar percentile keys are gated by this allowlist.

### Per-entry (apply via aggregator config)

- **Histograms — restrict per entry.**
  - **`inf_era5_analog_canesm5` and `inf_holdout_ssp585_ukesm`**:
    full histograms (`enabled: true`, `variables: null` to keep
    all). These are the two reference long-rollouts where the
    full distribution shape is the goal.
  - **All other long-rollouts** (variants, GFDL, MPI, INM,
    NorESM, in-sample 30yr): `enabled: true`,
    `variables: [PRATEsfc, hus10, hus50, ua10]` — precipitation
    + stratospheric tracers that we already know are
    distribution-tail sensitive (ACCESS-ESM1-5 / EC-Earth3
    stratosphere variability flagged earlier).
  - **Weather-scale (`inf_in_sample_5day`)**: `enabled: false`.
    Histograms over 5-day rollouts don't have enough samples to
    be informative.

- **Annual means — subset 3D pressure levels.**
  The default annual config emits one metric per `{var}{plev}`.
  Restrict the plev set on `AnnualMetricConfig.variables` to
  **stratosphere + tropopause + near-surface, plus one mid-trop
  representative**:
  - Keep: `ua10`, `ua250`, `ua700`, `va10`, `va250`, `va700`,
    `hus10`, `hus250`, `hus700`, `zg10`, `zg250`, `zg500`,
    `zg1000` (one mid-trop pick per variable).
  - Drop: every other plev for `ua`, `va`, `hus`, `zg` (50, 100,
    500 for the first three; 50, 100, 700, 850 for `zg`).
  - 2D variables (`TMP2m`, `pr`, `psl`, etc.) — keep all.

- **Zonal-mean evolutions — restrict by entry.**
  - **`inf_era5_analog_canesm5`**: full zonal-mean output
    (`enabled: true`, `variables: null`).
  - **`inf_holdout_variants_ecearth3_shape`**: subset to a sparse
    key-variable list — `variables: [TMP2m, amon_ts, hus250,
    ua250, ua10, zg10]`. The 3D picks target the
    stratospheric-shape probe; the surface picks anchor the
    response.
  - **All other entries**: `zonal_mean.enabled: false`. Zonal
    means are heavy and we don't need them for every rollout.

### Additional pruning

- **`step_means` and `ensembles`: step=5 only, drop step=20.**
  The default `InferenceEvaluatorAggregatorConfig` snapshots at
  step=20 (20 days in). The 5-day weather entry can't reach
  step=20 anyway, and on the long rollouts a 20-day snapshot is
  not meaningfully different from a 5-day one — both are early
  weather, neither captures the climate behaviour the long runs
  exist to probe. Override every entry to use step=5 for
  `step_means` and `ensembles` (ensembles only fires on the
  5-day weather entry since the others have `n_ensemble_per_ic:
  1`).

- **ENSO / IPO indices** — useful only on long ocean-tracking
  rollouts. Disable on (A) 10-year variants and on the
  weather-scale entry; keep on (B), (C), (D), and
  `inf_in_sample_30yr`.

- **Video metrics** — already off by default; keep off.

- **Seasonal metrics** — already off by default; keep off (the
  user removed seasonal forecasts from the inference list).

Open question: there's no per-cohort default-aggregator
override mechanism in `TrainConfig` today — each
`InlineInferenceConfig.aggregator` is set per entry. The cleanest
path is to author the per-entry aggregator configs explicitly
in the YAML; a small refactor to support "base aggregator +
per-entry override" would let us express the rules above more
compactly but isn't blocking.
