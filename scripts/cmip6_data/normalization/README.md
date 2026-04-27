# Normalization Investigation

Working doc for deciding how to normalize the CMIP6 daily pilot dataset for
training. This is an active investigation, not a finalized spec — open
questions get resolved into the **Decisions** section as analysis lands.

The dataset (`data/cmip6-daily-pilot/v0/`) is 76 ok-status datasets across
20 models. INM-CM4-8 is excluded for unphysical `zg` at 10 hPa. Per-dataset
statistics live in `data/cmip6-daily-pilot/v0/stats.csv` (4209 rows × 14
stat columns; see `compute_stats.py`).

## Goal

Pick a normalization scheme that gives the network well-conditioned inputs
without throwing away the physical comparability across models that the
label embedding depends on. "Best" here is judged against:

1. **Loss conditioning.** All variables and all (var, plev) levels should
   contribute on a comparable scale to the training loss; no variable
   should dominate the gradient by being O(1e5) and no variable should be
   ignored by being O(1e-6).
2. **Physical comparability.** The label embedding is supposed to encode
   *between-model* differences. If two models produce identical physics
   but different per-dataset normalization scales, the embedding has to
   spend capacity learning to undo the normalization.
3. **Distribution shape.** Heavy-tailed variables (pr, hus, siconc, rsus
   — all flagged with skew > 1, kurtosis > 20 in the per-dataset stats)
   give the network a hard time even after standardization. Log/Box-Cox
   transforms before standardization may matter more than the choice of
   per-dataset vs shared.

## Strategy candidates

We're comparing three schemes (and possibly hybrids):

- **A. Shared scales.** One `(mean, std)` per variable (and per plev for
  3D), computed across all training datasets pooled together with
  area-weighting. Same scale applied at inference regardless of label.
- **B. Per-dataset scales.** One `(mean, std)` per `(variable, label)`
  (i.e., per `source_id.p`). Normalization undone at inference using the
  label.
- **C. Hybrid.** Shared scales by default, with selective per-dataset
  treatment for variables whose inter-model dispersion is extreme, and/or
  log-transform-before-shared for heavy-tailed variables.

Prior (before looking at the numbers): C is most likely correct. Shared is
the right default because (a) physical comparability is structurally
preserved, (b) the label embedding can learn small per-model affine
corrections cheaply on top of shared scales, and (c) low-variance models
*should* contribute less to the loss, since they carry less signal — that's
a feature, not a bug. Per-dataset wins only where one or two models have
such extreme variance that they dominate gradients, or where distribution
shape (not just scale) varies wildly across models.

## Open questions

1. **How dispersed are model-level stds?** For each variable (and (var,
   plev) for 3D fields), what is the inter-model coefficient of variation
   of `std`? Tight clustering → shared is essentially free. Wide spread
   (e.g. >30%) → look at per-dataset for that variable.
2. **How dispersed are model-level means?** Same question for `mean`. Are
   there model-level biases that would leak into a shared zero?
3. **How dispersed are `d1_std` values?** ACE-style training cares about
   increment scales, not absolute. The loss-weighting that actually
   matters is `d1_std`, not `std`. Inter-model dispersion of `d1_std`
   may differ from dispersion of `std`.
4. **Outlier datasets per variable.** Which models are >2× the median std
   for some variable? Are they candidates for per-dataset scales, or for
   exclusion?
5. **Distribution shape consistency for heavy-tailed variables.** For pr,
   hus, siconc, rsus (and any others with skew > 1 or kurtosis > 20):
   overlay per-model histograms. If shapes are similar (just rescaled),
   shared log/Box-Cox + shared scale handles it. If shapes diverge,
   per-dataset transforms may be needed.
6. **Climatology consistency across models.** `clim_var_frac` in the
   stats — is it consistent across models? If some models are 80%
   climatology-dominated and others 20%, mean-removal acts very
   differently per model. Affects whether to standardize against
   `std` or `anom_std`.
7. **Plev-aware normalization.** For 3D fields (ua, va, hus, zg), does
   `std` vary strongly with plev? If yes, we want per-plev std (already
   computed); the question is whether the plev structure is similar
   across models.
8. **Forcings vs prognostic state.** `ts` and `siconc` are
   monthly-interpolated forcings. Their `d1_std` is artificially low
   (smooth interpolation), so normalizing by `d1_std` would inflate
   them. They should probably use `std`, not `d1_std`.
9. **Static fields.** `sftlf`, `orog` are time-invariant per cell. They
   need normalization (orog is in meters, sftlf in 0–100%) but `d1_std`
   is undefined. Per-cell mean/std collapse to the field itself, so
   pooled (over space) mean/std is the natural choice.
10. **Masks.** `below_surface_mask` and any `siconc_mask` are 0/1. Should
    they be standardized at all, or passed through? A 0/1 mask
    standardized by mean/std gives a bimodal input that's hard for the
    network — likely better to leave as-is.
11. **Loss-side vs input-side.** Do we use the same scales for input
    normalization and loss weighting? They serve different purposes:
    inputs want "well-conditioned for the network", loss wants "physical
    error scales we care about". Could differ in principle.
12. **Held-out models.** If we plan to fine-tune the label embedding on
    unseen models, the unseen model has no precomputed scales. With
    shared scales this is a non-issue; with per-dataset scales we need
    a story (fit scales on the held-out fine-tune data; bootstrap from
    nearest-neighbour model in label space; etc.).

## Tasks

Numbered to match the questions where applicable.

- [x] **T1/T2/T3 — Inter-model dispersion of std, mean, d1_std.** Per
  (variable, plev_index), tabulate median, IQR, max/median, CoV. Output:
  `outputs/{std,mean,d1_std}_dispersion.csv` +
  `outputs/dispersion_summary.md`. Addresses Q1–Q3.
- [ ] **T4 — Outlier-dataset list.** Per variable, datasets >2× median
  std (or <0.5×). Output: `outputs/outliers.md`. Addresses Q4.
- [x] **T5 — Heavy-tail histograms.** Overlaid per-model histograms for
  pr, hus (per plev), siconc, rsus, in linear and log space. Output:
  `outputs/heavy_tail_histograms/*.png`. Addresses Q5.
- [x] **T6 — `clim_var_frac` consistency check.** Per variable, spread
  of `clim_var_frac` across models. Output: `outputs/clim_var_frac.{csv,md}`.
  Addresses Q6.
- [ ] **T7 — Plev-structure check.** For each 3D variable, plot `std`
  vs plev with one line per model. Addresses Q7.
- [ ] **T8–T11 — Decision tasks.** No analysis needed; resolve via
  discussion once T1–T7 are in.

## Findings (T1, T3, T5, T6)

**Inter-model dispersion (T1/T3) is small everywhere except stratospheric humidity.**
Most (variable, plev) groups have CoV < 12% on `std` and on `d1_std`.
The only flagged variables at CoV > 30% are `hus` at plev_index 5/6/7
(stratosphere; ~50/20/10 hPa): 36–50% CoV in `std`, 41–55% in `d1_std`,
max/median ratios 2.3–2.9×. Mean-spread on those same levels is
1.2–1.6σ (vs <0.5σ for everything else except `ta_derived_layer_6` at
0.51σ — same stratospheric story since it derives from the affected
zg/hus). See `outputs/dispersion_summary.md`.

**Heavy-tail histograms (T5) — only `pr` clearly motivates a log transform.**

- `pr`: log10(pr + 1e-7) is approximately log-normal with a small dry-day
  spike. Models agree on the log-space shape remarkably well.
- `siconc`: bimodal at 0% and 100% (open ocean / full ice). This is
  *physical* multimodality, not heavy tails. Models agree extremely
  tightly. Log doesn't help.
- `rsus`: zero-floor + day/night bimodality. Models agree on shape. Log
  doesn't help.
- `hus`: surface levels (plev 0–4) cluster well in linear space; log
  helps but isn't necessary. Stratospheric levels (plev 5–7) are
  scattered in *both* linear and log space — log narrows them but
  doesn't resolve the inter-model spread (CanESM5 is 2–3× drier than
  the median model at plev 5). INM-CM4-8 also shows long left tails at
  plev 7, consistent with its known model-side issue.

**Climatology-fraction consistency (T6) — uniform across models except,
again, stratospheric humidity.**
Range across models in `clim_var_frac` is < 0.15 for the vast majority
of (var, plev) groups. The big exceptions are `hus` at plev 5/6/7 (range
0.39–0.72: some models are 80% climatology-dominated, others 10–20%),
and a handful of upper-level dynamics fields (psl, ua/zg at plev 5–7)
at range 0.23–0.32. Mean removal is a per-variable but model-agnostic
decision for everything else.

## Proposed decisions (subject to T4/T7 + discussion)

These are working hypotheses that the analysis so far supports; they
will move to the **Decisions** section after we close out the remaining
tasks.

1. **Standardize with shared per-variable (and per-plev for 3D) scales.**
   `(x - global_mean) / global_std`, pooled across all training
   datasets with area-weighting. The dispersion analysis says this
   loses essentially no conditioning vs per-dataset for >50 of 57
   (var, plev) groups.
2. **Standardize `pr` linearly (no log transform) for the baseline.**
   The histograms show `pr` is log-normal-shaped and a log transform
   would in principle help conditioning, but prior ACE-style training
   in this codebase has been done without a log transform on pr and we
   want the baseline to match. The log-transform-vs-not question is
   recorded as a deferred perturbation experiment (see Open
   experiments below) — promising enough to revisit, not promising
   enough to deviate from baseline up front.
3. **Standardize `siconc`, `rsus`, `hus` linearly.** Their heavy-tail
   stats reflect physical multimodality the network can handle; log
   transforms don't tighten the distributions.
4. **Use shared per-variable means for mean removal.** clim_var_frac
   consistency is high enough that one mean per (variable, plev) is
   meaningful across models, except for stratospheric humidity.
5. **Stratospheric hus (plev 5–7) gets special treatment** — open
   between (a) per-dataset std for those levels only, (b) log transform
   applied only there, (c) accept that the label embedding will spend
   capacity here since model physics genuinely differ. Decision
   deferred until we look at how training behaves with shared scales
   first; cheapest baseline is "do nothing special, see if it bites".
6. **Standardize statics (`sftlf`, `orog`) and forcings (`ts`,
   `siconc`) using `std` rather than `d1_std`.** Forcings are
   monthly-interpolated and have artificially low `d1_std`; statics
   have undefined `d1_std`. (Q8 → resolved.)
7. **Pass 0/1 masks (`below_surface_mask`, `siconc_mask`) through
   un-normalized.** Standardizing a 0/1 field gives a bimodal input the
   network has no easier time with; carrying it as 0/1 keeps semantics
   for the network to use directly. (Q10 → resolved.)

## Outputs layout

```
normalization/
├── README.md                       — this doc
├── analyze_dispersion.py           — T1, T2, T3
├── analyze_outliers.py             — T4
├── plot_heavy_tails.py             — T5
├── analyze_clim_var.py             — T6
├── plot_plev_structure.py          — T7
└── outputs/
    ├── std_dispersion.csv          — wide table, per-variable rows
    ├── std_dispersion.md           — human-readable summary
    ├── mean_dispersion.csv
    ├── d1_std_dispersion.csv
    ├── outliers.md
    ├── clim_var_frac.csv
    ├── heavy_tail_histograms/
    │   ├── pr.png
    │   ├── hus_plev*.png
    │   └── ...
    └── plev_structure/
        └── *.png
```

## Decisions

(Empty for now. Each item lands here as a one-paragraph decision +
pointer to the supporting output, similar to "Resolved Issues" in
`../README.md`.)

## Open experiments (perturbations to revisit)

Items the analysis suggests could improve training but that we're
*not* doing in the baseline, to keep parity with prior runs. Each
should be a single-knob perturbation experiment once the baseline
trains.

- **Log10 transform for `pr` (with additive offset, e.g. 1e-7 kg/m²/s
  ≈ 0.01 mm/day) on input + loss.** T5 shows pr is log-normal-shaped
  and consistent across models in log space; conditioning argument
  predicts measurable training improvement. Deferred from baseline to
  match prior ACE-style training, which uses linear pr. Revisit with
  an A/B perturbation once the baseline is trained — log-input + log-
  loss is the recommended form (linear loss on logged input is the
  worst of three options, see Q11 discussion).
- **Special treatment for stratospheric `hus` (plev_index 5–7).** Both
  std and clim_var_frac dispersion across models is large at these
  levels (T1, T6); model physics genuinely differ. Cheapest fix: keep
  shared scales for the baseline and check whether stratospheric hus
  loss converges. If it doesn't, candidates are per-dataset std for
  those levels only, or a log transform restricted to those levels.
