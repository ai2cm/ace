# Physics Uncertainty Ensembles from ERA5-Tuned Embeddings

## The application

Train a multi-model CMIP6 emulator with a low-dimensional model embedding
(~8–16 dimensions), where the shared network weights learn dynamics common
across climate models and the embedding captures model-specific physics.
Then:

1. **Freeze** the network weights.
2. **Fine-tune only the embedding** against the historical ERA5 record,
   minimizing one-step prediction error over the historical period.
3. Repeat step 2 from **many random initializations** of the embedding,
   producing an ensemble of embeddings that all fit ERA5 historically but
   may differ in the dimensions of embedding space unconstrained by the
   historical record.
4. Run each tuned embedding forward under SSP forcing scenarios. The
   **spread of projections** across the ensemble is a representation of
   physics uncertainty — the range of plausible future climates consistent
   with the observed historical record.

This is analogous to a **perturbed parameter ensemble** (PPE) in climate
modeling. In a PPE, uncertain physical parameters (cloud entrainment rate,
ice fall speed, etc.) are varied and configurations consistent with
observations are retained. Here the embedding is the analog of the
uncertain parameters, and the fine-tuning-to-ERA5 step is the analog
of the observational constraint.

---

## Strengths

### Grounded in CMIP6 physics diversity

The embedding space is trained on ~37 CMIP6 models spanning a wide range
of parameterization choices, climate sensitivities, and feedback strengths.
The network weights encode dynamics informed by this diversity. The ERA5
fine-tuning then asks: "which points in this space of learned model
behaviors are consistent with observations?" This is a principled way to
sample physics uncertainty — it is constrained by both the structural
diversity of CMIP6 and the observational record.

### Continuous embedding space fills gaps in the discrete CMIP6 ensemble

The CMIP6 multi-model ensemble is a sample of opportunity — it includes
whatever models groups chose to run, with no guarantee of uniform coverage
of the space of plausible physics. A continuous embedding space can
represent model behaviors *between* existing CMIP6 models, potentially
filling gaps in the discrete ensemble. ERA5-tuned embeddings need not
coincide with any training model; they can interpolate.

### Cheap ensemble generation

Fine-tuning an 8–16 dimensional embedding is extremely cheap compared to
running a full GCM ensemble or even retraining the emulator. Generating
hundreds of ensemble members from different random initializations is
feasible, enabling robust uncertainty quantification.

### Observationally constrained

Unlike the raw CMIP6 multi-model ensemble (which includes models with
known large biases), this ensemble is explicitly constrained to match
ERA5. Models that are observationally inconsistent are automatically
down-weighted — not by subjective expert judgment but by the optimization
itself.

### Interpretable uncertainty

If the embedding space is well-structured, the dimensions along which
ERA5-tuned embeddings vary should correspond to genuinely uncertain
aspects of climate physics. Analyzing the embedding ensemble (e.g.,
via PCA, correlating embedding dimensions with emergent constraints like
climate sensitivity) could provide physical insight into what drives
projection spread.

---

## Weaknesses and risks

### The embedding may not capture the right dimensions of uncertainty

The embedding learns whatever distinguishes CMIP6 models from each other
*in terms of one-step prediction loss*. This includes mean-state biases
(a model that is 2 K too cold globally), variability differences, and
dynamical/feedback differences. Only the last category is directly relevant
to projection uncertainty.

If most of the embedding capacity is spent encoding mean-state offsets —
because those dominate the inter-model variance in one-step loss — then
the dimensions that matter for projection spread (feedback strengths,
climate sensitivity) may be poorly represented. The ERA5-tuned ensemble
would then vary mostly in "what mean-state bias best fits ERA5" rather
than "what feedback strength best fits ERA5."

**This is the single most important risk, and it is where bias correction
has the most leverage.** See the bias correction section below.

### Finite training sample limits the learned manifold

The embedding space is trained on ~37 models (~57 physics-distinct labels).
The manifold of "physically plausible behaviors" is learned from this
finite sample. Embeddings that fit ERA5 but lie far from any training
model may produce unphysical behavior — the network has never been asked
to generate dynamics for that region of embedding space. This is an
out-of-distribution generalization problem.

Mitigation: monitor how far ERA5-tuned embeddings are from the nearest
training model embedding. If they cluster near training models, this is
less of a concern. If they are in sparsely-sampled regions, validate
carefully.

### Ensemble may be too narrow or too wide

**Too narrow:** If the fine-tuning loss tightly constrains all embedding
dimensions, different random initializations converge to similar solutions
and the ensemble spread underestimates true physics uncertainty.

This risk depends heavily on the choice of loss function. One-step
prediction loss provides ~20,000 constraints per year, heavily
over-determining an 8–16 dimensional embedding — the optimization likely
has a single sharp minimum and the ensemble collapses. Aggregate climate
metrics (global means, seasonal cycles, trends) provide far fewer
constraints and are more likely to leave projection-relevant dimensions
unconstrained. See the "Choice of fine-tuning loss" section.

**Too wide:** If some embedding dimensions are unconstrained by the
historical loss but produce large projection spread, the ensemble may
include physically implausible members that happen to fit the historical
record by chance. The ensemble would overestimate uncertainty.

The balance depends on the loss landscape geometry, which can only be
determined empirically.

### Equifinality is a feature, but also a trap

Models can agree on the historical climate but disagree wildly on future
projections — this is equifinality, and it is precisely the uncertainty
we want to capture. But equifinality also means the historical constraint
may be too weak to distinguish physically meaningful diversity from
overfitting artifacts. An embedding that fits ERA5 by exploiting a
spurious mode of the network (rather than by representing a genuine
physical configuration) will produce meaningless projections.

### Common CMIP6 biases cannot be corrected by the embedding

The network weights are trained on CMIP6 data. If all CMIP6 models share
a common bias (e.g., the double-ITCZ problem, systematic underestimation
of Arctic amplification), no embedding can correct this — it is baked into
the shared weights. The ERA5-tuned ensemble inherits these common biases.
This is a fundamental limitation of any approach that learns dynamics from
climate models.

### ERA5 is one realization of internal variability

Fine-tuning to ERA5 means fitting to one specific realization of internal
climate variability over the historical period. On interannual to decadal
timescales, the observed record includes ENSO events, volcanic eruptions,
and modes of variability that are specific to this realization. The
optimization must distinguish the forced response (which the embedding
should capture) from internal variability (which it should not).

With daily one-step loss, the embedding is unlikely to overfit to specific
events (it is predicting the next day, not the decadal mean). But the
embedding that minimizes daily loss will be influenced by how well the
emulator reproduces specific modes of variability. If a particular model's
physics produces the right ENSO amplitude, the ERA5-tuned embedding may
favor that physics for reasons specific to this realization.

### Projection depends on scenario encoding, not just the embedding

The CMIP6 emulator uses forcing scenario as a separate label (not part of
the model embedding). For projections, the emulator needs to produce
physically reasonable climate response to scenarios it was trained on
(ssp245, ssp585, etc.). The projection spread comes from the *interaction*
of the embedding with the scenario encoding through the network — the
embedding modulates the climate sensitivity, but the scenario encoding
provides the forcing trajectory.

If the scenario response is primarily learned in the shared weights
(because all models respond similarly to a given scenario) and the
embedding only controls the baseline state, then the projection spread
from the embedding ensemble may be small even if the embeddings differ.
The architecture must allow the embedding to meaningfully modulate the
forced response, not just the unforced state.

---

## Role of bias correction

### What the embedding learns depends on the training data distribution

The choice of how to normalize or bias-correct the training data
fundamentally determines what the embedding encodes:

| Training data treatment | Embedding captures | ERA5 ensemble spans |
|------------------------|-------------------|-------------------|
| No correction (raw CMIP6) | Mean state + variance + dynamics + feedbacks | All of the above — but likely dominated by mean-state variation |
| Per-model mean removal | Variance + dynamics + feedbacks | Dynamics and feedback differences (mean state handled by normalization) |
| Per-model z-scoring (mean + variance) | Distribution shape + dynamics + feedbacks | Shape and dynamics differences |
| Quantile mapping to ERA5 | Only differences QM cannot fix: temporal dynamics, spatial patterns, feedbacks | Narrower but more targeted |

**The key principle: removing nuisance variation from the training data
focuses the embedding on the dimensions that matter for projections.**

### Per-model normalization: the recommended baseline

Per-model normalization (subtracting each model's climatological mean and
dividing by its standard deviation, per variable and per grid cell or
globally) is the most important bias correction step for this application.

**Why it helps:**

1. **Focuses the embedding on dynamics.** With per-model normalization,
   the network operates in anomaly space. A model that is 2 K too cold
   globally looks the same as one that is 2 K too warm — the embedding
   does not need to encode this offset. Instead, the embedding captures
   how the model's dynamics differ: feedback strengths, variability
   patterns, teleconnections, response timescales.

2. **Makes ERA5 fine-tuning more meaningful.** When fine-tuning the
   embedding to match ERA5 (also normalized by its own statistics),
   the optimization finds embeddings that produce the right *dynamics*
   for ERA5, not just the right mean state. This is more likely to
   generalize to projections, because it is the dynamics and feedbacks
   that determine the forced response.

3. **Alleviates the `hus10` problem.** Per-model normalization brings
   all models into a comparable range, preventing models with extreme
   climatologies from dominating the loss (as documented in
   `training.md` issue 11).

4. **The model's own climatology can be reconstructed at inference
   time.** For ERA5-initialized runs, the ERA5 climatology is used for
   denormalization. The emulator predicts normalized anomalies; the
   output is denormalized using ERA5 statistics. This naturally produces
   output in the ERA5 distribution without explicit bias correction
   of the output.

**What per-model normalization means concretely:**

For each training model `m`, variable `v`, and (optionally) grid cell:
```
x_normalized(t) = (x_raw(t) - μ_m,v) / σ_m,v
```
where `μ` and `σ` are computed over the historical period. At inference
from ERA5:
```
x_normalized(t) = (x_ERA5(t) - μ_ERA5,v) / σ_ERA5,v
x_output(t) = x̂_normalized(t) × σ_ERA5,v + μ_ERA5,v
```

This is effectively a per-model additive-and-multiplicative bias
correction applied as a preprocessing step. It is the simplest form of
bias correction (variance scaling), but applied in the "right" direction:
normalizing models to a common space rather than adjusting one to match
another.

### Should normalization be per-grid-cell or global?

**Per-variable global normalization** (one mean and std per variable across
all grid cells) preserves spatial patterns of bias — the embedding must
encode that a model has, say, a warm Arctic and cold tropics. This
preserves more inter-model information but means the embedding still
partially encodes mean-state spatial patterns.

**Per-variable per-grid-cell normalization** (a mean and std at each grid
cell) removes spatial bias patterns entirely. The embedding only encodes
differences in dynamics, feedbacks, and higher-order statistics. This is
more aggressive and focuses the embedding more tightly on dynamics.

For this application, per-grid-cell normalization is likely better —
spatial bias patterns are nuisance variation for the purpose of learning
dynamics that transfer to ERA5. However, it requires more storage (a
normalization field per model per variable) and introduces the risk that
some models have very few timesteps at certain grid cells, leading to
noisy normalization statistics.

**Recommendation:** Start with per-variable global normalization (which the
pipeline already supports). Move to per-grid-cell normalization as a
second step if the embedding ensemble shows signs of encoding mean-state
spatial patterns rather than dynamics.

### Beyond per-model normalization: quantile mapping

Per-model z-scoring corrects the first two moments (mean and variance).
Per-model quantile mapping to a common reference corrects the full
marginal distribution. This could help if models differ substantially in
distribution shape (skewness of precipitation, tail behavior of
temperature) and these differences are not relevant to projection
uncertainty.

**When it helps:** For variables like precipitation, where the
distribution shape varies dramatically across models and is largely a
parameterization artifact rather than a signal of meaningful physics
uncertainty.

**When it hurts:** For variables where the distribution shape *does* carry
physics information. For example, the variance of interannual temperature
variability is related to climate sensitivity through the
fluctuation-dissipation theorem — correcting it away could remove a
genuine signal.

**Recommendation:** Do not apply quantile mapping as a default. Consider it
as a targeted intervention for specific variables (e.g., precipitation)
where distribution shape differences are known to be nuisance.

### What NOT to do: bias-correct to ERA5 before training

Mapping all CMIP6 models to the ERA5 distribution before training would
collapse the inter-model diversity that the embedding is designed to
capture. The embedding would have nothing meaningful to encode, and the
ERA5-tuned ensemble would collapse to a point. Additionally, this would
break physical consistency between variables (as discussed in
`bias_correction.md`).

---

## Choice of fine-tuning loss

The fine-tuning loss determines how tightly the embedding is constrained
by the historical record, and *which dimensions* of the embedding are
constrained. This is as important as the bias correction strategy for
determining the quality of the uncertainty ensemble.

### One-step prediction loss

The most natural choice: minimize the same loss used for training
(next-day prediction error) on ERA5 data. Each year of ERA5 provides
~365 × n_variables × n_gridcells constraints on an 8–16 dimensional
embedding. This heavily over-determines the embedding.

**Advantage:** Well-defined, differentiable, consistent with training,
computationally cheap (one forward pass per sample).

**Problem:** The one-step loss is dominated by weather-scale variability.
The day-to-day prediction error depends primarily on how well the
emulator captures synoptic dynamics (fronts, storms, planetary waves) —
which is largely governed by the shared network weights and initial
conditions, not the embedding. The embedding's contribution to one-step
loss is a small perturbation on top of chaotic dynamics. This means the
optimization landscape is nearly flat in the embedding dimensions that
control slow physics (feedbacks, climate sensitivity), and steep in the
dimensions that control fast physics (parameterization details that
affect day-to-day variability).O

The result: different random initializations converge to similar
embeddings that minimize weather-scale error, leaving little spread in
the dimensions that matter for projections. The ensemble collapses.

### Aggregate climate metrics

Fine-tune the embedding to match aggregate properties of the ERA5
historical record: global or regional means, seasonal cycles, interannual
variability, trends. These provide far fewer constraints — perhaps
tens to hundreds per variable rather than tens of thousands — leaving
more embedding dimensions unconstrained.

But the deeper argument is not just about constraint count. **Aggregate
metrics target the right dimensions of uncertainty.** Global mean
temperature trend is directly related to climate sensitivity. Seasonal
cycle amplitude encodes surface-atmosphere coupling strength. Interannual
temperature variance relates to feedback strength through the
fluctuation-dissipation relationship. These are the emergent properties
that distinguish climate models and that matter for projections.

An embedding that matches ERA5's global mean temperature trajectory but
has a different cloud feedback strength would produce a different warming
rate under ssp585 — exactly the physics uncertainty we want the ensemble
to span. One-step loss would not permit this: it would pin the embedding
to the single configuration that best predicts tomorrow's weather,
regardless of what that implies for decadal climate sensitivity.

This is also more analogous to how perturbed parameter ensembles are
constrained in practice. Emergent constraints in climate science
(e.g., Sherwood et al. 2020 on climate sensitivity) are almost
always formulated in terms of aggregate metrics, not daily weather
prediction skill.

**Candidate aggregate metrics:**

| Metric | Timescale | What it constrains |
|--------|-----------|-------------------|
| Global mean temperature (annual + seasonal) | Annual | Mean state, seasonal coupling |
| Meridional temperature gradient | Annual | Large-scale circulation, polar amplification |
| Global mean precipitation | Annual | Hydrological sensitivity |
| Interannual temperature variance (global + regional) | Interannual | Feedback strength, internal variability |
| Tropical Pacific SST variance (if ocean variables available) | Interannual | ENSO-like variability |
| Temperature trend over historical period | Multi-decadal | Transient climate response |
| Seasonal cycle amplitude per variable per region | Seasonal | Surface-atmosphere coupling |

**Practical considerations:** Computing aggregate metrics requires running
the emulator autoregressively for years to decades per evaluation,
making each loss evaluation much more expensive than one-step loss.
However, since the embedding is only 8–16 parameters, this is tractable:

- **Gradient-based:** Backpropagate through the full rollout
  (backprop-through-time). Memory-intensive for long rollouts but
  feasible with gradient checkpointing for a few decades at daily
  timestep.
- **Gradient-free:** With only 8–16 parameters, evolutionary strategies,
  Bayesian optimization, or even grid search over the embedding space
  are viable. This avoids the vanishing/exploding gradient problem of
  long autoregressions and naturally produces an ensemble of good
  solutions rather than a single optimum.

Gradient-free optimization is particularly appealing here: it naturally
samples the set of embeddings that achieve a comparable loss, rather
than converging to a single minimum. Running, say, CMA-ES with a
population of 100 would directly produce an ensemble of
observationally-consistent embeddings.

### Hybrid approach

Combine one-step and aggregate losses to get the benefits of both:

1. **Sequential:** First optimize on one-step loss to get into the right
   neighborhood of embedding space (fast, well-conditioned). Then switch
   to aggregate loss and continue from multiple perturbations of the
   one-step optimum. The one-step phase finds the "right region"; the
   aggregate phase explores the region's extent.

2. **Weighted sum:** `L = L_one_step + λ L_aggregate`, where λ controls
   the relative importance. The one-step term keeps the embedding in a
   region where the network produces physically realistic daily
   dynamics; the aggregate term pulls it toward configurations that
   match climate statistics.

3. **Aggregate loss with one-step regularization:** Use aggregate metrics
   as the primary objective, but add a penalty if the one-step loss
   exceeds a threshold. This ensures the embedding produces reasonable
   daily weather while allowing freedom in the climate-scale dimensions.

### Recommendation

**Use aggregate climate metrics as the primary fine-tuning objective,
with gradient-free optimization.** This directly targets the dimensions
of physics uncertainty that matter for projections, naturally produces
an ensemble of solutions, and avoids the ensemble-collapse problem of
one-step loss. The hybrid approach (option 3: aggregate primary +
one-step threshold) is a good fallback if pure aggregate optimization
produces embeddings with unrealistic daily dynamics.

The hold-out model test (see validation section) should be run with
both one-step and aggregate loss to empirically determine which
produces better projection skill — the theoretical arguments above
may not survive contact with the actual loss landscape.

---

## Validating the ensemble

Confidence that the projection ensemble represents physics uncertainty
requires validation. The main approaches:

### Hold-out model test (the critical validation)

1. Hold out a CMIP6 model from training.
2. Fine-tune the embedding to match the held-out model's *historical*
   record (same procedure as for ERA5).
3. Run the tuned embedding forward under SSP scenarios.
4. Compare to the held-out model's actual SSP projections.

If the emulator reproduces the held-out model's projection behavior from
only its historical record, this demonstrates that (a) the embedding
captures physics that generalizes to projections, and (b) the fine-tuning
procedure extracts the right information from the historical constraint.

Repeat for multiple held-out models to assess systematic performance.
Models with unusual physics (outliers in climate sensitivity, ENSO
behavior, etc.) are the most informative tests.

### Ensemble calibration against CMIP6 spread

The projection spread of the ERA5-tuned ensemble should be compared to the
CMIP6 multi-model spread for observationally consistent models. If the
ERA5 ensemble is much narrower, it may be underestimating uncertainty. If
much wider, it may include unphysical members.

Concretely: for a metric like global mean temperature change under ssp585
at 2050, compare the ERA5 ensemble's 5–95% range to the CMIP6 5–95%
range (optionally weighted by observational consistency, e.g., Tokarska
et al. 2020).

### Embedding space analysis

- Where do ERA5-tuned embeddings sit relative to training model embeddings?
  Clustering near known training models is reassuring; isolated points in
  embedding space are concerning.
- Do different random initializations converge to a connected region or
  discrete clusters? Clusters suggest genuinely distinct physical
  configurations; a single region suggests the constraint is tight.
- Correlate embedding dimensions with known emergent properties of the
  training models (climate sensitivity, ENSO amplitude, Arctic
  amplification). If the embedding encodes these, the ERA5 ensemble
  inherits physically meaningful variation.

### Hindcast skill at seasonal-to-decadal lead times

Run each ERA5-tuned ensemble member as a hindcast: initialize from a
historical ERA5 state, run forward for 1–10 years, and evaluate against
the observed future. The ensemble mean should be skillful (beating
climatology and persistence), and the ensemble spread should be
calibrated (observed outcomes fall within the ensemble range at the
expected frequency).

---

## Summary of recommendations

### Training-time

1. **Implement per-model normalization** as the baseline. Each model's
   data is centered and scaled by its own statistics before the network
   sees it. This focuses the embedding on dynamics and feedbacks rather
   than mean-state biases. Use per-variable global statistics initially.

2. **Move to the compact learned embedding** (issue 6 in `training.md`).
   The one-hot scheme does not support fine-tuning a continuous vector
   to ERA5. A learned `nn.Linear(n_labels, embed_dim)` projection with
   `embed_dim` of 8–16 is required.

3. **Do not apply quantile mapping or more aggressive bias correction**
   to training data as a default. It risks removing inter-model
   diversity that the embedding should capture.

### Fine-tuning to ERA5

4. **Normalize ERA5 by its own statistics** (same procedure as for CMIP6
   models) before fine-tuning. The embedding is optimized in the shared
   anomaly space.

5. **Use aggregate climate metrics as the primary fine-tuning loss**
   (global means, seasonal cycles, interannual variability, trends) rather
   than one-step prediction loss. Aggregate metrics target the dimensions
   of physics that matter for projections and leave more room for ensemble
   diversity. One-step loss over-determines the embedding and risks
   collapsing the ensemble. See the "Choice of fine-tuning loss" section
   for detail.

6. **Consider gradient-free optimization** (CMA-ES, evolutionary
   strategies) for the embedding fine-tuning. With only 8–16 parameters,
   gradient-free methods are tractable and naturally produce a population
   of good solutions rather than converging to a single minimum. This
   directly generates the ensemble.

7. **Diverse random initialization matters** regardless of optimization
   method. Sample initial embeddings from a distribution broad enough to
   cover the training model embeddings (e.g., uniform over the convex
   hull, or Gaussian with variance matching the training embedding
   variance). Track convergence: if all initializations converge to the
   same point, the ensemble is effectively a single model and the
   approach provides no uncertainty quantification.

### Validation

8. **Prioritize the hold-out model test** (train without model X,
   fine-tune embedding to model X's historical record, evaluate on
   model X's projections). This is the most direct test of whether the
   approach works. If it fails, the ERA5 ensemble should not be trusted.
   Run this test with both one-step and aggregate loss to empirically
   determine which produces better projection skill.

9. **Compare ensemble spread to CMIP6 spread** for standard metrics
   (global warming at 2050, regional precipitation change) under
   multiple SSP scenarios.

### Bias correction of output

10. If the emulator output is denormalized using ERA5 statistics (as
    recommended above), the output is already in the ERA5 distribution
    for the historical period. For projections, the output naturally
    drifts from the ERA5 distribution as the climate changes — this drift
    *is* the projection signal and should not be removed.

11. If users require output calibrated to observations (not just
    ERA5-like but matching station data, for example), apply **Quantile
    Delta Mapping** as a post-processing step. QDM preserves the
    emulator's projected changes at each quantile while adjusting the
    baseline distribution.
