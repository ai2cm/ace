# State of the research and avenues for exploration (2026-06-10)

A synthesis of the `experiment/2026-06-05-aimip-like` branch, the
`scripts/blowup_investigation` tooling and its accumulated outputs, recent
feature implementations, and research notes in the repo. (Round 3:
saturation-normalized humidity split into its own avenue with a
representation analysis; the Jacobian metric reframed as an inline evaluation
probe rather than an aggregator; probes redesigned to run from saved rollout
snapshots, eliminating the need for any new multi-year rollout. Code-reading
items — Avenue 1 implementation, Avenue 3b/4 attachment points — deferred to
the next round.)

## Current state of the research

**The setup.** The working model is a 4-degree, 8-layer, daily-timestep
`NoiseConditionedSFNO` (512 embed dim, 8 layers, ensemble CRPS + energy score
loss with `n_ensemble: 2`), trained on ERA5 1940–2025. The long game is mixing
ERA5 with climate-model data so observationally-calibrated behavior extends
into future forcing scenarios — which is why so much of the recent work is
about making the network's view of the state *climate-invariant*: if the
inputs and outputs the NN sees look statistically the same in a +4K world as
in the training record, statistical prediction can generalize physically.

**The experiment branch** is a ladder of 11 waves of runs, each adding a
feature aimed at that goal or at long-rollout stability:

- **Shared global-mean removal** (surface-temperature-referenced, applied to
  all temperature fields, with the removed mean appended as an input channel)
  — the proven win, dramatically improving interannual variability and
  out-of-sample behavior.
- **Qsat-scaled shared global-mean removal** (Waves 10–11, just merged):
  humidity-impacted fields (`specific_total_water_0-7`, `LHTFLsfc`,
  `PRATEsfc`, `Q2m`, advective tendency) are multiplied by the ratio of Bolton
  saturation vapor pressure at the climatological vs. sample reference
  temperature. This is essentially baking Clausius–Clapeyron into the
  normalization so moisture fields stay in-sample as the climate warms.
  Wave 11 pairs it with **CO2 removed from inputs** — testing whether the
  thermodynamic rescaling can carry the forcing response instead of an
  explicit CO2 channel.
- **Residual prediction** with separate residual normalization, plus
  **anomaly-only residual skips** (winds, or winds+temperature) for
  per-variable drift control, and a **tendency regularization** loss
  (weight 0.05) penalizing global-mean drift. The residual configuration is
  the strongest performer but also the one with the long-rollout instability
  — the non-residual runs do not share it.
- **`clip_latent_global_means`**: tracks the per-channel spatial-mean envelope
  of the post-encoder latent during training and, at eval, shifts the latent
  back into that envelope — a learned-statistics guardrail against
  latent-space drift.
- **Label conditioning** (`era5` vs. `c96-shield` dataset labels, Waves 4–7)
  — the mechanism for the eventual multi-source training, currently being
  ablated against era5-only controls, including whether labels still need
  residual prediction once LR-tuning and multistep rollout are in the recipe.
- **Finite-difference CRPS** (Wave 8, crps 0.7 / fd-crps 0.2 / energy 0.1),
  `filter_num_groups=8` probes, embed-dim ablations (256/384/512), and seed
  replicates.

**The blowup investigation** has already produced a fairly sharp mechanistic
picture for the residual checkpoint
(`train-4deg-daily-v1-era5-only-residual-rs0`):

- The failure is a compounding warming/moistening drift, not a sudden NaN. In
  the offline 8-year reproduction from the best-inference checkpoint,
  `specific_total_water_0` is the first variable to leave the training
  distribution (~step 2100, late 1984), followed by `eastward_wind_0` and
  `air_temperature_0`, then everything cascades within ~2 months.
- The Jacobian timeseries is the key quantitative finding: the global-mean
  scalar Jacobian **d(gm q0)/d(gm q0) sits almost exactly at 1.0** through the
  stable portion of the rollout. The global mean of top-level moisture is a
  *neutral mode* — the model has no restoring force on it, so it random-walks
  under the stochastic stepping until it exits the training distribution, and
  only then does the dynamics go unstable (the diagonals start swinging
  wildly, u0 from 0.54 to 1.8, in the window where blowup begins).
- The interventions confirm causality: globally clamping or relaxing
  (τ = 100–200 days) the global mean of q0 each step **prevents the blowup
  through the full 8 years** — every other variable stays in distribution with
  no other intervention. Conversely, clamping the global mean of u0
  destabilizes the model within ~80 days, so this is genuinely about the
  moisture mode, and crude state surgery on the wrong variable is itself
  out-of-distribution.
- A caveat discovered while reviewing the Jacobian tooling: the model draws
  fresh noise on every forward call (`fme.core.rand.randn`, no seed control in
  the scripts), so the base and perturbed passes of the finite difference see
  different noise. The diagonal entries look clean (the noise evidently has
  little projection onto output global means), but the huge raw off-diagonals
  (e.g. d(gm T0)/d(gm q0) ~ 1e7 with eps = 1e-8) are exactly what
  noise-induced global-mean differences of ~0.1 K divided by a tiny eps would
  produce, so the cross terms are likely noise-dominated artifacts. Future
  probes should fix the RNG state per pass (or average antithetic pairs) and
  report the matrix in per-σ normalized units.
- Supporting threads: checkpoint/EMA comparisons (best vs. final,
  EMA-swapped) and 3 baseline replicates showing blowup time is stochastic;
  `scripts/investigate_bad_timesteps/` chasing two specific ERA5 timesteps
  that produce huge residual targets during training; and
  `sfno_advection_mechanism.md`, a theory note proving real-isotropic spectral
  filters + pointwise nonlinearities *cannot* represent directional advection
  (reflection-symmetry argument) and that the complex spectral weights are
  what break the symmetry — with proposed ablation experiments.

## Avenue 1: training-time restoring force via perturbation augmentation

The diagnosis says the problem is a neutral mode, and the fix that works
(inference-time relaxation of gm q0) is a hack. The principled fix is to make
displaced states *in-sample at training time* so the model learns its own
restoring force. Below is a concrete procedure.

### Desirable properties

The first three were given; the rest follow from the diagnosis:

1. **Physically justifiable** — perturbed states should be physically
   plausible, and the supervised response should match a real physical
   process.
2. **General** — applicable to any near-neutral scalar mode, not just
   moisture.
3. **Efficient** — negligible training-time overhead.
4. **Quantitatively sized** — the restoring timescale must beat the
   stochastic stepping's noise-injection rate. If the per-step noise injects
   variance σ_step² into a mode and the learned relaxation has timescale τ,
   the mode behaves as an Ornstein–Uhlenbeck process with stationary std
   σ_step·sqrt(τ/2Δt). Choosing τ so this stays inside the training envelope
   turns "stable enough" into an arithmetic condition rather than a hope.
5. **Non-interfering** — most training samples should be unperturbed or
   barely perturbed, so deterministic skill and variability statistics are
   untouched; the perturbation directions (a handful of global scalars) are
   nearly orthogonal to where weather information lives.
6. **Verifiable** — the existing Jacobian diagnostic is a direct acceptance
   test: after retraining, d(gm q)/d(gm q) should drop from ~1.0 to
   ~(1 − Δt/τ), with no 46-year rollout needed.
7. **Composable** — must interact correctly with the existing input
   transforms (shared global-mean removal, qsat scaling), which deliberately
   make *some* global-mean directions neutral. The augmentation should target
   the directions that remain free after those transforms.

### Procedure: global-mean perturbation augmentation

Per training sample (or per batch), in physical units, *before* the
global-mean-removal/qsat transforms:

1. **Configure a list of perturbation specs**, each naming a group of fields,
   a perturbation type, an amplitude σ_δ, and a physical relaxation timescale
   τ. Initial set:
   - *Moisture* (`specific_total_water_0-7`, `Q2m`): **multiplicative**,
     q → (1+δ)q. Multiplicative perturbations preserve positivity and
     roughly preserve the relative-humidity structure, so the perturbed state
     is a plausible "anomalously moist/dry atmosphere." Physical relaxation:
     condensation/precipitation removes excess column water on the vapor
     residence timescale, τ ≈ 10 days (plausibly shorter for the top level —
     probe 1 below measures what the model itself does). Could be applied
     per-level or column-coherently; per-level matches the observed failure
     (q0 drifts alone).
   - *Temperature* (the shared-global-mean-removal field set): **additive**
     uniform offset. Physical relaxation: radiative damping of a uniform
     tropospheric temperature anomaly, τ ≈ 30–60 days. Note the interaction
     with shared global-mean removal: the offset is removed again by the
     forward transform and re-added on output, so the network sees it only
     through the appended mean channel — the augmentation here trains the
     *response to that channel* out to larger displacements, rather than a
     new pathway.
   - *Winds* (`eastward_wind_*`, `northward_wind_*`, 10 m winds):
     **additive** uniform offset (a barotropic superrotation anomaly).
     Physical relaxation: surface friction, τ ≈ 5–10 days. The violent
     response to the u0 clamp intervention suggests wind global means may
     also need this.
2. **Draw δ ~ N(0, σ_δ²)**, optionally zero with probability 1−p. Size σ_δ
   from the rollout data: large enough to cover the excursion range where the
   blowup begins (roughly the |gm z| ≈ 1–2 envelope from the diagnostics),
   i.e. several times the natural variability of the mode.
3. **Apply δ to the input state.**
4. **Apply the decayed perturbation δ·exp(−Δt/τ) to the target** (same fields,
   same type). For multistep rollout training, decay geometrically across the
   target window — the model then gets supervision on the entire relaxation
   trajectory, which is even better.
5. Loss, correctors, and transforms otherwise unchanged. Optional budget
   refinement for moisture: bump target `PRATEsfc` by the implied column-water
   removal rate so the moisture-budget corrector sees a consistent target;
   worth doing if the corrector fights the relaxation, skippable otherwise.

Cost: a handful of scalar draws and broadcast multiply/adds per batch — no
extra model evaluations, no measurable training-time increase. The natural
home is a config-driven transform in the data/stepper path next to
`global_mean_removal`, so it composes explicitly with the existing transforms.

The main risk is teaching the model an artificial damping that suppresses
*real* low-frequency variability of the perturbed modes. Mitigations: keep τ
at the physical value rather than something aggressive (the physical
relaxation is what the real system does to such anomalies anyway), keep σ_δ
moderate, and watch the interannual-variability metrics that the shared
global-mean removal previously improved.

USER: This is a sound plan. The next step here is to look at the code and propose how you would implement it, ideally isolating the responsibility to as few levels of the codebase as possible. It seems like this might live in the SingleModuleStep, after adding some set_train and set_eval methods as we have in branch feature/global-mean-relaxation. Part of this is we also need to propose how this should be configured - what should the configuration dataclass look like, and where should it live as an attribute on another configuration class?

## Avenue 2 (revised): qsat-scaling residual run as a data point, not a fix

Correction from review: the instability is specific to the **residual**
configuration, and no qsat-scaling **residual** run exists yet (Waves 10–11
are non-residual). One is being launched. Expectation, based on perturbation
tests so far, is that qsat scaling has small effects and is a normalization
device, not a stabilization mechanism — so this run is a control/data point
(does changing how the network perceives gm moisture move the q0 Jacobian at
all?) rather than a candidate fix. When the checkpoint lands, running it
through `diagnose_blowup.py` and the (seed-controlled) Jacobian timeseries is
cheap and tells us whether normalization-space changes touch the neutral mode
at all — useful either way for the design of Avenue 1.

USER: This is on hold until the run is done, leave this user comment in place until then.

## Avenue 3 (revised): physical constraints that actually bind on q0

Original idea — leaning on the moisture-budget corrector — doesn't work: q0
is tiny, so the column budget exerts essentially no leverage on it, and an
upper-level transport budget has no target data to support it. Two
constraints that *do* bind on upper-level moisture and need no new target
data:

- **Saturation bound as a corrector.** Specific humidity physically cannot
  much exceed saturation: q_i ≤ α·qsat(T_i, p_i), with α slightly above 1 to
  allow condensate within total water. Together with the existing
  `force_positive`, this two-side-bounds moisture everywhere. It needs only
  the predicted temperature and the level pressures (from `PRESsfc` and the
  vertical coordinate), both available; the Bolton helper already exists.
  This is a hard backstop on the runaway-moistening direction — though
  whether it bites *early* in the drift (vs. only after temperature has also
  drifted, raising qsat with it) is exactly probe 5 below, and worth checking
  before building it.

USER: Does the vapor pressure actually get anywhere close to exceeding saturation, even during blowup? Check as a diagnostic on the existing rollouts before planning out this feature.

This is complementary to Avenue 1, not competing: the saturation bound caps
the worst-case excursion; the perturbation augmentation removes the random
walk that drives states toward it.

## Avenue 3b: saturation-normalized humidity representation

Distinct from the saturation bound (a constraint clamped onto predictions),
this is a *change of variables*: the model's humidity state becomes a
saturation-relative quantity, the local pointwise generalization of the
global qsat scaling. A neutral random walk in the transformed variable can no
longer carry q itself out of physical range, because the physical range is
built into the representation.

### Representation: q/qsat vs. q − qsat

**q/qsat (RH-like) is the right choice; the deficit fails the purpose on
several counts.**

- *Conditioning and climate-invariance.* q/qsat is dimensionless and O(1) at
  every level, latitude, and climate — at cold upper levels both q and qsat
  are tiny and their ratio stays well-conditioned. The deficit q − qsat
  inherits qsat's own dynamic range: its natural magnitude spans orders of
  magnitude from the warm boundary layer (~1e-2 kg/kg) to the cold upper
  levels (~1e-6), which is exactly the problem normalization is supposed to
  remove. A per-level std for the deficit would be dominated by warm regions,
  leaving cold-region variations in the noise floor — the same failure mode
  full-field q normalization already has at upper levels.
- *Boundedness.* Reconstruction from RH is multiplicative, q = x·qsat, so
  positivity is automatic given x ≥ 0 (the existing `force_positive`
  machinery applied to the transformed variable). Reconstruction from the
  deficit is additive, q = x + qsat, which does not preserve positivity and
  leaves the lower side unbounded — a random walk in deficit space can still
  walk q below zero or arbitrarily far down.
- The deficit's one theoretical advantage — additivity, so advection acts
  linearly on it — is mostly illusory because qsat varies in space, which
  makes the deficit non-conserved under transport anyway (same issue as RH,
  analyzed next).
- One free choice within RH: do **not** hard-cap at 1. Total water includes
  condensate, so values modestly above 1 are physical; cap behavior, if any,
  belongs to the Avenue 3 bound (which becomes nearly free in this
  representation — a clamp at α on the normalized variable).

### Does it make advection harder?

Yes, somewhat, and it's worth being explicit about the mechanism so the
testing can target it. Outside condensation, q is approximately a conserved
tracer: Dq/Dt ≈ 0, and the SFNO demonstrably learns this operation. RH is
*not* conserved under transport: D(RH)/Dt = −RH·D(ln qsat)/Dt, so whenever
air crosses a temperature gradient, plain moisture transport shows up in RH
space as advection *plus* a source term ∝ RH·(u·∇ ln qsat). The network must
learn an additional coupled wind–temperature term to represent what used to
be linear. (That source term is real physics — it is why advection toward
cold air makes clouds — but it is extra work for the model.)

Implications:

- The risk is concentrated where temperature advection is strong and moisture
  advection dominates the budget: winter storm tracks, mid/lower troposphere.
  It is smallest at the top level — where qsat gradients are precisely what
  make q0's in-sample range so narrow, and where the neutral-mode problem
  lives. So the per-level cherry-picking isn't just nice to have; the
  sensible first experiment is RH-space prediction for q0 (perhaps q0–q1)
  only, keeping the advection-dominated levels in q space.
- There is a compensating effect to watch for: condensation/precipitation
  thresholds at RH ≈ 1, so precipitation may be *easier* to predict from an
  RH-space state.
- How it would bear out in testing: day-1/day-5 RMSE for q, `PRATEsfc`, and
  `tendency_of_total_water_path_due_to_advection`, with a regional breakdown
  in strongly baroclinic regions, compared level-by-level against the q-space
  control.

USER: Let's make sure we can configure this to be per-level, so we could isolate it to just the q0 level. Maybe by allowing a wildcard in the name? But also make it straighforward to configure uniformly across q-levels (e.g. not require a list of items, one per level).

### Input space vs. prediction space

These are separable decisions with different jobs. *Prediction* space is
where boundedness matters — it is the recurrent update that random-walks.
*Input* space is feature engineering — giving the network a well-conditioned
view of how close to saturation the air is. They compose freely, and the
input side is cheap in a way the output side is not: input channels can be
**redundant** (provide both q and q/qsat and let the network use either),
while the prediction must be exactly one representation per variable to avoid
reconstruction ambiguity. That makes "append RH input channels, keep q
predictions" the cheapest first experiment of the whole avenue — no change to
prediction semantics, normalization stats, or checkpoint compatibility logic
— and it isolates whether saturation *information* alone helps before
committing to the variable change.

In all cases qsat should be computed from the input-step temperature at the
same level (avoids circularity with predicted temperature) and the level
pressure from `PRESsfc` plus the vertical coordinate; the Bolton helper
already exists.

### Configuration shape

The combinatorial space is (which fields) × (rh | deficit) × (input |
prediction | both), with the input option further splitting into replace vs.
append. This stays flat as a list of entries with orthogonal knobs rather
than a matrix of variants:

```yaml
saturation_normalization:
  - names: [specific_total_water_0, specific_total_water_1]
    representation: rh        # rh | deficit
    prediction: true          # transform the prediction for these fields
    input: append             # none | replace | append
```

USER: Based on your reasoning earlier, let's remove the deficit option for now, and focus on RH. That means the representation key can be dropped.

Per-level cherry-picking is just the explicit `names` list (consistent with
how `qsat_scaled_names` and `force_positive_names` already work — no implicit
"all levels"). Validation in `__post_init__`: a field may appear in at most
one entry; overlap with `qsat_scaled_names` is an error (the global and local
scalings would double-count); each entry must do something (`prediction:
true` or `input != none`). With residual prediction, the residual is taken in
the transformed variable — which is the point: the random walk happens in the
bounded representation.

USER: You can proceed to planning the code implementation for this - at what level does it live, and what does the configuration look like?

(Code-reading round: settle the attachment point — where this transform
lives relative to normalization, `global_mean_removal`, and the corrector —
and reconcile the config sketch above with the existing dataclass structure.)

## Avenue 4: neutral-mode spectrum as a metric — design answers

**Validation or inference metric?** Validation, at least at first. The key
empirical fact licensing this: the measured d(gm q0)/d(gm q0) ≈ 1.0 from step
0 of the rollout — i.e. the neutral mode is visible *at in-sample states*,
not only at self-generated drifted states. So a one-step finite-difference
Jacobian of global means evaluated on validation batches (data states)
already detects it, runs early and often, and is far cheaper than anything
attached to the inline 46-year inference. An inference-aggregator version
probing states along the rollout is a sensible follow-up (it would catch
restoring forces that exist near data but vanish off-manifold), but it is not
where to start.

**Ready to code, or more probing first?** The structure is clear — batch the
base state plus N perturbed states into the batch dimension, one extra
forward pass set per validation epoch, report the diagonal (and optionally
top eigenvalue moduli) of the global-mean Jacobian in per-σ units. Three
design inputs need a short round of checkpoint probing first:

1. **Noise control.** As noted above, fresh noise per forward call pollutes
   finite differences. The metric must fix the RNG state across base and
   perturbed passes (or average antithetic noise pairs). This needs a small
   amount of validation on the existing checkpoint to confirm the diagonals
   are unchanged and the off-diagonals become interpretable.
2. **Epsilon per variable.** The scripts hand-picked eps for three variables;
   the metric needs a rule (e.g. a fixed fraction of each variable's residual
   std) validated for FD robustness across all ~50 prognostics.
3. **Mode set.** Per-variable global means vs. grouped modes (column-coherent
   moisture, barotropic wind). Also: with shared global-mean removal active,
   the temperature global mean is neutral *by design* — the metric must
   exclude designed-neutral directions or report them separately, otherwise
   it permanently flags a feature as a bug.

That probing is shared work with the probes below, so the metric falls out of
them nearly for free.

### Code-design iteration: where the complexities naturally live

The concern is right: this is not an aggregator, and forcing it to be one is
what would make it feel bolted-on. Two reframings dissolve the two
complexities.

**1. It's not an aggregator; it's an evaluation task — and that level already
exists.** The aggregator contract is passive: consume (target, prediction)
pairs from forward passes the trainer already runs. The Jacobian probe
violates that contract because it must *drive* forward passes of its own.
But the codebase already has a level whose responsibility is exactly "owns
its own data, drives its own forwards on an epoch cadence, reports metrics":
the inline inference blocks in the training config, each with its own loader,
`epochs: {start, step}` schedule, and aggregator. The probe is a natural
sibling of those — a small evaluation task with a tiny loader (a handful of
validation times), an epoch cadence, and a few hundred single-step calls
through the stepper's existing public predict API. Nothing in the aggregator
system changes; the trainer's only change is invoking configured probe tasks
at the same place it invokes inline inference. Evidence that no new
model-side capability is needed: `compute_jacobian_timeseries.py` already
does paired base/perturbed predictions entirely through public interfaces
(`get_forcing_data`, `predict_paired`, `PrognosticState`) — the probe is that
script, productionized at the inline-evaluation level.

This reframing also retires the "validation vs. inference metric"
distinction from above: as its own task, the probe samples whatever states
its loader provides (data states now, rollout snapshots later) without
belonging to either system.

USER: That sounds right to me. Make sure the way it lives alongside validation is clear, and that it's easy to not include the task type at all.

**2. Noise control is a generic RNG utility, not probe plumbing.** The
requirement is common random numbers across two forward calls. RNG policy
already has a single owner in the codebase: `fme.core.rand` (the `set_seed`
and `randn` wrappers exist precisely so RNG behavior is centralized). A
`fork_rng(seed)` context manager there — wrapping `torch.random.fork_rng`
over CPU and CUDA so the global RNG state is saved, seeded, and restored — is
a small, general-purpose tool: equally useful for regression tests,
reproducible inference ensembles, paired model comparisons, and antithetic
variance reduction. The probe simply enters the context once per forward
pass. Nothing Jacobian-specific leaks outside the probe module.

The rejected alternative, for the record: threading an optional `generator=`
parameter through predict/step/module-forward signatures would place noise
control "at the source" (the stochastic module draws the noise), but at the
cost of touching every layer of the call stack for one consumer. The context
manager gets identical semantics through the existing centralized-RNG
responsibility with zero API churn. If a second consumer someday needs
per-sample generators (e.g. noise-shared batched ensembles), revisit then.

Remaining before an implementation plan (code-reading round): confirm the
interface inline-inference tasks implement and where the trainer invokes
them; confirm a `PrognosticState` can be built from a validation batch
without loader gymnastics; then the config dataclass. The empirical
questions (eps rule, mode set) fold into probe 2 below.

## Avenue 5: the architecture thread

The advection-mechanism note suggests the SFNO represents directional
transport only via complex spectral weights, and the model has
`filter_preserves_global_mean` and the latent-clipping envelope available.
Probing whether the runaway feeds through the spectral l=0 channel (e.g.
enabling `filter_preserves_global_mean`, or extending latent clipping from
eval-only into rollout training) connects the stability work to the mechanism
work, and would tell us whether the neutral mode lives in the spectral
filters or in the pointwise/encoder paths. Lower priority than Avenues 1 and
the probes; kept as a thread.

## Proposed checkpoint probes (cheap, no training)

Things to measure on the existing residual checkpoint with the existing
harness, each directly informing the design of the training-time
intervention.

**None of these need the 8-year horizon — or any new long rollout at all.**
The 8-year runs already exist and saved the full prognostic state at every
step, and `compute_jacobians.py` already demonstrates re-initializing a
`PrognosticState` directly from those saved predictions. So every probe runs
from saved snapshots, and "initializing near the instability" is not just a
speedup but part of the experimental design: the restoring force should be
measured at on-manifold states (early rollout), mildly drifted states
(mid-rollout), and near-onset states (~step 2000), because its
stage-dependence is one of the questions. The only thing that genuinely
requires a multi-year rollout is end-to-end acceptance of a *retrained*
model (natural onset was ~5.8 years on this checkpoint, so a confirmation
run needs the full ~8 years) — which is exactly why the one-step Jacobian
acceptance test matters: it front-loads that verdict.

1. **Finite-amplitude response curves (sizes σ_δ and τ).** The Jacobian
   measures the infinitesimal response; the augmentation needs the
   finite-amplitude landscape. From 3 saved snapshots (early / mid /
   near-onset), displace gm q0 (and column q, T, barotropic u) by a ladder of
   amplitudes in both signs and run ~90 days. The whole amplitude ladder
   stacks into the sample/ensemble batch dimension, so this is ~3 batched
   90-step rollouts rather than ~50 sequential runs. Per-sample noise
   differences are fine here — the measurement is the mean relaxation
   trajectory over many steps, and a few replicate seeds per amplitude give
   error bars. 90 days cleanly separates τ ≈ 10 d relaxation from neutral
   drift; only if a curve looks neutral-but-bounded would a ~1-year extension
   of that single case be worth running. Reads off: whether the model is
   neutral everywhere or restoring-then-unstable beyond a threshold, the
   basin boundary (= σ_δ), τ, and their stage-dependence. **Minutes on GPU.**
2. **Full neutral-mode spectrum.** Extend the 3×3 Jacobian to all prognostic
   global means (~50×50, seed-controlled noise, per-σ units) at the same 3
   snapshots. Single-step finite differences — no rollout at all: ~51 seeded
   forward passes × 3 states × ~3 noise seeds ≈ 500 forwards. Finds *every*
   near-unit mode — if barotropic wind or another mode is also neutral, the
   augmentation covers it from the start rather than being discovered by the
   next blowup. Doubles as the prototype of the Avenue 4 probe task. Only
   dependency: the `fork_rng` utility. **Minutes on GPU.**
3. **Noise-injection rate.** Variance injection is a short-time statistic:
   the across-seed variance of each global-mean mode grows roughly linearly
   over the first weeks, so 16–32 seeds × 30–60 days from one saved snapshot
   — batched in the ensemble dimension, i.e. one short rollout — pins down
   σ_step per mode. The long horizon contributes nothing here. Combined with
   τ from probe 1, the OU stationary-variance formula gives the quantitative
   stability condition the trained restoring force must meet — the acceptance
   threshold for the retrained model's Jacobian, not just "less than 1."
   **Minutes on GPU.**
4. **Spatial structure of the neutral mode.** Perturb gm q0 via a uniform
   offset vs. equal-mean localized patterns (tropics-only, extratropics-only)
   in short runs from saved snapshots; also inspect the spatial pattern of
   the q0 drift in the existing outputs. If the undamped direction is not
   actually uniform, the augmentation should perturb with the right pattern,
   and a global-mean-only intervention may underconstrain.
5. **Saturation check on existing output.** From the saved drifting rollouts,
   compute upper-level RH against qsat(T, p) through the drift phase. If q0
   exceeds physical saturation early, the saturation-bound corrector (Avenue
   3) would have bitten early and is worth building; if the drift stays
   sub-saturated because T drifts in tandem, the bound alone won't prevent
   onset. Pure post-processing of existing files — no model runs.
6. **Causal direction of the q0–T coupling.** With seed-controlled noise, the
   off-diagonal Jacobians (d gm T/d gm q0 vs. d gm q0/d gm T, in per-σ units)
   become trustworthy and show which way the warming–moistening feedback loop
   runs, and whether breaking one leg (the augmentation on q alone) suffices
   or temperature needs its own restoring force trained through the
   appended-mean channel. Same harness and runs as probe 2.

## Suggested sequencing

Everything below is sub-day on a single GPU; nothing waits on a long rollout.

1. **`fork_rng` utility** in `fme.core.rand` — small, unblocks probes 2/6 and
   the Avenue 4 probe task.
2. **Probe 5** (free; gates the Avenue 3 corrector) and **probes 2+6
   together** (same harness; full spectrum + trustworthy cross terms; doubles
   as the Avenue 4 prototype).
3. **Probe 3** (batched noise-injection ensemble) → σ_step per mode.
4. **Probe 1** (batched amplitude ladders at 3 rollout stages) → σ_δ, τ,
   basin boundary, stage-dependence.
5. **Probe 4** if probe 2's results suggest the undamped direction has
   spatial structure worth resolving.

In parallel: the qsat-scaling residual run (Avenue 2) as a control. Then
implement the Avenue 1 perturbation augmentation with the probe-derived
numbers (σ_δ, τ, mode set, acceptance threshold) and the Avenue 4 probe as
its acceptance test, with the saturation bound (Avenue 3) as a cheap backstop
if probe 5 says it binds, and the Avenue 3b input-channel variant as the
cheapest representation experiment alongside.
