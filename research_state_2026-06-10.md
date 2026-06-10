# State of the research and avenues for exploration (2026-06-10)

A synthesis of the `experiment/2026-06-05-aimip-like` branch, the
`scripts/blowup_investigation` tooling and its accumulated outputs, recent
feature implementations, and research notes in the repo. (Round 2: expanded
with a concrete training-time procedure, corrected scope on the qsat-scaling
runs, alternative physical constraints on moisture, metric design answers, and
a proposed set of cheap checkpoint probes.)

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

- **Saturation-normalized prediction space.** The thorough version: predict
  upper-level moisture as q/qsat(T_input, p) — a *local, pointwise*
  generalization of the global qsat scaling. The model's moisture variable
  becomes RH-like, bounded ~[0, 1] and climate-invariant by construction, and
  a neutral random walk in the normalized variable can no longer take q
  itself out of physical range. Using input-step temperature for the
  normalization avoids circularity with the predicted temperature. This is a
  bigger change (touches normalization and checkpoint compatibility) but is
  the same philosophy that already paid off twice (shared mean removal, qsat
  scaling), taken to its limit.

Note these are complementary to Avenue 1, not competing: the saturation bound
caps the worst-case excursion; the perturbation augmentation removes the
random walk that drives states toward it.

USER: Saturation-normalized prediction space is a reasonable idea, split it into its own category since it's distinct from the saturation bound. It would be nice to be able to configure this to act only on q0, or to act on all levels of humidity, or to cherry-pick specific levels. Also, consider whether to use relative humidity q/qsat or humidity deficit q - qsat, and think about how this kind of representation of humidity might or might not make it harder to advect moisture (which would get bourne out by testing).

A separate version of this that can be used together or on its own is to saturation-normalize the input space, either in addition to or in replacement of the non-saturation-normalized input.

There's a considerable multiplication of configurations between input vs output, rh vs deficit, and per-level vs column-coherent, so the configuration dataclass should be designed to cleanly handle that combinatorial space without exploding in complexity. You can proceed to planning the code implementation for this - at what level does it live, and what does the configuration look like?

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

USER: OK, this seems reasonable so long as we're able to come up with a way to implement it without adding too much complexity to the codebase. I'm a little concerned that this requires changing the way the model does forward passes during validation - it's not simply another aggregator plugged into the existing aggregator system. The noise control requirement is also more complexity. Are you able to come up with a way to implement this where each of these complexities feels "natural" for the responsibility it lives on, instead of feeling specific to this jacobian aggregator that lives at a different level of complexity? Iterate on the code-design of this, but it's not ready for a fully fleshed out configuration and implementation plan until we've worked through these design details.

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
intervention:

1. **Finite-amplitude response curves (sizes σ_δ and τ).** The Jacobian
   measures the infinitesimal response; the augmentation needs the
   finite-amplitude landscape. From a mid-rollout state, displace gm q0 (and
   column q, T, barotropic u) by a ladder of amplitudes in both signs, run
   30–90 days each, and measure relaxation/amplification vs. amplitude. This
   answers whether the model is neutral everywhere or weakly restoring for
   small δ and unstable beyond a threshold, and reads off the basin boundary
   — which is exactly σ_δ for the augmentation and the validation that a
   relaxation-shaped target is the right supervision. (~50 short runs; hours
   on GPU.)
2. **Full neutral-mode spectrum.** Extend the 3×3 Jacobian to all prognostic
   global means (~50×50, with seed-controlled noise and per-σ units) at a few
   states along the rollout. Finds *every* near-unit mode — if barotropic
   wind or another mode is also neutral, the augmentation should cover it
   from the start rather than being discovered by the next blowup. Doubles as
   the prototyping work for the Avenue 4 metric.
3. **Noise-injection rate.** Run an ensemble of short rollouts from one IC
   with different seeds and measure the per-step variance injected into each
   global-mean mode. Combined with τ from probe 1, the OU stationary-variance
   formula gives the quantitative stability condition the trained restoring
   force must meet — i.e. the acceptance threshold for the retrained model's
   Jacobian, not just "less than 1."
4. **Spatial structure of the neutral mode.** Perturb gm q0 via a uniform
   offset vs. equal-mean localized patterns (tropics-only, extratropics-only)
   and compare responses; also inspect the spatial pattern of the q0 drift in
   the existing rollouts. If the undamped direction is not actually uniform,
   the augmentation should perturb with the right pattern, and a
   global-mean-only intervention may underconstrain.
5. **Saturation check on existing output.** From the saved drifting rollouts,
   compute upper-level RH against qsat(T, p) through the drift phase. If q0
   exceeds physical saturation early, the saturation-bound corrector (Avenue
   3) would have bitten early and is worth building; if the drift stays
   sub-saturated because T drifts in tandem, the bound alone won't prevent
   onset. Free — no model runs needed.
6. **Causal direction of the q0–T coupling.** With seed-controlled noise, the
   off-diagonal Jacobians (d gm T/d gm q0 vs. d gm q0/d gm T, in per-σ units)
   become trustworthy and show which way the warming–moistening feedback loop
   runs, and whether breaking one leg (the augmentation on q alone) suffices
   or temperature needs its own restoring force trained through the
   appended-mean channel.

## Suggested sequencing

Probes 5 and 6 are nearly free; probes 1–3 are a day or two of GPU time and
directly parameterize the Avenue 1 augmentation (σ_δ, τ, mode set, acceptance
threshold). The qsat-scaling residual run (Avenue 2) proceeds in parallel as
a control. Then implement the perturbation augmentation with the
probe-derived numbers and the Avenue 4 validation metric as its acceptance
test, with the saturation bound (Avenue 3) as a cheap backstop if probe 5
says it binds.

USER: These all sound worth doing. Is there a way to modify probes 1-3 so they can be completed faster? For example, is it reasonable to run them on a shorter series of data, either initializing near an instability or just running for fewer steps? Which probes if any require the full 8 years (which I chose arbitrarily)? Further develop the plans and ordering of these experiments.
