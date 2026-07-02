# CMIP6 Training: Issues and Plan

This document discusses features of the CMIP6 training data and goals for its use as a training dataset.

## CMIP6 Training Data

### Experiments and scenarios

CMIP6 models run standardized experiment protocols. The following are available and relevant to our goals. Model counts below are from the Pangeo GCS mirror's day table; monthly (Amon) coverage is typically 30–50% higher.

**Transient scenarios (the primary training data):**

- **historical** (1850–2014, 54 models daily, 65 monthly): forced by observed greenhouse gas concentrations, aerosol emissions, land use, and solar/volcanic activity. The longest transient run and the natural baseline for all models.
- **ssp126** (2015–2100, 37 daily, 45 monthly): strong mitigation, ~2.6 W/m² forcing by 2100. The low end of plausible warming.
- **ssp245** (2015–2100, 37 daily, 46 monthly): moderate mitigation, ~4.5 W/m² forcing by 2100. Middle of the road.
- **ssp370** (2015–2100, 36 daily, 42 monthly): weak mitigation, ~7.0 W/m² forcing by 2100. Between ssp245 and ssp585.
- **ssp585** (2015–2100, 39 daily, 47 monthly): high emissions, ~8.5 W/m² forcing by 2100. Upper bound of plausible warming.

The four SSP scenarios above have broad model coverage (36–39 models daily) and span the full range of forcing trajectories. Three additional SSPs exist but with limited coverage: ssp119 (11 daily), ssp434 (6), ssp460 (5), ssp534-over (8). These are less useful for training but could serve as held-out evaluation scenarios for testing generalization to unseen forcing trajectories.

**Idealized experiments:**

- **piControl** (pre-industrial control, 39 daily, 65 monthly): fixed pre-industrial forcing run for centuries. No forced trend — pure internal variability. Useful for characterizing each model's unforced drift and variability baseline, and potentially as a training signal (the emulator should reproduce stable climate under constant forcing). Long time series partially offsets the single-member-per-model limitation.
- **abrupt-4xCO2** (40 daily, 59 monthly): CO2 instantaneously quadrupled from pre-industrial levels. Produces a large, fast climate response useful for measuring climate sensitivity. The abrupt forcing is very different from the smooth transient scenarios — potentially valuable for testing whether the emulator can handle step-function forcing changes, but also risks confusing the learned dynamics if mixed into training naively.
- **1pctCO2** (39 daily, 59 monthly): CO2 increases at 1% per year from pre-industrial. A smooth idealized ramp, intermediate between piControl and the SSPs. Clean forcing signal without the complexity of aerosols/land-use, making it useful for isolating the CO2 response.

Each model typically runs multiple ensemble members per scenario — independent realizations from perturbed initial conditions that sample internal variability. Member counts vary widely: most models publish 1–5 members per scenario, but a few (CanESM5, MIROC6, EC-Earth3) publish 50+.

### Atmospheric data availability

Model counts below are from the Pangeo GCS mirror; ESGF generally has equal or greater coverage and is used to fill gaps. Data availability is not a constraint — our dataset count is limited by disk size and training time, not by what CMIP6 has published.

**Daily (day table).** ~37 models have the full daily core variable set needed for training (ua, va, hus, zg on 8 pressure levels; tas, huss, psl, pr at the surface), yielding ~146 eligible model/experiment/member combinations across historical, ssp245, and ssp585. Many more models publish a partial core set.

Key gaps at daily cadence: air temperature on pressure levels (ta) is published by only 3 models, and surface pressure (ps) by none. The pipeline works around these by deriving layer-mean temperatures from the hypsometric equation (using zg and hus) and using mean sea-level pressure (psl) as a substitute for ps.

Optional variables (TOA and surface radiation, surface turbulent fluxes, near-surface winds) are available for 35–44+ models and are included per-model when present.

**Monthly (Amon table).** Monthly atmospheric data is far more complete: ~79 models provide the full 3D state (ta, ua, va, hus, zg) — more than double the daily coverage. The key daily gaps disappear at monthly cadence: ta is available for ~79 models and ps for ~80. Surface variables (tas, psl, pr, etc.) are available for 72–84+ models. This broader coverage makes monthly data attractive for training, despite the non-uniform timestep challenge.

### Ocean data availability

**Daily (Oday table).** Daily ocean data is sparse. Only sea surface temperature (tos, ~43 models) has broad daily coverage. Other Oday variables (sos, tossq, sossq, omldamax) have very limited availability at daily cadence.

**Monthly (Omon table).** Monthly ocean data is substantially richer. Sea surface temperature (tos, ~68 models) and salinity (sos, ~64 models) have broad coverage, as do 3D fields: potential temperature (thetao, ~64 models), salinity (so, ~65 models), and velocity (uo/vo, ~60 models). Surface diagnostics useful for capturing ocean memory without 3D data volume include sea surface height (zos, ~60 models), net surface heat flux (hfds, ~56 models), and mixed layer depth (mlotst, ~53 models). Sea-ice concentration (siconc) is on the SImon table (~62 models).

Ocean data lives on curvilinear or tripolar model grids requiring regridding to the atmospheric target grid, which is more error-prone than atmosphere-to-atmosphere regridding.

### Forcings and static fields

Monthly surface temperature (ts, Amon, ~80 models) and sea-ice fraction (siconc, SImon, ~62 models) serve as lower boundary forcings — interpolated to daily for the daily pipeline, or used directly for monthly training. Static per-model fields include land fraction (sftlf, ~51 models) and orography (orog, ~47 models) from the fx table.

External forcings from input4MIPs (CO2 concentration, SO2/BC emissions, forest fraction) are shared across all models within a scenario and are planned but not yet in the pipeline.

## Goals

Basic goals (low risk):
- Sample physics-model uncertainty/spread using a single ML emulator, by successfully reproducing the climate response of each model.
- Emulate the model spread in response of both the atmosphere and ocean, based on forcing scenario and datetime or based or forcing variables themselves.
- Compression of the CMIP6 archive into a user-friendly form that enables climate research, should run on a single T4 GPU and be accessible to researchers without access to large compute resources.
- Individual model behavior is encapsulated by a low-dimensional model embedding space, which can be used to interpret similarity between models.
- Train an emulator of atmosphere and ocean dynamics on a unified 1-month timestep, which can rapidly produce climate projections.
- Train an emulator on a large ensemble of an existing model, and show that it captures the ensemble spread of that model within the regime it was trained on.

Wow goals (medium to high risk, high return):
- Emulator successfully captures the ensemble spread of individual models without being trained on such ensembles, suggesting it can be used to generate synthetic ensembles for models that have only a single run available.
- Low-dimensional model embedding space can be used to quickly fine-tune the emulator to capture a new model's behavior.
- (high risk) Having trained on historical and scenarios, the emulator can successfully predict the climate response of a model that was held out in a scenario, suggesting it can be used to generate synthetic scenarios for models that have only a few scenarios available.
- The emulator can produce robust piControl statistics for models that had their piControl runs held out of training, suggesting the model has learned the correct relationship between model physics and equilibrium climate variability.
- (high risk) Having trained for a given model on only one ensemble member for the historical scenario, the emulator can successfully predict the climate response that model in held-out SSP scenarios, suggesting it could be used to predict future climate from observations, provided the reanalysis data is sufficiently similar to the climate model data (whether by modifying the reanalysis data to resemble climate model data, or vice-versa).
- (high risk) The emulator can be fine-tuned on ERA5 data, producing accurate 5-day weather and 3-15 year climate predictions (time mean and higher-order metrics) in the future of its training data, suggesting it can be used to generate synthetic future climate projections that are consistent with historical observations.

Secondary research outputs (low risk, less key but worthwhile):
- Demonstrate that training with heterogeneous variable sets (i.e. adding models with some core variables masked for additional data) outperforms training on only the smaller set of models with complete coverage, due to the additional data available.
- Determine whether having been trained only on idealized scenarios for a certain model, the emulator can succeed on realistic scenarios (that it saw for other models), or vise-versa, as a way to inform the design of scenario sets for future emulator training.

## Model set-up

Use (source model, physics suite) as a model identifier label, which is bottlenecked into a learned lower-dimensional model embedding space.
Use forcing scenario as a separate label, without a bottleneck.
Use datetime to generate a time embedding, e.g. from sines and cosines over the simulation period, similar to Aurora.

## Discussion

### Mixed-layer ocean dynamics using a time and forcing scenario embedding

We can learn ocean mixed layer dynamics, which are a key driver of climate variability on decadal timescales, and allow the long-term ocean response to be learned from the forcing scenario label and the time embedding.
While the available ocean variables aren't sufficient for a physics-based estimation of future climate response, they may be sufficient for achieving the stated goals.
Notably, we have seen emulators do not produce physics-based estimations of climate response.
They are statistical tools.

This approach seems to preclude the emulator from being used on unseen forcing scenarios, but realistically even with a statistical emulator, an estimate of climate response to forcings should come from physics-based ocean modeling.
There is a high chance we simply do not have enough decades of independently forced ocean data to learn arbitrary ocean dynamics in a physical way from statistical learning.
For example, the forcing set used has significantly more degrees of freedom than the number of climate scenarios.
Given this, we should not expect to predict ocean forcing response to unseen scenarios with purely statistical approaches.
We can however reasonably represent the variability of the ocean response to the forcing scenarios that we have seen, and use this to capture the model spread in ocean response to these scenarios.
We may even be able to show we can learn a new forcing scenario with relatively few models, and generalize to additional models for that forcing scenario, which would be a useful result in its own right.
This is possible because we have many independent realizations of the ocean response to each forcing scenario.

Note we could investigate these dynamics separately and together from the atmospheric response, by training an emulator on the ocean variables alone, and then comparing the results to an emulator trained on both ocean and atmospheric variables.
However, I would suggest we maintain a unified timestep and focus on learning shorter-timescale mixed layer ocean dynamics, with long-term ocean dynamics learned mainly as a statistical response to the forcing scenario and time embedding, as this is more likely to be successful and is more consistent with the overall goals of the project.

### Daily or monthly timestep?

Monthly data is challenging because "monthly" is not a uniform length of time, breaking assumptions we make in our code.
To support it, we could remove those assumptions, or we could modify the datasets to use a unified 360-day calendar (either by restricting to such models, by "scaling" the time axis of the data to a 360-day calendar, or by mis-labelling the time axis and accepting rounding errors in some metrics).
On the other hand, monthly data is also much more widely available, with 3D data available for ocean models.
Training with a monthly step would be very consistent with our goals, as it removes much of the short-term variability that is not relevant to climate response, and allows us to focus on learning the long-term dynamics that are relevant to climate response.
It does run the risk that these dyanmics are difficult to learn.

While the available samples data drops by a factor of ~30, data availability is not our issue.
We have additional models we could add through processing the ESGF store, monthly data has more models available, and generally our dataset count right now is constrained by our disk size and training times, not data availability.

Using a 1-month timestep will also significantly improve the disparate scales of residuals for slow and fast ocean and atmosphere variables, which has been a major issue when training on a 6-hour timestep.

We should attempt a monthly-timestep model.
