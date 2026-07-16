# Ocean-sourced sea ice for coupled dataset creation

Lets `compute_coupled_sea_ice` take sea surface fraction and sea ice concentration from the
ocean store's 5-daily fields when no separate sea ice dataset is configured, applying the
coherence logic at 5-daily resolution and forward-filling the fractions onto the atmosphere's
6-hourly index (no window-averaging). The source is selected by presence priority — separate
sea ice dataset; else the ocean store when no `window_avg` is configured; else the existing
atmosphere source — rather than a config mode flag. A new
`use_atmosphere_sea_ice_fraction_fallback` switch (default True) lets production configs turn
the final atmosphere source into a loud error. Also passes `hfds_total_area` through from the input ocean store when it is already
present instead of re-deriving it. `create_coupled_datasets.py` and
`compute_coupled_atmosphere` are unchanged — the latter's existing ffill reindex of
ocean-cadence fractions already has the required semantics.

---

## `scripts/data_process/coupled_dataset_utils.py` (modified)

```python
class CoupledSeaIceConfig:
    # NEW — governs the last-priority source: taking the sea ice fraction from the
    # atmosphere when no sea ice dataset is configured and the ocean-sourced path is
    # not taken. True (default) keeps today's behavior; False makes reaching the
    # atmosphere source a ValueError, so a misconfigured production run (e.g. a
    # missing ocean sea ice variable, or a window_avg accidentally left configured)
    # fails loudly instead of silently training on the atmosphere's own field.
    use_atmosphere_sea_ice_fraction_fallback: bool = True


class OceanInputFieldsConfig:
    # NEW — full-cell sea ice area fraction in the ocean store (input to the
    # derived-concentration path: concentration = sea_ice_fraction / sea_surface_fraction)
    sea_ice_fraction_name: str = "sea_ice_fraction"
    # NEW — ocean-relative sea ice concentration computed natively by the upstream
    # pipeline; used directly when present (it is computed more accurately upstream
    # than the re-derived clipped ratio)
    ocean_sea_ice_fraction_name: str = "ocean_sea_ice_fraction"


def compute_coupled_sea_ice(
    atmos: xr.Dataset,
    config: CoupledSeaIceConfig,
    sea_ice: xr.Dataset | None = None,
    ocean: xr.Dataset | None = None,
    input_field_names: CoupledFieldNamesConfig | None = None,
    atmos_extras: ExtraFieldsConfig | None = None,
    sea_ice_extras: ExtraFieldsConfig | None = None,
) -> xr.Dataset:  # CHANGED — signature unchanged; sea-ice source selected by priority:
    # 1. sea_ice dataset when provided (legacy path, byte-for-byte unchanged);
    # 2. else the ocean dataset, when config.window_avg is None and the ocean carries
    #    sea ice fields (new 5-daily path below). Gating on window_avg keeps existing
    #    no-sea-ice-dataset configs whose ocean store contains ice fields (the E3SM
    #    coupled configs) on the legacy windowed path byte-for-byte;
    # 3. else the atmosphere's own sea ice fraction (legacy no-sea-ice path, including
    #    window_avg when configured) — allowed when
    #    config.use_atmosphere_sea_ice_fraction_fallback, else a ValueError.
    # Output stays on the atmosphere's 6-hourly index in all cases.
    ...


def _compute_fractions_from_ocean(
    atmos: xr.Dataset,
    ocean: xr.Dataset,
    input_field_names: CoupledFieldNamesConfig,
) -> xr.Dataset:  # NEW — the ocean-sourced 5-daily fraction path
    ...


class CoupledSeaSurfaceConfig:
    ...  # unchanged


def compute_coupled_ocean(...) -> xr.Dataset:  # CHANGED — hfds_total_area passthrough:
    # when hfds_total_area_name is already a data variable of the input ocean store
    # (the upstream pipeline now computes it natively), copy it (values + attrs)
    # instead of deriving hfds * sea_surface_fraction. Presence-checked in code rather
    # than relying on extra_fields config, so the derived and native values can never
    # silently coexist/diverge. Derivation is kept for legacy stores without it.
    ...
```

### Critical detail — ocean-sourced fraction path

- Taken only when `sea_ice is None`, `config.window_avg is None`, and the ocean dataset
  contains either configured sea ice variable. Configs with a `window_avg` (the E3SM coupled
  configs) or an ocean store without sea ice fields keep the legacy atmosphere-sourced
  behavior unchanged.
- Concentration source within the ocean dataset, also by presence: the native ocean-relative
  `ocean_sea_ice_fraction` when present, else derived as `sea_ice_fraction /
  sea_surface_fraction` clipped to [0, 1].
- Coherence is applied **at the ocean's 5-daily resolution**: land fraction is taken from the
  atmosphere at the ocean time instants (`.sel`, exact match — the ocean's snapshot instants
  coincide with 6-hourly atmosphere timestamps); modified sea surface fraction =
  `1 − land_fraction` where the ocean's `sea_surface_fraction > 0`, else 0; `sea_ice_fraction`
  / `ocean_fraction` are recomposed as `concentration × sfrac_mod` /
  `(1 − concentration) × sfrac_mod`. This is the same coherence logic as the legacy path,
  running at 5-daily instead of 6-hourly.
- The resulting fraction fields are reindexed onto the atmosphere's 6-hourly index with
  `reindex(method="ffill")` — matching the reindex semantics already used in
  `compute_coupled_atmosphere` — producing step-function fractions. No window average.
- `include_ts=True` keeps its blend semantics with the window average replaced by the ocean
  cadence: the blended field is `_interpolate_sst(ts=6-hourly ts, sst=ffill(ts at ocean
  instants), ofrac=reindexed ocean fraction)` — i.e. over ocean cells the surface temperature is
  the slowly-varying step function, over land it is the instantaneous atmosphere value, exactly
  as the legacy window-averaged blend behaves.

---

## Tests

## `scripts/data_process/test_coupled_dataset_utils.py` (new)

```python
# Pure-function tests on small synthetic datasets, following the style of
# test_compute_dataset.py (explicit helpers over fixtures).

def _synthetic_atmos(...) -> xr.Dataset:
    # 6-hourly atmosphere with land/sea-ice/ocean fractions and surface temperature
    ...

def _synthetic_ocean(...) -> xr.Dataset:
    # 5-daily ocean with sst, hfds, sea_surface_fraction and optional
    # sea_ice_fraction / ocean_sea_ice_fraction / hfds_total_area, time instants
    # aligned to the atmosphere's 6-hourly index
    ...

def test_sea_ice_source_priority():
    # GOAL: a provided sea_ice dataset wins even when the ocean carries sea ice
    # fields; the ocean wins over the atmosphere when window_avg is None; the
    # atmosphere is used only when neither other source applies.
    # PARAMETERIZE: available sources ∈ {sea_ice+ocean, ocean only, neither}.
    ...

def test_window_avg_keeps_legacy_path():
    # GOAL: with no sea_ice dataset, an ocean store carrying sea ice fields, and a
    # configured window_avg (the E3SM coupled config shape), output reproduces the
    # current atmosphere-sourced windowed behavior byte-for-byte.
    ...

def test_ocean_sourced_native_vs_derived_concentration():
    # GOAL: ocean_sea_ice_fraction is consumed verbatim when present; when absent,
    # concentration is derived as sea_ice_fraction / sea_surface_fraction clipped
    # to [0, 1] (including a ratio > 1 case).
    ...

def test_ocean_sourced_fractions_sum_to_one():
    # GOAL: land + ocean + sea ice fractions == 1 everywhere on the 6-hourly output.
    ...

def test_ocean_sourced_ffill_step_function():
    # GOAL: 5-daily fractions reindexed onto the 6-hourly index match hand-computed
    # step-function values (each 6-hourly step carries the most recent ocean instant).
    ...

def test_atmosphere_fallback_disabled_raises():
    # GOAL: with use_atmosphere_sea_ice_fraction_fallback=False, any configuration
    # that would use the atmosphere's sea ice fraction raises a ValueError; with the
    # default True the behavior is unchanged.
    # PARAMETERIZE: fallback cause ∈ {no ocean sea ice fields, window_avg configured}.
    ...

def test_ocean_sourced_include_ts_blend():
    # GOAL: with include_ts=True, output ts equals the documented blend
    # (instantaneous over land, ffilled ocean-cadence values over open ocean).
    ...

def test_legacy_mode_unchanged():
    # GOAL: regression guard — with a sea_ice dataset provided (or nothing but the
    # atmosphere), output reproduces the current compute_coupled_sea_ice behavior
    # on synthetic legacy inputs.
    ...

def test_ocean_sourced_full_chain():
    # GOAL: compute_coupled_sea_ice -> compute_coupled_ocean ->
    # compute_coupled_atmosphere on synthetic ocean-sourced inputs; the coupled
    # atmosphere carries step-function fractions summing to 1 on the 6-hourly index.
    ...

def test_hfds_total_area_passthrough():
    # GOAL: when the input ocean store provides hfds_total_area, compute_coupled_ocean
    # emits it unchanged (values distinct from hfds * sfrac to pin non-recomputation);
    # when absent, the derived product is emitted as before.
    # PARAMETERIZE: hfds_total_area present ∈ {True, False}.
    ...
```
