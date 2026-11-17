# Add Wright (1997) steric sea level ocean derived variable

Adds `steric_sea_level_wright97`, the local steric sea level η_steric [m] of
the static ocean column, next to `manometric_sea_level_wright97` (ai2cm/ace#1563,
which this PR stacks on). With both, `zos` splits into a steric and a manometric
part term by term. The global mean ⟨η_steric⟩ is the existing area-weighted
global-mean series the aggregators already report for every derived variable.

| symbol | meaning | units |
| --- | --- | --- |
| ρ0 | `RHO_0` (`fme/core/ocean_eos.py`) | kg m-3 |
| ρ'_k | `OceanData.rho_wright97` at level k: in-situ density minus ρ0, NaN below the sea floor | kg m-3 |
| dz_k | static layer thickness of level k (`DepthCoordinate.dz`) | m |
| C | Σ_k ρ'_k dz_k, `column_density_integral` (`fme/core/ocean_eos.py`; NaN ρ'_k contributes 0) | kg m-2 |
| η | `zos` | m |
| η_mano | `manometric_sea_level_wright97` (ai2cm/ace#1563) | m |
| ⟨·⟩_W | mean weighted by `area_weights_m2` over wet cells, as in η_mano | — |

```
k: depth level index, k in {0, ..., nz-1}

η_steric = −C / ρ0                                     NaN where mask_0 == 0
η_mano   = η + C/ρ0 − ⟨η + C/ρ0⟩_W                     (ai2cm/ace#1563)
⇒ η − ⟨η⟩_W = (η_steric − ⟨η_steric⟩_W) + η_mano       exact, wherever η is finite
```

---

## `fme/core/ocean_data.py` (modified)

```python
class OceanData:
    @property
    def steric_sea_level_wright97(self) -> torch.Tensor:  # NEW
        """η_steric = -C / RHO_0 [m], C = column_density_integral(rho_wright97, dz).

        Relative to a column of density RHO_0 over the static levels (no
        free-surface layer, as manometric_sea_level_wright97). Not demeaned: it
        carries a large, near-constant compressibility part, so its time changes
        and gen-minus-target differences are what it measures. NaN where
        mask_0 == 0.

        Raises:
            KeyError: potential temperature or salinity, or a level of either,
                is missing; raised before the depth coordinate is checked, so
                compute_ocean_derived_quantities skips it.
            ValueError: depth coordinate is not a HasOceanLayerGeometry, or
                potential temperature or salinity has more levels than it
                (inherited from rho_wright97).
        """
        ...
```

- No `zos` and no cell area provider needed, so it is computed on HEALPix and in
  configs without `zos` (unlike η_mano).
- `manometric_sea_level_wright97` is left unchanged; it keeps computing C itself.

## `fme/core/ocean_derived_variables.py` (modified)

```python
@register(  # NEW — placed directly above manometric_sea_level_wright97
    VariableMetadata("m", "Steric sea level of the static column, Wright (1997)")
)
def steric_sea_level_wright97(data: OceanData, timestep: datetime.timedelta) -> torch.Tensor:
    return data.steric_sea_level_wright97
```

- Default `exists_ok=False`: no dataset carries this name.
- Placement above η_mano lets η_mano later read it from the chained data; no
  ordering constraint with `zos` (η_steric does not read `zos`).

---

## Tests

## `fme/core/test_ocean_data.py` (modified)

```python
# Reuse _wright97_data / _wright97_ocean_data / _wright97_mask / _cell_area from the rho/η_mano tests.

def test_steric_sea_level_wright97_values_and_mask():
    # GOAL: equals -sum_k(rho_wright97_k * dz_k)/RHO_0 computed independently in float64,
    # NaN exactly where mask_0 == 0; masked-deep levels contribute 0.
    ...

def test_steric_sea_level_wright97_nan_wet_level_counts_as_zero():
    # GOAL: a NaN rho_wright97_k at a wet level contributes 0 to C (as in η_mano),
    # so η_steric stays finite there.
    ...

def test_steric_sea_level_wright97_sign():
    # GOAL: warming every wet level (same so) raises η_steric; freshening raises it too.
    # PARAMETERIZE: perturbation ∈ {thetao + 1, so - 1}.
    ...

def test_steric_plus_manometric_recovers_zos():
    # GOAL: η - ⟨η⟩_W == (η_steric - ⟨η_steric⟩_W) + η_mano wherever zos is finite,
    # ⟨·⟩_W with the same area weights and wet mask as η_mano.
    ...

def test_steric_sea_level_wright97_without_cell_area_or_zos():
    # GOAL: computes without a cell area provider and without zos.
    # PARAMETERIZE: missing ∈ {"zos", "cell_area"}.
    ...

# EXTEND the existing parametrizations with "steric_sea_level_wright97":
def test_wright97_without_layer_geometry_raises_value_error(name): ...  # CHANGED
def test_wright97_extra_levels_raise_value_error(name): ...  # CHANGED
def test_wright97_missing_profiles_raise_key_error_before_geometry(name): ...  # CHANGED
def test_wright97_one_dimensional_mask_matches_expanded(): ...  # CHANGED (loop)
```

## `fme/core/test_ocean_derived_variables.py` (modified)

```python
def test_steric_sea_level_wright97_derived_variable():
    # GOAL: compute_ocean_derived_quantities adds it with the OceanData property's value;
    # metadata units are "m".
    ...

def test_steric_sea_level_wright97_skipped_without_salinity():
    # GOAL: missing so -> label absent, no error.
    ...

# EXTEND test_manometric_sea_level_wright97_skipped_on_healpix:  # CHANGED
#   steric_sea_level_wright97 is present on HEALPix (no cell area needed).
# EXTEND test_wright97_skipped_without_profiles_or_layer_geometry:  # CHANGED
#   steric_sea_level_wright97 is absent too.
# EXTEND test_wright97_metadata_has_units:  # CHANGED
#   steric_sea_level_wright97 has units "m".
```

---

## Open Questions

- Thermosteric/halosteric decomposition: left to a follow-up PR (its reference
  state is a separate choice; see that PR).
- Name: `steric_sea_level_wright97` mirrors `manometric_sea_level_wright97`;
  CMIP `zostoga`/`zossga` are global scalars from the native grid and are not
  reused.
