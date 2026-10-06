# Add mld_wright97, rho_wright97 and pbo_wright97 ocean derived variables

Three ocean derived variables built on the Wright (1997) equation of state: a
density-threshold mixed layer depth, the per-level in-situ density anomaly, and
the globally demeaned bottom pressure anomaly. `register_multi` lets one
registered function emit one output per depth level. Every output carries units.

Symbols (units):

```
T_k     thetao_k, potential temperature           [degC]
S_k     so_k, practical salinity                  [PSU]
z_k     (idepth[k] + idepth[k+1]) / 2, level-centre depth, positive down  [m]
p_k     RHO_0 * G_EARTH * z_k, Boussinesq gauge pressure                  [Pa]
dz_k    DepthCoordinate.dz, wet layer thickness (partial bottom cells)    [m]
m_k     DepthCoordinate.mask[..., k], 1 ocean / 0 land or below sea floor
A       area_weights_m2 of the cell area provider                         [m**2]
RHO_0   DENSITY_OF_SEA_WATER_CM4 [kg/m**3];  G_EARTH  GRAVITY [m/s**2]
<x>     sum(A * w * x) / sum(A * w) over the horizontal dims, w = wet mask
```

Definitions:

```
rho_wright97_k = EOS_W97(S_k, T_k, p_k) - RHO_0                     [kg/m**3]
                 NaN where m_k == 0 or S_k, T_k not finite
C              = sum_k where(rho_wright97_k finite, rho_wright97_k * dz_k, 0)   [kg/m**2]
P              = RHO_0 * G_EARTH * zos + G_EARTH * C                [Pa]
pbo_wright97   = P - <P>,  w = (m_0 > 0) & isfinite(zos);  NaN where w == 0
mld_wright97   = depth where EOS_W97(S, T, 0)_k - EOS_W97(S, T, 0)_ref first exceeds
                 DELTA_RHO_THRESHOLD below MLD_REF_LAYER, linear between level
                 centres; sea floor depth if never; NaN where m_0 == 0   [m]
```

`pbo_wright97` is bottom pressure up to the static `RHO_0 * G_EARTH * deptho`
and the global mean.

---

## `fme/core/ocean_eos.py` (new)

```python
RHO_0 = DENSITY_OF_SEA_WATER_CM4  # [kg m-3]
G_EARTH = GRAVITY  # [m s-2]
DELTA_RHO_THRESHOLD = 0.03  # [kg m-3]
MLD_REF_LAYER = 1

def wright97_anomaly(S, theta, p, rho_ref: float = RHO_0) -> torch.Tensor:
    # rho - rho_ref, MOM6 MOM_EOS_Wright reduced-range fit, rho_ref branch
    # (keeps float32 precision by never forming rho)
    ...

def boussinesq_pressure(depth, rho_0=RHO_0, g_earth=G_EARTH) -> torch.Tensor: ...
def interface_to_center_depth(idepth) -> torch.Tensor: ...

def _sea_floor_depth(idepth, mask, deptho) -> torch.Tensor: ...

def _mixed_layer_depth(
    thetao, so, idepth, mask, deptho, delta_rho_threshold, ref_layer
) -> torch.Tensor: ...

def _density_anomaly(
    thetao: torch.Tensor,  # (..., nz)
    so: torch.Tensor,  # (..., nz)
    idepth: torch.Tensor,  # (nz + 1,)
    mask: torch.Tensor,  # broadcastable to thetao
) -> torch.Tensor:  # (..., nz), rho_wright97_k stacked on the last dim
    ...

def _column_density_integral(rho: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
    # C = sum_k rho_k dz_k, NaN rho -> 0
    ...
```

## `fme/core/ocean_data.py` (modified)

```python
@runtime_checkable
class HasOceanLayerGeometry(Protocol):  # NEW — what DepthCoordinate offers the wright97 variables
    @property
    def idepth(self) -> torch.Tensor: ...
    @property
    def mask(self) -> torch.Tensor: ...
    @property
    def dz(self) -> torch.Tensor: ...
    @property
    def deptho(self) -> torch.Tensor | None: ...

RHO_WRIGHT97_MAX_LEVELS = 100  # NEW — rho_wright97_{k} metadata is registered for k < this

class OceanData:
    @property
    def mld_wright97(self) -> torch.Tensor: ...  # NEW

    @property
    def rho_wright97(self) -> TensorDict: ...  # NEW — {"rho_wright97_{k}": ...}, k < nz

    @property
    def pbo_wright97(self) -> torch.Tensor: ...  # NEW

    def _layer_geometry(self, label: str) -> HasOceanLayerGeometry: ...  # NEW
    def _rho_wright97_stacked(self) -> torch.Tensor: ...  # NEW — shared by rho and pbo
```

### Critical detail — errors and skips

`compute_ocean_derived_quantities` skips a variable on `KeyError` and raises
anything else, as for `ocean_heat_content`.

| condition | mld | rho | pbo |
|---|---|---|---|
| depth coordinate is not `HasOceanLayerGeometry` | `ValueError` | `ValueError` | `ValueError` |
| `thetao_k` or `so_k` missing, `k < nz` | `KeyError` | `KeyError` | `KeyError` |
| fewer than `MLD_REF_LAYER + 2` levels | `KeyError` | — | — |
| `nz > RHO_WRIGHT97_MAX_LEVELS` | — | `ValueError` | — |
| `zos` or cell area provider missing | — | — | `KeyError` |

The global mean `<P>` goes through `Distributed.weighted_mean` over the
horizontal dims, so it is the global mean under spatial parallelism.

The EOS inputs are neither clamped nor NaN-substituted. Invalid points are NaN
in `rho_wright97_k` and contribute 0 to `C` through `torch.where`.

## `fme/core/ocean_derived_variables.py` (modified)

```python
OceanMultiDerivedVariableFunc = Callable[[OceanData, datetime.timedelta], TensorDict]  # NEW

_OCEAN_MULTI_DERIVED_VARIABLE_REGISTRY: MutableMapping[  # NEW
    str, tuple[OceanMultiDerivedVariableFunc, dict[str, VariableMetadata]]
] = {}

def get_ocean_derived_variable_metadata() -> dict[str, VariableMetadata]: ...  # CHANGED — adds every multi-registry output name

def register_multi(metadata: dict[str, VariableMetadata]): ...  # NEW — label collides with neither registry

def _compute_ocean_multi_derived_variable(
    data, depth_coordinate, timestep, label, func, cell_area_provider=None
) -> TensorDict: ...  # NEW — skip on KeyError; ValueError if an output name exists in data

def compute_ocean_derived_quantities(...) -> TensorDict: ...  # CHANGED — runs the multi registry after the single one

@register(VariableMetadata("m", "Mixed layer depth, Wright (1997) density threshold"))
def mld_wright97(data, timestep) -> torch.Tensor: ...  # NEW

@register_multi(
    {
        f"rho_wright97_{k}": VariableMetadata(
            "kg/m**3",
            f"In-situ density anomaly from {RHO_0:g} kg/m**3, Wright (1997), level {k}",
        )
        for k in range(RHO_WRIGHT97_MAX_LEVELS)
    }
)
def rho_wright97(data, timestep) -> TensorDict: ...  # NEW

@register(
    VariableMetadata("Pa", "Globally demeaned bottom pressure anomaly, Wright (1997)")
)
def pbo_wright97(data, timestep) -> torch.Tensor: ...  # NEW
```

---

## Tests

## `fme/core/test_ocean_eos.py` (new)

```python
def test_mom6_check_values(S, rho_chk, rho_ref):
    # GOAL: wright97_anomaly reproduces MOM6's published Wright check values.
    # PARAMETERIZE: rho_ref in {RHO_0, 1000.0}; the MOM6 check (S, T, p) points.
    ...

def test_float32_vs_float64():
    # GOAL: the anomaly form keeps float32 within tolerance of float64.
    ...

def test_monotonic_in_fit_range():
    # GOAL: density rises with S and p and falls with T inside the fit range.
    ...

def test_pressure_helpers():
    # GOAL: boussinesq_pressure and interface_to_center_depth on hand values.
    ...

def test_density_anomaly_values_and_mask():
    # GOAL: _density_anomaly equals wright97_anomaly at p_k per level; NaN where
    # m_k == 0 or an input is NaN.
    ...
```

## `fme/core/test_ocean_data.py` (modified)

```python
def test_mld_wright97_known_profile_and_land(): ...
def test_mld_wright97_no_crossing_is_sea_floor(): ...
def test_mld_wright97_missing_depth_coordinate_raises_value_error(): ...
def test_mld_wright97_depth_coordinate_without_idepth_raises_value_error(): ...
def test_mld_wright97_missing_salinity_raises_key_error(): ...
def test_mld_wright97_too_few_levels_raises_key_error(): ...

def test_rho_wright97_values_and_mask():
    # GOAL: one rho_wright97_k per level, equal to a hand computation from
    # wright97_anomaly; NaN on land, below the sea floor and at a NaN input.
    ...

def test_rho_wright97_too_many_levels_raises_value_error(): ...

def test_pbo_wright97_values_and_mask():
    # GOAL: equals a hand computation of P - <P> with partial bottom cells
    # (deptho) and non-uniform cell area; NaN on land; differs from the
    # full-cell column where deptho cuts a layer.
    ...

def test_pbo_wright97_global_mean_removed():
    # GOAL: <pbo_wright97> == 0; a uniform zos shift, and NaN zos off the wet
    # mask, leave pbo_wright97 unchanged.
    ...

def test_pbo_wright97_missing_inputs_raise_key_error():
    # PARAMETERIZE: missing in {zos, cell area provider}.
    ...

def test_wright97_without_layer_geometry_raises_value_error():
    # PARAMETERIZE: property in {rho_wright97, pbo_wright97}.
    ...
```

## `fme/core/test_ocean_derived_variables.py` (modified)

```python
def test_mld_wright97_derived_variable(): ...
def test_mld_wright97_skipped_without_salinity(): ...

def test_wright97_metadata_has_units():
    # GOAL: every name compute_ocean_derived_quantities emits for the three
    # variables is in get_derived_variable_metadata() with units; rho_wright97_k
    # is kg/m**3.
    ...

def test_pbo_wright97_skipped_without_cell_area_or_zos():
    # GOAL: rho_wright97_k still computed; pbo_wright97 absent.
    # PARAMETERIZE: missing in {cell area provider, zos}.
    ...

def test_register_multi_rejects_duplicate_label(): ...

def test_multi_derived_output_name_collision_raises(): ...
```

---

## Open Questions

- `RHO_WRIGHT97_MAX_LEVELS`: is a fixed metadata range acceptable, and is its
  value large enough for the vertical grids in use?
