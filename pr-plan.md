# Add per-level pooling factors to Samudra

`Samudra` and `SamudraBuilder` take `pooling`, one integer factor per U-Net level. When the
factors' product divides both grid dimensions, every level pools and upsamples exactly: no row
or column is dropped, padded or refilled. `pooling=None` (the default) keeps today's x2 floor
pooling at every level, bitwise.

Symbols: `H`, `W` = input grid rows, columns; `L = len(ch_width)`; `f_ℓ` = pooling factor at
level ℓ (ℓ = 1..L, applied to both axes); `P = ∏_ℓ f_ℓ`; `H_ℓ = H / ∏_{k≤ℓ} f_k`,
`W_ℓ = W / ∏_{k≤ℓ} f_k`; pool chain `𝒫 = AvgPool(f_L) ∘ … ∘ AvgPool(f_1)`, so `𝒫(x)` is the
bottleneck-resolution field of shape `(H_L, W_L)`.

---

## `fme/ace/models/ocean/m2lines/samudra.py` (modified)

```python
class Samudra(torch.nn.Module):
    def __init__(
        self,
        ...,
        pooling: list[int] | None = None,  # NEW — one factor per level; None → 2 at every level
    ):
        self.pooling: list[int]  # NEW — resolved factors, len == len(ch_width)
        self.exact_pooling: bool  # NEW — True iff pooling was given

        # CHANGED — encoder: level i pools by pooling[i]
        layers.append(AvgPool(pooling=self.pooling[i]))
        # CHANGED — decoder: the k-th upsample (k = 0 after the bottleneck) mirrors
        # the encoder, factor pooling[L - 1 - k], for both upsample classes
        layers.append(upsample_cls(in_channels=b, out_channels=b, upsampling=...))

    def forward(self, fts, context: Context | None = None):  # CHANGED — under exact_pooling,
        ...  # raise ValueError unless P divides fts.shape[-2] and fts.shape[-1]
```

`__init__` raises `ValueError` when `len(pooling) != len(ch_width)` or any factor is below 1.
The grid checks need `H`, `W`, so they live in `validate_exact_pooling_grid(pooling, img_shape,
lat_pad)`, called by `forward` on the input and by `SamudraBuilder.build` on the dataset grid. It
raises `ValueError` when:

- `P` does not divide `H` or `W`;
- `lat_pad="pole"` and any `W_ℓ` is odd.

`f_ℓ = 1` is allowed: that level keeps its resolution. Under `exact_pooling`, `forward` raises
instead of refilling when a decoder skip connection's shape differs from the upsampled tensor's.
Pooling adds no parameters, so state-dict keys and shapes depend only on `ch_width`. A
checkpoint loads under any `pooling` at equal `ch_width`, but it computes a different function.
The `Samudra` parameter docstring and the `SamudraBuilder` attribute docstring document
`pooling`.

## `fme/ace/registry/m2lines.py` (modified)

```python
class SamudraBuilder(ModuleConfig):
    pooling: list[int] | None = None  # NEW — see Samudra

    def __post_init__(self):  # NEW — length and positivity checks, so config parsing
        ...  # (fme.ace.validate_config) catches them

    def build(self, n_in_channels, n_out_channels, dataset_info):  # CHANGED — forward pooling;
        ...  # validate_exact_pooling_grid on dataset_info.img_shape
```

When `dataset_info` has no horizontal shape (`img_shape` raises `MissingDatasetInfo`), `build`
skips the divisibility check, and `forward` runs it instead.

### Critical detail — exactness and the default

- Explicit factors are a promise of exactness: `P | H` and `P | W`, checked at build and in
  `forward`. Then `H_ℓ`, `W_ℓ` are integers at every level, the decoder's skip-connection
  crop/refill in `Samudra.forward` is a no-op, and `MultiResolutionFiLM._resample` windows tile
  each feature cell exactly.
- `pooling=None` keeps floor pooling and its refill path unchanged; explicit
  `pooling=[2] * L` on a grid `2^L` does not divide raises. There is no fallback to padding.
- `lat_pad="pole"`: `P | H` with `W = 2H` makes every `W_ℓ` even, so `_rotate_half_longitude`
  shifts by exactly 180° at every level, and the first and last rows of every level have their
  outer edges at ±90°.
- `pad_latitude(mode="pole")` already raises when a pad exceeds the height. A ConvNeXt block's
  pad equals its dilation (kernel 3), so a config must keep `dilation[ℓ] ≤ H_ℓ` at the level it
  runs on, the bottleneck reusing `dilation[-1]` at `H_L`.
- `ZonallyPeriodicBilinearUpsample` already handles any integer `upsampling`; it needs no change.

---

## Tests

## `fme/ace/models/ocean/m2lines/test_samudra.py` (modified)

```python
# Existing locks stay untouched and green: test_samudra_output_is_unchanged,
# test_released_checkpoint_still_loads, test_samudra_default_lat_pad_matches_original_forward_bitwise,
# test_samudra_pole_lat_pad_matches_reference_forward_bitwise.
EXACT_1DEG = [[2, 2, 3], [2, 3, 3], [2, 2, 3, 3], [2, 2, 3, 1]]  # each P divides 180 and 360

def test_samudra_explicit_twos_match_default_bitwise():
    # GOAL: pooling=[2]*L equals pooling=None bitwise on a grid 2^L divides, same weights.
    # PARAMETERIZE: lat_pad × zonally_periodic_upsample.
    ...

def test_samudra_pooling_validation():
    # GOAL: wrong length, non-positive factor, a grid P does not divide, and an odd W_ℓ under
    # lat_pad="pole" each raise ValueError; f_ℓ = 1 is accepted.
    ...

def test_samudra_exact_pooling_level_shapes():
    # GOAL: at 180x360 every encoder and decoder level has shape (H_ℓ, W_ℓ); output shape
    # is (H, W); no skip-connection refill runs.
    # PARAMETERIZE: pooling ∈ EXACT_1DEG.
    ...

def test_exact_pool_chain_value_properties():
    # GOAL: at 180x360, on 𝒫 composed from the AvgPool modules of a built Samudra
    # (float64), all of:
    #   coverage:        ∂(Σ 𝒫(x))/∂x ≠ 0 for every input cell
    #   constant field:  𝒫(1) = 1
    #   uniform weight:  ∂(Σ 𝒫(x))/∂x = 1/P² for every input cell
    #   pole roll:       _rotate_half_longitude(𝒫(cos λ)) = -𝒫(cos λ), and likewise for sin λ,
    #                    i.e. the bottleneck column centred at λ maps to λ + 180°
    #   pole symmetry:   first and last bottleneck rows each draw on exactly P input rows
    # PARAMETERIZE: pooling ∈ EXACT_1DEG.
    ...

def test_samudra_exact_pooling_is_zonally_equivariant():
    # GOAL: with circular pad and zonally_periodic_upsample, rolling the input by P columns
    # rolls the output by P columns (allclose).
    # PARAMETERIZE: pooling ∈ EXACT_1DEG; lat_pad ∈ {"constant", "pole"}.
    ...

def test_samudra_exact_pooling_state_dict_matches_default():
    # GOAL: state-dict keys and shapes equal the default build's for the same ch_width.
    ...

def test_samudra_exact_pooling_with_noise_conditioning():
    # GOAL: a conditioned build with pooling runs at 180x360 and returns (H, W).
    ...

def test_zonally_periodic_upsample_is_zonally_periodic():  # CHANGED — PARAMETERIZE upsampling ∈ {2, 3}
    ...
```

## `fme/ace/registry/test_m2lines.py` (modified)

```python
def test_samudra_builder_pooling():
    # GOAL: the builder forwards pooling to Samudra; raises at construction on a wrong-length
    # or non-positive list; raises at build when P does not divide dataset_info.img_shape;
    # skips that check when dataset_info has no horizontal shape.
    ...
```

---

## Open Questions

- Name: `pooling` (matches `AvgPool(pooling=...)`, used by this PR and the `ocean_levpool`
  experiment) or `pooling_factors`? A rename at review means re-pinning the experiment.
- Per-axis factors are left out: at `W = 2H` one list gives square cells at every level. A later
  change can accept `[f_lat, f_lon]` pairs per entry without breaking `list[int]` configs.
- Should the builder also check `dilation[ℓ] ≤ H_ℓ` under `lat_pad="pole"`, failing at build
  rather than at the first forward?
- If the open `pad_pool` PR (#1589) merges first: under explicit `pooling`, `pad_pool` is a
  no-op. Allow the combination, or reject it?
