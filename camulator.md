# Plan: Port Camulator (CrossFormer) from ace2 to ace

## Context
The Camulator is a CrossFormer-based transformer model for atmospheric emulation originally developed in the `ai2cm/ace2` repo under the MILES-CREDIT project. This plan ports it to the `feature/swin_transformer` branch of `ai2cm/ace` so the model can be trained with the same 4-degree AIMIP infrastructure used by the existing SwinTransformer.

---

## Files to Create

### 1. `fme/ace/models/miles_credit/` (new package)
Copy the four source files from `/Users/alexeyy/Git/ace2/fme/ace/models/miles_credit/`:

| Target | Source | Notes |
|---|---|---|
| `__init__.py` | ace2 `__init__.py` | Empty file; no changes needed |
| `base_model.py` | ace2 `base_model.py` | Copy as-is; `load_model*` methods are CREDIT-specific and never called by ACE, but harmless to include |
| `boundary_padding.py` | ace2 `boundary_padding.py` | Copy as-is; dimension-agnostic, works for both 4D and 5D tensors |
| `crossformer.py` | ace2 `crossformer.py` | **Requires modifications** for noise conditioning (see below) |

The one commented-out import in crossformer.py (`# from credit.postblock import PostBlock`) is already a comment, so no change needed.

#### Modifications to `crossformer.py` for noise conditioning

CrossFormer uses a custom `LayerNorm` that normalizes along the channel dimension (dim 1) — the same operation as `ChannelLayerNorm` in the existing codebase. `ConditionalLayerNorm` wraps `ChannelLayerNorm` when `global_layer_norm=False`, so CrossFormer's norms can be replaced with CLN without any further restructuring. `NoiseConditionedModel` (already used by both SFNO and SwinTransformer) then generates the noise field and injects it through the context.

**New imports** (add at the top of `crossformer.py`):
```python
import dataclasses
from fme.core.models.conditional_sfno.layers import (
    ConditionalLayerNorm,
    Context,
    ContextConfig,
)
```
(`torch.nn.functional as F` is already imported by ace2.)

**Modify `FeedForward`** — accept an optional `context_config` and split the `nn.Sequential` so `context` can be threaded through the norm. Note: this refactor changes state-dict keys for `FeedForward` only (`layers.0.*`/`layers.1-4.*` → `norm.*`/`ff.*`). `Attention` is unaffected because its norm was already a named attribute `self.norm`, so the key `norm.*` is unchanged there. Pre-trained ace2 CrossFormer weights would need a key-remapping shim for the FeedForward layers. There are no such checkpoints yet, so this is not a practical concern.
```python
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0, context_config: ContextConfig | None = None):
        super().__init__()
        if context_config is not None:
            self.norm: nn.Module = ConditionalLayerNorm(
                dim, img_shape=(1, 1), context_config=context_config
            )
        else:
            self.norm = LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
        )

    def forward(self, x, context: Context | None = None):
        if isinstance(self.norm, ConditionalLayerNorm):
            assert context is not None, "context required when ConditionalLayerNorm is used"
            x_norm = self.norm(x, context)
        else:
            x_norm = self.norm(x)
        return self.ff(x_norm)
```
Note: `img_shape=(1, 1)` is a dummy — `ConditionalLayerNorm` only uses `img_shape` for the `global_layer_norm=True` path (global `nn.LayerNorm`), which we never use here.

**Modify `Attention`** — same pattern for its pre-norm:
```python
class Attention(nn.Module):
    def __init__(
        self, dim, attn_type, window_size, dim_head=32, dropout=0.0,
        context_config: ContextConfig | None = None,
    ):
        ...
        if context_config is not None:
            self.norm: nn.Module = ConditionalLayerNorm(
                dim, img_shape=(1, 1), context_config=context_config
            )
        else:
            self.norm = LayerNorm(dim)
        ...

    def forward(self, x, context: Context | None = None):
        # prenorm (replace bare `x = self.norm(x)`)
        if isinstance(self.norm, ConditionalLayerNorm):
            assert context is not None
            x = self.norm(x, context)
        else:
            x = self.norm(x)
        # rest of forward is unchanged
        ...
```

**Modify `Transformer`** — accept `context_config`, pass to sub-modules, and thread `context` through `forward`:
```python
class Transformer(nn.Module):
    def __init__(
        self, dim, *, local_window_size, global_window_size, depth=4,
        dim_head=32, attn_dropout=0.0, ff_dropout=0.0,
        context_config: ContextConfig | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, attn_type="short", window_size=local_window_size,
                          dim_head=dim_head, dropout=attn_dropout,
                          context_config=context_config),
                FeedForward(dim, dropout=ff_dropout, context_config=context_config),
                Attention(dim, attn_type="long", window_size=global_window_size,
                          dim_head=dim_head, dropout=attn_dropout,
                          context_config=context_config),
                FeedForward(dim, dropout=ff_dropout, context_config=context_config),
            ]))

    def forward(self, x, context: Context | None = None):
        for short_attn, short_ff, long_attn, long_ff in self.layers:
            x = short_attn(x, context=context) + x
            x = short_ff(x, context=context) + x
            x = long_attn(x, context=context) + x
            x = long_ff(x, context=context) + x
        return x
```

**Modify `CrossFormer`** — accept `context_config`, pass to each `Transformer`, and thread `context` (with per-stage noise downsampling) through `forward`:

In `__init__`, add:
```python
context_config: ContextConfig | None = None,
```
and pass it to each `Transformer(...)` call inside the `for` loop.

In `forward`, change the signature and downsample noise at each encoder stage. Use `F.interpolate` to match the actual post-CEL spatial size, which is robust to any `cross_embed_strides` configuration:
```python
def forward(self, x, context: Context | None = None):
    ...
    # squeeze frames dim (no-op for 4D input with frames=1)
    x = x.squeeze(2)

    encodings = []
    for cel, transformer in self.layers:
        x = cel(x)
        if context is not None and context.noise is not None:
            # Resize noise to match the current encoder stage spatial size.
            # F.interpolate is used (not avg_pool2d) so this works for any
            # cross_embed_strides, not just uniform stride-2.
            noise_i = F.interpolate(
                context.noise, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
            ctx_i = dataclasses.replace(context, noise=noise_i)
        else:
            ctx_i = context
        x = transformer(x, context=ctx_i)
        encodings.append(x)

    # decoder (up_blocks) is unchanged — no CLN in decoder
    ...
```

When `context_config=None` (deterministic mode), `context` will be `None` at every call, so all `isinstance(self.norm, ConditionalLayerNorm)` branches are dead code and the model behaves exactly as the ace2 original.

### 2. `fme/ace/step/camulator.py` (new)
Copy from `/Users/alexeyy/Git/ace2/fme/ace/step/camulator.py` with the following modifications. All `fme.core.*` imports are identical to those in `fme/ace/step/fcn3.py`. The CrossFormer model import path also matches the new `fme/ace/models/miles_credit/crossformer.py` location.

**Key differences from FCN3Step (intentional):**
- Uses a single unified `input_packer`/`output_packer` (FCN3 uses separate packers per variable group)
- Level ordering: atmosphere channels iterate name-major then level (FCN3 is level-major then name)
- Has `atmosphere_level_start: int = 0` parameter for level index offset

**Changes vs. ace2 source (must be applied when porting):**

1. **Remove dead fields from `CrossFormerConfig`** — the fields `channels`, `surface_channels`, `input_only_channels`, `output_only_channels`, `levels`, `image_height`, and `image_width` are declared as dataclass fields in ace2 but are never read by `CrossFormerConfig.build()` (channel counts and image dimensions are always supplied as arguments). Remove them so users cannot mistakenly set them in YAML and expect an effect. Keep them only in the `CrossFormer.__init__` signature (where they are still used by the model itself). Note: `image_height` and `image_width` have misleading defaults (640, 1280) that would silently have no effect — particularly important to remove. Also change the `frames` default from `2` (ace2 value) to `1`, matching `NoiseConditionedCrossFormerConfig` and the YAML configs. The ace2 default of `2` would silently route through the `F.avg_pool3d` branch in `CrossFormer.forward` if a user omits `frames` from their YAML.

2. **`diagnostic_names` property** — the ace2 version returns `[]`, but `surface_diagnostic_names` and `atmosphere_diagnostic_names` are genuine output-only variables. Replace with the correct implementation, matching `SingleModuleStepConfig` (single_module.py:166-168), which uses `self.output_names` (not `self.out_names`):
   ```python
   @property
   def diagnostic_names(self) -> list[str]:
       return list(set(self.output_names).difference(self.in_names))
   ```
   For `CrossFormerStepConfig`, `output_names` is defined as `self.out_names` (there is no secondary decoder), so the two expressions are equivalent. Using `self.output_names` keeps the pattern consistent with the rest of the codebase.

3. **`in_names` ordering** — change the derivation in `CrossFormerStepConfig.__post_init__` from atmosphere-before-surface to surface-before-atmosphere, to match the swin YAML channel ordering; and update the `input_packer` in `CrossFormerStep.__init__` to match:
   ```python
   # CrossFormerStepConfig.__post_init__ — ace2 original:
   self.in_names = self.forcing_names + self.atmosphere_input_names + self.surface_input_names
   # ace port (match swin ordering: forcing → surface → atmosphere):
   self.in_names = self.forcing_names + self.surface_input_names + self.atmosphere_input_names

   # CrossFormerStep.__init__ — ace2 original:
   self.input_packer = Packer(
       config.forcing_names + config.atmosphere_input_names + config.surface_input_names
   )
   # ace port (must match in_names ordering):
   self.input_packer = Packer(
       config.forcing_names + config.surface_input_names + config.atmosphere_input_names
   )
   ```

4. **`global_mean_removal`** — add the following import, field, validation, build, and forward-pass wiring (following `SingleModuleStep` in `fme/core/step/single_module.py`):
   ```python
   # new import
   from fme.core.step.global_mean_removal import (
       GlobalMeanRemoval,
       GlobalMeanRemovalConfigUnion,
       NoGlobalMeanRemoval,
   )

   # new field on CrossFormerStepConfig
   global_mean_removal: GlobalMeanRemovalConfigUnion | None = None

   # in __post_init__ (after in_names/out_names are set)
   if self.global_mean_removal is not None:
       self.global_mean_removal.validate_names(self.in_names, self.out_names)
   ```

   **Construction order constraint in `CrossFormerStep.__init__`:** `normalizer` is already a parameter (built in `get_step` and passed in — do **not** rebuild it here). GMR must be built second (requires normalizer), and the module third (so `n_aux_channels` can include the GMR extra). Do not reorder these steps:
   ```python
   # 1. normalizer is the __init__ parameter — already available, do not rebuild
   # 2. GMR second — requires normalizer
   if config.global_mean_removal is not None:
       self._global_mean_removal: GlobalMeanRemoval = config.global_mean_removal.build(
           normalizer=normalizer, in_names=config.in_names
       )
   else:
       self._global_mean_removal = NoGlobalMeanRemoval()
   # 3. module third — requires n_extra from GMR
   n_extra = self._global_mean_removal.n_extra_input_channels
   module: nn.Module = config.builder.build(
       ...
       n_aux_channels=len(config.forcing_names) + n_extra,
       ...
   )
   ```

   In `CrossFormerStep.step()`, update `network_call` to apply input masking and retrieve GMR extra
   channels. The masking line mirrors `single_module.py:386-387`; the GMR lines mirror
   `single_module.py:397-399`; `wrapper(self.module)` is the ace2 camulator form (CrossFormer is a
   plain `nn.Module`, not a `WrappedModule`):
   ```python
   def network_call(input_norm: TensorDict) -> TensorDict:
       if args.data_mask is not None:
           input_norm = _apply_input_mask(input_norm, args.data_mask)
       input_tensor = self.input_packer.pack(input_norm, axis=self.CHANNEL_DIM)
       extra = self._global_mean_removal.get_extra_channels()
       if extra is not None:
           input_tensor = torch.cat([input_tensor, extra], dim=self.CHANNEL_DIM)
       output_tensor = wrapper(self.module)(input_tensor)
       output_tensor = output_tensor.squeeze(2)
       return self.output_packer.unpack(output_tensor, axis=self.CHANNEL_DIM)
   ```

   Keep the existing labels guard at the top of `step()` (CrossFormer does not support labels):
   ```python
   if args.labels is not None:
       raise ValueError("Labels are not supported for CrossFormer")
   ```

   Update the `step_with_adjustments` import to also pull in `_apply_input_mask`:
   ```python
   from fme.core.step.single_module import _apply_input_mask, step_with_adjustments
   ```

   Also pass `global_mean_removal=self._global_mean_removal` and `data_mask=args.data_mask` to `step_with_adjustments`.

5. **`from_state` / `_remove_deprecated_keys`** — keep the ace2 `from_state` override as-is; it calls `_remove_deprecated_keys` before delegating to `dacite.from_dict`. Do **not** remove it or rely on the `StepConfigABC` default (which skips that step).

6. **Noise conditioning — two separate builder types (mirrors swin pattern exactly)** — the swin YAML uses `builder.type: SwinTransformer` for deterministic runs and `builder.type: NoiseConditionedSwinTransformer` for stochastic runs, with `noise_embed_dim` as a field at the same level as the architecture parameters. CrossFormer follows exactly the same pattern.

   **Keep `CrossFormerConfig` and `CrossFormerSelector` unchanged** (deterministic, `builder.type: CrossFormer`), except for the dead-field and `frames`-default fixes from item 1 above. `CrossFormerConfig.build()` requires no changes for noise conditioning: after `CrossFormer.__init__` gains `context_config: ContextConfig | None = None`, the existing `build()` call automatically passes `None` via the default — no explicit `context_config=None` argument is needed.

   **New imports** in `camulator.py`:
   ```python
   from fme.ace.registry.stochastic_sfno import NoiseConditionedModel
   from fme.core.models.conditional_sfno.layers import ContextConfig
   ```

   **New dataclass `NoiseConditionedCrossFormerConfig`** — same architecture fields as `CrossFormerConfig`, plus `noise_embed_dim`:
   ```python
   @dataclasses.dataclass
   class NoiseConditionedCrossFormerConfig:
       """CrossFormer with CLN-based noise conditioning via NoiseConditionedModel."""
       frames: int = 1
       patch_height: int = 1
       patch_width: int = 1
       dim: list[int] = dataclasses.field(default_factory=lambda: [256, 512, 1024, 2048])
       depth: list[int] = dataclasses.field(default_factory=lambda: [2, 2, 18, 2])
       dim_head: int = 32
       global_window_size: list[int] = dataclasses.field(
           default_factory=lambda: [4, 4, 2, 1]
       )
       local_window_size: int = 3
       cross_embed_kernel_sizes: list[list[int]] = dataclasses.field(
           default_factory=lambda: [[4, 8, 16, 32], [2, 4], [2, 4], [2, 4]]
       )
       cross_embed_strides: list[int] = dataclasses.field(
           default_factory=lambda: [2, 2, 2, 2]
       )
       attn_dropout: float = 0.0
       ff_dropout: float = 0.0
       use_spectral_norm: bool = True
       interp: bool = True
       padding_conf: dict | None = None
       post_conf: dict | None = None  # PostBlock is commented out in CrossFormer; this field is inert
       noise_embed_dim: int = 256  # ← the only functional addition vs CrossFormerConfig

       def build(
           self,
           n_atmo_channels: int,
           n_atmo_groups: int,
           n_surf_channels: int,
           n_aux_channels: int,
           n_atmo_diagnostic_channels: int,
           n_surf_diagnostic_channels: int,
           img_shape: tuple[int, int],
       ) -> nn.Module:
           context_config = ContextConfig(
               embed_dim_scalar=0,
               embed_dim_labels=0,
               embed_dim_noise=self.noise_embed_dim,
               embed_dim_pos=0,
           )
           crossformer = CrossFormer(
               image_height=img_shape[0],
               patch_height=self.patch_height,
               image_width=img_shape[1],
               patch_width=self.patch_width,
               ...,  # same remaining kwargs as CrossFormerConfig.build()
               context_config=context_config,
           )
           return NoiseConditionedModel(
               crossformer,
               img_shape=img_shape,
               embed_dim_noise=self.noise_embed_dim,
               embed_dim_pos=0,
               n_labels=0,
               label_embed_dim=0,
               inverse_sht=None,
           )
   ```

   **New selector `NoiseConditionedCrossFormerSelector`**:
   ```python
   @dataclasses.dataclass
   class NoiseConditionedCrossFormerSelector:
       type: Literal["NoiseConditionedCrossFormer"]
       config: NoiseConditionedCrossFormerConfig

       def build(self, ...) -> nn.Module:
           return self.config.build(...)
   ```

   **Update `CrossFormerStepConfig.builder` type** — change from bare `CrossFormerSelector` to the union:
   ```python
   builder: CrossFormerSelector | NoiseConditionedCrossFormerSelector
   ```
   dacite will resolve this from the `type` discriminant at parse time.

   **No changes to `CrossFormerStep`** — `NoiseConditionedModel.forward(input_tensor)` accepts a plain tensor (labels default to `None`), generates noise internally, and calls `CrossFormer.forward(x, Context(noise=noise))`. The step's `wrapper(self.module)(input_tensor)` call is unchanged. The labels guard remains in place.

   **`CrossFormerStep.modules` does not need to include `_global_mean_removal`** — none of the `GlobalMeanRemoval` implementations are `nn.Module` subclasses and none carry trainable `nn.Parameter`s. The `modules` property (used for DDP wrapping and the optimizer) correctly covers only `self.module`. This is consistent with `SingleModuleStep`, which also excludes GMR from `modules`.

   **Stochastic outputs** — repeated `step()` calls on the same input produce different outputs when using `NoiseConditionedCrossFormer`, enabling ensemble generation.

### 3. Config files
Two configs are created (one deterministic, one noise-conditioned), mirroring `ace-train-config-4deg-AIMIP-swin.yaml` / `ace-train-config-4deg-AIMIP-nc-swin.yaml`:

| File | `builder.type` | Stochastic? |
|---|---|---|
| `ace-train-config-4deg-AIMIP-crossformer.yaml` | `CrossFormer` | no |
| `ace-train-config-4deg-AIMIP-nc-crossformer.yaml` | `NoiseConditionedCrossFormer` | yes |

Both are derived from `ace-train-config-4deg-AIMIP-swin.yaml`. Differences vs. swin:
- `stepper.step.type: CrossFormer` (not `single_module`)
- Uses structured variable groups (`forcing_names`, `atmosphere_prognostic_names`, etc.) instead of explicit `in_names`/`out_names`
- CrossFormer hyperparameters tuned for 4-deg grid (see grid analysis below)
- **Requires `padding_conf`** to make window sizes work (see below)

---

## Files to Modify

### `fme/ace/step/__init__.py`
Add import to trigger `@StepSelector.register("CrossFormer")` decorator and expose all public symbols:
```python
from .camulator import (
    CrossFormerConfig,
    CrossFormerSelector,
    CrossFormerStepConfig,
    NoiseConditionedCrossFormerConfig,
    NoiseConditionedCrossFormerSelector,
)
```
The `fme/ace/__init__.py` already has `from . import step`, which loads this `__init__.py` at startup.

---

## 4-Degree Grid Sizing Analysis

The 4-deg ERA5 grid is 45×90. Without padding, each stride-2 CrossEmbed stage produces:
`45×90 → 22×45 → 11×22 → 5×11 → 2×5`

GCD(22, 45) = 1, so no useful `local_window_size` divides both dimensions — the CrossFormer **cannot** run on this grid without padding.

**Solution:** use `padding_conf: {activate: true, mode: earth, pad_lat: [2, 1], pad_lon: [3, 3]}` to pad 45→48, 90→96.

Padded stages: `48×96 → 24×48 → 12×24 → 6×12 → 3×6`

Window size divisibility check:
- `local_window_size=3`: 24%3=0, 48%3=0, 12%3=0, 24%3=0, 6%3=0, 12%3=0, 3%3=0, 6%3=0 ✓
- `global_window_size=[4,4,2,1]`: all four stages pass ✓

Decoder recovers padded size, then `interp=True` rescales back to 45×90.

The `TensorPadding` class uses `...` indexing and works on both 4D (B,C,H,W) and 5D (B,C,T,H,W) tensors.

---

## Config Design for CrossFormer (4-deg)

This produces the same **variable set** as the swin config. `in_names` ordering matches (`forcing → surface → atmosphere`) after the ace-port fix above. `out_names` ordering differs by design: CrossFormerStep produces `atmosphere_output + surface_output` (atmosphere first, because the CrossFormer model encodes atmosphere channels together), while the swin config has `surface_output + atmosphere_output`. This is intentional — CrossFormerStep uses its own `output_packer` which returns a dict, so downstream consumers are unaffected by the list ordering.

The complete `stepper:` section for the crossformer config (all other top-level keys — `seed`, `train_loader`, `validation`, `optimization`, `stepper_training`, `logging`, `inference` — carry over unchanged from the swin config):

```yaml
stepper:
  step:
    type: CrossFormer
    config:
      forcing_names:
      - land_fraction
      - ocean_fraction
      - sea_ice_fraction
      - DSWRFtoa
      - HGTsfc
      atmosphere_prognostic_names:
      - air_temperature
      - specific_total_water
      - eastward_wind
      - northward_wind
      atmosphere_levels: 8
      surface_prognostic_names:
      - PRESsfc
      - surface_temperature
      - TMP2m
      - Q2m
      - UGRD10m
      - VGRD10m
      atmosphere_diagnostic_names: []
      surface_diagnostic_names:
      - LHTFLsfc
      - SHTFLsfc
      - PRATEsfc
      - ULWRFsfc
      - ULWRFtoa
      - DLWRFsfc
      - DSWRFsfc
      - USWRFsfc
      - USWRFtoa
      - tendency_of_total_water_path_due_to_advection
      - TMP850
      - h500
      next_step_forcing_names:
      - DSWRFtoa
      residual_prediction: false
      builder:
        type: CrossFormer
        config:
          frames: 1
          dim: [256, 512, 1024, 2048]
          depth: [2, 2, 18, 2]
          global_window_size: [4, 4, 2, 1]
          local_window_size: 3
          cross_embed_kernel_sizes: [[4, 8, 16, 32], [2, 4], [2, 4], [2, 4]]
          cross_embed_strides: [2, 2, 2, 2]
          attn_dropout: 0.0
          ff_dropout: 0.0
          use_spectral_norm: true
          interp: true
          padding_conf:
            activate: true
            mode: earth
            pad_lat: [2, 1]
            pad_lon: [3, 3]
      normalization:
        network:
          global_means_path: /climate-default/2026-04-17-era5-4deg-8layer-daily-stats-1990-2019/2026-03-19-era5-4deg-8layer-1940-2025/centering.nc
          global_stds_path: /climate-default/2026-04-17-era5-4deg-8layer-daily-stats-1990-2019/2026-03-19-era5-4deg-8layer-1940-2025/scaling-full-field.nc
        residual:
          global_means_path: /climate-default/2026-04-17-era5-4deg-8layer-daily-stats-1990-2019/2026-03-19-era5-4deg-8layer-1940-2025/centering.nc
          global_stds_path: /climate-default/2026-04-17-era5-4deg-8layer-daily-stats-1990-2019/2026-03-19-era5-4deg-8layer-1940-2025/scaling-residual.nc
      ocean:
        surface_temperature_name: surface_temperature
        ocean_fraction_name: ocean_fraction
      corrector:
        conserve_dry_air: true
        moisture_budget_correction: advection_and_precipitation
        force_positive_names:
        - specific_total_water_0
        - specific_total_water_1
        - specific_total_water_2
        - specific_total_water_3
        - specific_total_water_4
        - specific_total_water_5
        - specific_total_water_6
        - specific_total_water_7
        - Q2m
        - PRATEsfc
        - ULWRFsfc
        - ULWRFtoa
        - DLWRFsfc
        - DSWRFsfc
        - USWRFsfc
        - USWRFtoa
      global_mean_removal:
        kind: shared
        append_as_input: true
```

The NC config (`ace-train-config-4deg-AIMIP-nc-crossformer.yaml`) is identical except the `builder` block becomes:
```yaml
      builder:
        type: NoiseConditionedCrossFormer
        config:
          frames: 1
          dim: [256, 512, 1024, 2048]
          depth: [2, 2, 18, 2]
          global_window_size: [4, 4, 2, 1]
          local_window_size: 3
          cross_embed_kernel_sizes: [[4, 8, 16, 32], [2, 4], [2, 4], [2, 4]]
          cross_embed_strides: [2, 2, 2, 2]
          attn_dropout: 0.0
          ff_dropout: 0.0
          use_spectral_norm: true
          interp: true
          padding_conf:
            activate: true
            mode: earth
            pad_lat: [2, 1]
            pad_lon: [3, 3]
          noise_embed_dim: 256
```
Also change `stepper_training.loss` from `type: MSE` to (matching nc-swin exactly):
```yaml
    loss:
      type: EnsembleLoss
      weights:
        air_temperature_0: 0.5
        air_temperature_1: 0.5
        eastward_wind_0: 0.5
        northward_wind_0: 0.5
        # ... (same per-variable weights as nc-swin)
      config:
        crps_weight: 0.9
        energy_score_weight: 0.1
```
`EnsembleLoss` combines CRPS (weight 0.9) and energy score (weight 0.1), both proper scoring rules for probabilistic forecasts. This requires multiple forward passes per training step (ensemble members) — the stochasticity from `NoiseConditionedModel` produces the spread.

---

## Tests to Add

### `fme/ace/models/miles_credit/test_crossformer.py`
- Forward pass with small spatial dims (e.g., 12×24 with padding to ensure divisibility)
- Verify output shape matches input spatial dims (with interp=True)
- Test both with and without padding_conf
- Forward pass with `context_config` (noise conditioning): verify output shape is unchanged and two calls with the same input produce different outputs

### `fme/ace/step/test_camulator.py`
Following the pattern of any existing step test in `fme/core/step/`:
- Instantiate `CrossFormerStepConfig` with `builder.type: CrossFormer` and minimal variables; call `get_step`; run one `step()` and verify output names match `out_names`
- Same for `NoiseConditionedCrossFormer` builder: verify two `step()` calls on identical input differ (stochastic check)

---

## Verification

1. **Unit tests:** `python -m pytest fme/ace/models/miles_credit/ fme/ace/step/test_camulator.py -v`
2. **Import check:** `python -c "import fme.ace; from fme.core.step.step import StepSelector; print(StepSelector.get_available_types())"` — should include "CrossFormer"
3. **Config parse check (deterministic):** `python -c "import yaml, dacite, fme.ace; from fme.ace.train import TrainConfig; cfg = yaml.safe_load(open('configs/experiments/2026-05-28-swin-transformer/ace-train-config-4deg-AIMIP-crossformer.yaml')); dacite.from_dict(TrainConfig, cfg, config=dacite.Config(strict=True))"` — should not raise
4. **Config parse check (NC):** same command with `ace-train-config-4deg-AIMIP-nc-crossformer.yaml`
5. **Pre-commit:** `pre-commit run --all-files` to pass ruff, ruff-format, mypy
