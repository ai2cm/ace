# Plan: SwinTransformer Backbone for ACE

## Context

ACE models use a 2D spatial interface `(B, C, H, W)` where all atmospheric vertical levels are stacked into the channel dimension. ArchesWeather's SwinTransformer works with 3D inputs `(pressure_levels, lat, lon)`. This plan adapts its core ideas — U-Net encoder-decoder, 2D shifted windowed attention, column-wise interaction, AdaLN conditioning — to ACE's 2D interface and model conventions.

---

## Architecture: what the paper and code actually say

**From the ArchesWeather paper (arXiv 2412.12971):**
- 3D Swin U-Net with two orthogonal interaction mechanisms:
  1. **Horizontal 2D local attention**: window shape `(1, 6, 10)` — vertical window = 1 so no vertical mixing inside the attention
  2. **Cross-Level Attention (CLA)**: full self-attention across pressure levels at each lat/lon column, applied as a separate sub-block *inside each transformer block*, between the window attention and MLP sub-blocks

**From the code (verified against paper):**

- `axis_attn=True` in all released configs enables CLA inside every `EarthSpecificBlock`
- A separate `LinVert` (linear layer across all levels at each spatial point) is applied **once before all transformer blocks** as a preprocessing step — this is *not described in the paper* but present in the code
- Exact block forward order:
  ```
  1. norm1 → [AdaLN scale/shift] → 3D windowed attention
                                 → axis_attn added to attn output (no separate residual)
                                 → [AdaLN gate_msa] + shortcut residual
  2. norm2 → [AdaLN scale/shift] → MLP → [AdaLN gate_mlp] + shortcut residual
  ```
- AdaLN (6-vector shift/scale/gate) is applied to both sub-blocks; `axis_attn` has no conditioning and no separate residual — it is folded into the window-attention output before the gated shortcut

---

## ACE 2D Adaptation

**2D windowed attention** is a direct translation: window shape `(ws_h, ws_w)` over `(H, W)`, alternating regular/shifted windows per block.

**Column interaction (CLA / axis_attn equivalent)**

In ArchesWeather's 3D model, CLA attends across `n_levels` tokens (one per pressure level) at each `(lat, lon)` point — O(n_levels²) per-point complexity. In ACE, all levels are stacked into channels, so there is no separate level-token sequence. After the Conv2d encoder maps `C` input channels to `embed_dim`, each spatial location has a single `embed_dim`-dimensional vector encoding all levels.

The faithful ACE equivalent is a **ColumnMixer**: a pointwise `Linear(embed_dim, embed_dim)` applied independently at every `(h, w)` location. Matching ArchesWeather's `axis_attn`, it has no separate norm and no separate residual — its output is added to the window-attention output before the gated shortcut. It follows the same spirit as LinVert (a linear cross-level transform, not full attention) but applied in embedding space since ACE has no explicit level dimension at that point.

**LinVert preprocessing (before all blocks)**

Following the code, a `ChannelMixer` is applied once before the transformer blocks. The actual `LinVert` has **no LayerNorm** — it is just a linear + residual: `Linear(embed_dim, embed_dim)` pointwise + residual. In ArchesWeather, LinVert operates on `8*C` jointly (mixing all pressure levels × channels in one matrix), but since ACE has no explicit level dimension the pointwise `Linear(embed_dim, embed_dim)` is the natural 2D equivalent. The absent LayerNorm matches the original.

---

## Full Block Structure

```
SwinTransformerBlock:
  1. norm1 → [AdaLN scale/shift on norm1 output] → WindowAttention2D
             → [ColumnMixer added to attn output] → drop_path + residual, gated by AdaLN gate_msa
  2. norm2 → [AdaLN scale/shift on norm2 output] → MLP
             → drop_path + residual, gated by AdaLN gate_mlp
```

**Note on ColumnMixer placement**: In ArchesWeather, `axis_attn` output is **folded into the window-attention residual** — it is added to the raw attention output before the gated shortcut, not treated as a separate sub-block. The faithful 2D adaptation preserves this: ColumnMixer (`Linear(dim, dim)` pointwise) is applied to the window-attention output and its result is added before the `gate_msa` shortcut. No separate norm or residual for ColumnMixer. No AdaLN on ColumnMixer (matches ArchesWeather).

```python
# block forward (with conditioning):
shortcut = x
x = norm1(x)
x = x * (1 + scale_msa) + shift_msa          # AdaLN
x = window_attention(x)
x = x + column_mixer(x)                       # ColumnMixer folded in here
x = shortcut + gate_msa * drop_path(x)
mlp_in = norm2(x) * (1 + scale_mlp) + shift_mlp  # AdaLN
x = x + drop_path(gate_mlp * mlp(mlp_in))
```

---

## Full Network Structure

```
(B, C_in, H, W)
 ↓  pad to nearest multiple of window_size * 2  (crop output at end)
 ↓  Conv2d encoder:  C_in → embed_dim
 ↓  ChannelMixer (LinVert equivalent: Linear(embed_dim, embed_dim) pointwise + residual, no LayerNorm)
 ↓  Layer 1: BasicLayer(embed_dim,   depth=2*d, num_heads[0], window_size)  @ (H, W)   [saved as skip]
 ↓  PatchMerging:  2×2 patch concat → LayerNorm(4*embed_dim) → Linear(4*embed_dim, 2*embed_dim),  (H/2, W/2)
 ↓  Layer 2: BasicLayer(embed_dim*2, depth=6*d, num_heads[1], window_size)  @ (H/2, W/2)
 ↓  Layer 3: BasicLayer(embed_dim*2, depth=6*d, num_heads[2], window_size)  @ (H/2, W/2)  [bottleneck]
 ↓  PatchExpanding:  Linear(embed_dim*2, embed_dim*4) → pixel-shuffle 2× → LayerNorm(embed_dim) → Linear(embed_dim, embed_dim),  (H, W)
 ↓  cat(expanded, skip_layer1) → embed_dim*2
 ↓  Layer 4: BasicLayer(embed_dim*2, depth=2*d, num_heads[3], window_size)  @ (H, W)
 ↓  Linear:  embed_dim*2 → embed_dim
 ↓  Conv2d decoder:  embed_dim → C_out
 ↓  crop to original (H, W)
(B, C_out, H, W)
```

Depths: `[2*d, 6*d, 6*d, 2*d]` where `d = depth_multiplier` (same as ArchesWeather).

---

## Conditioning (AdaLN)

**Why**: ACE uses the same registry for deterministic and diffusion models. `Context.embedding_scalar` carries per-sample scalar embeddings (e.g., diffusion timestep). `Context.labels` carries dataset-level labels (forcing scenarios, etc.). AdaLN (from DiT/ArchesWeather) is the right mechanism.

**How**: Per `BasicLayer`, one projection `SiLU → Linear(cond_dim → 6*dim)` produces `[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]`. Applied to both sub-blocks (window_attn and MLP). ColumnMixer has no conditioning — it is folded into the window_attn sub-block without its own AdaLN, matching ArchesWeather.

`cond_dim = embed_dim_scalar + embed_dim_labels`. If both are 0, no conditioning is built and Context is ignored.

**Why not `noise`/`embedding_pos`**: These are `(B, C, H, W)` per-pixel tensors — AdaLN requires a global per-sample vector. Spatial conditioning needs a different mechanism (e.g., extra input channels) and is out of scope.

---

## Files to Create

### `fme/core/models/swin_transformer/swin_layers.py`
- `window_partition_2d(x, ws_h, ws_w)` — `(B, H, W, C)` → `(B*nW, ws_h, ws_w, C)`
- `window_reverse_2d(windows, ws_h, ws_w, H, W)` — inverse
- `WindowAttention2D` — MHSA with 2D learnable relative position bias; optional shift mask
- `ColumnMixer` — `Linear(dim, dim)` pointwise (no norm, no separate residual); folded into the window-attention output before the gated shortcut, matching ArchesWeather's axis_attn placement
- `ChannelMixer` — `Linear(embed_dim, embed_dim)` pointwise + residual (no LayerNorm, matching LinVert); applied once before all blocks
- `SwinTransformerBlock` — two gated sub-blocks: (window_attn + ColumnMixer folded in), then MLP; AdaLN on both; alternates regular/shifted windows by block index; optional `mlp_layer` arg supports SwiGLU (released ArchesWeather config default)
- `PatchMerging` — 2×2 patch concat → `LayerNorm(4C)` → `Linear(4C, 2C)` (norm comes first, matching ArchesWeather `DownSample`)
- `PatchExpanding` — `Linear(C, 4C)` → pixel-shuffle 2× → `LayerNorm(C)` → `Linear(C, C)` (two linears + norm, matching ArchesWeather `UpSample`)
- `BasicLayer` — stack of `SwinTransformerBlock`s; owns AdaLN projection if conditioning active; DropPath schedule matches ArchesWeather: `linspace(0, drop_path_rate/depth_multiplier, 8*depth_multiplier)`, with layers 1&4 sharing the first `2*d` values and layers 2&3 sharing the last `6*d` values (not a single global linear ramp)

### `fme/core/models/swin_transformer/swin_transformer.py`
`SwinTransformerNet(nn.Module)`:
- Args: `in_chans, out_chans, img_shape, embed_dim, depth_multiplier, num_heads, window_size, mlp_ratio, drop_path_rate, use_skip, context_config`
- `context_config: ContextConfig | None`; conditioning dim = `embed_dim_scalar + embed_dim_labels`
- `forward(self, x, context: Context | None = None) -> Tensor`

### `fme/core/models/swin_transformer/__init__.py`
Exports `SwinTransformerNet`.

### `fme/core/models/swin_transformer/test_swin_transformer.py`
- `test_forward_no_conditioning` — `img_shape=(16,32)`, no context; output shape == input shape
- `test_forward_with_padding` — `img_shape=(9,18)` (not divisible by window*2); output shape == input shape
- `test_forward_with_conditioning` — `embedding_scalar` + `labels` via Context; output shape correct
- `test_no_skip` — `use_skip=False`
- `test_column_mixer` — unit test: ColumnMixer output == input when its Linear weights are zeroed (residual in block preserved)

### `fme/ace/registry/swin_transformer.py`
```python
@ModuleSelector.register("SwinTransformer")
@dataclasses.dataclass
class SwinTransformerBuilder(ModuleConfig):
    embed_dim: int = 96
    depth_multiplier: int = 1         # depths = [2,6,6,2] * depth_multiplier
    num_heads: tuple[int,...] = (3,6,6,3)
    window_size: tuple[int,int] = (4, 8)
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.2
    use_skip: bool = True
    embed_dim_scalar: int = 0         # 0 = no AdaLN conditioning
    embed_dim_labels: int = 0

    def build(self, n_in_channels, n_out_channels, dataset_info) -> nn.Module:
        # build SwinTransformerNet, wrap with _ContextWrappedModule
```

`_ContextWrappedModule` (same as `fme/ace/registry/local_net.py:24`) converts `labels=` kwarg into a `Context` object, populating only `context.labels`. This is sufficient for label conditioning (`embed_dim_labels > 0`).

**Note**: `embed_dim_scalar > 0` will silently do nothing with this wrapper because `_ContextWrappedModule` never populates `context.embedding_scalar`. Scalar conditioning (e.g. diffusion timestep) requires a separate wrapper analogous to `NoiseConditionedModel` (`fme/ace/registry/stochastic_sfno.py`) that embeds the timestep and injects it as `embedding_scalar`. This is out of scope for the initial implementation; `embed_dim_scalar` should be left at 0 until that wrapper exists.

---

## Files to Modify

| File | Change |
|------|--------|
| `fme/ace/registry/__init__.py` | Add `from . import swin_transformer as _swin` (and delete alias at bottom) |
| `fme/core/registry/module.py` | Add `"SwinTransformer"` to `CONDITIONAL_BUILDERS` |
| `fme/core/models/__init__.py` | Add `from . import swin_transformer` |

---

## Reused Utilities
- `Context`, `ContextConfig` — `fme/core/models/conditional_sfno/layers.py`
- `ModuleConfig`, `ModuleSelector` — `fme/core/registry/module.py`
- `DatasetInfo` — `fme/core/dataset_info.py`
- `DropPath`, `trunc_normal_` — `timm.models.layers` (existing dependency)
- `_ContextWrappedModule` pattern — `fme/ace/registry/local_net.py:24`

---

## Verification
```bash
# New tests
python -m pytest fme/core/models/swin_transformer/test_swin_transformer.py -v
# Existing tests still pass
python -m pytest fme/ace/registry/ fme/core/models/ -v
# Lint + types
pre-commit run --all-files
```
