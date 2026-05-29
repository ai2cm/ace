# Plan: Replace RPB with Physically-Informed CPB in Swin Transformer

## Context

The Swin Transformer in ACE uses a learned Relative Position Bias (RPB) lookup table in
`WindowAttention2D`. This treats pixel-index offsets `(Δi, Δj)` as the relative position
signal. For climate/atmospheric data on a lat/lon grid, this ignores that 1° of longitude
represents very different physical distances at different latitudes (cos(lat) scaling). The
Swin V2 Continuous Positional Bias (CPB) replaces the lookup table with a small MLP over
continuous coordinates — enabling physically meaningful inputs.

**Goals:**
- Replace RPB with a 2-layer CPB MLP (reference: `~/Git/Swin-Transformer-V2/swin_transformer_v2/model_parts.py`)
- Feed physically-informed coordinates into the MLP: `(Δlat, Δlon × cos(lat_window_mean))` using log-spacing
- Thread `lat_coords` (1D latitude tensor) through the full model stack so each window gets its own latitude-scaled input
- Direct replacement of RPB (no backward-compat flag); `cpb_hidden_dim` becomes a config parameter

---

## Files to Modify

### 1. `fme/core/coordinates.py`

Add a `lat_1d: torch.Tensor | None` property to `HorizontalCoordinates` (abstract base) returning `None` by default. Override in `LatLonCoordinates` to return `self.lat` (the existing 1D lat field). This avoids isinstance checks in the registry.

```python
# HorizontalCoordinates (add non-abstract property):
@property
def lat_1d(self) -> torch.Tensor | None:
    return None

# LatLonCoordinates (override):
@property
def lat_1d(self) -> torch.Tensor:
    return self.lat
```

---

### 2. `fme/core/models/swin_transformer/swin_layers.py`

**`WindowAttention2D.__init__` changes:**
- Add `cpb_hidden_dim: int = 64` parameter
- Remove: `relative_position_bias_table` (nn.Parameter), `relative_position_index` buffer, `trunc_normal_` call
- Add `cpb_mlp`: `nn.Sequential(Linear(2, cpb_hidden_dim), ReLU(), Linear(cpb_hidden_dim, num_heads))`
  - Zero-initialize final layer: `nn.init.zeros_(cpb_mlp[-1].weight); nn.init.zeros_(cpb_mlp[-1].bias)`
- Add `relative_coords_base` buffer: shape `(N*N, 2)` — raw pixel-index offsets `(Δi, Δj)` without log transform or shifting; range `[-(ws_h-1), ws_h-1] × [-(ws_w-1), ws_w-1]`
- Precompute `relative_coords_log` buffer from `relative_coords_base` (log-scaled base case, reused when lat_mean is None)
- Remove `trunc_normal_` import if no other uses remain in the file

**`WindowAttention2D.forward` changes — signature `forward(x, mask=None, lat_mean=None)`:**

- `lat_mean`: optional `(nW,)` tensor of mean latitude in degrees for each spatial window
- If `lat_mean is None` (no physical scaling): use precomputed `relative_coords_log` buffer
  ```python
  bias = self.cpb_mlp(self.relative_coords_log)   # (N*N, heads)
  bias = bias.permute(1, 0).reshape(num_heads, N, N)
  attn = attn + bias.unsqueeze(0)
  ```
- If `lat_mean is not None` (per-window lat scaling):
  ```python
  lat_rad = lat_mean * (math.pi / 180.0)            # (nW,)
  h_coords = self.relative_coords_base[:, 0]         # (N*N,) broadcast
  w_coords = (self.relative_coords_base[:, 1].unsqueeze(0)
               * torch.cos(lat_rad).unsqueeze(1))    # (nW, N*N)
  coords = torch.stack([
      h_coords.unsqueeze(0).expand(nW, -1),
      w_coords], dim=-1)                             # (nW, N*N, 2)
  coords_log = torch.sign(coords) * torch.log(1.0 + coords.abs())
  bias = self.cpb_mlp(coords_log)                    # (nW, N*N, heads)
  bias = bias.permute(0, 2, 1).reshape(nW, num_heads, N, N)
  attn = attn.view(B_ // nW, nW, num_heads, N, N) + bias.unsqueeze(0)
  attn = attn.view(B_, num_heads, N, N)
  ```
- Mask handling (shifted-window) proceeds unchanged after position bias is applied

**`SwinTransformerBlock.__init__` changes:**
- Add `cpb_hidden_dim: int = 64` and `lat_coords: torch.Tensor | None = None` parameters
- `register_buffer("lat_coords", lat_coords)` (persistent=False, matches attn_mask)
- Pass `cpb_hidden_dim` to `WindowAttention2D`

**`SwinTransformerBlock.forward` changes:**
- Compute `lat_mean` once before the CLN/AdaLN branching:
  ```python
  if self.lat_coords is not None:
      lat_shifted = (torch.roll(self.lat_coords, -sh)
                     if sh != 0 else self.lat_coords)
      nH_win = H // ws_h
      nW_win = W // ws_w
      lat_mean_h = lat_shifted[:H].reshape(nH_win, ws_h).mean(1)  # (nH_win,)
      lat_mean = lat_mean_h.unsqueeze(1).expand(-1, nW_win).reshape(-1)
  else:
      lat_mean = None
  ```
- Pass `lat_mean` in both branches: `self.attn(h_windows, mask=self.attn_mask, lat_mean=lat_mean)`

**`BasicLayer.__init__` changes:**
- Add `cpb_hidden_dim: int = 64` and `lat_coords: torch.Tensor | None = None` parameters
- Pass both to each `SwinTransformerBlock` in the list comprehension

---

### 3. `fme/core/models/swin_transformer/swin_transformer.py`

**`SwinTransformerNet.__init__` changes:**
- Add `cpb_hidden_dim: int = 64` and `lat_coords: torch.Tensor | None = None` parameters
- Compute lat vectors for each resolution (after computing `Hp`):
  ```python
  if lat_coords is not None:
      # Pad H0 → Hp by repeating the last value
      pad_h = Hp - H0
      lat_full = (torch.cat([lat_coords, lat_coords[-1:].expand(pad_h)])
                  if pad_h > 0 else lat_coords)        # (Hp,)
      # Half-resolution (after PatchMerging): average adjacent pairs
      lat_half = (lat_full[::2] + lat_full[1::2]) / 2  # (Hp//2,)
  else:
      lat_full = lat_half = None
  ```
- Pass `lat_full` and `cpb_hidden_dim` to `layer1` and `layer4`; `lat_half` and `cpb_hidden_dim` to `layer2` and `layer3`

---

### 4. `fme/ace/registry/swin_transformer.py`

**Both `SwinTransformerBuilder` and `NoiseConditionedSwinTransformerBuilder`:**
- Add dataclass field: `cpb_hidden_dim: int = 64`
- In `build()`, extract lat_coords from dataset_info:
  ```python
  try:
      lat_coords = dataset_info.horizontal_coordinates.lat_1d
  except MissingDatasetInfo:
      lat_coords = None
  ```
- Pass `lat_coords` and `cpb_hidden_dim` when constructing `SwinTransformerNet`

---

## Testing

### `fme/core/models/swin_transformer/test_swin_transformer.py`

- All existing tests continue to pass (CPB without lat_coords falls back to lat-independent behavior)
- Add `test_cpb_lat_coords_changes_output`: build two nets with the same weights but different `lat_coords` (e.g., 10° vs 60°) and assert outputs differ — verifies lat_mean actually flows into attention
- Add `test_cpb_backward_with_lat_coords`: verify gradients reach `cpb_mlp` parameters when `lat_coords` is provided
- Update `_build_net()` helper: no changes needed (lat_coords defaults to None)

### `fme/ace/registry/test_swin_transformer.py`

- Update dataset_info fixtures to use `LatLonCoordinates` (instead of img_shape-only DatasetInfo) so lat_coords gets extracted
- Add assertion that the built model's `layer1.blocks[0].attn.cpb_mlp` exists and `relative_position_bias_table` does not

### Run

```bash
python -m pytest fme/core/models/swin_transformer/ fme/ace/registry/test_swin_transformer.py -x
```
