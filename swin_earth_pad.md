# Plan: Earth Padding for Swin Transformer

## Context

CrossFormer uses `TensorPadding` with `mode="earth"` to apply physically correct boundary
conditions: circular wrap for longitude and pole-reflection (180° shift + lat flip) for
latitude. The 4-degree AIMIP config expands 45×90 → 48×96 before the model and unpadding
after.

The goal is to add the same capability to both Swin Transformer builder variants
(`SwinTransformerBuilder`, `NoiseConditionedSwinTransformerBuilder`), following CrossFormer's
pattern exactly: `padding_conf` propagates to the model constructor, and `TensorPadding` is
stored as `self.padding_opt` on the model itself.

**Import constraint**: `fme/core` may not import from `fme/ace`. `TensorPadding` currently lives
in `fme/ace`, but it has zero `fme/ace` dependencies (pure torch). Moving it to `fme/core`
resolves the constraint while being the right long-term home for a shared geophysical utility.

---

## Approach: Mirror CrossFormer exactly

1. Move `TensorPadding` to `fme/core`; re-export from old location for backward compat.
2. Add `padding_conf` to `SwinTransformerNet.__init__`; apply pad/unpad inside `forward()`.
3. Add `padding_conf` to both builder dataclasses; pass it through to the model constructor.

This is structurally identical to CrossFormer: builder passes `padding_conf` to model, model
holds `self.padding_opt`, padding/unpadding happen inside `forward()`.

---

## Files to modify

### 1. `fme/core/models/boundary_padding.py` (new file)

Copy `TensorPadding` class verbatim from `fme/ace/models/miles_credit/boundary_padding.py`.

### 2. `fme/ace/models/miles_credit/boundary_padding.py` (backward-compat re-export)

Replace file contents with:
```python
from fme.core.models.boundary_padding import TensorPadding  # noqa: F401
```
CrossFormer's import continues to work unchanged.

### 3. `fme/core/models/swin_transformer/swin_transformer.py`

**`__init__` changes** — add `padding_conf: dict | None = None` parameter. After the existing
`window_size`/`pad_mult` lines, and before the current `padded_shape` calculation:

```python
from fme.core.models.boundary_padding import TensorPadding

# replace the existing H0, W0 = img_shape + padded_shape block with:
if padding_conf is None:
    padding_conf = {"activate": False}
self.use_padding = padding_conf["activate"]
if self.use_padding:
    self.padding_opt = TensorPadding(**padding_conf)
    pl = padding_conf["pad_lat"]
    pw = padding_conf["pad_lon"]
    H0 = img_shape[0] + pl[0] + pl[1]   # earth-padded dims drive layer sizing
    W0 = img_shape[1] + pw[0] + pw[1]
else:
    H0, W0 = img_shape

Hp = math.ceil(H0 / self.pad_mult[0]) * self.pad_mult[0]
Wp = math.ceil(W0 / self.pad_mult[1]) * self.pad_mult[1]
self.padded_shape = (Hp, Wp)

# Before the existing lat_full/lat_half block, prepend/append reflected rows:
if self.use_padding and lat_coords is not None:
    north = torch.flip(lat_coords[: pl[0]], dims=[0])
    south = torch.flip(lat_coords[-pl[1] :], dims=[0])
    lat_coords = torch.cat([north, lat_coords, south])
# existing lat_full / lat_half computation and all layer construction unchanged
```

**`forward()` changes** — earth-pad at start, earth-unpad at end:

```python
def forward(self, x, context=None):
    if self.use_padding:
        x = self.padding_opt.pad(x)

    _, _, H, W = x.shape        # earth-padded dims, e.g. 48x96
    Hp, Wp = self.padded_shape
    pad_h, pad_w = Hp - H, Wp - W
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))

    # CLN path: also earth-pad noise to match x
    if self.conditioning == "cln" and self.embed_dim_noise > 0:
        noise = context.noise
        if self.use_padding:
            noise = self.padding_opt.pad(noise)
        if pad_h > 0 or pad_w > 0:
            noise = F.pad(noise, (0, pad_w, 0, pad_h))
        noise_half = noise[..., ::2, ::2]
        ctx_full = dataclasses.replace(context, noise=noise)
        ctx_half = dataclasses.replace(context, noise=noise_half)

    # ... existing U-Net layers (layer1 ... layer4) unchanged ...

    x = x[..., :H, :W]          # crop to earth-padded shape (e.g. 48x96)

    if self.use_padding:
        x = self.padding_opt.unpad(x)   # restore original shape (e.g. 45x90)
    return x
```

### 4. `fme/ace/registry/swin_transformer.py`

Add `padding_conf: dict | None = None` field to both builder dataclasses.

In both `build()` methods, pass it to `SwinTransformerNet` — one extra line each:
```python
net = SwinTransformerNet(
    ...
    padding_conf=self.padding_conf,
)
```

`NoiseConditionedModel` still receives the original `dataset_info.img_shape` (unchanged) — it
generates noise at the original resolution, which is then earth-padded inside
`SwinTransformerNet.forward()`.

No changes to `_ContextWrappedModule` or `NoiseConditionedModel`.

### 5. `fme/core/models/swin_transformer/test_swin_transformer.py`

Add two tests (update `_build_cln_net` helper to accept optional `padding_conf`):

```python
def test_earth_padding_forward():
    """Earth padding: output shape equals the original (unpadded) img_shape."""
    img_shape = (9, 18)
    padding_conf = {"activate": True, "mode": "earth", "pad_lat": [2, 1], "pad_lon": [2, 2]}
    net = SwinTransformerNet(
        3, 3, img_shape, embed_dim=32, num_heads=(2, 4, 4, 2),
        window_size=(4, 4), mlp_ratio=2.0, drop_path_rate=0.0,
        padding_conf=padding_conf,
    ).to(get_device())
    x = torch.randn(2, 3, *img_shape, device=get_device())
    assert net(x).shape == (2, 3, *img_shape)


def test_earth_padding_cln_forward():
    """Earth padding with CLN noise conditioning: output shape preserved."""
    img_shape = (9, 18)
    padding_conf = {"activate": True, "mode": "earth", "pad_lat": [2, 1], "pad_lon": [2, 2]}
    net = _build_cln_net(3, 3, img_shape, padding_conf=padding_conf).to(get_device())
    noise = torch.randn(2, _EMBED_DIM_NOISE, *img_shape, device=get_device())
    ctx = Context(embedding_scalar=None, embedding_pos=None, labels=None, noise=noise)
    assert net(torch.randn(2, 3, *img_shape, device=get_device()), ctx).shape == (2, 3, *img_shape)
```

### 6. `fme/ace/registry/test_swin_transformer.py`

Add two builder-level tests:

```python
_PAD_CONF = {"activate": True, "mode": "earth", "pad_lat": [2, 1], "pad_lon": [2, 2]}


def test_swin_transformer_earth_padding():
    """Builder with earth padding returns output in original (unpadded) spatial shape."""
    module = _builder(padding_conf=_PAD_CONF).build(5, 3, _get_dataset_info()).to(fme.get_device())
    x = torch.randn(2, 5, *IMG_SHAPE, device=fme.get_device())
    assert module(x).shape == (2, 3, *IMG_SHAPE)


def test_nc_swin_transformer_earth_padding():
    """NoiseConditioned builder with earth padding returns output in original spatial shape."""
    module = _nc_builder(padding_conf=_PAD_CONF).build(5, 3, _get_dataset_info()).to(fme.get_device())
    x = torch.randn(2, 5, *IMG_SHAPE, device=fme.get_device())
    assert module(x).shape == (2, 3, *IMG_SHAPE)
```

---

## Verification

```bash
conda run -n fme python -m pytest \
    fme/core/models/swin_transformer/test_swin_transformer.py \
    fme/ace/registry/test_swin_transformer.py \
    fme/ace/models/miles_credit/test_crossformer.py \
    -v
```

All existing tests pass (no behavioral change when `padding_conf=None`). Four new tests pass.
CrossFormer tests confirm the re-export did not break anything.
