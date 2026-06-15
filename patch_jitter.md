# Patch Jitter for ACE Swin Transformer

Analysis of whether patch jitter from the Walrus paper (Polymathic AI, arxiv 2511.15684)
would benefit the ACE Swin Transformer.

---

## What Patch Jitter Does (Walrus)

Applies a random spatial roll to the input before encoding, with magnitude ≤ half the
patch size. For periodic boundaries: `torch.roll`. For non-periodic: learned padding
tokens at the boundary before rolling. The jitter is reversed exactly after decoding, so
inference remains deterministic. Motivation: stride-based downsampling always groups the
same spatial pixels together, and errors accumulate at those fixed locations across
autoregressive time steps.

Key implementation: `walrus/models/shared_utils/patch_jitterers.py`, `walrus/models/isotropic_model.py`

---

## Why the Motivation Transfers to ACE

ACE runs multi-step autoregressive rollouts, so the same error accumulation mechanism
applies. The `PatchMerging` layer (2×2 stride) always groups the same spatial pixels
together. Even though Swin alternates regular/shifted windows *within* each forward pass,
those window tilings reset identically at every time step — boundary-aligned errors can
compound across a long rollout.

---

## Why the Benefit Is Probably Smaller Than in Walrus

1. **Shifted windows are Swin's built-in answer**: Within each forward pass, two
   complementary window tilings cover every boundary. Walrus has no equivalent. This
   reduces (but doesn't eliminate) the across-step accumulation problem.

2. **Only one downsampling scale**: ACE downsamples 2× once. Walrus has a deeper
   multi-stride encoder with more opportunities for error to accumulate.

3. **No strided patch embedding**: ACE's initial `Conv2d(kernel=3, padding=1)` preserves
   full spatial resolution. The first processing layer doesn't create fixed downsampling
   artifacts.

---

## Complication: Lat/Lon Grid

Walrus treats space as uniform. ACE's Continuous Positional Bias (CPB) already does
lat-aware longitude scaling via `cos(lat)` in `WindowAttention2D`. A pixel-space jitter
shifts the model to a slightly different physical location, which the CPB partially but
not perfectly compensates for. A proper implementation should likely use:

- **Longitude jitter only** (periodic, `torch.roll`) — conceptually clean since longitude
  is periodic on a global grid
- Or a **small joint jitter** capped at `window_size // 2` in both dimensions so that
  the existing pad-to-`pad_mult` logic still ensures correct window alignment

---

## Empirical Check First

Before implementing, look at spatial maps of multi-step rollout error variance. If a
periodic artifact appears with spatial period ~8 lat × 16 lon (the window size), that is
direct evidence jitter would help. If error variance is spatially smooth, the shifted
windows are probably sufficient and jitter isn't worth the complexity.

---

## Implementation Sketch (if proceeding)

A `PatchJitter` module that wraps `SwinTransformerNet.forward()`:

```python
@dataclass
class PatchJitterConfig:
    enabled: bool = False
    max_shift_h: int | None = None   # default: window_size[0] // 2
    max_shift_w: int | None = None   # default: window_size[1] // 2
```

```python
class PatchJitter(nn.Module):
    def jitter(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        # random roll in both dims, return (jittered_x, shift)
        ...

    def unjitter(self, x: Tensor, shift: tuple[int, int]) -> Tensor:
        # reverse roll
        ...
```

In `SwinTransformerNet.forward()`:

```python
if self.training:
    x, shift = self.jitterer.jitter(x)
else:
    shift = (0, 0)

# ... full forward pass (Conv2d encoder → layers → decoder → crop) ...

x = self.jitterer.unjitter(x, shift)
```

Longitude: `torch.roll` (periodic). Latitude: reflect-pad + roll + crop (non-periodic).
Shift range: `[-max_shift, max_shift]`, default `max_shift = window_size // 2`.

The pad-to-`pad_mult` step already happens inside the forward pass after jitter, so
window partitioning alignment is unaffected.

---

## Files to Modify

- `fme/core/models/swin_transformer/swin_transformer.py` — `SwinTransformerNet`
- `fme/ace/registry/swin_transformer.py` — add `patch_jitter` field to config
- New file: `fme/core/models/swin_transformer/patch_jitter.py`
- New test: `fme/core/models/swin_transformer/test_patch_jitter.py`
