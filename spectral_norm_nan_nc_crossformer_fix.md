# Fix: NaN model output in nc-crossformer

## Problem

`ConditionalLayerNorm.reset_parameters()` zero-initializes `W_scale_2d` and `W_bias_2d` (Conv2d) so noise conditioning starts as a no-op. Then `CrossFormer.__init__` calls `apply_spectral_norm(self)`, which walks every submodule and applies `nn.utils.spectral_norm` to all Conv2d layers — including these zero-weight ones.

PyTorch spectral norm computes `weight = weight_orig / sigma` where `sigma = u · W v`. With `W = 0`, power iteration gives `u = v = 0` and `sigma = 0`, so `weight = 0 / 0 = NaN`. Every subsequent `W_scale_2d(noise)` call returns NaN, the CLN scale becomes NaN, and the entire forward pass produces NaN outputs → NaN loss.

SFNO doesn't hit this because it never calls `apply_spectral_norm`. Plain CrossFormer doesn't hit this because it has no `ConditionalLayerNorm` layers (no zero-initialized Conv2d).

## Verification check (already added)

`EnsembleLoss.forward` now raises `"gen_norm contains NaN"` — confirmed firing, proving NaN is in the model output, not downstream in loss computation.

## Proposed fix

### Step 1 — Diagnostic check in `apply_spectral_norm`

Add a `ValueError` that fires if spectral norm is about to be applied to an all-zero weight. This confirms the hypothesis without changing behavior.

**File:** `fme/ace/models/miles_credit/crossformer.py`
```python
def apply_spectral_norm(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d | nn.Linear | nn.ConvTranspose2d):
            if not torch.any(module.weight != 0):
                raise ValueError(
                    f"apply_spectral_norm: {type(module).__name__} has all-zero "
                    "weights. Spectral norm computes sigma=0 → weight=0/0=NaN."
                )
            nn.utils.spectral_norm(module)
```

### Step 2 — Once confirmed, apply the real fix

Exclude CLN conditioning layers from spectral norm in `apply_spectral_norm`:

```python
def apply_spectral_norm(model):
    from fme.core.models.conditional_sfno.layers import ConditionalLayerNorm
    skip: set[nn.Module] = set()
    for m in model.modules():
        if isinstance(m, ConditionalLayerNorm):
            for attr in ("W_scale_2d", "W_bias_2d", "W_scale_pos", "W_bias_pos"):
                layer = getattr(m, attr, None)
                if layer is not None:
                    skip.add(layer)
    for module in model.modules():
        if module not in skip and isinstance(
            module, nn.Conv2d | nn.Linear | nn.ConvTranspose2d
        ):
            nn.utils.spectral_norm(module)
```

Also remove the temporary diagnostic checks added to `EnsembleLoss.forward` (loss.py) and `get_energy_score` (ensemble.py).

### Step 3 — Add a unit test

In `fme/ace/models/miles_credit/test_crossformer.py`, add a test that constructs `NoiseConditionedCrossFormer` with `use_spectral_norm=True`, runs one forward pass, and asserts no NaN in the output.

## Files changed

| File | Change |
|------|--------|
| `fme/ace/models/miles_credit/crossformer.py` | `apply_spectral_norm`: skip CLN conditioning layers |
| `fme/core/loss.py` | Remove temporary NaN check in `EnsembleLoss.forward` |
| `fme/core/ensemble.py` | Already reverted; no change needed |
| `fme/ace/models/miles_credit/test_crossformer.py` | Add no-NaN forward pass test with spectral norm |
