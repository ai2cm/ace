# Fix: NaN Loss in nc-crossformer Training

## Context

The `nc-crossformer` experiment crashes with `ValueError: Loss is NaN-valued during training.` after ~1.5 minutes into the first epoch. The error path is:

```
train_one_epoch → train_on_batch → _accumulate_loss → optimization.accumulate_loss → _validate_loss → ValueError
```

The NaN appears in the model's output (not in the loss computation itself), which then propagates through `EnsembleLoss` (CRPS + energy score).

**Root cause: no gradient clipping.** The `nc-crossformer` uses `NoiseConditionedCrossFormer` where every `Attention` and `FeedForward` block has `ConditionalLayerNorm` (CLN) with a learnable 2D noise-conditioned scale (`W_scale_2d`). Unlike the plain `CrossFormer` (which uses MSE and static `LayerNorm`), the nc variant:

1. Injects fresh N(0,1) noise of shape `(B, 256, H, W)` each forward pass, interpolated to each encoder stage.
2. Adds `W_scale_2d(noise)` (a 1×1 Conv2d) directly to the CLN scale, so the effective scale at every norm is `1 + W_scale_2d(noise)`.
3. Uses `EnsembleLoss` (CRPS 0.9 + spectral energy score 0.1), which can produce larger per-step gradient spikes than MSE.

Without gradient clipping, gradient spikes from the energy score term grow the CLN scale parameters, which amplifies feature magnitudes, which drives attention logits toward ±∞. Once any activation becomes inf, downstream ops (e.g. `0 * inf`, `inf - inf`) silently produce NaN model outputs, and the loss check fires.

The `Optimization` class (`fme/core/optimization.py`) does not support `max_grad_norm` at all — no config field, no call to `clip_grad_norm_`. This is the only change needed.

The regular `crossformer` is unaffected: its static `LayerNorm` is bounded, and MSE gradients are smoother, so it trains stably without clipping at the same LR.

---

## Implementation Plan

### 1. `fme/core/optimization.py` — add gradient clipping support

**`Optimization.__init__`**: accept and store `max_grad_norm: float | None = None`.

```python
def __init__(
    self,
    parameters: Iterable[torch.nn.Parameter],
    optimizer_type: ...,
    lr: float,
    max_epochs: int,
    scheduler: ...,
    enable_automatic_mixed_precision: bool,
    kwargs: Mapping[str, Any],
    use_gradient_accumulation: bool = False,
    get_checkpoint: ... = ...,
    max_grad_norm: float | None = None,   # NEW
):
    ...
    self._max_grad_norm = max_grad_norm
```

**`_step_weights`**: clip before optimizer step (and unscale first if AMP is active):

```python
def _step_weights(self):
    if self._max_grad_norm is not None:
        if self.gscaler is not None:
            self.gscaler.unscale_(self.optimizer)
        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(params, self._max_grad_norm)
    if self.gscaler is not None:
        self.gscaler.step(self.optimizer)
    else:
        self.optimizer.step()
```

**`OptimizationConfig`**: add optional field, wire it through to `build()`:

```python
@dataclasses.dataclass
class OptimizationConfig:
    ...
    max_grad_norm: float | None = None   # NEW
```

Find where `OptimizationConfig` is built into an `Optimization` instance (likely in `train.py` or a builder) and pass `max_grad_norm=self.max_grad_norm`.

### 2. `configs/experiments/2026-05-28-swin-transformer/ace-train-config-4deg-AIMIP-nc-crossformer.yaml`

Add one line under `optimization:`:

```yaml
optimization:
  max_grad_norm: 1.0      # NEW — prevent NaN from noise-conditioned CLN gradient spikes
  enable_automatic_mixed_precision: false
  lr: 0.0001
  ...
```

`max_grad_norm: 1.0` is the standard value used across attention-based climate models (e.g., Pangu, FuXi). It clips the global gradient norm before each optimizer step without affecting convergence when gradients are already small.

---

## Files to Change

| File | Change |
|------|--------|
| `fme/core/optimization.py` | Add `max_grad_norm` to `Optimization.__init__` and `OptimizationConfig`; call `clip_grad_norm_` in `_step_weights` |
| Wherever `OptimizationConfig.build()` constructs `Optimization` | Pass `max_grad_norm` through |
| `configs/experiments/2026-05-28-swin-transformer/ace-train-config-4deg-AIMIP-nc-crossformer.yaml` | Add `max_grad_norm: 1.0` |

---

## Verification

1. Run `python -m pytest fme/core/test_optimization.py` (or the relevant test file) — ensure existing tests pass.
2. Add a test: construct `Optimization` with `max_grad_norm=1.0`, call `accumulate_loss` + `step_weights` with a large gradient (manually set `param.grad = large_tensor`), assert `param.grad.norm() <= 1.0 + tol` after clipping.
3. Launch a short smoke run of nc-crossformer with `max_epochs: 1` and a single batch — confirm no NaN is raised in the first training step.
