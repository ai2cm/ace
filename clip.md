# Plan: Implement Gradient Clipping

## Context

`ace2-var-mask-nc-sfno-mask10-gmron-steps2-iid-nonoise` is crashing with `ValueError: Loss is NaN-valued during training` around step 200. The nonoise + 2-step autoregressive training combination is numerically less stable: step 2 is trained on the model's own step-1 predictions (which ran under `no_grad()`), and without noise conditioning there's no stochastic regularization. Gradient clipping is the standard fix for this kind of instability.

No gradient clipping exists anywhere in the codebase currently.

## Changes

### 1. `fme/core/optimization.py`

**`Optimization.__init__`** — add `max_grad_norm: float | None = None` parameter, store as `self._max_grad_norm`.

**`Optimization._step_weights()`** — insert clipping before the optimizer step:

```python
def _step_weights(self):
    if self._max_grad_norm is not None:
        if self.gscaler is not None:
            self.gscaler.unscale_(self.optimizer)  # must unscale before clipping with AMP
        torch.nn.utils.clip_grad_norm_(
            itertools.chain.from_iterable(
                g["params"] for g in self.optimizer.param_groups
            ),
            max_norm=self._max_grad_norm,
        )
    if self.gscaler is not None:
        self.gscaler.step(self.optimizer)
    else:
        self.optimizer.step()
```

Note: `gscaler.step()` is safe to call after `unscale_()` — it detects the call was already made and skips re-unscaling.

**`OptimizationConfig`** — add field:

```python
max_grad_norm: float | None = None
```

with a docstring entry explaining it clips the global gradient norm before each optimizer step. `None` disables clipping (default, backward compatible).

**`OptimizationConfig.build()`** — pass `max_grad_norm=self.max_grad_norm` to the `Optimization(...)` constructor.

### 2. Config files — enable clipping for nonoise runs

Add `max_grad_norm: 1.0` to the `optimization:` section of all 6 nonoise configs in `configs/experiments/2026-05-27-var-masking/`:

- `ace-train-config-4deg-AIMIP-nc-sfno-mask0-gmron-steps1-nonoise.yaml`
- `ace-train-config-4deg-AIMIP-nc-sfno-mask0-gmron-steps2-nonoise.yaml`
- `ace-train-config-4deg-AIMIP-nc-sfno-mask10-gmron-steps1-nonoise.yaml`
- `ace-train-config-4deg-AIMIP-nc-sfno-mask10-gmron-steps2-nonoise.yaml`
- `ace-train-config-4deg-AIMIP-nc-sfno-mask10-gmron-steps1-iid-nonoise.yaml`
- `ace-train-config-4deg-AIMIP-nc-sfno-mask10-gmron-steps2-iid-nonoise.yaml`

Do this in `generate_masking_configs.py`.

### 3. Tests — `fme/core/test_optimization.py`

Add a test verifying that gradient clipping actually clips large gradients:

```python
def test_gradient_clipping():
    # Set up a model with a parameter that would produce a large gradient
    # Configure Optimization with max_grad_norm=1.0
    # After accumulate_loss + step_weights, assert that param.grad.norm() <= 1.0
```

## Critical files

- `fme/core/optimization.py` — `Optimization.__init__`, `_step_weights`, `OptimizationConfig`, `OptimizationConfig.build`
- `fme/core/test_optimization.py` — new test
- `configs/experiments/2026-05-27-var-masking/ace-train-config-4deg-AIMIP-nc-sfno-*-nonoise.yaml` (6 files)

## Verification

Run `python -m pytest fme/core/test_optimization.py -x` to confirm the new test passes and existing tests are unaffected.
