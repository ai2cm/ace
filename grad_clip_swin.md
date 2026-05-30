# Gradient Clipping Fix for NaN Loss in NoiseConditionedSwinTransformer

## Root Cause

Training crashes at step ~800 with `ValueError: Loss is NaN-valued during training.`

`Optimization.step_weights()` (`fme/core/optimization.py:185`) stores `self._max_grad_norm` in `__init__` but **never applies it** — the clipping code is absent from the method body. The config `max_grad_norm: 1.0` therefore has **no effect**, leaving gradients unclipped.

Over ~800 steps, gradient norms grow until weights go NaN → next forward pass produces NaN outputs → crash.

### Why noise-conditioned Swin fails but not regular Swin

Three factors compound the missing gradient clipping:

1. **Higher gradient variance from stochastic sampling** — each forward pass draws fresh N(0,1) noise. Different noise realizations produce different activations, so gradient norms random-walk upward faster than in a deterministic model.
2. **Extra gradient path through the noise embedding** (`noise_embed_dim: 32`) — an additional high-variance gradient source absent in regular Swin.
3. **EnsembleLoss gradients are noisier** — early in training the two ensemble members are nearly identical (model hasn't learned to use noise yet), so the diversity-incentivizing energy score gradient oscillates sign as members' relative ordering flips batch-to-batch.

---

## Fix 1 (primary, required): Implement gradient clipping in `step_weights()`

**File:** `fme/core/optimization.py`, `step_weights()` at line 185.

Add, between the backward call and `_step_weights()`:
```python
if self._max_grad_norm is not None:
    if self.gscaler is not None:
        self.gscaler.unscale_(self.optimizer)
    params = [p for group in self.optimizer.param_groups for p in group["params"]]
    total_norm = torch.nn.utils.clip_grad_norm_(params, self._max_grad_norm)
    if not torch.isfinite(total_norm):
        # Gradients are NaN/inf — skip this step to avoid corrupting weights
        self.optimizer.zero_grad()
        if self.gscaler is not None:
            self.gscaler.update()
        self._accumulated_loss = torch.tensor(0.0, device=get_device())
        return
```

The `torch.isfinite(total_norm)` guard prevents NaN gradients from being written into weights. This mirrors what PyTorch's `GradScaler.step()` does internally for AMP.

Note: when `gscaler` is not None, `unscale_()` must be called before `clip_grad_norm_()` and before `gscaler.step()`. The existing `_step_weights()` calls `gscaler.step()` which expects unscaling to have already happened.

---

## Fix 2 (secondary, optional): Change `_validate_loss` to warn-and-skip instead of crash

**File:** `fme/core/optimization.py`, `_validate_loss()` at line 277.

Once Fix 1 is in place, a NaN loss at `accumulate_loss` means the model already has corrupt weights. Rather than crashing, log a warning and skip accumulation + backward for that batch.

```python
def _validate_loss(self, loss: torch.Tensor) -> bool:
    with torch.no_grad():
        if torch.isnan(loss):
            logging.warning("Loss is NaN-valued; skipping this batch.")
            return False
    return True
```

`accumulate_loss` would check the return value and skip if False.

---

## Verification

1. Run a short training job and confirm gradient norms are being clipped (add a logging statement printing the norm each step).
2. Confirm training runs past step 800 without NaN crash.
3. Run `python -m pytest fme/core/test_optimization.py` to verify no regression.
