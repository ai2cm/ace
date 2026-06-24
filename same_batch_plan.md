# Plan: `same_mask_per_batch` option for input-dropout variable masking

## Context

This branch (`feature/var_masking_simple_same_batch`) adds training-only input
dropout, which masks (zeroes) input channels to improve robustness. Today every
batch member draws an **independent** mask — `sample_mask` returns a
`[batch_size, n_channels]` tensor where each row is sampled separately.

We want a config-level boolean: when `True`, all members of a batch share the
**same** masked variables; when `False` (default), keep the current per-sample
behavior. This lets experiments compare "whole batch sees the same corruption"
against "each sample sees its own."

## Approach

Add `same_mask_per_batch: bool = False` to **both** mask configs in
`fme/core/var_masking.py`:
- `UniformMaskingConfig`
- `PerVariableMaskingConfig`

Keep the two helpers (`_sample_uniform`, `_sample_per_variable`) **unchanged**.
Do the work in the `sample_mask` method: when `same_mask_per_batch` is `True`,
call the helper with `batch_size=1` (one count / one Bernoulli draw per channel),
producing `[1, n_channels]`, then `.repeat(batch_size, 1)` to the full batch.

`.repeat` makes a contiguous copy (no shared memory, no stride-0 view), so the
returned `[batch_size, n_channels]` mask is identical in shape/dtype/layout to
the per-sample path. Cost is trivial — the mask is a tiny bool tensor.

No change needed downstream — `make_input_dropout_mask`
(`fme/core/step/single_module.py:435`) and the training-layer ensemble broadcast
`_repeat_interleaved_tensor_mapping` (`fme/ace/stepper/single_module.py:485`)
operate on whatever `[batch_size, n_channels]` mask comes back. With the flag on,
the base batch is uniform and the ensemble repeat keeps it uniform — consistent.

The spatial-parallel `broadcast_spatial` call is orthogonal (cross-rank
agreement) and unaffected.

### Sketch (`fme/core/var_masking.py`)

Add field to both dataclasses:

```python
@dataclasses.dataclass
class UniformMaskingConfig:
    kind: Literal["uniform"] = "uniform"
    max_masked_vars: int | str = "max"
    same_mask_per_batch: bool = False
    # ...
```

```python
@dataclasses.dataclass
class PerVariableMaskingConfig:
    kind: Literal["per_variable"] = "per_variable"
    rate: float = 0.0
    same_mask_per_batch: bool = False
    # ...
```

In each `sample_mask`, sample one row and expand when the flag is set
(helpers stay as-is):

```python
def sample_mask(self, n_channels, batch_size, device):
    if self.same_mask_per_batch:
        mask = _sample_uniform(self, n_channels, 1, device)
        return mask.repeat(batch_size, 1)
    return _sample_uniform(self, n_channels, batch_size, device)
```

Same pattern in `PerVariableMaskingConfig.sample_mask` calling
`_sample_per_variable`.

## Files to modify

- `fme/core/var_masking.py` — add field to both configs; branch in both
  `sample_mask` methods. Helpers unchanged.

## Tests

Add to `fme/core/test_var_masking.py`:
- `same_mask_per_batch=True` → all rows of the returned mask identical, for both
  config kinds.
- `same_mask_per_batch=True` (uniform) → masked count identical across all rows.
- Default (`False`) → existing per-sample tests still pass (rows may differ).
- Shape/dtype unchanged (`[batch_size, n_channels]`, bool) in both modes.

(Optional) one step-level check in `fme/core/step/test_step.py` that a
`same_mask_per_batch=True` config yields the same dropped channels for every
batch member through `make_input_dropout_mask`.

## Verification

```
python -m pytest fme/core/test_var_masking.py
python -m pytest fme/core/step/test_step.py -k input_dropout
```

Run pre-commit (ruff, ruff-format, mypy):
```
pre-commit run --files fme/core/var_masking.py fme/core/test_var_masking.py
```
