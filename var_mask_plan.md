Adds training-time input channel dropout to `SingleModuleStepConfig` so models can be trained to be robust to missing inputs. `VariableMaskingConfig` supports two modes: a uniform count-based mode and a per-variable independent Bernoulli mode. Dropout operates at the `network_call` level on the full channel tensor (including any appended extra channels such as global-mean-removal inputs), so channel indices rather than variable names determine eligibility. If `include_channel_mask_inputs=True`, binary mask indicator channels are generated for every input channel (named and unnamed) and are merged with any existing `allow_missing_variables` masks.

## Configuration options

### `VariableMaskingConfig`

Exactly one of the two sections must be provided:

- `uniform` (`UniformMaskingConfig`): mask a uniformly sampled count of channels per sample.
- `per_variable` (`PerVariableMaskingConfig`): mask each channel independently with a fixed Bernoulli rate.

Example — mask between 1 and 3 channels per sample:

```yaml
input_dropout:
  uniform:
    min_vars: 1
    max_vars: 3
```

Example — mask each channel independently with probability 0.0 (effectively disabled, useful as a placeholder):

```yaml
input_dropout:
  per_variable:
    rate: 0.0
```

### `UniformMaskingConfig`

Nested under `VariableMaskingConfig.uniform`. Masks a uniformly sampled count of channels per sample:

- `min_vars` (int or `"min"`, default `"min"`): minimum number of channels to mask. `"min"` resolves to `0`.
- `max_vars` (int or `"max"`, default `"max"`): maximum number of channels to mask. `"max"` resolves to the total number of input channels.

Each sample independently draws a random integer `n` from `[min_vars, max_vars]` and selects `n` channels uniformly at random.

### `PerVariableMaskingConfig`

Nested under `VariableMaskingConfig.per_variable`. Masks each channel independently:

- `rate` (float in `[0, 1]`, required): probability that any single channel is masked for a given sample.

### `SingleModuleStepConfig.input_dropout`

Set to `VariableMaskingConfig` or `null` to disable. Dropout is only active when the underlying PyTorch module is in training mode; it is automatically disabled during inference.

## Effect on training

Dropout is applied inside `network_call`, after the full input channel tensor (including appended extra channels) has been assembled but before the forward pass:

1. A per-sample boolean mask (`True` = present, `False` = masked) is sampled over all input channels, including appended channels such as global-mean-removal inputs.
2. Masked channels are zeroed in the input tensor.
3. If `include_channel_mask_inputs=True`, binary indicator channels (`0.0` = absent, `1.0` = present) are generated for every input channel (named and unnamed). These indicators are merged with any existing `allow_missing_variables` masks so all absence sources are handled uniformly.

Changes:
- `fme.core.var_masking.VariableMaskingConfig`: config with exactly one of `uniform` or `per_variable`.
- `fme.core.var_masking.UniformMaskingConfig`: uniform count-based channel masking with `min_vars` and `max_vars`.
- `fme.core.var_masking.PerVariableMaskingConfig`: independent Bernoulli masking with a single `rate`.
- `fme.core.step.single_module.SingleModuleStepConfig.input_dropout`: new optional `VariableMaskingConfig | None` field.
- `fme.core.step.single_module.network_call`: dropout sampling and channel zeroing moved here, applied to the full assembled input tensor including extra channels.
- `fme.core.step.single_module`: channel mask indicators now cover all input channels (named + unnamed) when `include_channel_mask_inputs=True`, merged with `allow_missing_variables` masks.
- `fme.ace.__init__`: exports `VariableMaskingConfig`, `UniformMaskingConfig`, and `PerVariableMaskingConfig`.

- [x] Tests added
- [x] No dependencies changed; deps-only image rebuild and `latest_deps_only_image.txt` update not needed.
