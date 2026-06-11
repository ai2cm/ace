# n_ic_timesteps=2 Issue

## Root Cause

The config validation fails because `n_ic_timesteps: 2` appears in the YAML under
`stepper.step.config`, but `SingleModuleStepConfig` (`fme/core/step/single_module.py:87`)
does **not** have this as a field — it only has it as a hardcoded property:

```python
@property
def n_ic_timesteps(self) -> int:
    return 1
```

When dacite deserializes the YAML into `TrainConfig`, it tries to match the `stepper` dict
against the union `StepperConfig | CheckpointStepperConfig`. For `StepperConfig`, it
recursively tries to build `StepSelector` from `{type: "single_module", config: {...}}`,
which calls `SingleModuleStepConfig.from_state(config_dict)` via the registry. That call
uses `dacite.from_dict(..., config=dacite.Config(strict=True))` — and since `n_ic_timesteps`
is **not a declared field** of `SingleModuleStepConfig`, `strict=True` causes
`StrictUnexpectedDataError`. Dacite treats this as "StepperConfig doesn't match" and since
`CheckpointStepperConfig` also doesn't match (it only has `checkpoint_path`), the whole
union fails with `UnionMatchError`.

There is also a secondary bug in `validate_config.py:59`: the `except UnionMatchError`
handler unconditionally re-raises the original error (`raise err`) even after successfully
validating the stepper as a `StepperConfig` directly, so the fallback diagnostic path never
suppresses the error.

## What the feature is supposed to do

Per `swin_diff.md` §6, `n_ic_timesteps: 2` enables **2-step input** (ArchesWeather's
`use_prev=True`): concatenate the previous state with the current state as input channels to
the network. `swin_diff.md` notes this "is currently hardcoded to 1 as a property on
`SingleModuleStepConfig`… A small code change is required to make it a configurable field
(default 1)."

## Full scope of changes needed

The config fix alone (making dacite happy) is just one piece. For the training job to
actually work, three components need updates:

1. **`fme/core/step/single_module.py`** — replace the `n_ic_timesteps` property with a
   dataclass field `n_ic_timesteps: int = 1`, and multiply `n_in_channels` by
   `n_ic_timesteps` in `SingleModuleStep.__init__` so the builder receives the right number
   of input channels.

2. **`fme/core/step/args.py`** — add `history: list[TensorMapping] | None = None` to
   `StepArgs` so previous state(s) can flow into the step call.

3. **`fme/ace/stepper/single_module.py` (`predict_generator`)** — for `n_ic_timesteps > 1`,
   extract multiple IC states from `ic_dict` instead of squeezing the singleton time dim,
   maintain a history list across steps, and pass it in `StepArgs`. Inside
   `SingleModuleStep`'s `network_call`, normalize the history states (using
   `self._normalizer`), pack them into channel tensors, and concatenate them before the
   current-state channels before feeding the network.

### Key complication: normalization

`step_with_adjustments` normalizes `args.input` via the standard normalizer, but history
tensors need separate normalization using the same normalizer before being packed into
channels. Since the normalizer filters to known variable names, history channels cannot be
slipped in via the existing `input` dict — they must be handled explicitly inside
`network_call`, which has access to `self._normalizer` and `args.history` via closure.
