# Per-component learning rates and LR schedules for coupled training

`TrainConfig.optimization` in `fme/coupled` takes either today's single
`OptimizationConfig` or a mapping with one `OptimizationConfig` per component
(`atmosphere`, `ocean`), each building its own optimizer and LR scheduler.
`CoupledStepper.modules` stays a `ModuleList`, so model weights, EMA names and
single-form optimizer state in existing coupled checkpoints load unchanged.

Config, dict form:

```yaml
optimization:
  atmosphere:
    optimizer_type: AdamW
    lr: <lr_atmosphere>
    enable_automatic_mixed_precision: true
    max_grad_norm: <clip_atmosphere>
    scheduler: {type: CosineAnnealingLR}
  ocean:
    optimizer_type: Muon
    lr: <lr_ocean>
    kwargs: {adamw_lr: <adamw_lr_ocean>}
    enable_automatic_mixed_precision: true
    resume_optimizer_ckpt_path: <uncoupled ocean ckpt.tar>   # param names 0.<name> match uncoupled ACE
    scheduler: {type: ReduceLROnPlateau, kwargs: {factor: <f>, patience: <p>}}
```

Saved `ckpt.tar["optimization"]`:

```
single form:  {optimizer_state_dict, scheduler_state_dict, gscaler_state_dict}   # unchanged
dict form:    {components: {atmosphere: {optimizer_state_dict, scheduler_state_dict},
                            ocean:      {optimizer_state_dict, scheduler_state_dict}},
               gscaler_state_dict}
```

---

## `fme/core/generics/optimization.py` (modified)

```python
OptimizationModules: TypeAlias = nn.ModuleList | Mapping[str, nn.ModuleList]  # NEW

class OptimizationABC(abc.ABC):
    @property
    def learning_rates(self) -> dict[str, float]:  # NEW — per-component lr; {} by default
        ...
```

## `fme/core/generics/train_stepper.py` (modified)

```python
class TrainStepperABC:
    @property
    def optimization_modules(self) -> OptimizationModules:  # NEW — default self.modules; same params as modules
        ...
```

## `fme/core/optimization.py` (modified)

```python
def flatten_optimization_modules(modules: OptimizationModules) -> nn.ModuleList:  # NEW — concat mapping values in key order
    ...

def _step_scheduler(scheduler, valid_loss, is_iteration) -> None:  # NEW — lifted from Optimization.step_scheduler
    ...

def _load_optimizer_state_for_finetuning(optimizer, optimizer_state_dict) -> None:  # NEW — lifted from Optimization
    ...

def _read_optimization_state(checkpoint_path: str) -> dict[str, Any]:  # NEW — replaces _load_finetune_optimization_state
    ...

class Optimization:
    def load_state(self, state):  # CHANGED — raises ValueError on a per-component ("components") state
        ...

class OptimizationDict(OptimizationABC):  # NEW — one optimizer + scheduler per key, one backward, one shared GradScaler
    learning_rate: float                    # first component's lr
    learning_rates: dict[str, float]
    def set_learning_rate(self, lr): ...    # raises NotImplementedError (no lr_tuning, D5)
    def step_scheduler(self, valid_loss=None, is_iteration=False): ...  # steps each component's scheduler on its own step_each_iteration
    def step_weights(self): ...             # skips a component with no grads; clips each component's norm alone
    def load_optimizer_state_for_finetuning(self, key, state): ...  # single-form or per-component state into one key
    def get_state(self): ...                # {components: {...}, gscaler_state_dict}
    def load_state(self, state): ...        # raises on single-form state or on different component keys

class OptimizationConfig:
    def build_optimizer_and_scheduler(self, modules, max_epochs) -> tuple[Optimizer, LRScheduler]:  # NEW
        ...
    def build(self, modules: OptimizationModules, max_epochs) -> Optimization:  # CHANGED — a mapping flattens in key order
        ...

OptimizationConfigDict: TypeAlias = dict[str, OptimizationConfig]  # NEW

def validate_optimization_config_dict(configs, keys) -> None:  # NEW — exact keys; amp/accumulation/checkpoint agree
    ...

def has_lr_schedule(config: OptimizationConfig | OptimizationConfigDict) -> bool:  # NEW — any entry
    ...

def build_optimization_dict(configs, modules: Mapping[str, nn.ModuleList], max_epochs) -> OptimizationDict:  # NEW
    ...
```

### Critical detail — mixed precision with several optimizers

- One `GradScaler`: `unscale_(opt_c)` and `step(opt_c)` per component `c`, one
  `update()`. An inf in one component skips only that component's step.
- `GradScaler.step(opt_c)` raises when no parameter of `c` has a grad (frozen
  component, or a single-component batch that does not reach `c`), so such a
  component is skipped; `update()` is skipped when no component stepped.

## `fme/coupled/stepper.py` (modified)

```python
class CoupledTrainStepper:
    @property
    def optimization_modules(self) -> dict[str, nn.ModuleList]:  # NEW — {"atmosphere": ..., "ocean": ...}
        ...  # key order matches CoupledStepper.modules: the single form flattens to today's param order
```

## `fme/coupled/train/train_config.py` (modified)

```python
class TrainConfig:
    optimization: OptimizationConfig | OptimizationConfigDict  # CHANGED — single form first in the union

    def __post_init__(self): ...  # CHANGED — validate dict keys and shared fields; reject lr_tuning with the dict form
    def _get_optimization(self, modules: OptimizationModules) -> OptimizationABC: ...  # CHANGED — dispatch on form
```

## `fme/core/generics/trainer.py` (modified)

```python
class Trainer:
    def __init__(self, ..., build_optimization: Callable[[OptimizationModules], OptimizationABC], ...):  # CHANGED
        ...  # builds from stepper.optimization_modules; EMA and wandb.watch keep stepper.modules
    # train / train_one_epoch: CHANGED — also log lr/<component>

def _component_lr_logs(optimization: OptimizationABC) -> dict[str, float]:  # NEW
    ...
```

## `fme/core/generics/lr_tuning.py` (modified)

```python
def run_lr_tuning_trial(..., build_optimization: Callable[[OptimizationModules], OptimizationABC], ...):  # CHANGED
    ...  # baseline and candidate built from <stepper>.optimization_modules
```

## `fme/ace/train/train_config.py` (modified)

```python
def _get_optimization(self, modules: OptimizationModules) -> Optimization:  # CHANGED — annotation only
    ...
```

---

## Tests

## `fme/core/test_optimization.py` (modified)

```python
def test_optimization_dict_distinct_lr_and_schedules():
    # GOAL: AdamW + CosineAnnealingLR and Adam + StepLR each match a lone torch optimizer + scheduler every epoch.
def test_optimization_dict_matches_separate_optimizations():
    # GOAL: one step on a summed loss == one OptimizationConfig.build per component stepped alone.
def test_optimization_dict_state_round_trip():
    # GOAL: get_state -> fresh build -> load_state continues identically (weights and lr).
def test_optimization_dict_load_state_rejects_single_form(): ...
def test_optimization_load_state_rejects_dict_form(): ...
def test_optimization_dict_load_state_rejects_other_components(): ...
def test_optimization_dict_skips_component_without_grad():
    # PARAMETERIZE: scaler in {off, cpu GradScaler, cuda}; frozen component unchanged, other updated, no raise.
def test_optimization_dict_no_component_with_grad_does_not_raise():
    # PARAMETERIZE: {cpu GradScaler, cuda}; scale unchanged.
def test_optimization_dict_clips_per_component():
    # GOAL: a large-grad component does not shrink the other's clipped grad (read from Adam exp_avg).
def test_optimization_dict_set_learning_rate_not_supported(): ...
def test_optimization_dict_per_component_lr_trajectories():
    # PARAMETERIZE: atmosphere {constant, growing} x ocean decaying x {iteration, epoch} stepping per component;
    # each component's lr per batch matches its closed-form schedule.
def test_validate_optimization_config_dict_rejects():
    # PARAMETERIZE: missing key, unknown key, amp / accumulation / checkpoint differ.
def test_build_optimization_dict_resume_from_uncoupled_ckpt():
    # GOAL: per-entry resume_optimizer_ckpt_path restores that component's moments, keeps its lr, other fresh.
```

## `fme/coupled/train/test_train_config.py` (modified)

```python
def test_train_config_optimization_union_parse():
    # PARAMETERIZE: {} and {lr} -> OptimizationConfig; {atmosphere, ocean} -> dict of OptimizationConfig.
def test_train_config_optimization_union_hides_entry_error():
    # GOAL: documents that dacite's union reports an invalid entry as UnionMatchError.
def test_train_config_rejects_lr_tuning_with_per_component_form(): ...
def test_train_config_rejects_wrong_optimization_components(): ...
```

## `fme/coupled/test_stepper.py` (modified)

```python
def test_optimization_modules_cover_modules():
    # GOAL: keys atmosphere, ocean in order; flattened params == modules params in order.
def test_single_form_optimization_unchanged():
    # GOAL: OptimizationConfig.build from optimization_modules == from modules after two train_on_batch.
```

## `fme/coupled/test_train.py` (modified)

```python
def test_train_per_component_optimization():
    # GOAL: dict-form coupled training logs lr/atmosphere and lr/ocean following distinct StepLR schedules;
    # saves the per-component state, resumes, and continues the same schedules.
def test_train_per_component_lr_schedules():
    # PARAMETERIZE: atmosphere constant / growing, ocean decaying, mixed iteration / epoch stepping;
    # logged lr/atmosphere and lr/ocean per batch match each closed-form schedule.
```

## `fme/core/generics/test_trainer.py` (modified)

```python
# build_optimization helper and test_finetune_optimization_checkpoint_loads_optimizer_state
# narrow OptimizationModules / OptimizationABC with isinstance (Trainer.optimization is now OptimizationABC).
```

---

## Open Questions

- Old coupled checkpoint resumed under the dict form: refuse (implemented), convert by param-index split, or start fresh with a warning?
- `CoupledTrainStepper.optimization_modules` under the single form: always the mapping, flattened in key order (implemented), or `self.modules`?
- Per-entry fields: full `OptimizationConfig` with amp/accumulation/checkpoint required to agree (implemented), or a narrower per-entry class?
- Gradient clipping: per component (implemented) or one global norm?
- Logged `lr` and `lr_tuning`: decided — atmosphere's lr plus `lr/<component>`; `lr_tuning` rejected with the dict form; distinct per-component schedules required.
- Missing component key: raise (implemented), freeze it, or fall back to a default?
