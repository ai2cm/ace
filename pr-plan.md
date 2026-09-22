# Add per-component `corrector_loss` to the coupled train stepper

`ComponentTrainingConfig` gains `corrector_loss: CorrectorLossConfig | None`,
the coupled counterpart of `TrainStepperConfig.corrector_loss`: each realm's
`StepLoss` becomes a `StepOutputLoss`, and the corrector deltas of each
component step reach it. With the field `None`, or a corrector emitting no
deltas, every realm's loss is exactly today's.

---

## `fme/coupled/stepper.py` (modified)

```python
class ComponentEnsembleStepPrediction:
    def __init__(
        self,
        realm: Literal["ocean", "atmosphere"],
        data: EnsembleTensorDict,
        step: int,
        deltas: EnsembleTensorDict,  # NEW — required, not defaulted: this is the seam where the component step's deltas would otherwise be dropped (as `ComponentStepPrediction.corrector_diagnostics` is). Empty when the corrector was inactive.
    ): ...

    @property
    def deltas(self) -> EnsembleTensorDict:  # NEW
        ...

    def detach_if_using_gradient_accumulation(self, optimizer) -> "ComponentEnsembleStepPrediction":  # CHANGED — deltas carried through, treated like data
        ...

    def detach(self) -> "ComponentEnsembleStepPrediction":  # CHANGED — deltas carried through, treated like data
        ...


class CoupledStepperTrainLoss:
    def __init__(
        self,
        ocean_loss: StepOutputLoss,       # CHANGED — was StepLoss
        atmosphere_loss: StepOutputLoss,  # CHANGED — was StepLoss
        ocean_schedule: ComponentLossSchedule,
        atmosphere_schedule: ComponentLossSchedule,
        optimize_single_component_per_batch: bool = False,
    ): ...

    _loss_objs: dict[str, StepOutputLoss]  # CHANGED

    def compute_loss(self, prediction, target_data) -> torch.Tensor:  # CHANGED — `self._loss_objs[realm](prediction.data, target_data, prediction.step, deltas=prediction.deltas)`
        ...

    def __call__(self, prediction, target_data) -> torch.Tensor | None:  # CHANGED — same call, same `.total()` (now `StepOutputLossOutput.total()`, main plus weighted penalty)
        ...


@dataclasses.dataclass
class ComponentTrainingConfig:
    loss: StepLossConfig
    n_steps: TimeLengthProbabilities | int | None = None
    optimize_last_step_only: bool = False
    loss_weight: float = 1.0
    parameter_init: ParameterInitializationConfig = ...
    corrector_loss: CorrectorLossConfig | None = None  # NEW — this component's consumption of its corrector's deltas, semantics of TrainStepperConfig.corrector_loss


class CoupledTrainStepperConfig:
    def _build_loss(self, stepper, n_coupled_steps) -> CoupledStepperTrainLoss:  # CHANGED — per realm: StepOutputLoss(stepper.<realm>.build_loss(self.<realm>.loss), stepper.<realm>.build_corrector_loss(self.<realm>.corrector_loss))
        ...


class CoupledTrainStepper:
    def _accumulate_step_loss(self, gen_step: ComponentStepPrediction, ...) -> None:  # CHANGED — builds the ensemble step with deltas=unfold_ensemble_dim(dict(gen_step.corrector_diagnostics.delta), n_ensemble)
        ...
```

### Critical detail — the seam

- Deltas come from `ComponentStepPrediction.corrector_diagnostics.delta`, the
  training-path carriage; never from the inference-only `StepDiagnostics`.
- Build-time validation of the selected names against each realm's
  `loss_names`, and the runtime check against the delta keys on the first
  non-empty delta, are `Stepper.build_corrector_loss` and
  `StepOutputLoss.forward` as they stand. Nothing coupled-specific is added.
- The coupled path passes no `data_mask`, as today.

---

## Tests

## `fme/coupled/test_loss.py` (modified)

```python
def _mock_step_loss(fn):  # CHANGED — Mock(spec=StepOutputLoss); side effect accepts the `deltas` kwarg and returns a StepOutputLossOutput
    ...

def step_and_target_gen(n_atmos_per_ocean=2):  # CHANGED — constructs ComponentEnsembleStepPrediction with empty deltas
    ...

def test_coupled_loss_forwards_each_realms_deltas():
    # GOAL: the ocean StepOutputLoss receives the ocean prediction's deltas and
    # the atmosphere StepOutputLoss the atmosphere's, in both __call__ and
    # compute_loss.
    ...
```

## `fme/coupled/test_stepper.py` (modified)

```python
# Build on get_train_stepper_and_batch and the corrector injection of
# _get_coupler_and_ic_for_step_diagnostics (ConstantOffsetCorrection on
# "sst" for the ocean, "a_prog" for the atmosphere). Mirror
# fme/ace/stepper/test_single_module.py::test_train_on_batch_pre_corrector_equivalence
# and the penalty assertion of its corrector-loss tests, once per realm.

def test_component_training_config_corrector_loss_round_trip():
    # GOAL: dacite.from_dict(CoupledTrainStepperConfig, ..., strict=True) with
    # `ocean.corrector_loss` and `atmosphere.corrector_loss` set lands each on
    # its component; with neither key, both are None.
    ...

@pytest.mark.parametrize("realm", ["ocean", "atmosphere"])
def test_train_on_batch_corrector_loss_inert(realm):
    # GOAL: with `corrector_loss` configured on `realm` and no corrector
    # installed, every per-step loss metric equals the unconfigured stepper's.
    ...

@pytest.mark.parametrize("realm", ["ocean", "atmosphere"])
def test_train_on_batch_pre_corrector_equivalence(realm):
    # GOAL: a constant-offset corrector on `realm` plus precorrector_optimization
    # of that variable gives `realm`'s step losses equal to a stepper with no
    # corrector, the other realm's losses untouched, and gen_data still
    # carrying the offset.
    ...

@pytest.mark.parametrize("realm", ["ocean", "atmosphere"])
def test_train_on_batch_corrector_penalty(realm):
    # GOAL: regularization on `realm` adds the weighted penalty of the offset to
    # that realm's step losses and to metrics["loss"]; the other realm's step
    # losses are unchanged.
    ...
```

---

## Open Questions

- `ComponentEnsembleStepPrediction.deltas` required (mirrors the
  `ComponentStepPrediction.corrector_diagnostics` precedent) or defaulted to
  empty, which leaves existing constructor sites in tests untouched?
