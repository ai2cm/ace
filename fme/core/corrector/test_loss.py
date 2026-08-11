import pytest
import torch

from fme.core.corrector.loss import (
    CorrectorLossConfig,
    CorrectorRegularizationConfig,
    PreCorrectorOptimizationConfig,
)
from fme.core.corrector.output import CorrectorOutput
from fme.core.corrector.registry import CorrectorABC
from fme.core.corrector.state import CorrectorState
from fme.core.loss import LossConfig, StepLossCorrectorArgs
from fme.core.normalizer import StandardNormalizer
from fme.core.typing_ import TensorMapping


class _StubCorrector(CorrectorABC):
    """A minimal CorrectorABC with settable modified/keep-gradient names."""

    def __init__(
        self,
        modified_names: frozenset[str] | None = None,
        keep_gradient_names: frozenset[str] = frozenset(),
    ):
        self._modified_names = modified_names
        self._keep_gradient_names = keep_gradient_names

    @property
    def modified_names(self) -> frozenset[str] | None:
        return self._modified_names

    @property
    def keep_gradient_names(self) -> frozenset[str]:
        return self._keep_gradient_names

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        forcing_data: TensorMapping,
        corrector_state: CorrectorState | None,
    ) -> CorrectorOutput:
        return CorrectorOutput(
            corrected=dict(gen_data), corrector_state=corrector_state
        )


def _normalizer(names) -> StandardNormalizer:
    return StandardNormalizer(
        means={name: torch.as_tensor(0.0) for name in names},
        stds={name: torch.as_tensor(1.0) for name in names},
    )


def _build(
    config: CorrectorLossConfig,
    corrector: CorrectorABC | None,
    prescribed_prognostic_names=(),
) -> StepLossCorrectorArgs:
    names: list[str] = []
    if corrector is not None and corrector.modified_names is not None:
        names = sorted(corrector.modified_names)
    return config.build(
        corrector,
        prescribed_prognostic_names=prescribed_prognostic_names,
        normalizer=_normalizer(names),
        gridded_operations=None,
    )


def test_config_post_init_errors():
    # GOAL: both features None; a present feature with names_and_prefixes=None;
    # weight <= 0; EnsembleLoss / NaN / global_mean_type — each raises in
    # __post_init__.
    with pytest.raises(ValueError, match="at least one"):
        CorrectorLossConfig()
    with pytest.raises(ValueError, match="names_and_prefixes"):
        PreCorrectorOptimizationConfig()
    with pytest.raises(ValueError, match="names_and_prefixes"):
        CorrectorRegularizationConfig()
    with pytest.raises(ValueError, match="weight"):
        CorrectorRegularizationConfig(names_and_prefixes=["a"], weight=0.0)
    with pytest.raises(ValueError, match="weight"):
        CorrectorRegularizationConfig(names_and_prefixes=["a"], weight=-1.0)
    with pytest.raises(ValueError, match="EnsembleLoss"):
        CorrectorRegularizationConfig(
            names_and_prefixes=["a"], loss=LossConfig(type="EnsembleLoss")
        )
    with pytest.raises(ValueError, match="NaN"):
        CorrectorRegularizationConfig(
            names_and_prefixes=["a"], loss=LossConfig(type="NaN")
        )
    with pytest.raises(ValueError, match="global_mean_type"):
        CorrectorRegularizationConfig(
            names_and_prefixes=["a"], loss=LossConfig(global_mean_type="LpLoss")
        )


@pytest.mark.parametrize("entry", ["missing_var", "missing_"])
def test_build_errors_on_entry_matching_no_modified_name(entry):
    # GOAL: an entry matching no corrector-modified name raises at build.
    corrector = _StubCorrector(modified_names=frozenset({"a", "b_0", "b_1"}))
    config = CorrectorLossConfig(
        precorrector_optimization=PreCorrectorOptimizationConfig(
            names_and_prefixes=[entry]
        )
    )
    with pytest.raises(ValueError, match=entry):
        _build(config, corrector)


def test_build_errors_without_corrector_or_discovery():
    # GOAL: corrector None, modified_names empty, and modified_names None
    # (discovery never ran) each raise with distinct messages.
    config = CorrectorLossConfig(
        precorrector_optimization=PreCorrectorOptimizationConfig(
            names_and_prefixes=["a"]
        )
    )
    with pytest.raises(ValueError, match="no corrector"):
        _build(config, None)
    with pytest.raises(ValueError, match="modifies no variables"):
        _build(config, _StubCorrector(modified_names=frozenset()))
    with pytest.raises(RuntimeError, match="discover_modified_names"):
        _build(config, _StubCorrector(modified_names=None))


def test_build_excludes_prescribed_prognostics():
    # GOAL: an entry matching only a prescribed prognostic name raises —
    # validation runs against the loss-visible set.
    corrector = _StubCorrector(modified_names=frozenset({"a", "b"}))
    config = CorrectorLossConfig(
        precorrector_optimization=PreCorrectorOptimizationConfig(
            names_and_prefixes=["a"]
        )
    )
    with pytest.raises(ValueError, match=r"\['a'\]"):
        _build(config, corrector, prescribed_prognostic_names={"a"})
    # the same entry is valid when the name is loss-visible
    args = _build(config, corrector)
    assert args.precorrector_names == ["a"]


@pytest.mark.parametrize("feature", ["precorrector_optimization", "regularization"])
@pytest.mark.parametrize("entry", ["sea_ice_0", "sea_ice_"])
def test_keep_gradient_selection_raises_at_build(feature, entry):
    # GOAL: an entry matching a keep_gradient name errors at build, for both
    # features — a straight-through clamp's delta is detached, so neither
    # feature can act on it.
    corrector = _StubCorrector(
        modified_names=frozenset({"sea_ice_0", "other"}),
        keep_gradient_names=frozenset({"sea_ice_0"}),
    )
    if feature == "precorrector_optimization":
        config = CorrectorLossConfig(
            precorrector_optimization=PreCorrectorOptimizationConfig(
                names_and_prefixes=[entry]
            )
        )
    else:
        config = CorrectorLossConfig(
            regularization=CorrectorRegularizationConfig(names_and_prefixes=[entry])
        )
    with pytest.raises(ValueError, match="keep-gradient"):
        _build(config, corrector)


def test_build_regularizer_packs_matched_names():
    # GOAL: the built WeightedMappingLoss packs exactly
    # selection.matched(loss_visible_names); a prefix entry matches all its
    # level names.
    corrector = _StubCorrector(
        modified_names=frozenset({"thetao_0", "thetao_1", "so_0", "zos"})
    )
    config = CorrectorLossConfig(
        regularization=CorrectorRegularizationConfig(
            names_and_prefixes=["thetao_", "zos"], weight=2.0
        )
    )
    args = _build(config, corrector)
    assert args.precorrector_names is None
    assert args.regularization_weight == 2.0
    assert args.regularizer is not None
    assert list(args.regularizer.packer.names) == ["thetao_0", "thetao_1", "zos"]
