import dataclasses
import datetime
import pathlib
import tempfile

import pytest
import torch

import fme
from fme.ace.step.cmip6 import Cmip6Step, Cmip6StepConfig
from fme.ace.testing.fv3gfs_data import get_scalar_dataset
from fme.core.coordinates import LatLonCoordinates, NullVerticalCoordinate
from fme.core.dataset_info import DatasetInfo
from fme.core.labels import BatchLabels
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.per_source_normalizer import PerSourceNormalizationConfig
from fme.core.registry import ModuleSelector
from fme.core.step.args import StepArgs
from fme.core.step.step import StepSelector
from fme.core.typing_ import TensorDict

IMG_SHAPE = (16, 32)
TIMESTEP = datetime.timedelta(days=1)
N_SAMPLES = 2


def _get_normalization(
    names: list[str],
) -> NetworkAndLossNormalizationConfig:
    return NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(
            means={name: 0.0 for name in names},
            stds={name: 1.0 for name in names},
        ),
    )


def _get_dataset_info(all_labels: set[str] | None = None) -> DatasetInfo:
    device = fme.get_device()
    return DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.zeros(IMG_SHAPE[0], device=device),
            lon=torch.zeros(IMG_SHAPE[1], device=device),
        ),
        vertical_coordinate=NullVerticalCoordinate(),
        timestep=TIMESTEP,
        all_labels=all_labels,
    )


def _tensor_dict(names: list[str]) -> TensorDict:
    device = fme.get_device()
    return {name: torch.rand(N_SAMPLES, *IMG_SHAPE, device=device) for name in names}


def _get_cmip6_config(
    in_names: list[str],
    out_names: list[str],
    residual_prediction: bool = False,
    per_source_normalization: PerSourceNormalizationConfig | None = None,
) -> Cmip6StepConfig:
    all_names = list(set(in_names) | set(out_names))
    normalization = _get_normalization(all_names)
    return Cmip6StepConfig(
        builder=ModuleSelector(
            type="SphericalFourierNeuralOperatorNet",
            config={"scale_factor": 1, "embed_dim": 4, "num_layers": 2},
        ),
        in_names=in_names,
        out_names=out_names,
        normalization=normalization,
        per_source_normalization=per_source_normalization,
        residual_prediction=residual_prediction,
    )


def _get_cmip6_config_conditional(
    in_names: list[str],
    out_names: list[str],
    all_labels: set[str],
    residual_prediction: bool = False,
    per_source_normalization: PerSourceNormalizationConfig | None = None,
) -> Cmip6StepConfig:
    all_names = list(set(in_names) | set(out_names))
    normalization = _get_normalization(all_names)
    return Cmip6StepConfig(
        builder=ModuleSelector(
            type="NoiseConditionedSFNO",
            config={"embed_dim": 4, "num_layers": 2},
            conditional=True,
        ),
        in_names=in_names,
        out_names=out_names,
        normalization=normalization,
        per_source_normalization=per_source_normalization,
        residual_prediction=residual_prediction,
    )


def _build_step(
    config: Cmip6StepConfig, all_labels: set[str] | None = None
) -> Cmip6Step:
    return config.get_step(
        _get_dataset_info(all_labels=all_labels), init_weights=lambda _: None
    )


def test_step_forward():
    """Basic forward pass produces all expected outputs."""
    config = _get_cmip6_config(
        in_names=["ta1000", "forcing"],
        out_names=["ta1000"],
    )
    step = _build_step(config)
    input_data = _tensor_dict(step.input_names)
    next_step = _tensor_dict(step.next_step_input_names)
    output = step.step(
        args=StepArgs(input=input_data, next_step_input_data=next_step, labels=None)
    )
    assert "ta1000" in output
    assert output["ta1000"].shape == (N_SAMPLES, *IMG_SHAPE)


def test_residual_prediction():
    """Residual prediction runs without error."""
    config = _get_cmip6_config(
        in_names=["ta1000", "ua1000"],
        out_names=["ta1000", "ua1000"],
        residual_prediction=True,
    )
    step = _build_step(config)
    input_data = _tensor_dict(step.input_names)
    next_step = _tensor_dict(step.next_step_input_names)
    output = step.step(
        args=StepArgs(input=input_data, next_step_input_data=next_step, labels=None)
    )
    assert "ta1000" in output
    assert "ua1000" in output


def test_selector_round_trip():
    """Config can be serialized via StepSelector and reconstructed."""
    config = _get_cmip6_config(
        in_names=["ta1000", "ua1000"],
        out_names=["ta1000", "ua1000"],
    )
    selector = StepSelector(type="cmip6", config=dataclasses.asdict(config))
    step = selector.get_step(_get_dataset_info(), init_weights=lambda _: None)
    assert isinstance(step, Cmip6Step)


def test_state_round_trip():
    """Step state can be saved and loaded."""
    config = _get_cmip6_config(
        in_names=["ta1000", "ua1000"],
        out_names=["ta1000", "ua1000"],
    )
    step = _build_step(config)
    state = step.get_state()
    step2 = _build_step(config)
    step2.load_state(state)
    input_data = _tensor_dict(step.input_names)
    next_step = _tensor_dict(step.next_step_input_names)
    args = StepArgs(input=input_data, next_step_input_data=next_step, labels=None)
    torch.manual_seed(0)
    out1 = step.step(args=args)
    torch.manual_seed(0)
    out2 = step2.step(args=args)
    for name in out1:
        torch.testing.assert_close(out1[name], out2[name])


def test_deprecated_mask_variable_prefix_ignored():
    """Old configs with mask_variable_prefix can still load."""
    config = _get_cmip6_config(
        in_names=["ta1000"],
        out_names=["ta1000"],
    )
    state = dataclasses.asdict(config)
    state["mask_variable_prefix"] = "below_surface_mask"
    restored = Cmip6StepConfig.from_state(state)
    assert restored.in_names == ["ta1000"]


def test_per_source_normalization_requires_conditional_builder():
    """per_source_normalization without a conditional builder raises."""
    per_source = PerSourceNormalizationConfig(
        sources={
            "model_a": NormalizationConfig(
                means={"ta1000": 10.0}, stds={"ta1000": 2.0}
            ),
        }
    )
    with pytest.raises(ValueError, match="conditional builder"):
        _get_cmip6_config(
            in_names=["ta1000"],
            out_names=["ta1000"],
            per_source_normalization=per_source,
        )


def test_per_source_normalization_config_in_step():
    """Step with per-source normalization applies label-specific constants."""
    device = fme.get_device()
    per_source = PerSourceNormalizationConfig(
        sources={
            "model_a": NormalizationConfig(
                means={"ta1000": 10.0}, stds={"ta1000": 2.0}
            ),
        }
    )
    config = _get_cmip6_config_conditional(
        in_names=["ta1000"],
        out_names=["ta1000"],
        per_source_normalization=per_source,
        all_labels={"model_a"},
    )
    step = _build_step(config, all_labels={"model_a"})
    input_data = _tensor_dict(step.input_names)
    next_step = _tensor_dict(step.next_step_input_names)
    label_tensor = torch.ones(N_SAMPLES, 1, device=device)
    labels = BatchLabels(tensor=label_tensor, names=["model_a"])
    output = step.step(
        args=StepArgs(input=input_data, next_step_input_data=next_step, labels=labels)
    )
    assert "ta1000" in output
    assert output["ta1000"].shape == (N_SAMPLES, *IMG_SHAPE)


def test_per_source_normalization_from_directory():
    """PerSourceNormalizationConfig loads from directory and wires into step."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        norm_dir = tmp_path / "per_source_normalization"
        label_dir = norm_dir / "model_a"
        label_dir.mkdir(parents=True)
        get_scalar_dataset(["ta1000"], fill_value=5.0).to_netcdf(
            label_dir / "centering.nc"
        )
        get_scalar_dataset(["ta1000"], fill_value=3.0).to_netcdf(
            label_dir / "scaling.nc"
        )
        per_source = PerSourceNormalizationConfig(data_dir=str(tmp_path))
        per_source.load()

    config = _get_cmip6_config_conditional(
        in_names=["ta1000"],
        out_names=["ta1000"],
        per_source_normalization=per_source,
        all_labels={"model_a"},
    )
    step = _build_step(config, all_labels={"model_a"})
    assert "model_a" in step._per_source_normalizer._normalizers


def test_allow_missing_variables_delegates_to_builder():
    """``Cmip6StepConfig.allow_missing_variables`` reads off the
    builder's flag, mirroring SingleModuleStepConfig. The trainer
    keys on this to decide whether the data loader can emit
    masked-variable batches — needed for multi-model cohorts where
    sub-universal outputs (radiation / heat-flux diagnostics) are
    absent from a substantial fraction of datasets."""
    in_names = ["ta1000"]
    out_names = ["ta1000"]
    normalization = _get_normalization(in_names)

    default_config = Cmip6StepConfig(
        builder=ModuleSelector(
            type="SphericalFourierNeuralOperatorNet",
            config={"scale_factor": 1, "embed_dim": 4, "num_layers": 2},
        ),
        in_names=in_names,
        out_names=out_names,
        normalization=normalization,
    )
    assert default_config.allow_missing_variables is False

    permissive_config = Cmip6StepConfig(
        builder=ModuleSelector(
            type="SphericalFourierNeuralOperatorNet",
            config={"scale_factor": 1, "embed_dim": 4, "num_layers": 2},
            allow_missing_variables=True,
        ),
        in_names=in_names,
        out_names=out_names,
        normalization=normalization,
    )
    assert permissive_config.allow_missing_variables is True
