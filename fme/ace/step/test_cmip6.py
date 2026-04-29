import dataclasses
import datetime

import torch

import fme
from fme.ace.step.cmip6 import Cmip6Step, Cmip6StepConfig
from fme.core.coordinates import LatLonCoordinates, NullVerticalCoordinate
from fme.core.dataset_info import DatasetInfo
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
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


def _get_dataset_info() -> DatasetInfo:
    device = fme.get_device()
    return DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.zeros(IMG_SHAPE[0], device=device),
            lon=torch.zeros(IMG_SHAPE[1], device=device),
        ),
        vertical_coordinate=NullVerticalCoordinate(),
        timestep=TIMESTEP,
    )


def _tensor_dict(names: list[str]) -> TensorDict:
    device = fme.get_device()
    return {name: torch.rand(N_SAMPLES, *IMG_SHAPE, device=device) for name in names}


def _get_cmip6_config(
    in_names: list[str],
    out_names: list[str],
    mask_variable_prefix: str = "below_surface_mask",
    residual_prediction: bool = False,
) -> Cmip6StepConfig:
    non_mask = [
        n
        for n in set(in_names) | set(out_names)
        if not n.startswith(mask_variable_prefix)
    ]
    normalization = _get_normalization(non_mask)
    return Cmip6StepConfig(
        builder=ModuleSelector(
            type="SphericalFourierNeuralOperatorNet",
            config={"scale_factor": 1, "embed_dim": 4, "num_layers": 2},
        ),
        in_names=in_names,
        out_names=out_names,
        normalization=normalization,
        mask_variable_prefix=mask_variable_prefix,
        residual_prediction=residual_prediction,
    )


def _build_step(config: Cmip6StepConfig) -> Cmip6Step:
    return config.get_step(_get_dataset_info(), init_weights=lambda _: None)


# ---------------------------------------------------------------------------
# Config-level tests
# ---------------------------------------------------------------------------


def test_mask_names_discovered():
    config = _get_cmip6_config(
        in_names=["ta1000", "below_surface_mask1000"],
        out_names=["ta1000", "below_surface_mask1000"],
    )
    assert config.mask_in_names == ["below_surface_mask1000"]
    assert config.mask_out_names == ["below_surface_mask1000"]


def test_mask_names_excluded_from_normalize_names():
    config = _get_cmip6_config(
        in_names=["ta1000", "below_surface_mask1000"],
        out_names=["ta1000", "below_surface_mask1000"],
    )
    assert "below_surface_mask1000" not in config._normalize_names
    assert "ta1000" in config._normalize_names


def test_no_mask_variables():
    config = _get_cmip6_config(
        in_names=["ta1000", "ua1000"],
        out_names=["ta1000", "ua1000"],
    )
    assert config.mask_in_names == []
    assert config.mask_out_names == []


# ---------------------------------------------------------------------------
# Step-level tests
# ---------------------------------------------------------------------------


def test_step_forward():
    """Basic forward pass produces all expected outputs."""
    config = _get_cmip6_config(
        in_names=["ta1000", "forcing", "below_surface_mask1000"],
        out_names=["ta1000", "below_surface_mask1000"],
    )
    step = _build_step(config)
    input_data = _tensor_dict(step.input_names)
    next_step = _tensor_dict(step.next_step_input_names)
    output = step.step(
        args=StepArgs(input=input_data, next_step_input_data=next_step, labels=None)
    )
    assert "ta1000" in output
    assert "below_surface_mask1000" in output
    assert output["ta1000"].shape == (N_SAMPLES, *IMG_SHAPE)


def test_mask_outputs_are_probabilities():
    """Mask outputs should be in [0, 1] (sigmoid applied)."""
    config = _get_cmip6_config(
        in_names=["ta1000", "below_surface_mask1000"],
        out_names=["ta1000", "below_surface_mask1000"],
    )
    step = _build_step(config)
    input_data = _tensor_dict(step.input_names)
    next_step = _tensor_dict(step.next_step_input_names)
    output = step.step(
        args=StepArgs(input=input_data, next_step_input_data=next_step, labels=None)
    )
    mask = output["below_surface_mask1000"]
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0


def test_residual_prediction_excludes_masks():
    """Residual prediction should apply to non-mask prognostics only."""
    config = _get_cmip6_config(
        in_names=["ta1000", "below_surface_mask1000"],
        out_names=["ta1000", "below_surface_mask1000"],
        residual_prediction=True,
    )
    step = _build_step(config)
    assert "ta1000" in step._non_mask_prognostic_names
    assert "below_surface_mask1000" not in step._non_mask_prognostic_names
    input_data = _tensor_dict(step.input_names)
    next_step = _tensor_dict(step.next_step_input_names)
    output = step.step(
        args=StepArgs(input=input_data, next_step_input_data=next_step, labels=None)
    )
    assert "ta1000" in output
    mask = output["below_surface_mask1000"]
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0


def test_selector_round_trip():
    """Config can be serialized via StepSelector and reconstructed."""
    config = _get_cmip6_config(
        in_names=["ta1000", "below_surface_mask1000"],
        out_names=["ta1000", "below_surface_mask1000"],
    )
    selector = StepSelector(type="cmip6", config=dataclasses.asdict(config))
    step = selector.get_step(_get_dataset_info(), init_weights=lambda _: None)
    assert isinstance(step, Cmip6Step)


def test_state_round_trip():
    """Step state can be saved and loaded."""
    config = _get_cmip6_config(
        in_names=["ta1000", "below_surface_mask1000"],
        out_names=["ta1000", "below_surface_mask1000"],
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


def test_nan_inputs_in_masked_regions_produce_valid_outputs():
    """NaN values in below-surface input cells should not poison outputs."""
    config = _get_cmip6_config(
        in_names=["ta1000", "ta850", "below_surface_mask1000", "below_surface_mask850"],
        out_names=[
            "ta1000",
            "ta850",
            "below_surface_mask1000",
            "below_surface_mask850",
        ],
    )
    step = _build_step(config)
    input_data = _tensor_dict(step.input_names)
    next_step = _tensor_dict(step.next_step_input_names)

    # Mark some cells as below-surface (mask=1) and put NaN in data there.
    mask_1000 = torch.zeros(N_SAMPLES, *IMG_SHAPE, device=fme.get_device())
    mask_1000[:, :2, :] = 1.0  # first two latitude rows are below surface
    input_data["below_surface_mask1000"] = mask_1000
    input_data["ta1000"] = input_data["ta1000"].clone()
    input_data["ta1000"][:, :2, :] = float("nan")

    mask_850 = torch.zeros(N_SAMPLES, *IMG_SHAPE, device=fme.get_device())
    input_data["below_surface_mask850"] = mask_850  # nothing masked at 850

    output = step.step(
        args=StepArgs(input=input_data, next_step_input_data=next_step, labels=None)
    )

    # No output should contain NaN — input NaN must not leak through the
    # network.  (Output masking is left to the loss / downstream consumer.)
    for name in output:
        assert not torch.isnan(output[name]).any(), name


def test_loss_normalizer_excludes_masks():
    """get_loss_normalizer should not include mask variables."""
    config = _get_cmip6_config(
        in_names=["ta1000", "below_surface_mask1000"],
        out_names=["ta1000", "below_surface_mask1000"],
    )
    normalizer = config.get_loss_normalizer()
    assert "below_surface_mask1000" not in normalizer._names
    assert "ta1000" in normalizer._names
