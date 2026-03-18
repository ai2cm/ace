import dataclasses
import datetime
import pathlib
from collections.abc import Iterable
from typing import Any

import dacite
import pytest
import torch
import yaml

import fme
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.labels import LabelEncoding
from fme.core.rand import set_seed
from fme.core.registry.module import Module

from .module import CONDITIONAL_BUILDERS, ModuleConfig, ModuleSelector

DATA_DIR = pathlib.Path(__file__).parent / "testdata"


class MockModule(torch.nn.Module):
    def __init__(self, param_shapes: Iterable[tuple[int, ...]]):
        super().__init__()
        for i, shape in enumerate(param_shapes):
            setattr(self, f"param{i}", torch.nn.Parameter(torch.randn(shape)))


@ModuleSelector.register("mock")
@dataclasses.dataclass
class MockModuleBuilder(ModuleConfig):
    param_shapes: list[tuple[int, ...]]

    def build(self, n_in_channels, n_out_channels, dataset_info):
        return MockModule(self.param_shapes)

    @classmethod
    def from_state(cls, state):
        return dacite.from_dict(cls, state, config=dacite.Config(strict=True))

    def get_state(self):
        return {
            "param_shapes": self.param_shapes,
        }


def test_register():
    """Make sure that the registry is working as expected."""
    selector = ModuleSelector(type="mock", config={"param_shapes": [(1, 2, 3)]})
    dataset_info = DatasetInfo(img_shape=(16, 32))
    module = selector.build(
        n_in_channels=1, n_out_channels=1, dataset_info=dataset_info
    )
    assert isinstance(module, Module)
    assert isinstance(module.torch_module, MockModule)
    assert module._label_encoding is None


def test_build_conditional():
    """Make sure that the registry is working as expected."""
    try:
        CONDITIONAL_BUILDERS.append("mock")
        selector = ModuleSelector(
            type="mock", conditional=True, config={"param_shapes": [(1, 2, 3)]}
        )
        module = selector.build(
            n_in_channels=1,
            n_out_channels=1,
            dataset_info=DatasetInfo(all_labels={"a", "b"}, img_shape=(16, 32)),
        )
        assert isinstance(module, Module)
        assert isinstance(module.torch_module, MockModule)
        assert isinstance(module._label_encoding, LabelEncoding)
    finally:
        CONDITIONAL_BUILDERS.remove("mock")


def test_module_selector_raises_with_bad_config():
    with pytest.raises(dacite.UnexpectedDataError):
        ModuleSelector(type="mock", config={"non_existent_key": 1})


def get_dbc2925_ncsfno_module() -> tuple[ModuleSelector, Module]:
    img_shape = (9, 18)
    n_in_channels = 5
    n_out_channels = 6
    all_labels = {"a", "b"}
    timestep = datetime.timedelta(hours=6)
    device = fme.get_device()
    horizontal_coordinate = LatLonCoordinates(
        lat=torch.zeros(img_shape[0], device=device),
        lon=torch.zeros(img_shape[1], device=device),
    )
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7, device=device), bk=torch.arange(7, device=device)
    )
    dataset_info = DatasetInfo(
        horizontal_coordinates=horizontal_coordinate,
        vertical_coordinate=vertical_coordinate,
        timestep=timestep,
        all_labels=all_labels,
    )
    selector = ModuleSelector(
        type="NoiseConditionedSFNO",
        config={
            "embed_dim": 8,
            "noise_embed_dim": 4,
            "noise_type": "isotropic",
            "filter_type": "linear",
            "use_mlp": True,
            "num_layers": 4,
            "operator_type": "dhconv",
            "affine_norms": True,
            "spectral_transform": "sht",
        },
    )
    module = selector.build(
        n_in_channels=n_in_channels,
        n_out_channels=n_out_channels,
        dataset_info=dataset_info,
    )
    return selector, module


def get_noise_conditioned_sfno_module() -> tuple[ModuleSelector, Module]:
    img_shape = (9, 18)
    n_in_channels = 5
    n_out_channels = 6
    all_labels = {"a", "b"}
    timestep = datetime.timedelta(hours=6)
    device = fme.get_device()
    horizontal_coordinate = LatLonCoordinates(
        lat=torch.zeros(img_shape[0], device=device),
        lon=torch.zeros(img_shape[1], device=device),
    )
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7, device=device), bk=torch.arange(7, device=device)
    )
    dataset_info = DatasetInfo(
        horizontal_coordinates=horizontal_coordinate,
        vertical_coordinate=vertical_coordinate,
        timestep=timestep,
        all_labels=all_labels,
    )
    selector = ModuleSelector(
        type="NoiseConditionedSFNO",
        config={
            "embed_dim": 8,
            "noise_embed_dim": 4,
            "noise_type": "isotropic",
            "filter_type": "linear",
            "use_mlp": True,
            "num_layers": 4,
            "operator_type": "dhconv",
            "affine_norms": True,
            "spectral_transform": "sht",
        },
    )
    module = selector.build(
        n_in_channels=n_in_channels,
        n_out_channels=n_out_channels,
        dataset_info=dataset_info,
    )
    return selector, module


def load_state(selector_name: str) -> dict[str, torch.Tensor]:
    state_dict_path = DATA_DIR / f"{selector_name}_state_dict.pt"
    if not state_dict_path.exists():
        raise RuntimeError(
            f"State dict for {selector_name} not found at {state_dict_path}. "
            "Please make sure the checkpoint exists and is committed to the repo."
        )
    return torch.load(state_dict_path)


def load_or_cache_state(
    selector_name: str, module: Module, module_config: ModuleConfig | None = None
) -> dict[str, torch.Tensor]:
    state_dict_path = DATA_DIR / f"{selector_name}_state_dict.pt"
    if state_dict_path.exists():
        return torch.load(state_dict_path)
    else:
        state_dict = module.get_state()
        torch.save(state_dict, state_dict_path)
        raise AssertionError(
            f"State dict for {selector_name} not found. "
            f"Created a new one at {state_dict_path}. "
            "Please commit it to the repo and run the test again."
        )


def load_or_cache_module_config(
    selector_name: str, module_config: dict[str, Any]
) -> dict[str, Any]:
    module_config_path = DATA_DIR / f"{selector_name}_module_config.yaml"
    if module_config_path.exists():
        with open(module_config_path) as f:
            data = yaml.safe_load(f)
        return data
    else:
        with open(module_config_path, "w") as f:
            yaml.safe_dump(module_config, f)
        raise AssertionError(
            f"Module config for {selector_name} not found. "
            f"Created a new one at {module_config_path}. "
            "Please commit it to the repo and run the test again."
        )


FROZEN_BUILDERS = {
    "dbc2925_ncsfno": get_dbc2925_ncsfno_module,
}


@pytest.mark.parametrize(
    "selector_name",
    FROZEN_BUILDERS.keys(),
)
def test_frozen_module_backwards_compatibility(selector_name: str):
    """
    Backwards compatibility for frozen releases from specific commits.
    """
    set_seed(0)
    _, module = FROZEN_BUILDERS[selector_name]()
    loaded_state_dict = load_state(selector_name)
    module.load_state(loaded_state_dict)


LATEST_BUILDERS = {
    "NoiseConditionedSFNO": get_noise_conditioned_sfno_module,
}


@pytest.mark.parametrize(
    "selector_name",
    LATEST_BUILDERS.keys(),
)
def test_latest_module_backwards_compatibility(selector_name: str):
    """
    Backwards compatibility for the latest module implementations.

    Should be kept up-to-date with the latest code changes.
    """
    set_seed(0)
    selector, module = LATEST_BUILDERS[selector_name]()
    loaded_state_dict = load_or_cache_state(selector_name, module)
    module.load_state(loaded_state_dict)
    # check if config has new keys and fail so we update the checkpoint if it does
    module_config = dataclasses.asdict(selector.module_config)
    loaded_module_config = load_or_cache_module_config(selector_name, module_config)
    new_keys = set(module_config.keys()).difference(loaded_module_config.keys())
    assert not new_keys, (
        f"New keys {new_keys} were added to the module config of {selector_name}. "
        "If you want to ensure backwards compatibility of this new feature, "
        "you must update the configuration for this module to use that feature, "
        "then run this test to update the cached config and checkpoint, and "
        "commit those files to the repo. If you do not want to ensure backwards "
        "compatibility of this feature, you must still re-generate the checkpoint "
        "to remove this error. In either case update the checkpoint "
        "(and configuration) as its own isolated commit."
    )
