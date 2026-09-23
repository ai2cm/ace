import dataclasses
import datetime
import pathlib
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar
from unittest import mock

import dacite
import pytest
import torch
import yaml

import fme
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.labels import LabelEncoding
from fme.core.rand import set_seed
from fme.core.registry.module import Module, compile_torch_module
from fme.core.testing import dynamo_hygiene  # noqa: F401  autouse in this module

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

    @classmethod
    def remove_deprecated_keys(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        return dict(state)

    def build(self, n_in_channels, n_out_channels, dataset_info):
        return MockModule(self.param_shapes)

    def get_state(self):
        return {
            "param_shapes": self.param_shapes,
        }


@ModuleSelector.register("mock_with_default")
@dataclasses.dataclass
class MockModuleBuilderWithDefault(ModuleConfig):
    param_shapes: list[tuple[int, ...]]
    pad: str = "reflect"

    @classmethod
    def remove_deprecated_keys(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        return dict(state)

    def build(self, n_in_channels, n_out_channels, dataset_info):
        return MockModule(self.param_shapes)


@ModuleSelector.register("mock_with_deprecation")
@dataclasses.dataclass
class MockModuleBuilderWithDeprecation(ModuleConfig):
    """Mock builder whose hook drops one key and renames another."""

    param_shapes: list[tuple[int, ...]]
    new_name: str = "default"

    @classmethod
    def remove_deprecated_keys(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(state)
        result.pop("old_dropped_key", None)
        if "old_name" in result:
            result["new_name"] = result.pop("old_name")
        return result

    def build(self, n_in_channels, n_out_channels, dataset_info):
        return MockModule(self.param_shapes)


def test_remove_deprecated_keys_drops_and_renames():
    """A builder whose hook drops one deprecated key and renames another
    should build successfully, applying the renamed value."""
    config = {
        "param_shapes": [(2, 3)],
        "old_dropped_key": "garbage",
        "old_name": "renamed_value",
    }
    selector = ModuleSelector(type="mock_with_deprecation", config=config)
    module_config = selector.module_config
    assert isinstance(module_config, MockModuleBuilderWithDeprecation)
    assert module_config.new_name == "renamed_value"


def test_remove_deprecated_keys_preserves_selector_config():
    """ModuleSelector.config should be the raw dict passed in (after
    normalization to defaults), not the cleaned dict."""
    raw_config: dict[str, Any] = {
        "param_shapes": [(2, 3)],
        "old_dropped_key": "garbage",
        "old_name": "renamed_value",
    }
    selector = ModuleSelector(type="mock_with_deprecation", config=raw_config)
    # After __post_init__, selector.config is normalized from the built
    # dataclass (dataclasses.asdict), so it should contain the current field
    # names, not the deprecated ones.
    assert "old_dropped_key" not in selector.config
    assert "old_name" not in selector.config
    assert selector.config["new_name"] == "renamed_value"


def test_remove_deprecated_keys_does_not_mutate_input():
    """The hook must not mutate the input mapping."""
    original: dict[str, Any] = {
        "param_shapes": [(2, 3)],
        "old_dropped_key": "garbage",
        "old_name": "renamed_value",
    }
    original_copy = dict(original)
    ModuleSelector(type="mock_with_deprecation", config=original)
    assert original == original_copy


def test_module_selector_config_includes_defaults():
    """ModuleSelector.config should be normalized to include the default
    values of the built ModuleConfig, so that defaults are captured when the
    config is serialized (e.g. logged to wandb). See issue #596.
    """
    selector = ModuleSelector(
        type="mock_with_default", config={"param_shapes": [(1, 2, 3)]}
    )
    serialized_config = dataclasses.asdict(selector)["config"]
    assert serialized_config["pad"] == "reflect"


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
            "label_embed_dim": 3,
            "clip_latent_global_means": True,
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
    return torch.load(state_dict_path, map_location="cpu")


def load_or_cache_state(
    selector_name: str, module: Module, module_config: ModuleConfig | None = None
) -> dict[str, torch.Tensor]:
    state_dict_path = DATA_DIR / f"{selector_name}_state_dict.pt"
    if state_dict_path.exists():
        return torch.load(state_dict_path, map_location="cpu")
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


class _LinearNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.linear(x))


@pytest.mark.medium_duration
def test_module_compile_matches_uncompiled_and_keeps_state():
    """compile() changes only the forward callable: same outputs, same
    state-dict keys, same underlying torch module; wrapping preserves it.

    Uses the ``aot_eager`` backend so the test exercises dynamo tracing
    without paying inductor's codegen cost.
    """
    torch.manual_seed(0)
    net = _LinearNet().to(fme.get_device())
    module = Module(net, label_encoding=None)
    compiled = module.compile(backend="aot_eager")
    assert not module.is_compiled
    assert compiled.is_compiled
    assert compiled.torch_module is module.torch_module
    assert compiled.get_state().keys() == module.get_state().keys()

    x = torch.randn(3, 4, device=fme.get_device())
    torch.testing.assert_close(compiled(x), module(x))

    assert compiled.forward_module is not compiled.torch_module

    # a wrapper with an observable effect, so this checks the wrapper really
    # sits on the compiled forward callable rather than only that the
    # compiled flag survives wrapping
    wrapped = compiled.wrap_module(lambda m: (lambda *args: 2 * m(*args)))
    assert wrapped.is_compiled
    torch.testing.assert_close(wrapped(x), 2 * module(x))

    with pytest.raises(RuntimeError, match="before Module.compile"):
        compiled.to(fme.get_device())


@ModuleSelector.register("mock_compile_unsupported")
@dataclasses.dataclass
class MockModuleBuilderCompileUnsupported(ModuleConfig):
    """Mock builder that declares torch.compile unsupported."""

    compile_unsupported_reason: ClassVar[str] = (
        "the mock module's forward pass is not traceable"
    )

    param_shapes: list[tuple[int, ...]]

    @classmethod
    def remove_deprecated_keys(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        return dict(state)

    def build(self, n_in_channels, n_out_channels, dataset_info):
        return MockModule(self.param_shapes)


def _build_compile_unsupported_module() -> Module:
    selector = ModuleSelector(
        type="mock_compile_unsupported", config={"param_shapes": [(2, 3)]}
    )
    return selector.build(n_in_channels=1, n_out_channels=1, dataset_info=DatasetInfo())


def test_compile_unsupported_builder_raises_without_tracing():
    """A builder declaring compilation unsupported must fail fast: compile()
    raises with the declared reason and never reaches torch.compile, since
    tracing an untraceable module can take minutes before falling back."""
    module = _build_compile_unsupported_module()
    with mock.patch("torch.compile") as mock_compile:
        with pytest.raises(NotImplementedError, match="not traceable"):
            module.compile()
    mock_compile.assert_not_called()


def test_compile_unsupported_reason_survives_wrapping_and_to():
    """The reason must follow the Module through the transformations a step
    applies before compiling it (device placement, distributed wrapping)."""
    module = _build_compile_unsupported_module().to(fme.get_device())
    module = module.wrap_module(lambda m: m)
    with pytest.raises(NotImplementedError, match="not traceable"):
        module.compile()


@pytest.mark.medium_duration
def test_compile_sets_fail_on_recompile_limit_hit():
    """Compiling must opt out of dynamo's silent eager fallback, so that a
    compiled run which hits the recompile limit raises instead of quietly
    running uncompiled."""
    torch._dynamo.config.fail_on_recompile_limit_hit = False
    net = _LinearNet().to(fme.get_device())
    Module(net, label_encoding=None).compile(backend="aot_eager")
    assert torch._dynamo.config.fail_on_recompile_limit_hit is True


@pytest.mark.medium_duration
def test_compile_torch_module_compiles_and_sets_flag():
    torch._dynamo.config.fail_on_recompile_limit_hit = False
    net = _LinearNet().to(fme.get_device())
    compiled = compile_torch_module(net, backend="aot_eager")
    # isinstance is the assertion here: the contract of compile_torch_module is
    # that it returns torch.compile's wrapper type, not the original module.
    assert isinstance(compiled, torch._dynamo.eval_frame.OptimizedModule)
    assert compiled._orig_mod is net
    assert torch._dynamo.config.fail_on_recompile_limit_hit is True


@pytest.mark.parametrize("selector_name", sorted(ModuleSelector.get_available_types()))
def test_compile_unsupported_reason_is_not_a_dataclass_field(selector_name: str):
    """compile_unsupported_reason must be a ClassVar on every builder.

    If a builder annotated it without ClassVar it would become a dataclass
    field, and so would leak into ModuleSelector.config and into serialized
    checkpoint configs.
    """
    builder_cls = ModuleSelector.registry._types[selector_name]
    field_names = {field.name for field in dataclasses.fields(builder_cls)}
    assert "compile_unsupported_reason" not in field_names
