import dataclasses
from typing import Any, Literal, Mapping, Optional, Protocol, Tuple, Type

import dacite
import torch
from modulus.models.sfno.sfnonet import SphericalFourierNeuralOperatorNet
from torch import nn

from fme.core.device import get_device


class ModuleConfig(Protocol):
    """
    A protocol for a class that can build a nn.Module given information about the input
    and output channels and the image shape.

    This is a "Config" as in practice it is a dataclass loaded directly from yaml,
    allowing us to specify details of the network architecture in a config file.
    """

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: Tuple[int, int],
    ) -> nn.Module:
        """
        Build a nn.Module given information about the input and output channels
        and the image shape.

        Args:
            n_in_channels: number of input channels
            n_out_channels: number of output channels
            img_shape: last two dimensions of data, corresponding to lat and
                lon when using FourCastNet conventions

        Returns:
            a nn.Module
        """
        ...

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "ModuleConfig":
        """
        Create a ModuleSelector from a dictionary containing all the information
        needed to build a ModuleConfig.
        """
        return cls(**state)


# this is based on the call signature of SphericalFourierNeuralOperatorNet at
# https://github.com/NVIDIA/modulus/blob/b8e27c5c4ebc409e53adaba9832138743ede2785/modulus/models/sfno/sfnonet.py#L292  # noqa: E501
@dataclasses.dataclass
class SphericalFourierNeuralOperatorBuilder(ModuleConfig):
    spectral_transform: str = "sht"
    filter_type: str = "non-linear"
    operator_type: str = "diagonal"
    scale_factor: int = 16
    embed_dim: int = 256
    num_layers: int = 12
    hard_thresholding_fraction: float = 1.0
    normalization_layer: str = "instance_norm"
    use_mlp: bool = True
    activation_function: str = "gelu"
    encoder_layers: int = 1
    pos_embed: bool = True
    big_skip: bool = True
    rank: float = 1.0
    factorization: Optional[str] = None
    separable: bool = False
    complex_network: bool = True
    complex_activation: str = "real"
    spectral_layers: int = 1
    checkpointing: int = 0
    data_grid: Literal["legendre-gauss", "equiangular"] = "legendre-gauss"

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: Tuple[int, int],
    ):
        sfno_net = SphericalFourierNeuralOperatorNet(
            params=self,
            in_chans=n_in_channels,
            out_chans=n_out_channels,
            img_shape=img_shape,
        )

        return sfno_net


@dataclasses.dataclass
class PreBuiltBuilder(ModuleConfig):
    """
    A simple module configuration which returns a pre-defined module.

    Used mainly for testing.
    """

    module: nn.Module

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: Tuple[int, int],
    ) -> nn.Module:
        return self.module


@dataclasses.dataclass
class ModuleSelector:
    """
    A dataclass containing all the information needed to build a ModuleConfig,
    including the type of the ModuleConfig and the data needed to build it.

    This is helpful as ModuleSelector can be serialized and deserialized
    without any additional information, whereas to load a ModuleConfig you
    would need to know the type of the ModuleConfig being loaded.

    It is also convenient because ModuleSelector is a single class that can be
    used to represent any ModuleConfig, whereas ModuleConfig is a protocol
    that can be implemented by many different classes.

    Attributes:
        type: the type of the ModuleConfig
        config: data for a ModuleConfig instance of the indicated type
    """

    type: Literal["SphericalFourierNeuralOperatorNet", "prebuilt", "BuilderWithWeights"]
    config: Mapping[str, Any]

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: Tuple[int, int],
    ) -> nn.Module:
        """
        Build a nn.Module given information about the input and output channels
        and the image shape.

        Args:
            n_in_channels: number of input channels
            n_out_channels: number of output channels
            img_shape: last two dimensions of data, corresponding to lat and
                lon when using FourCastNet conventions

        Returns:
            a nn.Module
        """
        return (
            NET_REGISTRY[self.type]
            .from_state(self.config)
            .build(
                n_in_channels=n_in_channels,
                n_out_channels=n_out_channels,
                img_shape=img_shape,
            )
        )

    def get_state(self) -> Mapping[str, Any]:
        """
        Get a dictionary containing all the information needed to build a ModuleConfig.
        """
        return {"type": self.type, "config": self.config}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "ModuleSelector":
        """
        Create a ModuleSelector from a dictionary containing all the information
        needed to build a ModuleConfig.
        """
        return dacite.from_dict(
            data_class=ModuleSelector, data=state, config=dacite.Config(strict=True)
        )


def _strip_leading_module(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Remove the leading "module." from the keys of a state dict.

    This is necessary because SingleModuleStepper wraps the module in either
    a DistributedDataParallel layer or DummyWrapper layer, which adds a leading
    "module." to the keys of the state dict.
    """
    return {
        k[len("module.") :] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }


@dataclasses.dataclass
class BuilderWithWeights(ModuleConfig):
    """
    A builder which initializes a model from another builder and loads weights
    from disk, and then initializes each parameter in the built model with
    the corresponding parameter in the loaded model.

    When the built model has a larger number of parameters than the loaded model,
    only the initial slice is initialized. For example, if the loaded model has
    a parameter `a` of shape [10, 10], and the built model has a parameter `a`
    of shape [20, 10], then only the first 10 rows of `a` will be initialized
    from the weights on disk.

    This is particularly helpful for fine-tuning a model, as it allows us to
    initialize a model with weights from a pre-trained model and then train
    the model on a new dataset potentially with new weights. For example, these
    weights could correspond to new inputs or output variables, or
    increased model resolution.

    Attributes:
        module: configuration to build the model
        weights_path: path to a SingleModuleStepper checkpoint
            containing weights to load
        allow_missing_parameters: if True, allow the built model to have new
            parameters not defined in the loaded model. The built model is still
            not allowed to be missing parameters defined in the loaded model.
    """

    module: ModuleSelector
    weights_path: str
    allow_missing_parameters: bool = False

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: Tuple[int, int],
    ) -> nn.Module:
        """
        Build a nn.Module given information about the input and output channels
        and the image shape.

        Args:
            n_in_channels: number of input channels
            n_out_channels: number of output channels
            img_shape: last two dimensions of data, corresponding to lat and lon

        Returns:
            a nn.Module
        """
        model = self.module.build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            img_shape=img_shape,
        )
        checkpoint = torch.load(self.weights_path, map_location=get_device())
        loaded_builder = ModuleSelector.from_state(
            checkpoint["stepper"]["config"]["builder"]
        )
        if "data_shapes" in checkpoint["stepper"]:
            # included for backwards compatibility
            data_shapes = checkpoint["stepper"]["data_shapes"]
            loaded_img_shape = data_shapes[list(data_shapes.keys())[0]][-2:]
        else:
            loaded_img_shape = checkpoint["stepper"]["img_shape"]
        loaded_model = loaded_builder.build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            img_shape=loaded_img_shape,
        )
        state_dict = _strip_leading_module(checkpoint["stepper"]["module"])
        loaded_model.load_state_dict(state_dict)

        _overwrite_weights(loaded_model, model)

        return model

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "ModuleConfig":
        """
        Create a ModuleSelector from a dictionary containing all the information
        needed to build a ModuleConfig.
        """
        state = dict(state)  # make a copy so we can modify it
        module_selector = ModuleSelector.from_state(state.pop("module"))
        return cls(
            module=module_selector,
            **state,
        )


def _set_nested_parameter(module, param_name, new_param):
    *path, name = param_name.split(".")
    for p in path:
        module = getattr(module, p)
    if not isinstance(new_param, nn.Parameter):
        new_param = nn.Parameter(new_param)
    setattr(module, name, new_param)


def _overwrite_weights(from_module: torch.nn.Module, to_module: torch.nn.Module):
    """
    Overwrite the weights in to_module with the weights in from_module.

    When an axis is larger in to_module than in from_module, only the initial
    slice is overwritten. For example, if the from module has a parameter `a`
    of shape [10, 10], and the to module has a parameter `a` of shape [20, 10],
    then only the first 10 rows of `a` will be overwritten.

    If an axis is larger in from_module than in to_module, an exception is raised.

    Args:
        from_module: module containing weights to be copied
        to_module: module whose weights will be overwritten
    """
    from_names = set(from_module.state_dict().keys())
    to_names = set(to_module.state_dict().keys())
    if not from_names.issubset(to_names):
        missing_parameters = from_names - to_names
        raise ValueError(
            f"Dest module is missing parameters {missing_parameters}, "
            "which is not allowed"
        )
    for name in from_names:
        from_param = from_module.state_dict()[name]
        to_param = to_module.state_dict()[name]
        if len(from_param.shape) != len(to_param.shape):
            raise ValueError(
                f"Dest parameter {name} has "
                f"{len(to_param.shape.shape)} "
                "dimensions which needs to be equal to the loaded "
                f"parameter dimension {len(from_param.shape)}"
            )
        for from_size, to_size in zip(from_param.shape, to_param.shape):
            if from_size > to_size:
                raise ValueError(
                    f"Dest parameter has size {to_size} along one of its "
                    "dimensions which needs to be greater than loaded "
                    f"parameter size {from_size}"
                )
        slices = tuple(slice(0, size) for size in from_param.shape)
        with torch.no_grad():
            new_param_data = to_param.data.clone()
            new_param_data[slices] = from_param.data
            _set_nested_parameter(to_module, name, new_param_data)


NET_REGISTRY: Mapping[str, Type[ModuleConfig]] = {
    "SphericalFourierNeuralOperatorNet": SphericalFourierNeuralOperatorBuilder,  # type: ignore  # noqa: E501
    "prebuilt": PreBuiltBuilder,
    "BuilderWithWeights": BuilderWithWeights,
}
