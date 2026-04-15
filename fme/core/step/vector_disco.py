"""VectorDisco step: scalar-vector spherical network with wind as vectors.

Wraps a VectorDiscoNetwork in the StepABC interface. Winds are treated
as vectors (not individual scalar channels), normalized by a single
speed scale with no bias. Coriolis and kinetic energy are pre-computed
and injected as additional scalar inputs.
"""

import dataclasses
import logging
import math
import re
from collections.abc import Callable
from typing import Any

import dacite
import torch
from torch import nn

from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.corrector.registry import CorrectorABC
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.models.vector_disco import VectorDiscoNetwork, VectorDiscoNetworkConfig
from fme.core.normalizer import NetworkAndLossNormalizationConfig, StandardNormalizer
from fme.core.ocean import Ocean, OceanConfig
from fme.core.optimization import NullOptimization
from fme.core.packer import Packer
from fme.core.registry import CorrectorSelector
from fme.core.step.args import StepArgs
from fme.core.step.step import StepABC, StepConfigABC, StepSelector
from fme.core.typing_ import TensorDict, TensorMapping


def _find_level_names(names: list[str], prefix: str) -> list[str]:
    """Find variable names matching ``prefix_0``, ``prefix_1``, … in order."""
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    matches: dict[int, str] = {}
    for name in names:
        m = pattern.match(name)
        if m:
            matches[int(m.group(1))] = name
    return [matches[k] for k in sorted(matches)]


def _compute_wind_scale(
    normalizer: StandardNormalizer,
    u_names: list[str],
    v_names: list[str],
) -> float:
    """Derive a single wind speed scale from per-variable normalizer stds.

    Combines u and v stds as a vector magnitude and averages across levels:
        wind_scale = mean_over_levels(sqrt(std_u_k² + std_v_k²))
    """
    total = 0.0
    K = len(u_names)
    for u_name, v_name in zip(u_names, v_names):
        std_u = normalizer.stds[u_name].item()
        std_v = normalizer.stds[v_name].item()
        total += math.sqrt(std_u**2 + std_v**2)
    return total / max(K, 1)


@StepSelector.register("vector_disco")
@dataclasses.dataclass
class VectorDiscoStepConfig(StepConfigABC):
    """Configuration for a VectorDisco stepper.

    Parameters:
        network: Configuration for the VectorDiscoNetwork.
        in_names: Names of all input variables.
        out_names: Names of all output variables.
        eastward_wind_name: Prefix for eastward wind levels (e.g.
            "eastward_wind" matches "eastward_wind_0", "eastward_wind_1", …).
        northward_wind_name: Prefix for northward wind levels.
        normalization: Normalization configuration for inputs/outputs.
        residual_prediction: Whether prognostic scalar variables are
            predicted as residuals (added to input). Diagnostic variables
            are always predicted full-field. Wind vectors always use
            residual prediction regardless of this flag.
        ocean: Optional ocean model configuration.
        corrector: Corrector configuration.
        next_step_forcing_names: Input-only variables from the output timestep.
    """

    network: VectorDiscoNetworkConfig
    in_names: list[str]
    out_names: list[str]
    normalization: NetworkAndLossNormalizationConfig
    eastward_wind_name: str = "eastward_wind"
    northward_wind_name: str = "northward_wind"
    residual_prediction: bool = False
    ocean: OceanConfig | None = None
    corrector: AtmosphereCorrectorConfig | CorrectorSelector | None = None
    next_step_forcing_names: list[str] = dataclasses.field(default_factory=list)

    @property
    def n_ic_timesteps(self) -> int:
        return 1

    def get_state(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state) -> "VectorDiscoStepConfig":
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    @property
    def _all_names(self) -> list[str]:
        return list(set(self.in_names).union(self.out_names))

    @property
    def input_names(self) -> list[str]:
        if self.ocean is None:
            return self.in_names
        return list(set(self.in_names).union(self.ocean.forcing_names))

    @property
    def output_names(self) -> list[str]:
        return self.out_names

    @property
    def next_step_input_names(self) -> list[str]:
        input_only = set(self.input_names).difference(self.output_names)
        result = set(input_only)
        if self.ocean is not None:
            result = result.union(self.ocean.forcing_names)
        return list(result)

    @property
    def diagnostic_names(self) -> list[str]:
        return list(set(self.output_names).difference(self.in_names))

    @property
    def loss_names(self) -> list[str]:
        return self.output_names

    def get_next_step_forcing_names(self) -> list[str]:
        return self.next_step_forcing_names

    def get_loss_normalizer(
        self,
        extra_names: list[str] | None = None,
        extra_residual_scaled_names: list[str] | None = None,
    ) -> StandardNormalizer:
        if extra_names is None:
            extra_names = []
        if extra_residual_scaled_names is None:
            extra_residual_scaled_names = []
        return self.normalization.get_loss_normalizer(
            names=self._all_names + extra_names,
            residual_scaled_names=self.prognostic_names + extra_residual_scaled_names,
        )

    def replace_ocean(self, ocean: OceanConfig | None):
        self.ocean = ocean

    def get_ocean(self) -> OceanConfig | None:
        return self.ocean

    def load(self):
        self.normalization.load()

    def get_step(
        self,
        dataset_info: DatasetInfo,
        init_weights: Callable[[list[nn.Module]], None],
    ) -> "VectorDiscoStep":
        logging.info("Initializing VectorDisco stepper from provided config")
        corrector = (
            self.corrector.get_corrector(dataset_info)
            if self.corrector is not None
            else None
        )
        normalizer = self.normalization.get_network_normalizer(self._all_names)
        return VectorDiscoStep(
            config=self,
            dataset_info=dataset_info,
            corrector=corrector,
            normalizer=normalizer,
            init_weights=init_weights,
        )


class VectorDiscoStep(StepABC):
    """Step class for a VectorDiscoNetwork.

    Separates wind variables from scalars. Winds are normalized by a
    single speed scale (no bias), paired into vectors, and fed to the
    network's vector path. Coriolis and kinetic energy are computed and
    appended to the scalar inputs.
    """

    def __init__(
        self,
        config: VectorDiscoStepConfig,
        dataset_info: DatasetInfo,
        corrector: CorrectorABC | None,
        normalizer: StandardNormalizer,
        init_weights: Callable[[list[nn.Module]], None],
    ):
        super().__init__()
        self._config = config
        self._normalizer = normalizer
        self._no_optimization = NullOptimization()

        # Identify wind variables by prefix
        u_in = _find_level_names(config.in_names, config.eastward_wind_name)
        v_in = _find_level_names(config.in_names, config.northward_wind_name)
        u_out = _find_level_names(config.out_names, config.eastward_wind_name)
        v_out = _find_level_names(config.out_names, config.northward_wind_name)
        if len(u_in) != len(v_in):
            raise ValueError(f"Mismatched wind levels: {len(u_in)} u vs {len(v_in)} v")
        if len(u_out) != len(v_out):
            raise ValueError(
                f"Mismatched output wind levels: {len(u_out)} u vs {len(v_out)} v"
            )
        n_levels = len(u_in)
        self._u_in_names = u_in
        self._v_in_names = v_in
        self._u_out_names = u_out
        self._v_out_names = v_out
        wind_names_in = set(u_in + v_in)
        wind_names_out = set(u_out + v_out)

        # Scalar names (everything except wind)
        scalar_in_names = [n for n in config.in_names if n not in wind_names_in]
        scalar_out_names = [n for n in config.out_names if n not in wind_names_out]
        self._scalar_in_packer = Packer(scalar_in_names)
        self._scalar_out_packer = Packer(scalar_out_names)

        # Wind speed scale from normalizer statistics
        self._wind_scale = _compute_wind_scale(normalizer, u_in, v_in)

        # Coriolis parameter: f = 2ω sin(lat), shape (1, 1, H, W)
        assert dataset_info.horizontal_coordinates is not None
        lat_grid, _ = dataset_info.horizontal_coordinates.meshgrid
        omega = 7.292e-5  # Earth rotation rate (rad/s)
        coriolis = (2.0 * omega * torch.sin(torch.deg2rad(lat_grid))).float()
        self.coriolis = coriolis.unsqueeze(0).unsqueeze(0).to(get_device())

        # Build ocean
        if config.ocean is not None:
            self.ocean: Ocean | None = config.ocean.build(
                config.in_names, config.out_names, dataset_info.timestep
            )
        else:
            self.ocean = None

        # Scalar input count: scalar fields + KE per level + Coriolis
        n_in_scalars = len(scalar_in_names) + n_levels + 1
        n_out_scalars = len(scalar_out_names)

        # Build network
        assert dataset_info.img_shape is not None
        network = VectorDiscoNetwork(
            config=config.network,
            n_in_scalars=n_in_scalars,
            n_out_scalars=n_out_scalars,
            n_in_vectors=n_levels,
            n_out_vectors=n_levels,
            img_shape=dataset_info.img_shape,
        )
        self.network = network.to(get_device())

        dist = Distributed.get_instance()
        init_weights(self.modules)
        self.network = dist.wrap_module(self.network)

        self._corrector = corrector
        self._img_shape = dataset_info.img_shape

    @property
    def config(self) -> VectorDiscoStepConfig:
        return self._config

    @property
    def normalizer(self) -> StandardNormalizer:
        return self._normalizer

    @property
    def surface_temperature_name(self) -> str | None:
        if self._config.ocean is not None:
            return self._config.ocean.surface_temperature_name
        return None

    @property
    def ocean_fraction_name(self) -> str | None:
        if self._config.ocean is not None:
            return self._config.ocean.ocean_fraction_name
        return None

    def prescribe_sst(
        self,
        mask_data: TensorMapping,
        gen_data: TensorMapping,
        target_data: TensorMapping,
    ) -> TensorDict:
        if self.ocean is None:
            raise RuntimeError("Ocean interface is missing.")
        return self.ocean.prescriber(mask_data, gen_data, target_data)

    @property
    def modules(self) -> nn.ModuleList:
        return nn.ModuleList([self.network])

    def _pack_wind_vectors(
        self, data: TensorMapping, u_names: list[str], v_names: list[str]
    ) -> torch.Tensor:
        """Pack u/v wind components into (B, K, H, W, 2) vector tensor."""
        us = [data[n] for n in u_names]
        vs = [data[n] for n in v_names]
        u = torch.stack(us, dim=1)  # (B, K, H, W)
        v = torch.stack(vs, dim=1)
        return torch.stack([u, v], dim=-1)  # (B, K, H, W, 2)

    def _unpack_wind_vectors(
        self, vectors: torch.Tensor, u_names: list[str], v_names: list[str]
    ) -> TensorDict:
        """Unpack (B, K, H, W, 2) vector tensor into per-variable dict."""
        result: TensorDict = {}
        for k, (u_name, v_name) in enumerate(zip(u_names, v_names)):
            result[u_name] = vectors[:, k, :, :, 0]
            result[v_name] = vectors[:, k, :, :, 1]
        return result

    def step(
        self,
        args: StepArgs,
        wrapper: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> TensorDict:
        input_data = args.input

        # ── Normalize scalars (standard per-variable normalization) ────────
        scalar_norm = self._normalizer.normalize(input_data)

        # ── Normalize winds (single scale, no bias) ───────────────────────
        wind_in = self._pack_wind_vectors(
            input_data, self._u_in_names, self._v_in_names
        )
        wind_norm = wind_in / self._wind_scale  # (B, K, H, W, 2)

        # ── Compute additional scalar inputs ──────────────────────────────
        B = wind_norm.shape[0]
        ke = 0.5 * (wind_norm[..., 0] ** 2 + wind_norm[..., 1] ** 2)  # (B, K, H, W)
        coriolis = self.coriolis.expand(B, 1, *self._img_shape)  # (B, 1, H, W)

        # ── Assemble scalar tensor ────────────────────────────────────────
        scalar_packed = self._scalar_in_packer.pack(scalar_norm, axis=-3)
        scalar_input = torch.cat(
            [scalar_packed, ke, coriolis], dim=-3
        )  # (B, N_in_s, H, W)

        # ── Network forward ───────────────────────────────────────────────
        scalar_out, vector_out = wrapper(self.network)(scalar_input, wind_norm)

        # ── Unpack scalar outputs and denormalize ─────────────────────────
        output_norm = self._scalar_out_packer.unpack(scalar_out, axis=-3)

        # Residual for prognostic scalars
        if self._config.residual_prediction:
            prognostic_scalar_names = [
                n
                for n in self._config.prognostic_names
                if n not in set(self._u_in_names + self._v_in_names)
            ]
            for name in prognostic_scalar_names:
                if name in scalar_norm and name in output_norm:
                    output_norm[name] = output_norm[name] + scalar_norm[name]

        output = self._normalizer.denormalize(output_norm)

        # ── Wind output: always residual, denormalize ─────────────────────
        wind_out_norm = wind_norm + vector_out  # residual in normalized space
        wind_out = wind_out_norm * self._wind_scale
        output.update(
            self._unpack_wind_vectors(wind_out, self._u_out_names, self._v_out_names)
        )

        # ── Corrector and ocean ───────────────────────────────────────────
        if self._corrector is not None:
            output = self._corrector(input_data, output, args.next_step_input_data)
        if self.ocean is not None:
            output = self.ocean(input_data, output, args.next_step_input_data)

        return output

    def get_regularizer_loss(self):
        return torch.tensor(0.0)

    def get_state(self):
        return {"network": self.network.state_dict()}

    def load_state(self, state: dict[str, Any]) -> None:
        self.network.load_state_dict(state["network"])
