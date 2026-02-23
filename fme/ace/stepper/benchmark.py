import dataclasses
from datetime import timedelta
from typing import Self

import numpy as np
import torch

from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNOBuilder
from fme.ace.stepper.single_module import Stepper, StepperConfig
from fme.core.benchmark.benchmark import BenchmarkABC, register_benchmark
from fme.core.benchmark.timer import Timer
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.labels import BatchLabels
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.registry.module import ModuleSelector
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepABC, StepSelector
from fme.core.typing_ import TensorDict


def get_stepper(
    stepper_config: StepperConfig,
    img_shape: tuple[int, int],
    all_labels: set[str] | None,
    n_levels: int,
) -> StepABC:
    device = get_device()
    lat_edges = torch.linspace(0, 180, img_shape[0] + 1, device=device)
    lon_edges = torch.linspace(0, 360, img_shape[1] + 1, device=device)
    # equiangular is fine for just benchmarking,
    # even if the code grid is not equiangular
    lat = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon = (lon_edges[:-1] + lon_edges[1:]) / 2
    horizontal_coordinate = LatLonCoordinates(
        lat=lat,
        lon=lon,
    )
    vertical_coordinate = HybridSigmaPressureCoordinate(
        # pressure = ak * surface_pressure + bk
        # ak from 0 to 1 to 0
        ak=torch.sin(torch.arange(n_levels + 1, device=device) * np.pi / n_levels),
        # bk 1 at surface, 0 at top, evenly spaced in between
        bk=1 - torch.arange(n_levels + 1, device=device) / n_levels,
    )
    dataset_info = DatasetInfo(
        horizontal_coordinates=horizontal_coordinate,
        vertical_coordinate=vertical_coordinate,
        timestep=timedelta(hours=6),
        all_labels=all_labels,
    )
    return stepper_config.get_stepper(dataset_info)


class CSFNOPredict(BenchmarkABC):
    def __init__(
        self,
        stepper: Stepper,
        initial_condition: PrognosticState,
        forcing: BatchData,
        compute_derived_variables: bool,
        compute_derived_forcings: bool,
    ):
        self.stepper = stepper
        self.initial_condition = initial_condition
        self.forcing = forcing
        self.compute_derived_variables = compute_derived_variables
        self.compute_derived_forcings = compute_derived_forcings

    def run_instance(self, timer: Timer) -> TensorDict:
        with torch.no_grad():
            self.stepper.predict(
                initial_condition=self.initial_condition,
                forcing=self.forcing,
                compute_derived_variables=self.compute_derived_variables,
                compute_derived_forcings=self.compute_derived_forcings,
                timer=timer,
            )
        return {}  # no regression implemented yet

    @classmethod
    def new(cls) -> Self:
        B = 2
        C = 384
        H = 180
        L = 360
        G = 1
        conditional_embed_dim_noise = 32
        conditional_embed_dim_labels = 3
        conditional_embed_dim_pos = 16
        return cls._new_with_params(
            B=B,
            C=C,
            H=H,
            L=L,
            G=G,
            embed_dim_noise=conditional_embed_dim_noise,
            embed_dim_pos=conditional_embed_dim_pos,
            num_blocks=8,
            vertical_gridded_names=[
                "air_temperature",
                "specific_total_water",
                "u",
                "v",
            ],
            sfc_in_names=[
                "land_fraction",
                "ocean_fraction",
                "sea_ice_fraction",
                "DSWRFtoa",
                "HGTsfc",
                "global_mean_co2",
                "PRESsfc",
                "surface_temperature",
                "TMP2m",
                "Q2m",
                "UGRD10m",
                "VGRD10m",
            ],
            sfc_out_names=[
                "PRESsfc",
                "surface_temperature",
                "TMP2m",
                "Q2m",
                "UGRD10m",
                "VGRD10m",
                "LHTFLsfc",
                "SHTFLsfc",
                "PRATEsfc",
                "ULWRFsfc",
                "ULWRFtoa",
                "DLWRFsfc",
                "DSWRFsfc",
                "USWRFsfc",
                "USWRFtoa",
                "tendency_of_total_water_path_due_to_advection",
                "TMP850",
                "h500",
            ],
            n_labels=conditional_embed_dim_labels,
            n_levels=8,
            n_forward_steps=10,
        )

    @classmethod
    def _new_with_params(
        cls,
        B: int,
        C: int,
        H: int,
        L: int,
        G: int,
        embed_dim_noise: int,
        embed_dim_pos: int,
        num_blocks: int,
        vertical_gridded_names: list[str],
        sfc_in_names: list[str],
        sfc_out_names: list[str],
        n_labels: int,
        n_levels: int,
        n_forward_steps: int,
    ) -> Self:
        expanded_vertical_names = []
        for name in vertical_gridded_names:
            for level in range(n_levels):
                expanded_vertical_names.append(f"{name}_{level}")
        stepper = cls._get_stepper(
            embed_dim=C,
            embed_dim_noise=embed_dim_noise,
            embed_dim_pos=embed_dim_pos,
            num_blocks=num_blocks,
            expanded_vertical_names=expanded_vertical_names,
            sfc_in_names=sfc_in_names,
            sfc_out_names=sfc_out_names,
            n_labels=n_labels,
            n_levels=n_levels,
            H=H,
            L=L,
            G=G,
        )
        if n_labels > 0:
            labels = BatchLabels(
                tensor=torch.zeros((B, n_labels), device=get_device()),
                names=[f"label_{i}" for i in range(n_labels)],
            )
        else:
            labels = None
        initial_condition = PrognosticState(
            BatchData.new_for_testing(
                names=expanded_vertical_names + sfc_in_names,
                n_samples=B,
                n_timesteps=1,
                img_shape=(H, L),
                horizontal_dims=["lat", "lon"],
                labels=labels,
            )
        )
        forcing = BatchData.new_for_testing(
            names=sfc_in_names,
            n_samples=B,
            n_timesteps=n_forward_steps + 1,
            img_shape=(H, L),
            horizontal_dims=["lat", "lon"],
            labels=labels,
        )
        return cls(
            stepper=stepper,
            initial_condition=initial_condition,
            forcing=forcing,
            compute_derived_variables=True,
            compute_derived_forcings=False,
        )

    @staticmethod
    def _get_stepper(
        embed_dim: int,
        embed_dim_noise: int,
        embed_dim_pos: int,
        num_blocks: int,
        expanded_vertical_names: list[str],
        sfc_in_names: list[str],
        sfc_out_names: list[str],
        n_labels: int,
        n_levels: int,
        H: int = 180,
        L: int = 360,
        G: int = 16,
    ):
        csfno_config = NoiseConditionedSFNOBuilder(
            embed_dim=embed_dim,
            noise_embed_dim=embed_dim_noise,
            context_pos_embed_dim=embed_dim_pos,
            pos_embed=False,
            num_layers=num_blocks,
            normalize_big_skip=True,
            affine_norms=True,
            filter_num_groups=G,
        )
        all_names = set(expanded_vertical_names + sfc_in_names + sfc_out_names)
        step_config = SingleModuleStepConfig(
            builder=ModuleSelector(
                type="NoiseConditionedSFNO",
                config=dataclasses.asdict(csfno_config),
                conditional=n_labels > 0,
            ),
            in_names=expanded_vertical_names + sfc_in_names,
            out_names=expanded_vertical_names + sfc_out_names,
            normalization=NetworkAndLossNormalizationConfig(
                network=NormalizationConfig(
                    means={name: 0.0 for name in all_names},
                    stds={name: 1.0 for name in all_names},
                )
            ),
        )
        config = StepperConfig(
            step=StepSelector(
                type="single_module",
                config=dataclasses.asdict(step_config),
            ),
        )
        stepper = get_stepper(
            config,
            img_shape=(H, L),
            all_labels=set(f"label_{i}" for i in range(n_labels))
            if n_labels > 0
            else None,
            n_levels=n_levels,
        )
        return stepper


register_benchmark("predict_csfno")(CSFNOPredict)
