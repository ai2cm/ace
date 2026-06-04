import copy
import dataclasses
import pathlib
import subprocess
import tempfile
import unittest.mock
from typing import Literal

import dacite
import numpy as np
import pytest
import torch
import xarray as xr
import yaml

import fme
from fme.ace.aggregator.inference.main import InferenceEvaluatorAggregatorConfig
from fme.ace.aggregator.one_step.main import OneStepAggregatorConfig
from fme.ace.aggregator.one_step.map import OneStepMapMetricConfig
from fme.ace.aggregator.one_step.snapshot import OneStepSnapshotMetricConfig
from fme.ace.data_loading.config import DataLoaderConfig
from fme.ace.data_loading.inference import (
    InferenceDataLoaderConfig,
    InferenceInitialConditionIndices,
)
from fme.ace.inference.data_writer.file_writer import FileWriterConfig
from fme.ace.inference.data_writer.main import DataWriterConfig
from fme.ace.inference.evaluator import InferenceEvaluatorConfig
from fme.ace.inference.evaluator import main as inference_evaluator_main
from fme.ace.registry.test_hpx import (
    conv_next_block_config,
    decoder_config,
    down_sampling_block_config,
    encoder_config,
    output_layer_config,
    up_sampling_block_config,
)
from fme.ace.stepper.derived_forcings import DerivedForcingsConfig
from fme.ace.stepper.insolation.config import InsolationConfig, NameConfig, ValueConfig
from fme.ace.stepper.single_module import (
    CheckpointStepperConfig,
    StepperConfig,
    TrainStepperConfig,
)
from fme.ace.stepper.time_length_probabilities import (
    TimeLength,
    TimeLengthMilestone,
    TimeLengthProbabilities,
    TimeLengthProbability,
    TimeLengthSchedule,
)
from fme.ace.testing import (
    DimSizes,
    MonthlyReferenceData,
    patch_cm4_solar_constant,
    save_nd_netcdf,
    save_scalar_netcdf,
    save_stepper_checkpoint,
)
from fme.ace.train.train import build_trainer, prepare_directory
from fme.ace.train.train import main as train_main
from fme.ace.train.train_config import (
    InlineInferenceConfig,
    InlineValidationConfig,
    TrainBuilders,
    TrainConfig,
)
from fme.core.coordinates import (
    HEALPixCoordinates,
    HorizontalCoordinates,
    LatLonCoordinates,
)
from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.dataset.concat import ConcatDatasetConfig
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.generics.trainer import _restore_checkpoint
from fme.core.logging_utils import LoggingConfig
from fme.core.loss import StepLossConfig
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.ocean import OceanConfig
from fme.core.optimization import OptimizationConfig
from fme.core.rand import set_seed
from fme.core.registry.corrector import CorrectorSelector
from fme.core.registry.module import ModuleSelector
from fme.core.scheduler import SchedulerConfig
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepSelector
from fme.core.testing.model import compare_parameters, compare_restored_parameters
from fme.core.testing.wandb import mock_wandb

JOB_SUBMISSION_SCRIPT_PATH = (
    pathlib.PurePath(__file__).parent / "run-train-and-inference.sh"
)


def _make_validation_entries(
    *,
    valid_data_path,
    spatial_dimensions_str,
    conditional,
    log_validation_maps,
    multi_validation=False,
) -> InlineValidationConfig | list[InlineValidationConfig]:
    primary = InlineValidationConfig(
        loader=DataLoaderConfig(
            dataset=XarrayDataConfig(
                data_path=str(valid_data_path),
                spatial_dimensions=spatial_dimensions_str,
                labels=["era5"] if conditional else None,
            ),
            batch_size=2,
            num_data_workers=0,
        ),
        aggregator=OneStepAggregatorConfig(
            snapshot=OneStepSnapshotMetricConfig(enabled=log_validation_maps),
            mean_map=OneStepMapMetricConfig(enabled=log_validation_maps),
        ),
    )
    if not multi_validation:
        return primary
    secondary = InlineValidationConfig(
        loader=DataLoaderConfig(
            dataset=XarrayDataConfig(
                data_path=str(valid_data_path),
                spatial_dimensions=spatial_dimensions_str,
                labels=["era5"] if conditional else None,
            ),
            batch_size=2,
            num_data_workers=0,
        ),
        name="val_extra",
        weight=0.0,
    )
    return [primary, secondary]


def _get_test_yaml_files(
    *,
    train_data_path,
    valid_data_path,
    monthly_data_filename: pathlib.Path | None,
    results_dir,
    global_means_path,
    global_stds_path,
    in_variable_names,
    out_variable_names,
    mask_name,
    n_forward_steps=2,
    nettype="SphericalFourierNeuralOperatorNet",
    log_to_wandb=False,
    max_epochs=1,
    segment_epochs=1,
    inference_forward_steps=2,
    use_healpix=False,
    crps_training=False,
    save_per_epoch_diagnostics=False,
    log_validation_maps=False,
    skip_inline_inference=False,
    time_buffer=1,
    use_time_length_probabilities=True,
    use_schedule=False,
    validate_using_ema=False,
    derived_forcings=None,
    multi_validation=False,
    partial_train_data_path: pathlib.Path | None = None,
    batch_size: int = 2,
    sample_with_replacement: int | None = 10,
):
    if derived_forcings is None:
        derived_forcings = DerivedForcingsConfig()
    if nettype == "HEALPixUNet":
        in_channels = len(in_variable_names)
        conv_next_block = conv_next_block_config(in_channels=in_channels)
        down_sampling_block = down_sampling_block_config()
        encoder = encoder_config(
            conv_next_block, down_sampling_block, n_channels=[16, 8, 4]
        )
        up_sampling_block = up_sampling_block_config()
        output_layer = output_layer_config()
        decoder = decoder_config(
            conv_next_block,
            up_sampling_block,
            output_layer,
            n_channels=[4, 8, 16],
        )
        net_config = dict(
            encoder=encoder,
            decoder=decoder,
        )
        spatial_dimensions_str: Literal["healpix", "latlon"] = "healpix"
    elif nettype == "Samudra":
        net_config = dict(
            ch_width=[8, 16],
            dilation=[2, 4],
            n_layers=[1, 1],
        )
        spatial_dimensions_str = "latlon"
    elif nettype == "SphericalFourierNeuralOperatorNet":
        net_config = dict(
            num_layers=2,
            embed_dim=12,
        )
        spatial_dimensions_str = "latlon"
    elif nettype == "NoiseConditionedSFNO":
        net_config = dict(
            num_layers=2,
            embed_dim=12,
            label_embed_dim=3,
        )
        if use_healpix:
            net_config["data_grid"] = "healpix"
            spatial_dimensions_str = "healpix"
        else:
            spatial_dimensions_str = "latlon"

    if nettype == "NoiseConditionedSFNO":
        conditional = True
    else:
        conditional = False

    if nettype == "SphericalFourierNeuralOperatorNet":
        corrector_config: AtmosphereCorrectorConfig | CorrectorSelector = (
            CorrectorSelector(
                type="atmosphere_corrector",
                config=dataclasses.asdict(
                    AtmosphereCorrectorConfig(conserve_dry_air=True)
                ),
            )
        )
    else:
        corrector_config = AtmosphereCorrectorConfig()

    logging_config = LoggingConfig(
        log_to_screen=True,
        log_to_wandb=log_to_wandb,
        log_to_file=False,
        project="fme",
        entity="ai2cm",
    )
    if skip_inline_inference:
        inference_configs: list[InlineInferenceConfig] = []
    else:
        inference_configs = [
            InlineInferenceConfig(
                aggregator=InferenceEvaluatorAggregatorConfig(
                    monthly_reference_data=(
                        str(monthly_data_filename)
                        if monthly_data_filename is not None
                        else None
                    ),
                ),
                loader=InferenceDataLoaderConfig(
                    dataset=XarrayDataConfig(
                        data_path=str(valid_data_path),
                        spatial_dimensions=spatial_dimensions_str,
                        labels=[] if conditional else None,
                    ),
                    start_indices=InferenceInitialConditionIndices(
                        first=0,
                        n_initial_conditions=2,
                        interval=1,
                    ),
                ),
                n_forward_steps=inference_forward_steps,
                forward_steps_in_memory=2,
                n_ensemble_per_ic=2,
            ),
            InlineInferenceConfig(
                name="weather_eval",
                aggregator=InferenceEvaluatorAggregatorConfig(
                    monthly_reference_data=(
                        str(monthly_data_filename)
                        if monthly_data_filename is not None
                        else None
                    ),
                ),
                loader=InferenceDataLoaderConfig(
                    dataset=XarrayDataConfig(
                        data_path=str(valid_data_path),
                        spatial_dimensions=spatial_dimensions_str,
                        labels=["era5"] if conditional else None,
                    ),
                    start_indices=InferenceInitialConditionIndices(
                        first=0,
                        n_initial_conditions=2,
                        interval=1,
                    ),
                ),
                n_forward_steps=inference_forward_steps,
                forward_steps_in_memory=2,
                n_ensemble_per_ic=2,
            ),
        ]

    if use_time_length_probabilities:
        n_forward_steps_arg: TimeLength | TimeLengthSchedule = TimeLengthProbabilities(
            outcomes=[
                TimeLengthProbability(steps=1, probability=0.5),
                TimeLengthProbability(steps=n_forward_steps, probability=0.5),
            ]
        )
    elif use_schedule:
        n_forward_steps_arg = TimeLengthSchedule(
            start_value=TimeLengthProbabilities(
                outcomes=[
                    TimeLengthProbability(steps=1, probability=0.5),
                    TimeLengthProbability(steps=n_forward_steps, probability=0.5),
                ]
            ),
            milestones=[TimeLengthMilestone(epoch=1, value=n_forward_steps + 1)],
        )
        max_epochs = 2
    else:
        n_forward_steps_arg = n_forward_steps

    if crps_training:
        loss = StepLossConfig(
            type="EnsembleLoss",
            kwargs={
                "crps_weight": 1.0,
                "energy_score_weight": 0.0,
                "finite_difference_crps_weight": 0.05,
            },
        )
        n_ensemble: int = 2
    else:
        loss = StepLossConfig(type="MSE")
        n_ensemble = 1

    train_config = TrainConfig(
        train_loader=DataLoaderConfig(
            dataset=ConcatDatasetConfig(
                concat=[
                    XarrayDataConfig(
                        data_path=str(train_data_path),
                        labels=["era5"] if conditional else None,
                        spatial_dimensions=spatial_dimensions_str,
                    ),
                    XarrayDataConfig(
                        data_path=str(partial_train_data_path or train_data_path),
                        labels=[] if conditional else None,
                        spatial_dimensions=spatial_dimensions_str,
                    ),
                ],
                strict=(partial_train_data_path is None),
            ),
            batch_size=batch_size,
            num_data_workers=0,
            time_buffer=time_buffer,
            sample_with_replacement=sample_with_replacement,
        ),
        validation=_make_validation_entries(
            valid_data_path=valid_data_path,
            spatial_dimensions_str=spatial_dimensions_str,
            conditional=conditional,
            log_validation_maps=log_validation_maps,
            multi_validation=multi_validation,
        ),
        optimization=OptimizationConfig(
            use_gradient_accumulation=True,
            enable_automatic_mixed_precision=True,
            optimizer_type="Adam",
            lr=0.0001,
            kwargs=dict(weight_decay=0.01),
            scheduler=SchedulerConfig(
                type="CosineAnnealingLR",
                kwargs=dict(T_max=1),
            ),
        ),
        stepper=StepperConfig(
            derived_forcings=derived_forcings,
            step=StepSelector(
                type="single_module",
                config=dataclasses.asdict(
                    SingleModuleStepConfig(
                        in_names=in_variable_names,
                        out_names=out_variable_names,
                        normalization=NetworkAndLossNormalizationConfig(
                            network=NormalizationConfig(
                                global_means_path=str(global_means_path),
                                global_stds_path=str(global_stds_path),
                            ),
                            residual=NormalizationConfig(
                                global_means_path=str(global_means_path),
                                global_stds_path=str(global_stds_path),
                            ),
                        ),
                        builder=ModuleSelector(
                            type=nettype,
                            conditional=conditional,
                            config=net_config,
                            allow_missing_variables=(
                                partial_train_data_path is not None
                            ),
                        ),
                        ocean=OceanConfig(
                            surface_temperature_name=in_variable_names[0],
                            ocean_fraction_name=mask_name,
                        ),
                        corrector=corrector_config,
                    )
                ),
            ),
        ),
        stepper_training=TrainStepperConfig(
            loss=loss,
            n_ensemble=n_ensemble,
            n_forward_steps=n_forward_steps_arg,
        ),
        inference=inference_configs,
        validate_using_ema=validate_using_ema,
        max_epochs=max_epochs,
        segment_epochs=segment_epochs,
        save_checkpoint=True,
        logging=logging_config,
        experiment_dir=str(results_dir),
        save_per_epoch_diagnostics=save_per_epoch_diagnostics,
    )

    inference_config = InferenceEvaluatorConfig(
        experiment_dir=str(results_dir),
        n_forward_steps=6,
        forward_steps_in_memory=2,
        checkpoint_path=str(results_dir / "training_checkpoints" / "best_ckpt.tar"),
        data_writer=DataWriterConfig(
            save_monthly_files=False,
            save_prediction_files=False,
            files=[FileWriterConfig("autoregressive")],
        ),
        aggregator=InferenceEvaluatorAggregatorConfig(),
        logging=logging_config,
        loader=InferenceDataLoaderConfig(
            dataset=XarrayDataConfig(
                data_path=str(valid_data_path),
                spatial_dimensions=spatial_dimensions_str,
                labels=["era5"] if conditional else None,
            ),
            start_indices=InferenceInitialConditionIndices(
                first=0,
                n_initial_conditions=2,
                interval=1,
            ),
        ),
    )

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f_train:
        f_train.write(yaml.dump(dataclasses.asdict(train_config)))

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml"
    ) as f_inference:
        f_inference.write(yaml.dump(dataclasses.asdict(inference_config)))

    return f_train.name, f_inference.name


def get_sizes(
    spatial_dims: HorizontalCoordinates = LatLonCoordinates(
        lon=torch.Tensor(np.arange(32)),
        lat=torch.Tensor(np.arange(16)),
    ),
    n_time=3,
    nz_interface=3,
) -> DimSizes:
    return DimSizes(
        n_time=n_time,
        horizontal=copy.deepcopy(spatial_dims.loaded_sizes),
        nz_interface=nz_interface,
    )


def _setup(
    path,
    nettype,
    log_to_wandb=False,
    max_epochs=1,
    segment_epochs=1,
    n_time=10,
    timestep_days=5,
    inference_forward_steps=2,
    use_healpix=False,
    save_per_epoch_diagnostics=False,
    crps_training=False,
    log_validation_maps=False,
    skip_inline_inference=False,
    time_buffer=1,
    use_time_length_probabilities=True,
    derived_forcings=None,
    use_schedule: bool = False,
    validate_using_ema: bool = False,
    stats_std_fill_value: float | None = None,
    multi_validation: bool = False,
    use_variable_masking: bool = False,
):
    if not path.exists():
        path.mkdir()
    if derived_forcings is None:
        derived_forcings = DerivedForcingsConfig()
    seed = 0
    set_seed(seed)
    in_variable_names = [
        "PRESsfc",
        "specific_total_water_0",
        "specific_total_water_1",
        "surface_temperature",
        "baz",
    ]
    if derived_forcings.insolation is not None:
        in_variable_names.append(derived_forcings.insolation.insolation_name)
    out_variable_names = [
        "PRESsfc",
        "specific_total_water_0",
        "specific_total_water_1",
        "surface_temperature",
    ]
    mask_name = "mask"
    all_variable_names = list(set(in_variable_names + out_variable_names))

    if use_healpix:
        hpx_coords = HEALPixCoordinates(
            face=torch.Tensor(np.arange(12)),
            width=torch.Tensor(np.arange(16)),
            height=torch.Tensor(np.arange(16)),
        )
        dim_sizes = get_sizes(spatial_dims=hpx_coords, n_time=n_time)
    else:
        dim_sizes = get_sizes(n_time=n_time)

    data_dir = path / "data"
    stats_dir = path / "stats"
    results_dir = path / "results"
    data_dir.mkdir()
    stats_dir.mkdir()
    results_dir.mkdir()
    on_disk_names = all_variable_names + [mask_name]
    if derived_forcings.insolation is not None:
        if isinstance(derived_forcings.insolation.solar_constant, NameConfig):
            on_disk_names.append(derived_forcings.insolation.solar_constant.name)
    save_nd_netcdf(
        data_dir / "data.nc",
        dim_sizes,
        variable_names=on_disk_names,
        timestep_days=timestep_days,
    )
    partial_data_dir = None
    if use_variable_masking:
        partial_data_dir = path / "data_partial"
        partial_data_dir.mkdir()
        partial_names = [n for n in on_disk_names if n != "specific_total_water_1"]
        save_nd_netcdf(
            partial_data_dir / "data.nc",
            dim_sizes,
            variable_names=partial_names,
            timestep_days=timestep_days,
        )
    save_scalar_netcdf(
        stats_dir / "stats-mean.nc",
        variable_names=all_variable_names,
    )
    save_scalar_netcdf(
        stats_dir / "stats-stddev.nc",
        variable_names=all_variable_names,
        fill_value=stats_std_fill_value,
    )

    monthly_dim_sizes: DimSizes
    if use_healpix:
        # monthly reference functionality not supported for HEALPix
        # see https://github.com/ai2cm/full-model/issues/1561
        monthly_data_filename = None
    else:
        monthly_dim_sizes = get_sizes(n_time=10 * 12, nz_interface=1)
        monthly_reference_data = MonthlyReferenceData(
            path=data_dir,
            names=out_variable_names,
            dim_sizes=monthly_dim_sizes,
            n_ensemble=3,
        )
        monthly_data_filename = monthly_reference_data.data_filename

    train_config_filename, inference_config_filename = _get_test_yaml_files(
        train_data_path=data_dir,
        valid_data_path=data_dir,
        monthly_data_filename=monthly_data_filename,
        results_dir=results_dir,
        global_means_path=stats_dir / "stats-mean.nc",
        global_stds_path=stats_dir / "stats-stddev.nc",
        in_variable_names=in_variable_names,
        out_variable_names=out_variable_names,
        mask_name=mask_name,
        nettype=nettype,
        log_to_wandb=log_to_wandb,
        max_epochs=max_epochs,
        segment_epochs=segment_epochs,
        inference_forward_steps=inference_forward_steps,
        use_healpix=use_healpix,
        crps_training=crps_training,
        save_per_epoch_diagnostics=save_per_epoch_diagnostics,
        log_validation_maps=log_validation_maps,
        skip_inline_inference=skip_inline_inference,
        time_buffer=time_buffer,
        use_time_length_probabilities=use_time_length_probabilities,
        derived_forcings=derived_forcings,
        use_schedule=use_schedule,
        validate_using_ema=validate_using_ema,
        multi_validation=multi_validation,
        partial_train_data_path=partial_data_dir,
    )
    return train_config_filename, inference_config_filename


@dataclasses.dataclass
class TrainAndInferenceTestSettings:
    nettype: str = "SphericalFourierNeuralOperatorNet"
    crps_training: bool = False
    log_validation_maps: bool = False
    use_healpix: bool = False
    use_schedule: bool = False
    validate_using_ema: bool = False
    multi_validation: bool = False
    use_variable_masking: bool = False


_TRAIN_AND_INFERENCE_CASES = [
    pytest.param(
        TrainAndInferenceTestSettings(
            nettype="NoiseConditionedSFNO",
            crps_training=True,
            use_schedule=True,
        ),
        id="SFNO-crps-schedule",
    ),
    pytest.param(
        TrainAndInferenceTestSettings(
            log_validation_maps=True,
            multi_validation=True,
        ),
        id="SFNO-val-maps-multi",
    ),
    pytest.param(
        TrainAndInferenceTestSettings(
            nettype="HEALPixUNet",
            use_healpix=True,
        ),
        id="HEALPix",
    ),
    pytest.param(
        TrainAndInferenceTestSettings(nettype="Samudra"),
        id="Samudra",
    ),
    pytest.param(
        TrainAndInferenceTestSettings(
            nettype="NoiseConditionedSFNO",
            use_variable_masking=True,
        ),
        id="SFNO-masking",
        marks=pytest.mark.filterwarnings(
            "ignore:Metadata for each ensemble member:UserWarning"
        ),
    ),
    pytest.param(
        TrainAndInferenceTestSettings(
            nettype="NoiseConditionedSFNO",
            crps_training=True,
            use_schedule=True,
            validate_using_ema=True,
        ),
        id="SFNO-crps-schedule-ema",
    ),
]


@pytest.mark.parametrize("settings", _TRAIN_AND_INFERENCE_CASES)
def test_train_and_inference(
    tmp_path,
    settings: TrainAndInferenceTestSettings,
    very_fast_only: bool,
):
    """Ensure that training and standalone inference run without errors."""
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")
    # need multi-year to cover annual aggregator
    train_config, inference_config = _setup(
        tmp_path,
        settings.nettype,
        log_to_wandb=True,
        timestep_days=20,
        n_time=int(366 * 3 / 20 + 1),
        inference_forward_steps=int(366 * 3 / 20 / 2 - 1) * 2,  # must be even
        use_healpix=settings.use_healpix,
        crps_training=settings.crps_training,
        save_per_epoch_diagnostics=True,
        use_schedule=settings.use_schedule,
        validate_using_ema=settings.validate_using_ema,
        log_validation_maps=settings.log_validation_maps,
        multi_validation=settings.multi_validation,
        use_variable_masking=settings.use_variable_masking,
    )
    # using pdb requires calling main functions directly
    with mock_wandb() as wandb:
        train_main(
            yaml_config=train_config,
        )
        wandb_logs = wandb.get_logs()
        for log in wandb_logs:
            # ensure inference time series is not logged
            assert "inference_0/mean/forecast_step" not in log

        epoch_logs = wandb_logs[-1]
        assert "inference_0/mean_step_20_norm/weighted_rmse/channel_mean" in epoch_logs
        primary_val_name = "val_0" if settings.multi_validation else "val"
        assert f"{primary_val_name}/mean_norm/weighted_rmse/channel_mean" in epoch_logs
        if settings.multi_validation:
            assert "val_extra/mean/loss" in epoch_logs
        ensemble_step_20_keys = [
            k for k in epoch_logs if "inference_0/ensemble_step_20/" in k
        ]
        assert ensemble_step_20_keys, (
            "expected at least one ensemble_step_20 metric in inline inference "
            "epoch log"
        )
        weather_eval_keys = [k for k in epoch_logs if k.startswith("weather_eval/")]
        assert (
            weather_eval_keys
        ), "expected at least one weather_eval metric in inference epoch log"
        weather_eval_ensemble_keys = [
            k for k in epoch_logs if "weather_eval/ensemble_step_20/" in k
        ]
        assert weather_eval_ensemble_keys, (
            "expected at least one ensemble_step_20 metric in weather_eval "
            "inference epoch log"
        )

    validation_output_dir = (
        tmp_path / "results" / "output" / primary_val_name / "epoch_0001"
    )
    assert validation_output_dir.exists()
    validation_diags = ["mean"]
    validation_map_diags = ["snapshot", "mean_map"]
    for diagnostic in validation_diags + validation_map_diags:
        diagnostic_output = validation_output_dir / f"{diagnostic}_diagnostics.nc"
        if diagnostic in validation_map_diags and not settings.log_validation_maps:
            assert not diagnostic_output.exists()
        else:
            assert diagnostic_output.exists()
            ds = xr.open_dataset(diagnostic_output, decode_timedelta=False)
            assert len(ds) > 0

    inline_inference_output_dir = (
        tmp_path / "results" / "output" / "inference_0" / "epoch_0001"
    )
    assert inline_inference_output_dir.exists()
    for diagnostic in (
        "mean_step_20",
        "time_mean",
        "time_mean_norm",
        "annual",
    ):
        diagnostic_output = inline_inference_output_dir / f"{diagnostic}_diagnostics.nc"
        assert diagnostic_output.exists()
        ds = xr.open_dataset(diagnostic_output, decode_timedelta=False)
        assert len(ds) > 0

    # inference should not require stats files
    (tmp_path / "stats" / "stats-mean.nc").unlink()
    (tmp_path / "stats" / "stats-stddev.nc").unlink()

    with mock_wandb() as wandb:
        wandb.configure(log_to_wandb=True)
        inference_evaluator_main(yaml_config=inference_config)
        inference_logs = wandb.get_logs()

    prediction_output_path = tmp_path / "results" / "autoregressive_predictions.nc"
    best_checkpoint_path = (
        tmp_path / "results" / "training_checkpoints" / "best_ckpt.tar"
    )
    best_inference_checkpoint_path = (
        tmp_path / "results" / "training_checkpoints" / "best_inference_ckpt.tar"
    )
    assert best_checkpoint_path.exists()
    checkpoint_training_history = torch.load(
        best_checkpoint_path, map_location="cpu", weights_only=False
    )["stepper"].get("training_history")
    assert checkpoint_training_history is not None
    assert len(checkpoint_training_history) == 1
    assert "git_sha" in checkpoint_training_history[0].keys()
    assert best_inference_checkpoint_path.exists()
    n_ic_timesteps = 1
    n_forward_steps = 6
    n_summary_steps = 1
    assert len(inference_logs) == n_ic_timesteps + n_forward_steps + n_summary_steps
    assert prediction_output_path.exists()
    ds_prediction = xr.open_dataset(prediction_output_path, decode_timedelta=False)
    assert np.sum(np.isnan(ds_prediction["PRESsfc"].values)) == 0
    assert np.sum(np.isnan(ds_prediction["specific_total_water_0"].values)) == 0
    assert np.sum(np.isnan(ds_prediction["specific_total_water_1"].values)) == 0
    assert np.sum(np.isnan(ds_prediction["total_water_path"].values)) == 0
    ds_target = xr.open_dataset(
        tmp_path / "results" / "autoregressive_target.nc", decode_timedelta=False
    )
    assert np.sum(np.isnan(ds_target["baz"].values)) == 0


@pytest.mark.parametrize("nettype", ["SphericalFourierNeuralOperatorNet"])
def test_resume(tmp_path, nettype, very_fast_only: bool):
    """Make sure the training is resumed from a checkpoint when restarted."""
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")

    mock = unittest.mock.MagicMock(side_effect=_restore_checkpoint)
    with unittest.mock.patch("fme.core.generics.trainer._restore_checkpoint", new=mock):
        train_config, _ = _setup(
            tmp_path, nettype, log_to_wandb=True, max_epochs=2, segment_epochs=1
        )
        with mock_wandb() as wandb:
            train_main(yaml_config=train_config)
            assert (
                min([val["epoch"] for val in wandb.get_logs() if "epoch" in val]) == 1
            )
            assert (
                max([val["epoch"] for val in wandb.get_logs() if "epoch" in val]) == 1
            )
            assert not mock.called
            # need to persist the id since mock_wandb doesn't
            id = wandb.get_id()
        with mock_wandb() as wandb:
            # set the id so that we can check it matches what's in the experiment dir
            wandb.set_id(id)
            train_main(yaml_config=train_config)
            mock.assert_called()
            assert (
                min([val["epoch"] for val in wandb.get_logs() if "epoch" in val]) == 2
            )
            assert (
                max([val["epoch"] for val in wandb.get_logs() if "epoch" in val]) == 2
            )


def _get_reproducible_trainer(config_dict, seed):
    set_seed(seed)
    # TrainConfig objects may create rng (e.g., TimeLengthProbabilities), so it
    # has to be rebuilt after setting the seed.
    config = dacite.from_dict(
        data_class=TrainConfig, data=config_dict, config=dacite.Config(strict=True)
    )
    prepare_directory(config.experiment_dir, config_dict)
    builders = TrainBuilders(config)
    return build_trainer(builders, config)


@pytest.mark.parametrize("nettype", ["NoiseConditionedSFNO"])
def test_set_seed(tmp_path, nettype, very_fast_only: bool):
    """Test that set_seed leads to identical training outcomes."""
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")

    config_path, _ = _setup(tmp_path, nettype)
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    trainer1 = _get_reproducible_trainer(config_dict, seed=0)
    trainer2 = _get_reproducible_trainer(config_dict, seed=0)

    compare_parameters(
        trainer1.stepper.modules.named_parameters(),
        trainer2.stepper.modules.named_parameters(),
    )

    set_seed(0)
    trainer1.train_one_epoch()

    set_seed(0)
    trainer2.train_one_epoch()

    compare_parameters(
        trainer1.stepper.modules.named_parameters(),
        trainer2.stepper.modules.named_parameters(),
    )


@pytest.mark.parametrize("nettype", ["NoiseConditionedSFNO"])
@pytest.mark.parametrize("save_type", ["restart", "all"])
def test_restore_checkpoint(
    tmp_path,
    nettype: str,
    save_type: Literal["restart", "all"],
    very_fast_only: bool,
):
    """Test that restoring a checkpoint works."""
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")

    # this test will fail if the config has rng
    config_path, _ = _setup(tmp_path, nettype, use_time_length_probabilities=False)
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    set_seed(0)
    config = dacite.from_dict(
        data_class=TrainConfig, data=config_dict, config=dacite.Config(strict=True)
    )
    prepare_directory(config.experiment_dir, config_dict)
    builders = TrainBuilders(config)

    base_trainer = build_trainer(builders, config)
    restored_trainer1 = build_trainer(builders, config)
    restored_trainer2 = build_trainer(builders, config)

    # run one epoch
    base_trainer.train_one_epoch()
    if save_type == "all":
        base_trainer.save_all_checkpoints(0.1, 0.2)
    elif save_type == "restart":
        base_trainer._save_restart_checkpoints()

    # reload and check model parameters and optimizer state
    restored_trainer1.restore_checkpoint(
        base_trainer.paths.latest_checkpoint_path,
    )
    restored_trainer2.restore_checkpoint(
        base_trainer.paths.latest_checkpoint_path,
    )
    compare_restored_parameters(
        base_trainer.stepper.modules.parameters(),
        restored_trainer1.stepper.modules.parameters(),
        base_trainer.optimization.optimizer,
        restored_trainer1.optimization.optimizer,
    )
    with base_trainer._ema_context():
        with restored_trainer1._ema_context():
            compare_restored_parameters(
                base_trainer.stepper.modules.parameters(),
                restored_trainer1.stepper.modules.parameters(),
                base_trainer.optimization.optimizer,
                restored_trainer1.optimization.optimizer,
            )

    base_ema_state = base_trainer._ema.get_state()
    base_params = base_ema_state.pop("ema_params")
    restored_ema_state = restored_trainer1._ema.get_state()
    restored_params = restored_ema_state.pop("ema_params")
    assert base_ema_state == restored_ema_state
    torch.testing.assert_close(base_params, restored_params)

    set_seed(0)
    base_trainer.train_one_epoch()
    set_seed(0)
    restored_trainer1.train_one_epoch()
    set_seed(0)
    restored_trainer2.train_one_epoch()

    compare_restored_parameters(
        restored_trainer2.stepper.modules.parameters(),
        restored_trainer1.stepper.modules.parameters(),
        restored_trainer2.optimization.optimizer,
        restored_trainer1.optimization.optimizer,
    )
    with restored_trainer2._ema_context():
        with restored_trainer1._ema_context():
            compare_restored_parameters(
                restored_trainer2.stepper.modules.parameters(),
                restored_trainer1.stepper.modules.parameters(),
                restored_trainer2.optimization.optimizer,
                restored_trainer1.optimization.optimizer,
            )

    compare_restored_parameters(
        base_trainer.stepper.modules.parameters(),
        restored_trainer1.stepper.modules.parameters(),
        base_trainer.optimization.optimizer,
        restored_trainer1.optimization.optimizer,
    )

    with base_trainer._ema_context():
        with restored_trainer1._ema_context():
            compare_restored_parameters(
                base_trainer.stepper.modules.parameters(),
                restored_trainer1.stepper.modules.parameters(),
                base_trainer.optimization.optimizer,
                restored_trainer1.optimization.optimizer,
            )

    with base_trainer._ema_context():
        with pytest.raises(AssertionError):  # should not be equal
            compare_restored_parameters(
                base_trainer.stepper.modules.parameters(),
                restored_trainer1.stepper.modules.parameters(),
                base_trainer.optimization.optimizer,
                restored_trainer1.optimization.optimizer,
            )


@pytest.mark.serial
@pytest.mark.parametrize("nettype", ["SphericalFourierNeuralOperatorNet"])
@pytest.mark.skipif(
    fme.get_device().type == "mps", reason="MPS does not support multi-device training."
)
def test_resume_two_workers(tmp_path, nettype, skip_slow: bool, tmpdir: pathlib.Path):
    """Make sure the training is resumed from a checkpoint when restarted, using
    torchrun with NPROC_PER_NODE set to 2."""
    if skip_slow:
        # script is slow as everything is re-imported when it runs
        pytest.skip("Skipping slow tests")
    train_config, inference_config = _setup(tmp_path, nettype)
    subprocess_args = [
        JOB_SUBMISSION_SCRIPT_PATH,
        train_config,
        inference_config,
        "2",  # this makes the training run on two GPUs
    ]
    initial_process = subprocess.run(subprocess_args, cwd=tmpdir)
    initial_process.check_returncode()
    resume_subprocess_args = [
        "torchrun",
        "--nproc_per_node",
        "2",
        "-m",
        "fme.ace.train",
        train_config,
    ]
    resume_process = subprocess.run(resume_subprocess_args, cwd=tmpdir)
    resume_process.check_returncode()


def _create_copy_weights_after_batch_config(
    path_to_train_config_yaml: str, path_to_checkpoint: str, experiment_dir: str
):
    with open(path_to_train_config_yaml) as config_file:
        config_data = yaml.safe_load(config_file)
        config_data["stepper_training"]["parameter_init"] = {
            "weights_path": path_to_checkpoint
        }
        config_data["copy_weights_after_batch"] = [{"include": ["*"], "exclude": None}]
        config_data["experiment_dir"] = experiment_dir
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".yaml"
        ) as new_config_file:
            new_config_file.write(yaml.dump(config_data))

    return new_config_file.name


@pytest.mark.parametrize("nettype", ["SphericalFourierNeuralOperatorNet"])
def test_copy_weights_after_batch(tmp_path, nettype, skip_slow: bool):
    """Check that fine tuning config using copy_weights_after_batch
    runs without errors."""
    if skip_slow:
        pytest.skip("Skipping slow tests")

    train_config, _ = _setup(tmp_path, nettype)

    train_main(
        yaml_config=train_config,
    )

    results_dir = tmp_path / "results"
    ckpt = f"{results_dir}/training_checkpoints/best_ckpt.tar"

    fine_tuning_config = _create_copy_weights_after_batch_config(
        train_config, ckpt, experiment_dir=str(tmp_path / "fine_tuning_dir")
    )
    train_main(yaml_config=fine_tuning_config)


def test_train_without_inline_inference(tmp_path, very_fast_only: bool):
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")
    nettype = "SphericalFourierNeuralOperatorNet"
    crps_training = False
    log_validation_maps = False
    train_config, inference_config = _setup(
        tmp_path,
        nettype,
        log_to_wandb=True,
        timestep_days=20,
        n_time=int(366 * 3 / 20 + 1),
        inference_forward_steps=int(366 * 3 / 20 / 2 - 1) * 2,  # must be even
        use_healpix=False,
        crps_training=crps_training,
        save_per_epoch_diagnostics=True,
        log_validation_maps=log_validation_maps,
        skip_inline_inference=True,
        time_buffer=2,
        multi_validation=True,
    )
    with mock_wandb() as wandb:
        train_main(
            yaml_config=train_config,
        )
        wandb_logs = wandb.get_logs()
    assert np.isinf(wandb_logs[-1]["best_inference_error"])
    assert not any("inference/" in key for key in wandb_logs[-1])
    epoch_logs = wandb_logs[-1]
    assert "val_0/mean/loss" in epoch_logs
    assert "val_extra/mean/loss" in epoch_logs
    val_extra_output = tmp_path / "results" / "output" / "val_extra" / "epoch_0001"
    assert val_extra_output.exists()


@pytest.mark.skipif(torch.cuda.is_available(), reason="flaky on GPU")
@pytest.mark.parametrize(
    "insolation_config",
    [
        pytest.param(
            InsolationConfig("DSWRFtoa", ValueConfig(1360.0)),
            id="solar-constant-as-value",
        ),
        pytest.param(
            InsolationConfig("DSWRFtoa", NameConfig("solar_constant")),
            id="solar-constant-as-name",
        ),
    ],
)
def test_train_and_inference_with_derived_forcings(
    tmp_path, insolation_config: InsolationConfig, very_fast_only: bool
):
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")

    nettype = "SphericalFourierNeuralOperatorNet"
    crps_training = False
    log_validation_maps = False
    derived_forcings = DerivedForcingsConfig(insolation_config)
    train_config, inference_config = _setup(
        tmp_path,
        nettype,
        log_to_wandb=True,
        timestep_days=0.25,
        n_time=12,
        inference_forward_steps=10,  # must be even
        save_per_epoch_diagnostics=True,
        crps_training=crps_training,
        log_validation_maps=log_validation_maps,
        derived_forcings=derived_forcings,
        stats_std_fill_value=1.0,
    )
    with patch_cm4_solar_constant(1.0):
        with mock_wandb() as wandb:
            train_main(
                yaml_config=train_config,
            )
        with mock_wandb() as wandb:
            wandb.configure(log_to_wandb=True)
            inference_evaluator_main(yaml_config=inference_config)


def test_train_with_non_local_experiment_dir_error():
    """Test that an error is raised if the experiment_dir is not local during
    training. This test can be removed when we support non-local experiment
    directories in training."""
    non_local_experiment_dir = "memory://path/to/experiment_dir"

    # Construct dummy configurations for the rest of the training config, since
    # all we are testing is that an error is raised upon construction.
    step = StepSelector(
        type="single_module",
        config=dataclasses.asdict(
            SingleModuleStepConfig(
                in_names=[],
                out_names=[],
                normalization=NetworkAndLossNormalizationConfig(
                    network=NormalizationConfig(
                        global_means_path="",
                        global_stds_path="",
                    ),
                ),
                builder=ModuleSelector(
                    type="SphericalFourierNeuralOperatorNet", config={}
                ),
            ),
        ),
    )
    stepper = StepperConfig(step=step)
    dummy_data_loader = DataLoaderConfig(
        dataset=XarrayDataConfig(data_path=""),
        batch_size=1,
    )

    with pytest.raises(ValueError, match="local directory"):
        TrainConfig(
            experiment_dir=non_local_experiment_dir,
            stepper=stepper,
            train_loader=dummy_data_loader,
            validation=InlineValidationConfig(loader=dummy_data_loader),
            optimization=OptimizationConfig(),
            logging=LoggingConfig(),
            max_epochs=1,
            save_checkpoint=False,
            inference=[],
        )


def test_train_config_with_checkpoint_stepper(tmp_path: pathlib.Path):
    checkpoint_path = tmp_path / "checkpoint.tar"
    original_config = save_stepper_checkpoint(checkpoint_path)
    dummy_data_loader = DataLoaderConfig(
        dataset=XarrayDataConfig(data_path=""),
        batch_size=1,
    )
    train_config = TrainConfig(
        experiment_dir=str(tmp_path / "experiment"),
        stepper=CheckpointStepperConfig(checkpoint_path=str(checkpoint_path)),
        stepper_training=TrainStepperConfig(n_forward_steps=2),
        train_loader=dummy_data_loader,
        validation=InlineValidationConfig(loader=dummy_data_loader),
        optimization=OptimizationConfig(),
        logging=LoggingConfig(),
        max_epochs=1,
        save_checkpoint=False,
        inference=[],
    )
    assert isinstance(train_config.stepper_config, StepperConfig)
    assert (
        train_config.stepper_config.derived_forcings == original_config.derived_forcings
    )
    assert train_config.stepper_config.step.type == original_config.step.type


def test_train_config_with_stepper_config_sets_stepper_config(tmp_path: pathlib.Path):
    step = StepSelector(
        type="single_module",
        config=dataclasses.asdict(
            SingleModuleStepConfig(
                in_names=[],
                out_names=[],
                normalization=NetworkAndLossNormalizationConfig(
                    network=NormalizationConfig(
                        global_means_path="",
                        global_stds_path="",
                    ),
                ),
                builder=ModuleSelector(
                    type="SphericalFourierNeuralOperatorNet", config={}
                ),
            ),
        ),
    )
    stepper = StepperConfig(step=step)
    dummy_data_loader = DataLoaderConfig(
        dataset=XarrayDataConfig(data_path=""),
        batch_size=1,
    )
    train_config = TrainConfig(
        experiment_dir=str(tmp_path / "experiment"),
        stepper=stepper,
        stepper_training=TrainStepperConfig(n_forward_steps=2),
        train_loader=dummy_data_loader,
        validation=InlineValidationConfig(loader=dummy_data_loader),
        optimization=OptimizationConfig(),
        logging=LoggingConfig(),
        max_epochs=1,
        save_checkpoint=False,
        inference=[],
    )
    assert train_config.stepper_config is stepper


def _setup_task_sampling_data(tmp_path, variable_names, n_time=4, timestep_days=5):
    """Create minimal data and stats for task sampling integration tests."""
    dim_sizes = get_sizes(n_time=n_time)
    data_dir = tmp_path / "data"
    stats_dir = tmp_path / "stats"
    results_dir = tmp_path / "results"
    data_dir.mkdir()
    stats_dir.mkdir()
    results_dir.mkdir()
    save_nd_netcdf(
        data_dir / "data.nc",
        dim_sizes,
        variable_names=variable_names,
        timestep_days=timestep_days,
    )
    save_scalar_netcdf(
        stats_dir / "stats-mean.nc",
        variable_names=variable_names,
    )
    save_scalar_netcdf(
        stats_dir / "stats-stddev.nc",
        variable_names=variable_names,
    )
    return data_dir, stats_dir, results_dir


def test_train_task_sampling_single_module_step(tmp_path):
    """Integration test: SingleModuleStep with prediction-only task sampling."""
    from fme.ace.stepper.task import TaskConfig, TaskSamplingConfig, TaskWeights

    in_names = ["a", "b"]
    out_names = ["a", "b"]
    all_names = list(set(in_names + out_names))
    data_dir, stats_dir, results_dir = _setup_task_sampling_data(
        tmp_path, all_names, n_time=4
    )

    train_config = TrainConfig(
        train_loader=DataLoaderConfig(
            dataset=XarrayDataConfig(
                data_path=str(data_dir),
                spatial_dimensions="latlon",
            ),
            batch_size=2,
            num_data_workers=0,
            sample_with_replacement=10,
        ),
        validation=InlineValidationConfig(
            loader=DataLoaderConfig(
                dataset=XarrayDataConfig(
                    data_path=str(data_dir),
                    spatial_dimensions="latlon",
                ),
                batch_size=2,
                num_data_workers=0,
            ),
        ),
        optimization=OptimizationConfig(
            optimizer_type="Adam",
            lr=0.0001,
        ),
        stepper=StepperConfig(
            step=StepSelector(
                type="single_module",
                config=dataclasses.asdict(
                    SingleModuleStepConfig(
                        in_names=in_names,
                        out_names=out_names,
                        normalization=NetworkAndLossNormalizationConfig(
                            network=NormalizationConfig(
                                global_means_path=str(stats_dir / "stats-mean.nc"),
                                global_stds_path=str(stats_dir / "stats-stddev.nc"),
                            ),
                        ),
                        builder=ModuleSelector(
                            type="SphericalFourierNeuralOperatorNet",
                            config=dict(num_layers=2, embed_dim=12),
                        ),
                    )
                ),
            ),
        ),
        stepper_training=TrainStepperConfig(
            loss=StepLossConfig(type="MSE"),
            n_forward_steps=2,
            task_sampling=TaskSamplingConfig(
                tasks=TaskWeights(
                    auto_encode=TaskConfig(probability=0.0),
                    infill=TaskConfig(probability=0.0),
                    prediction=TaskConfig(probability=1.0),
                    infill_prediction=TaskConfig(probability=0.0),
                    combined_all=TaskConfig(probability=0.0),
                ),
            ),
        ),
        inference=[],
        max_epochs=1,
        save_checkpoint=False,
        logging=LoggingConfig(
            log_to_screen=True, log_to_wandb=False, log_to_file=False
        ),
        experiment_dir=str(results_dir),
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml.dump(dataclasses.asdict(train_config)))
    with mock_wandb():
        train_main(yaml_config=f.name)


def test_train_task_sampling_infill_prediction_step(tmp_path):
    """Integration test: InfillPredictionStep with mixed task modes."""
    from fme.ace.stepper.task import TaskConfig, TaskSamplingConfig, TaskWeights
    from fme.core.step.infill_prediction import (
        InferenceSchemeConfig,
        InfillPredictionStepConfig,
    )

    all_names = ["a", "b", "forcing_x"]
    forcing_names = ["forcing_x"]
    data_dir, stats_dir, results_dir = _setup_task_sampling_data(
        tmp_path, all_names, n_time=4
    )

    train_config = TrainConfig(
        train_loader=DataLoaderConfig(
            dataset=XarrayDataConfig(
                data_path=str(data_dir),
                spatial_dimensions="latlon",
            ),
            batch_size=2,
            num_data_workers=0,
            sample_with_replacement=10,
        ),
        validation=InlineValidationConfig(
            loader=DataLoaderConfig(
                dataset=XarrayDataConfig(
                    data_path=str(data_dir),
                    spatial_dimensions="latlon",
                ),
                batch_size=2,
                num_data_workers=0,
            ),
        ),
        optimization=OptimizationConfig(
            optimizer_type="Adam",
            lr=0.0001,
        ),
        stepper=StepperConfig(
            step=StepSelector(
                type="infill_prediction",
                config=dataclasses.asdict(
                    InfillPredictionStepConfig(
                        builder=ModuleSelector(
                            type="SphericalFourierNeuralOperatorNet",
                            config=dict(num_layers=2, embed_dim=12),
                        ),
                        all_names=all_names,
                        forcing_names=forcing_names,
                        normalization=NetworkAndLossNormalizationConfig(
                            network=NormalizationConfig(
                                global_means_path=str(stats_dir / "stats-mean.nc"),
                                global_stds_path=str(stats_dir / "stats-stddev.nc"),
                            ),
                        ),
                        inference_scheme=InferenceSchemeConfig(
                            in_names=all_names,
                            out_names=["a", "b"],
                        ),
                    )
                ),
            ),
        ),
        stepper_training=TrainStepperConfig(
            loss=StepLossConfig(type="MSE"),
            n_forward_steps=2,
            task_sampling=TaskSamplingConfig(
                tasks=TaskWeights(
                    auto_encode=TaskConfig(probability=1.0),
                    infill=TaskConfig(probability=1.0),
                    prediction=TaskConfig(probability=1.0),
                    infill_prediction=TaskConfig(probability=1.0),
                    combined_all=TaskConfig(probability=1.0),
                ),
            ),
        ),
        inference=[],
        max_epochs=1,
        save_checkpoint=False,
        logging=LoggingConfig(
            log_to_screen=True, log_to_wandb=False, log_to_file=False
        ),
        experiment_dir=str(results_dir),
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml.dump(dataclasses.asdict(train_config)))
    with mock_wandb():
        train_main(yaml_config=f.name)
