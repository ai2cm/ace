import copy
import dataclasses
import pathlib
import subprocess
import tempfile
import unittest.mock
from typing import Literal
from pathlib import Path
import dacite
import numpy as np
import pytest
import torch
import xarray as xr
import yaml
import fme
from fme.ace.aggregator.inference.main import InferenceEvaluatorAggregatorConfig
from fme.ace.aggregator.one_step.main import OneStepAggregatorConfig
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
    recurrent_block_config,
    up_sampling_block_config,
)
from fme.ace.stepper.single_module import StepperConfig
from fme.ace.stepper.time_length_probabilities import (
    TimeLengthProbabilities,
    TimeLengthProbability,
)
from fme.ace.testing import (
    DimSizes,
    MonthlyReferenceData,
    save_nd_netcdf,
    save_scalar_netcdf,
)
from fme.ace.train.train import build_trainer, prepare_directory
from fme.ace.train.train import main as train_main
from fme.ace.train.train_config import (
    InlineInferenceConfig,
    TrainBuilders,
    TrainConfig,
    WeatherEvaluationConfig,
)
from fme.core.coordinates import (
    HEALPixCoordinates,
    HorizontalCoordinates,
    LatLonCoordinates,
)
from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.generics.trainer import (
    _restore_checkpoint,
    count_parameters,
    epoch_checkpoint_enabled,
)
from fme.core.logging_utils import LoggingConfig
from fme.core.loss import StepLossConfig
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.ocean import OceanConfig
from fme.core.optimization import OptimizationConfig
from fme.core.registry.corrector import CorrectorSelector
from fme.core.registry.module import ModuleSelector
from fme.core.scheduler import SchedulerConfig
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepSelector
from fme.core.testing.model import compare_restored_parameters
from fme.core.testing.wandb import mock_wandb
from fme.core.typing_ import Slice
from fme.core.distributed import Distributed
JOB_SUBMISSION_SCRIPT_PATH = (
    pathlib.PurePath(__file__).parent / "run-train-and-inference.sh"
)

# @pytest.fixture
# def custom_tmp_path(request):
#     # Create a temporary directory
#     temp_dir = tempfile.mkdtemp()
#     # Yield the path to the temporary directory
#     yield Path(temp_dir)

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
):
    input_time_size = 1
    output_time_size = 1
    if nettype == "HEALPixRecUNet":
        in_channels = len(in_variable_names)
        out_channels = len(out_variable_names)
        prognostic_variables = min(
            out_channels, in_channels
        )  # how many variables in/out share.
        # in practice, we will need to compare variable names, since there
        # are some input-only and some output-only channels.
        # TODO: https://github.com/ai2cm/full-model/issues/1046
        n_constants = 0
        decoder_input_channels = 0  # was 1, to indicate insolation - now 0
        input_time_size = 1  # TODO: change to 2 (issue #1177)
        output_time_size = 1  # TODO: change to 4 (issue #1177)

        conv_next_block = conv_next_block_config(in_channels=in_channels)
        down_sampling_block = down_sampling_block_config()
        recurrent_block = recurrent_block_config()
        encoder = encoder_config(
            conv_next_block, down_sampling_block, n_channels=[16, 8, 4]
        )
        up_sampling_block = up_sampling_block_config()
        output_layer = output_layer_config()
        decoder = decoder_config(
            conv_next_block,
            up_sampling_block,
            output_layer,
            recurrent_block,
            n_channels=[4, 8, 16],
        )
        net_config = dict(
            encoder=encoder,
            decoder=decoder,
            prognostic_variables=prognostic_variables,
            n_constants=n_constants,
            decoder_input_channels=decoder_input_channels,
            input_time_size=input_time_size,
            output_time_size=output_time_size,
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
        )
        if use_healpix:
            net_config["data_grid"] = "healpix"
            spatial_dimensions_str = "healpix"
        else:
            spatial_dimensions_str = "latlon"

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
        log_to_file=True,
        project="fme",
        entity="ai2cm",
    )
    if skip_inline_inference:
        inline_inference_config = None
        weather_evaluation_config = None
    else:
        inline_inference_config = InlineInferenceConfig(
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
                ),
                start_indices=InferenceInitialConditionIndices(
                    first=0,
                    n_initial_conditions=4,
                    interval=1,
                ),
            ),
            n_forward_steps=inference_forward_steps,
            forward_steps_in_memory=2,
        )
        weather_evaluation_config = WeatherEvaluationConfig(
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
                ),
                start_indices=InferenceInitialConditionIndices(
                    first=0,
                    n_initial_conditions=4,
                    interval=1,
                ),
            ),
            n_forward_steps=inference_forward_steps,
            forward_steps_in_memory=2,
        )

    train_config = TrainConfig(
        train_loader=DataLoaderConfig(
            dataset=XarrayDataConfig(
                data_path=str(train_data_path),
                spatial_dimensions=spatial_dimensions_str,
            ),
            batch_size=4,
            num_data_workers=0,
            time_buffer=time_buffer,
            sample_with_replacement=10,
        ),
        validation_loader=DataLoaderConfig(
            dataset=XarrayDataConfig(
                data_path=str(valid_data_path),
                spatial_dimensions=spatial_dimensions_str,
            ),
            batch_size=4,
            num_data_workers=0,
        ),
        optimization=OptimizationConfig(
            use_gradient_accumulation=True,
            optimizer_type="Adam",
            lr=0.001,
            kwargs=dict(weight_decay=0.01),
            scheduler=SchedulerConfig(
                type="CosineAnnealingLR",
                kwargs=dict(T_max=1),
            ),
        ),
        stepper=StepperConfig(
            loss=StepLossConfig(type="MSE"),
            crps_training=crps_training,
            train_n_forward_steps=TimeLengthProbabilities(
                outcomes=[
                    TimeLengthProbability(steps=1, probability=0.5),
                    TimeLengthProbability(steps=n_forward_steps, probability=0.5),
                ]
            ),
            step=StepSelector(
                type="single_module",
                config=dataclasses.asdict(
                    SingleModuleStepConfig(
                        crps_training=crps_training,
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
                            config=net_config,
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
        inference=inline_inference_config,
        weather_evaluation=weather_evaluation_config,
        max_epochs=max_epochs,
        segment_epochs=segment_epochs,
        save_checkpoint=True,
        logging=logging_config,
        experiment_dir=str(results_dir),
        save_per_epoch_diagnostics=save_per_epoch_diagnostics,
        validation_aggregator=OneStepAggregatorConfig(
            log_snapshots=log_validation_maps,
            log_mean_maps=log_validation_maps,
        ),
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
        aggregator=InferenceEvaluatorAggregatorConfig(
            log_video=True,
        ),
        logging=logging_config,
        loader=InferenceDataLoaderConfig(
            dataset=XarrayDataConfig(
                data_path=str(valid_data_path),
                spatial_dimensions=spatial_dimensions_str,
            ),
            start_indices=InferenceInitialConditionIndices(
                first=0,
                n_initial_conditions=4,
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
):
    if not path.exists():
        path.mkdir()
    seed = 0
    np.random.seed(seed)
    in_variable_names = [
        "PRESsfc",
        "specific_total_water_0",
        "specific_total_water_1",
        "surface_temperature",
        "baz",
    ]
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
    save_nd_netcdf(
        data_dir / "data.nc",
        dim_sizes,
        variable_names=all_variable_names + [mask_name],
        timestep_days=timestep_days,
    )
    save_scalar_netcdf(
        stats_dir / "stats-mean.nc",
        variable_names=all_variable_names,
    )
    save_scalar_netcdf(
        stats_dir / "stats-stddev.nc",
        variable_names=all_variable_names,
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
    )
    return train_config_filename, inference_config_filename


@pytest.mark.parametrize(
    "nettype, crps_training, log_validation_maps, use_healpix",
    [
        ("SphericalFourierNeuralOperatorNet", False, True, False),
    ],
)
def test_train_and_inference(
    tmp_path,
    nettype,
    crps_training,
    log_validation_maps: bool,
    use_healpix: bool,
    very_fast_only: bool,
):
    """Ensure that ACE training and subsequent standalone inference run without errors.

    Args:
        tmp_path: pytext fixture for temporary workspace.
        nettype: parameter indicating model architecture to use.
        very_fast_only: parameter indicating whether to skip slow tests.
    """
    if very_fast_only:
      pytest.skip("Skipping non-fast tests")
    # Let's generate the configuration file on a single processor.
    with Distributed.non_distributed():
      train_config, inference_config = _setup(
        tmp_path,
        nettype,
        log_to_wandb=False,
        timestep_days=20,
        n_time=int(366 * 3 / 20 + 1),
        inference_forward_steps=50,#int(366 * 3 / 20 / 2 - 1) * 2,  # must be even
        use_healpix=use_healpix,
        crps_training=crps_training,
        save_per_epoch_diagnostics=True,
        log_validation_maps=log_validation_maps,
      )
      # return
    # with mock_wandb() as wandb:
    train_main(
            yaml_config=train_config
    )
      # wandb_logs = wandb.get_logs()
        # for log in wandb_logs:
        #     # ensure inference time series is not logged
        #     assert "inference/mean/forecast_step" not in log

        # epoch_logs = wandb_logs[-1]
        # assert "inference/mean_step_20_norm/weighted_rmse/channel_mean" in epoch_logs
        # assert "val/mean_norm/weighted_rmse/channel_mean" in epoch_logs

    # train_main(
    #         yaml_config=train_config,
    #     )
