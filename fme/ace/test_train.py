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
from fme.ace.data_loading.config import DataLoaderConfig
from fme.ace.data_loading.inference import (
    InferenceDataLoaderConfig,
    InferenceInitialConditionIndices,
)
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
from fme.core.dataset.config import XarrayDataConfig
from fme.core.generics.trainer import (
    _restore_checkpoint,
    count_parameters,
    epoch_checkpoint_enabled,
)
from fme.core.logging_utils import LoggingConfig
from fme.core.loss import WeightedMappingLossConfig
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

JOB_SUBMISSION_SCRIPT_PATH = (
    pathlib.PurePath(__file__).parent / "run-train-and-inference.sh"
)


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
    if use_healpix:
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
        log_to_file=False,
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
                    n_initial_conditions=2,
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
                    n_initial_conditions=2,
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
            batch_size=2,
            num_data_workers=0,
            time_buffer=time_buffer,
            sample_with_replacement=10,
        ),
        validation_loader=DataLoaderConfig(
            dataset=XarrayDataConfig(
                data_path=str(valid_data_path),
                spatial_dimensions=spatial_dimensions_str,
            ),
            batch_size=2,
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
            loss=WeightedMappingLossConfig(type="MSE"),
            crps_training=crps_training,
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
        n_forward_steps=n_forward_steps,
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
            save_prediction_files=True,
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
    "nettype, crps_training, log_validation_maps",
    [
        ("SphericalFourierNeuralOperatorNet", False, True),
        ("NoiseConditionedSFNO", True, False),
        ("HEALPixRecUNet", False, False),
        ("Samudra", False, False),
        ("NoiseConditionedSFNO", False, False),
    ],
)
def test_train_and_inference(
    tmp_path, nettype, crps_training, log_validation_maps: bool, very_fast_only: bool
):
    """Ensure that ACE training and subsequent standalone inference run without errors.

    Args:
        tmp_path: pytext fixture for temporary workspace.
        nettype: parameter indicating model architecture to use.
        very_fast_only: parameter indicating whether to skip slow tests.
    """
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")
    # need multi-year to cover annual aggregator
    train_config, inference_config = _setup(
        tmp_path,
        nettype,
        log_to_wandb=True,
        timestep_days=20,
        n_time=int(366 * 3 / 20 + 1),
        inference_forward_steps=int(366 * 3 / 20 / 2 - 1) * 2,  # must be even
        use_healpix=(nettype == "HEALPixRecUNet"),
        crps_training=crps_training,
        save_per_epoch_diagnostics=True,
        log_validation_maps=log_validation_maps,
    )
    # using pdb requires calling main functions directly
    with mock_wandb() as wandb:
        train_main(
            yaml_config=train_config,
        )
        wandb_logs = wandb.get_logs()

        for log in wandb_logs:
            # ensure inference time series is not logged
            assert "inference/mean/forecast_step" not in log

    validation_output_dir = tmp_path / "results" / "output" / "val" / "epoch_0001"
    assert validation_output_dir.exists()
    validation_diags = ["mean"]
    validation_map_diags = ["snapshot", "mean_map"]
    for diagnostic in validation_diags + validation_map_diags:
        diagnostic_output = validation_output_dir / f"{diagnostic}_diagnostics.nc"
        if diagnostic in validation_map_diags and not log_validation_maps:
            assert not diagnostic_output.exists()
        else:
            assert diagnostic_output.exists()
            ds = xr.open_dataset(diagnostic_output, decode_timedelta=False)
            assert len(ds) > 0

    inline_inference_output_dir = (
        tmp_path / "results" / "output" / "inference" / "epoch_0001"
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
    checkpoint_training_history = torch.load(best_checkpoint_path, weights_only=False)[
        "stepper"
    ].get("training_history")
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


def test_restore_checkpoint(tmp_path, very_fast_only: bool):
    """Test that restoring a checkpoint works."""
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")

    config_path, _ = _setup(tmp_path, "SphericalFourierNeuralOperatorNet")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    config = dacite.from_dict(
        data_class=TrainConfig, data=config_dict, config=dacite.Config(strict=True)
    )
    prepare_directory(config.experiment_dir, config_dict)

    builders = TrainBuilders(config)
    trainer1 = build_trainer(builders, config)

    # run one epoch
    trainer1.train_one_epoch()
    trainer1.save_all_checkpoints(0.1, 0.2)

    # reload and check model parameters and optimizer state
    trainer2 = build_trainer(builders, config)
    _restore_checkpoint(
        trainer2,
        trainer1.paths.latest_checkpoint_path,
        trainer1.paths.ema_checkpoint_path,
    )

    compare_restored_parameters(
        trainer1.stepper.modules.parameters(),
        trainer2.stepper.modules.parameters(),
        trainer1.optimization.optimizer,
        trainer2.optimization.optimizer,
    )


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


def _create_fine_tuning_config(path_to_train_config_yaml: str, path_to_checkpoint: str):
    # TODO(gideond) rename to "overwrite" or something of that nature
    with open(path_to_train_config_yaml) as config_file:
        config_data = yaml.safe_load(config_file)
        config_data["stepper"] = {"checkpoint_path": path_to_checkpoint}
        current_experiment_dir = config_data["experiment_dir"]
        new_experiment_dir = pathlib.Path(current_experiment_dir) / "fine_tuning"
        config_data["experiment_dir"] = str(new_experiment_dir)
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".yaml"
        ) as new_config_file:
            new_config_file.write(yaml.dump(config_data))

    return new_config_file.name, new_experiment_dir


@pytest.mark.parametrize("nettype", ["SphericalFourierNeuralOperatorNet"])
def test_fine_tuning(tmp_path, nettype, very_fast_only: bool):
    """Check that fine tuning config runs without errors."""
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")
    train_config, _ = _setup(tmp_path, nettype)

    train_main(yaml_config=train_config)

    results_dir = tmp_path / "results"
    ckpt = f"{results_dir}/training_checkpoints/best_ckpt.tar"

    fine_tuning_config, new_results_dir = _create_fine_tuning_config(train_config, ckpt)

    train_main(yaml_config=fine_tuning_config)
    assert (new_results_dir / "training_checkpoints" / "ckpt.tar").exists()


def _create_copy_weights_after_batch_config(
    path_to_train_config_yaml: str, path_to_checkpoint: str, experiment_dir: str
):
    with open(path_to_train_config_yaml) as config_file:
        config_data = yaml.safe_load(config_file)
        config_data["stepper"]["parameter_init"] = {"weights_path": path_to_checkpoint}
        config_data["copy_weights_after_batch"] = [{"include": ["*"], "exclude": []}]
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


@pytest.mark.parametrize(
    "checkpoint_save_epochs,expected_save_epochs",
    [(None, []), (Slice(start=-2), [2, 3]), (Slice(step=2), [0, 2])],
)
def test_epoch_checkpoint_enabled(checkpoint_save_epochs, expected_save_epochs):
    max_epochs = 4
    for i in range(max_epochs):
        if i in expected_save_epochs:
            assert epoch_checkpoint_enabled(i, max_epochs, checkpoint_save_epochs)
        else:
            assert not epoch_checkpoint_enabled(i, max_epochs, checkpoint_save_epochs)


@pytest.mark.parametrize(
    "module_list,expected_num_parameters",
    [
        (torch.nn.ModuleList([torch.nn.Linear(10, 5), torch.nn.Linear(5, 2)]), 67),
        (torch.nn.ModuleList([]), 0),
    ],
)
def test_count_parameters(module_list, expected_num_parameters):
    num_parameters = count_parameters(module_list)
    assert num_parameters == expected_num_parameters


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
        use_healpix=(nettype == "HEALPixRecUNet"),
        crps_training=crps_training,
        save_per_epoch_diagnostics=True,
        log_validation_maps=log_validation_maps,
        skip_inline_inference=True,
        time_buffer=2,
    )
    with mock_wandb() as wandb:
        train_main(
            yaml_config=train_config,
        )
        wandb_logs = wandb.get_logs()
    assert np.isinf(wandb_logs[-1]["best_inference_error"])
    assert not any("inference/" in key for key in wandb_logs[-1])
