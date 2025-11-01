import subprocess
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import xarray as xr
import yaml

from fme.core.dataset.time import TimeSlice
from fme.core.logging_utils import LoggingConfig
from fme.downscaling.data import Topography
from fme.downscaling.generate.generate import Downscaler, GenerationConfig
from fme.downscaling.generate.output import EventConfig, OutputTarget, RegionConfig
from fme.downscaling.models import (
    CheckpointModelConfig,
    DiffusionModelConfig,
    DiffusionModuleRegistrySelector,
    LossConfig,
    NormalizationConfig,
    PairedNormalizationConfig,
)
from fme.downscaling.predictors import PatchPredictionConfig, PatchPredictor
from fme.downscaling.test_evaluator import LinearDownscalingDiffusion

# Fixtures


@pytest.fixture
def mock_model():
    """Create a mock model with coarse_shape attribute."""
    model = MagicMock()
    model.coarse_shape = (16, 16)
    model.fine_shape = (32, 32)
    return model


@pytest.fixture
def mock_output_target():
    """Create a mock OutputTarget."""
    target = MagicMock(spec=OutputTarget)
    target.name = "test_target"
    target.save_vars = ["var1", "var2"]
    target.n_ens = 4
    target.patch = PatchPredictionConfig()
    target.data = []
    return target


@pytest.fixture
def mock_topography():
    """Create a mock Topography with shape."""

    def create_topo(shape=(16, 16)):
        topo = MagicMock(spec=Topography)
        data_mock = MagicMock()
        data_mock.shape = shape
        topo.data = data_mock
        topo.coords = MagicMock()
        return topo

    return create_topo


# Tests for Downscaler initialization


def test_downscaler_initialization(mock_model, mock_output_target):
    """Test that Downscaler can be instantiated with required fields."""
    downscaler = Downscaler(
        model=mock_model,
        output_targets=[mock_output_target],
        output_dir="/test/output",
    )

    assert downscaler.model is mock_model
    assert downscaler.output_targets == [mock_output_target]
    assert downscaler.output_dir == "/test/output"


# Tests for Downscaler._get_generation_model


def test_get_generation_model_exact_match(
    mock_model, mock_output_target, mock_topography
):
    """
    Test _get_generation_model returns model unchanged when shapes match exactly.
    """
    mock_model.fine_shape = (16, 16)
    topo = mock_topography(shape=(16, 16))

    downscaler = Downscaler(
        model=mock_model,
        output_targets=[mock_output_target],
    )

    result = downscaler._get_generation_model(
        topography=topo,
        target=mock_output_target,
    )

    assert result is mock_model


@pytest.mark.parametrize("topo_shape", [(8, 16), (16, 8), (8, 8)])
def test_get_generation_model_raises_when_domain_too_small(
    mock_model, mock_output_target, mock_topography, topo_shape
):
    """
    Test _get_generation_model raises ValueError when domain is
    smaller than model.
    """
    mock_model.fine_shape = (16, 16)
    topo = mock_topography(shape=topo_shape)

    downscaler = Downscaler(
        model=mock_model,
        output_targets=[mock_output_target],
    )

    with pytest.raises(ValueError):
        downscaler._get_generation_model(
            topography=topo,
            target=mock_output_target,
        )


def test_get_generation_model_creates_patch_predictor_when_needed(
    mock_model, mock_output_target, mock_topography
):
    """
    Test _get_generation_model creates PatchPredictor for
    large domains with patching.
    """
    mock_model.fine_shape = (16, 16)
    topo = mock_topography(shape=(32, 32))  # Larger than model

    patch_config = PatchPredictionConfig(
        divide_generation=True,
        composite_prediction=True,
        coarse_horizontal_overlap=2,
    )
    mock_output_target.patch = patch_config
    downscaler = Downscaler(
        model=mock_model,
        output_targets=[mock_output_target],
    )

    model = downscaler._get_generation_model(
        topography=topo,
        target=mock_output_target,
    )

    assert isinstance(model, PatchPredictor)
    assert model.coarse_horizontal_overlap == 2


def test_get_generation_model_raises_when_large_domain_without_patching(
    mock_model, mock_output_target, mock_topography
):
    """
    Test _get_generation_model raises when domain is large but patching
    not configured.
    """
    mock_model.fine_shape = (16, 16)
    topo = mock_topography(shape=(32, 32))  # Larger than model
    mock_output_target.patch = PatchPredictionConfig(divide_generation=False)

    downscaler = Downscaler(
        model=mock_model,
        output_targets=[mock_output_target],
    )

    with pytest.raises(ValueError):
        downscaler._get_generation_model(
            topography=topo,
            target=mock_output_target,
        )


def test_run_target_generation_skips_padding_items(
    mock_model, mock_output_target, mock_topography
):
    """Test run_target_generation skips writing output for padding items."""
    # Create padding work item
    mock_work_item = MagicMock()
    mock_work_item.is_padding = True
    mock_work_item.n_ens = 4
    mock_work_item.batch = MagicMock()

    mock_topo = mock_topography(shape=(16, 16))

    mock_output_target.data = [(mock_work_item, mock_topo)]
    mock_model.fine_shape = (16, 16)

    mock_output = {
        "var1": torch.randn(1, 4, 16, 16),
        "var2": torch.randn(1, 4, 16, 16),
    }
    mock_model.generate_on_batch_no_target.return_value = mock_output

    mock_writer = MagicMock()
    mock_output_target.get_writer.return_value = mock_writer

    downscaler = Downscaler(
        model=mock_model,
        output_targets=[mock_output_target],
    )

    downscaler.run_target_generation(target=mock_output_target)

    # Verify model was still called
    mock_model.generate_on_batch_no_target.assert_called_once()

    # Verify writer did NOT record the output
    mock_writer.record_batch.assert_not_called()


def get_generate_model_config():
    return DiffusionModelConfig(
        DiffusionModuleRegistrySelector(
            "prebuilt",
            {
                "module": LinearDownscalingDiffusion(
                    factor=1,
                    fine_img_shape=(16, 16),
                    n_channels=3,  # x, y, topography
                    n_out_channels=2,  # x, y
                )
            },
            expects_interpolated_input=True,
        ),
        loss=LossConfig("NaN"),
        in_names=["x", "y"],
        out_names=["x", "y"],
        normalization=PairedNormalizationConfig(
            NormalizationConfig(means={"x": 0.0, "y": 0.0}, stds={"x": 1.0, "y": 1.0}),
            NormalizationConfig(means={"x": 0.0, "y": 0.0}, stds={"x": 1.0, "y": 1.0}),
        ),
        p_mean=0,
        p_std=1,
        sigma_min=1,
        sigma_max=2,
        churn=1,
        num_diffusion_generation_steps=2,
        predict_residual=True,
    )


@pytest.fixture
def checkpointed_model_config(tmp_path):
    """Create and return a path to a checkpointed model for testing."""
    model_config = get_generate_model_config()
    # TODO: ensure this gets connected to centralized data helper
    coarse_shape = (8, 8)
    model_config.use_fine_topography = True
    model = model_config.build(coarse_shape, 2)

    checkpoint_path = tmp_path / "model_checkpoint.pth"
    model.get_state()
    torch.save({"model": model.get_state()}, checkpoint_path)

    return CheckpointModelConfig(
        checkpoint_path=str(checkpoint_path),
    )


@pytest.fixture
def generation_config(tmp_path, loader_config, checkpointed_model_config):
    """Create a GenerationConfig for testing."""
    region_config = RegionConfig(
        name="test_region",
        time_range=TimeSlice("2000-01-01T00:00:00", "2000-01-03T00:00:00"),
        n_ens=2,
        save_vars=["x", "y"],
        zarr_chunks={"time": 1, "ens": 1},
        zarr_shards={"time": 3, "ens": 2},
    )
    event_config = EventConfig(
        name="test_event",
        event_time="2000-01-02T00:00:00",
        n_ens=4,
        save_vars=["x"],
        zarr_chunks={"time": 1, "ens": 1},
        zarr_shards={"time": 1, "ens": 2},
    )

    output_dir = tmp_path / "generation_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return GenerationConfig(
        model=checkpointed_model_config,
        data=loader_config,
        experiment_dir=str(output_dir),
        output_targets=[region_config, event_config],
        logging=LoggingConfig(log_to_screen=True, log_to_wandb=False),
        patch=PatchPredictionConfig(divide_generation=True),
    )


@pytest.fixture
def generation_config_path(generation_config):
    output_dir = generation_config.experiment_dir

    # save generation config to yaml
    config_path = Path(output_dir) / "generation_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(asdict(generation_config), f)

    return config_path


@pytest.mark.parametrize(
    "multi_gpu",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                (not torch.cuda.is_available() or torch.cuda.device_count() < 2),
                reason="Skipping multi-GPU test: less than 2 GPUs available.",
            ),
        ),
    ],
)
def test_generation_main(generation_config_path, multi_gpu, skip_slow):
    """Test the main generation process end-to-end."""
    if skip_slow:
        pytest.skip("Skipping slow test.")

    if multi_gpu:
        command = [
            "torchrun",
            "--nproc_per_node",
            "2",
            "-m",
            "fme.downscaling.generate",
            str(generation_config_path),
        ]
    else:
        command = [
            "python",
            "-m",
            "fme.downscaling.generate",
            str(generation_config_path),
        ]

    subprocess.run(command, check=True)
    output_dir = generation_config_path.parent
    test_region_zarr = output_dir / "test_region.zarr"
    test_event_zarr = output_dir / "test_event.zarr"

    assert (output_dir / "test_region.zarr").exists()
    assert (output_dir / "test_event.zarr").exists()
    region = xr.open_zarr(test_region_zarr)
    assert "x" in region.data_vars
    assert "y" in region.data_vars
    assert region["x"].notnull().all()

    event = xr.open_zarr(test_event_zarr)
    assert "x" in event.data_vars
    assert "y" not in event.data_vars
    assert event["x"].notnull().all()
