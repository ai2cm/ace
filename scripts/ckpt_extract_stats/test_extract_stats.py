import dataclasses
import datetime
import pathlib
import tempfile

import pytest
import torch
import xarray as xr
from extract_stats import _find_normalization, extract_stats, write_stats

from fme.ace.registry import ModuleSelector
from fme.ace.stepper.single_module import StepperConfig
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset_info import DatasetInfo
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.step.multi_call import MultiCallStepConfig
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepSelector

TIMESTEP = datetime.timedelta(hours=6)
IN_NAMES = ["a", "b"]
OUT_NAMES = ["a", "b"]


class PlusOne(torch.nn.Module):
    def forward(self, x):
        return x + 1


def _build_stepper(
    network_means: dict[str, float],
    network_stds: dict[str, float],
    residual_stds: dict[str, float] | None = None,
):
    """Build a minimal stepper and return it."""
    residual = None
    if residual_stds is not None:
        residual = NormalizationConfig(
            means={k: 0.0 for k in residual_stds},
            stds=residual_stds,
        )
    config = StepperConfig(
        step=StepSelector(
            type="multi_call",
            config=dataclasses.asdict(
                MultiCallStepConfig(
                    wrapped_step=StepSelector(
                        type="single_module",
                        config=dataclasses.asdict(
                            SingleModuleStepConfig(
                                builder=ModuleSelector(
                                    type="prebuilt", config={"module": PlusOne()}
                                ),
                                in_names=IN_NAMES,
                                out_names=OUT_NAMES,
                                normalization=NetworkAndLossNormalizationConfig(
                                    network=NormalizationConfig(
                                        means=network_means,
                                        stds=network_stds,
                                    ),
                                    residual=residual,
                                ),
                            ),
                        ),
                    ),
                    include_multi_call_in_loss=False,
                ),
            ),
        ),
    )
    dataset_info = DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.zeros(4), lon=torch.zeros(8)
        ),
        vertical_coordinate=HybridSigmaPressureCoordinate(
            ak=torch.arange(7, dtype=torch.float32),
            bk=torch.arange(7, dtype=torch.float32),
        ),
        timestep=TIMESTEP,
        variable_metadata={
            OUT_NAMES[0]: VariableMetadata(units="K", long_name="temperature"),
        },
    )
    return config.get_stepper(dataset_info=dataset_info)


def _save_checkpoint(path: pathlib.Path, stepper):
    torch.save({"stepper": stepper.get_state()}, path)


class TestFindNormalization:
    def test_finds_normalization_in_nested_config(self):
        config = {
            "step": {
                "type": "multi_call",
                "config": {
                    "wrapped_step": {
                        "type": "single_module",
                        "config": {
                            "normalization": {
                                "network": {
                                    "means": {"x": 1.0},
                                    "stds": {"x": 2.0},
                                },
                                "residual": None,
                                "loss": None,
                            }
                        },
                    }
                },
            }
        }
        norm = _find_normalization(config)
        assert norm["network"]["means"] == {"x": 1.0}
        assert norm["network"]["stds"] == {"x": 2.0}

    def test_raises_when_no_normalization(self):
        with pytest.raises(ValueError, match="Could not find normalization"):
            _find_normalization({"step": {"config": {}}})


class TestExtractStatsModern:
    """Test extraction from modern-format checkpoints."""

    def test_network_only(self):
        means = {"a": 1.5, "b": -0.3}
        stds = {"a": 0.5, "b": 1.2}
        stepper = _build_stepper(network_means=means, network_stds=stds)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = pathlib.Path(tmpdir) / "ckpt.tar"
            _save_checkpoint(ckpt_path, stepper)
            stats = extract_stats(ckpt_path)

        assert "network-means.nc" in stats
        assert "network-stds.nc" in stats
        assert "residual-means.nc" not in stats
        assert "residual-stds.nc" not in stats
        assert "loss-means.nc" not in stats
        assert "loss-stds.nc" not in stats

        for name, expected in means.items():
            assert float(stats["network-means.nc"][name].values) == pytest.approx(
                expected
            )
        for name, expected in stds.items():
            assert float(stats["network-stds.nc"][name].values) == pytest.approx(
                expected
            )

    def test_network_with_residual(self):
        means = {"a": 1.5, "b": -0.3}
        stds = {"a": 0.5, "b": 1.2}
        residual_stds = {"a": 0.1, "b": 0.05}
        stepper = _build_stepper(
            network_means=means,
            network_stds=stds,
            residual_stds=residual_stds,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = pathlib.Path(tmpdir) / "ckpt.tar"
            _save_checkpoint(ckpt_path, stepper)
            stats = extract_stats(ckpt_path)

        assert "network-means.nc" in stats
        assert "network-stds.nc" in stats
        assert "residual-means.nc" in stats
        assert "residual-stds.nc" in stats

        for name, expected in residual_stds.items():
            assert float(stats["residual-stds.nc"][name].values) == pytest.approx(
                expected
            )


class TestExtractStatsLegacy:
    """Test extraction from legacy-format checkpoints."""

    def test_legacy_checkpoint(self):
        means = {"a": 2.0, "b": -1.0}
        stds = {"a": 0.8, "b": 1.5}
        loss_stds = {"a": 0.2, "b": 0.4}

        stepper = _build_stepper(network_means=means, network_stds=stds)
        modern_state = stepper.get_state()

        module_weights = modern_state["step"]["wrapped_step"]["module"]
        dataset_info = modern_state["dataset_info"]
        legacy_config = {
            "builder": {"type": "prebuilt", "config": {"module": PlusOne()}},
            "in_names": IN_NAMES,
            "out_names": OUT_NAMES,
            "normalization": {
                "global_means_path": None,
                "global_stds_path": None,
                "means": means,
                "stds": stds,
            },
        }
        legacy_stepper_state = {
            "config": legacy_config,
            "normalizer": {"means": means, "stds": stds},
            "loss_normalizer": {"means": means, "stds": loss_stds},
            "module": module_weights,
            "vertical_coordinate": dataset_info["vertical_coordinate"],
            "gridded_operations": {
                "type": "LatLonOperations",
                "state": {"area_weights": torch.ones(4, 8)},
            },
            "img_shape": [4, 8],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = pathlib.Path(tmpdir) / "legacy_ckpt.tar"
            torch.save({"stepper": legacy_stepper_state}, ckpt_path)
            stats = extract_stats(ckpt_path)

        assert "network-means.nc" in stats
        assert "network-stds.nc" in stats

        for name, expected in means.items():
            assert float(stats["network-means.nc"][name].values) == pytest.approx(
                expected
            )
        for name, expected in stds.items():
            assert float(stats["network-stds.nc"][name].values) == pytest.approx(
                expected
            )

        # Legacy loss_normalizer is converted to the "loss" normalization key
        assert "loss-means.nc" in stats
        assert "loss-stds.nc" in stats
        for name, expected in loss_stds.items():
            assert float(stats["loss-stds.nc"][name].values) == pytest.approx(expected)


class TestWriteStats:
    def test_write_creates_files(self):
        stats = {
            "network-means.nc": xr.Dataset({"x": xr.DataArray(1.0)}),
            "network-stds.nc": xr.Dataset({"x": xr.DataArray(2.0)}),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            write_stats(stats, tmpdir)
            for filename in stats:
                path = pathlib.Path(tmpdir) / filename
                assert path.exists()
                ds = xr.open_dataset(path)
                assert "x" in ds
