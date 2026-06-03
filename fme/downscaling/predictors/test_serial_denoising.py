import dataclasses
from unittest.mock import MagicMock

import pytest
import torch

from fme.core.coordinates import LatLonCoordinates
from fme.downscaling.data import StaticInputs
from fme.downscaling.predictors.serial_denoising import (
    DenoisingExpertCheckpointConfig,
    DenoisingMoEConfig,
    DenoisingMoEPredictor,
    _SigmaDispatchModule,
    _validate_experts_compatible,
    _validate_sigma_ranges,
)
from fme.downscaling.test_models import (
    _get_diffusion_model,
    make_fine_coords,
    make_paired_batch_data,
)


def _range(sigma_min: float, sigma_max: float) -> DenoisingExpertCheckpointConfig:
    return DenoisingExpertCheckpointConfig(
        checkpoint_config=MagicMock(),
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )


def test_validate_sigma_ranges_min_ge_max_raises():
    with pytest.raises(ValueError, match="sigma_min < sigma_max"):
        _validate_sigma_ranges([(10.0, 10.0)])


def test_validate_sigma_ranges_gap_raises():
    with pytest.raises(ValueError, match="contiguous"):
        _validate_sigma_ranges([(0.0, 10.0), (11.0, 80.0)])


def test_denoising_moe_config_sorts_ranges():
    cfg = DenoisingMoEConfig(
        denoising_expert_configs=[_range(10.0, 80.0), _range(0.002, 10.0)],
        num_diffusion_generation_steps=10,
    )
    assert cfg.denoising_expert_configs[0].sigma_min == 0.002
    assert cfg.denoising_expert_configs[1].sigma_min == 10.0


class _TrackedExpert(torch.nn.Module):
    def __init__(self, which: int, calls: list[int]):
        super().__init__()
        self.which = which
        self._calls = calls

    def forward(self, x, x_lr, sigma):
        self._calls.append(self.which)
        return float(self.which)


def _two_expert_dispatch():
    calls: list[int] = []
    e0 = _TrackedExpert(0, calls)
    e1 = _TrackedExpert(1, calls)
    dispatch = _SigmaDispatchModule([(1.0, 10.0), (10.0, 20.0)], [e0, e1])
    return dispatch, calls


def test_sigma_dispatch_routes_to_correct_expert():
    dispatch, calls = _two_expert_dispatch()
    x = torch.ones(1, 1, 4, 4)
    assert dispatch(x, x, torch.tensor(0.0)) == 0.0
    assert dispatch(x, x, torch.tensor(5.0)) == 0.0
    assert dispatch(x, x, torch.tensor(15.0)) == 1.0
    assert dispatch(x, x, torch.tensor(25.0)) == 1.0
    assert calls == [0, 0, 1, 1]


def test_sigma_dispatch_boundary_prefers_lower_sigma_range():
    dispatch, calls = _two_expert_dispatch()
    x = torch.ones(1, 1, 4, 4)
    out = dispatch(x, x, torch.tensor(10.0))
    assert out == 0.0
    assert calls == [0]


_SHARED_METADATA = object()


@dataclasses.dataclass
class _MockMetadata:
    in_names: tuple[str, ...] = ("a", "b")
    out_names: tuple[str, ...] = ("c",)
    coarse_shape: tuple[int, int] = (2, 2)
    downscale_factor: int = 4
    predict_residual: bool = False
    static_inputs: StaticInputs | None = None
    full_fine_coords: LatLonCoordinates = dataclasses.field(
        default_factory=lambda: LatLonCoordinates(
            lat=torch.linspace(-90, 90, 16), lon=torch.linspace(0, 360, 32)
        )
    )
    has_static_inputs: bool = False
    static_inputs_coords: LatLonCoordinates | None = dataclasses.field(
        default_factory=lambda: LatLonCoordinates(
            lat=torch.linspace(-90, 90, 16), lon=torch.linspace(0, 360, 32)
        )
    )


def _mock_expert(metadata=_SHARED_METADATA):
    expert = MagicMock()
    expert.in_packer.names = ["a", "b"]
    expert.out_packer.names = ["c"]
    expert.coarse_shape = (4, 8)
    expert.downscale_factor = 4
    expert.config.predict_residual = False
    expert.static_inputs = _mock_static_inputs()
    expert.metadata = metadata
    return expert


def _mock_static_inputs(n_lat: int = 16, n_lon: int = 32):
    si = MagicMock()
    si.coords = LatLonCoordinates(
        lat=torch.linspace(-90, 90, n_lat), lon=torch.linspace(0, 360, n_lon)
    )
    return si


def test_validate_experts_compatible():
    e0 = _mock_expert(metadata=_MockMetadata(static_inputs=None))
    e1 = _mock_expert(
        metadata=_MockMetadata(static_inputs=_mock_static_inputs(n_lat=16, n_lon=32))
    )
    with pytest.raises(ValueError, match="metadata"):
        _validate_experts_compatible([e0, e1])


def test_generate_on_batch_returns_scalar_loss():
    """``DenoisingMoEPredictor.generate_on_batch`` must reduce the per-component
    loss list returned by ``self._primary.loss(...)`` to a scalar tensor, the
    same way ``DiffusionModel.generate_on_batch`` does.

    Regression for the post-#1159 loss-architecture refactor: the loss module
    now returns ``list[LossComponent]``; if the MoE predictor forgets to unpack
    and ``.mean()`` it, ``ModelOutputs.loss`` is a list and downstream
    consumers (``PatchPredictor`` accumulator, loss aggregator) explode with a
    confusing ``TypeError`` or ``AttributeError`` deep in unrelated code.
    """
    coarse_shape = (8, 16)
    fine_shape = (16, 32)
    fine_coords = make_fine_coords(fine_shape)
    experts = [
        _get_diffusion_model(
            coarse_shape=coarse_shape,
            downscale_factor=2,
            full_fine_coords=fine_coords,
            predict_residual=False,
            use_fine_topography=False,
            static_inputs=StaticInputs(fields=[], coords=fine_coords),
        )
        for _ in range(2)
    ]
    predictor = DenoisingMoEPredictor(
        experts=experts,
        sigma_ranges=[(0.1, 0.5), (0.5, 1.0)],
        num_diffusion_generation_steps=4,
        churn=0.0,
    )
    batch = make_paired_batch_data(coarse_shape, fine_shape, batch_size=1)
    outputs = predictor.generate_on_batch(batch, n_samples=1)
    assert isinstance(outputs.loss, torch.Tensor)
    assert outputs.loss.ndim == 0
