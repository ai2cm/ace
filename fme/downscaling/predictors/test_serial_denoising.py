import dataclasses
from unittest.mock import MagicMock

import pytest
import torch

from fme.core.coordinates import LatLonCoordinates
from fme.downscaling.data import StaticInput, StaticInputs
from fme.downscaling.models import CheckpointModelConfig
from fme.downscaling.predictors.serial_denoising import (
    DenoisingExpertCheckpointConfig,
    DenoisingMoEBundledConfig,
    DenoisingMoEConfig,
    DenoisingMoEPredictor,
    _SigmaDispatchModule,
    _validate_experts_compatible,
    _validate_sigma_ranges,
)
from fme.downscaling.test_models import (
    _get_diffusion_model,
    _get_monotonic_coordinate,
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


def _build_two_expert_predictor() -> DenoisingMoEPredictor:
    """Two-expert MoE with real DiffusionModel experts (different weights)."""
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
    return DenoisingMoEPredictor(
        experts=experts,
        sigma_ranges=[(0.1, 0.5), (0.5, 1.0)],
        num_diffusion_generation_steps=4,
        churn=0.25,
    )


def test_save_and_load_roundtrip_preserves_predictor(tmp_path):
    predictor = _build_two_expert_predictor()
    ckpt = tmp_path / "moe.pt"
    predictor.save(str(ckpt))

    loaded = DenoisingMoEBundledConfig(mixture_of_experts_path=str(ckpt)).build()

    assert loaded._sigma_ranges == predictor._sigma_ranges
    assert (
        loaded._num_diffusion_generation_steps
        == predictor._num_diffusion_generation_steps
    )
    assert loaded._churn == predictor._churn
    assert len(loaded._experts) == len(predictor._experts)
    for orig_expert, new_expert in zip(predictor._experts, loaded._experts):
        for p_orig, p_new in zip(
            orig_expert.module.parameters(), new_expert.module.parameters()
        ):
            assert torch.equal(p_orig.cpu(), p_new.cpu())


def test_loaded_predictor_dispatches_to_same_experts(tmp_path):
    """The reloaded predictor must route the same sigma to the same expert
    (i.e. produce bitwise-identical generations)."""
    predictor = _build_two_expert_predictor()
    ckpt = tmp_path / "moe.pt"
    predictor.save(str(ckpt))
    loaded = DenoisingMoEBundledConfig(mixture_of_experts_path=str(ckpt)).build()

    x = torch.randn(
        1, 1, 16, 32, device=next(predictor._experts[0].module.parameters()).device
    )
    x_lr = torch.randn_like(x)
    for sigma_val in [0.2, 0.7]:
        sigma = torch.tensor(sigma_val)
        orig = predictor._dispatch_module(x, x_lr, sigma)
        new = loaded._dispatch_module(x, x_lr, sigma)
        assert torch.allclose(orig.cpu(), new.cpu(), atol=1e-5, rtol=1e-5)


def test_checkpoint_config_data_requirements(tmp_path):
    predictor = _build_two_expert_predictor()
    ckpt = tmp_path / "moe.pt"
    predictor.save(str(ckpt))

    reqs = DenoisingMoEBundledConfig(
        mixture_of_experts_path=str(ckpt)
    ).data_requirements
    assert reqs.fine_names == ["x"]
    assert set(reqs.coarse_names) == {"x"}
    assert reqs.n_timesteps == 1
    assert reqs.use_fine_topography is False


def test_save_preserves_rename_applied_by_checkpoint_model_config(tmp_path):
    """``CheckpointModelConfig.rename`` mutates in_names/out_names at load time,
    so the renamed names are part of the built ``DiffusionModel.config`` and must
    survive the MoE save/reload roundtrip.
    """
    coarse_shape = (8, 16)
    fine_shape = (16, 32)
    fine_coords = make_fine_coords(fine_shape)

    expert_ckpts: list[str] = []
    for i in range(2):
        # Train-time names use "x"; downstream consumers will see "renamed_x".
        model = _get_diffusion_model(
            coarse_shape=coarse_shape,
            downscale_factor=2,
            full_fine_coords=fine_coords,
            predict_residual=False,
            use_fine_topography=False,
            static_inputs=StaticInputs(fields=[], coords=fine_coords),
        )
        path = tmp_path / f"expert_{i}.ckpt"
        torch.save({"model": model.get_state()}, path)
        expert_ckpts.append(str(path))

    moe_config = DenoisingMoEConfig(
        denoising_expert_configs=[
            DenoisingExpertCheckpointConfig(
                checkpoint_config=CheckpointModelConfig(
                    checkpoint_path=p, rename={"x": "renamed_x"}
                ),
                sigma_min=lo,
                sigma_max=hi,
            )
            for p, (lo, hi) in zip(expert_ckpts, [(0.1, 0.5), (0.5, 1.0)])
        ],
        num_diffusion_generation_steps=4,
        churn=0.25,
    )
    predictor = moe_config.build()
    # Sanity: each expert exposes runtime (renamed) names while config retains
    # the original training-time names. Rename is owned by the predictor, not
    # the model.
    for expert in predictor._experts:
        assert expert.in_names == ["renamed_x"]
        assert expert.out_names == ["renamed_x"]
        assert expert.config.in_names == ["x"]
        assert expert.config.out_names == ["x"]
    assert predictor._expert_renames == [{"x": "renamed_x"}, {"x": "renamed_x"}]

    bundle_path = tmp_path / "moe.pt"
    predictor.save(str(bundle_path))
    loaded = DenoisingMoEBundledConfig(mixture_of_experts_path=str(bundle_path)).build()

    for expert in loaded._experts:
        assert expert.in_names == ["renamed_x"]
        assert expert.out_names == ["renamed_x"]
        assert expert.config.in_names == ["x"]
        assert expert.config.out_names == ["x"]
    assert loaded._expert_renames == [{"x": "renamed_x"}, {"x": "renamed_x"}]

    reqs = DenoisingMoEBundledConfig(
        mixture_of_experts_path=str(bundle_path)
    ).data_requirements
    assert reqs.fine_names == ["renamed_x"]
    assert set(reqs.coarse_names) == {"renamed_x"}


def _make_global_fine_coords_and_static(fine_shape: tuple[int, int]):
    """Return a global-covering LatLonCoordinates and matching StaticInputs."""
    step = 360 / fine_shape[1]
    global_fine_lon = torch.arange(fine_shape[1]) * step + step / 2
    global_fine_lat = _get_monotonic_coordinate(fine_shape[0], stop=fine_shape[0])
    full_fine_coords = LatLonCoordinates(lat=global_fine_lat, lon=global_fine_lon)
    static_field = torch.arange(
        fine_shape[0] * fine_shape[1], dtype=torch.float32
    ).reshape(*fine_shape)
    static_inputs = StaticInputs(
        fields=[StaticInput(static_field)], coords=full_fine_coords
    )
    return full_fine_coords, static_inputs


def test_denoising_moe_predictor_rejects_mismatched_expert_grids():
    """Experts on different grids are rejected at construction (shared-grid)."""

    fine_coords_a, static_a = _make_global_fine_coords_and_static((16, 32))
    fine_coords_b, static_b = _make_global_fine_coords_and_static((16, 16))
    expert_a = _get_diffusion_model(
        coarse_shape=(8, 16),
        downscale_factor=2,
        full_fine_coords=fine_coords_a,
        static_inputs=static_a,
    )
    expert_b = _get_diffusion_model(
        coarse_shape=(8, 8),
        downscale_factor=2,
        full_fine_coords=fine_coords_b,
        static_inputs=static_b,
    )
    with pytest.raises(ValueError, match="metadata"):
        DenoisingMoEPredictor(
            experts=[expert_a, expert_b],
            sigma_ranges=[(0.0, 0.5), (0.5, 1.0)],
            num_diffusion_generation_steps=2,
            churn=0.0,
        )
