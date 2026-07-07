import dataclasses
from unittest.mock import MagicMock

import pytest
import torch

from fme.core.coordinates import LatLonCoordinates
from fme.downscaling.data import StaticInputs
from fme.downscaling.models import CheckpointModelConfig
from fme.downscaling.predictors.serial_denoising import (
    DenoisingExpertCheckpointConfig,
    DenoisingMoEBundledConfig,
    DenoisingMoEConfig,
    DenoisingMoEPredictor,
    DenoisingMoEStudentConfig,
    DenoisingMoEStudentPredictor,
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

    # A teacher bundle (no sampler_type tag) loads as the EDM predictor, not the
    # student cascade — the backward-compatible side of the dispatch.
    assert type(loaded) is DenoisingMoEPredictor
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


def _build_student_predictor(
    steps_per_range: list[int],
    sigma_ranges: list[tuple[float, float]] | None = None,
    predict_residual: bool = False,
) -> DenoisingMoEStudentPredictor:
    """Two-student cascade predictor with real DiffusionModel experts."""
    coarse_shape = (8, 16)
    fine_shape = (16, 32)
    fine_coords = make_fine_coords(fine_shape)
    experts = [
        _get_diffusion_model(
            coarse_shape=coarse_shape,
            downscale_factor=2,
            full_fine_coords=fine_coords,
            predict_residual=predict_residual,
            use_fine_topography=False,
            static_inputs=StaticInputs(fields=[], coords=fine_coords),
        )
        for _ in range(2)
    ]
    return DenoisingMoEStudentPredictor(
        experts=experts,
        sigma_ranges=sigma_ranges or [(0.1, 0.5), (0.5, 1.0)],
        steps_per_range=steps_per_range,
    )


def _record_dispatch_sigmas(
    predictor: DenoisingMoEStudentPredictor,
) -> tuple[list[float], list[float]]:
    """Register forward-pre-hooks capturing the sigma each expert is queried at.

    Returns (lo_sigmas, hi_sigmas) — experts are sorted ascending, so index 0 is
    the low-noise segment and index 1 the high-noise segment.
    """
    lo_sigmas: list[float] = []
    hi_sigmas: list[float] = []
    recorders = [lo_sigmas, hi_sigmas]

    def make_hook(sink: list[float]):
        def hook(_module, args):
            sink.append(float(args[2].reshape(-1)[0].item()))

        return hook

    for expert, sink in zip(predictor._experts, recorders):
        expert.module.register_forward_pre_hook(make_hook(sink))
    return lo_sigmas, hi_sigmas


def test_student_predictor_cascade_routes_high_then_low():
    """The student cascade queries the high segment above the boundary and the
    low segment at/below it — the fastgen predict-x0-renoise handoff, not the
    teacher's continuous EDM grid."""
    predictor = _build_student_predictor(steps_per_range=[1, 1])
    lo_sigmas, hi_sigmas = _record_dispatch_sigmas(predictor)

    batch = make_paired_batch_data((8, 16), (16, 32), batch_size=1)
    predictor.generate_on_batch_no_target(batch.coarse, n_samples=1)

    assert len(hi_sigmas) == 1 and hi_sigmas[0] > 0.5
    assert len(lo_sigmas) == 1 and lo_sigmas[0] <= 0.5


def test_student_predictor_uses_boundary_aligned_t_list():
    """The per-step sigmas equal ``boundary_aligned_t_list`` (with the boundary
    node present), distinguishing the fastgen cascade from the teacher's
    continuous Karras grid."""
    from fme.downscaling.samplers import boundary_aligned_t_list

    predictor = _build_student_predictor(steps_per_range=[1, 1])
    lo_sigmas, hi_sigmas = _record_dispatch_sigmas(predictor)

    batch = make_paired_batch_data((8, 16), (16, 32), batch_size=1)
    predictor.generate_on_batch_no_target(batch.coarse, n_samples=1)

    expected = boundary_aligned_t_list(
        predictor._sigma_ranges, predictor._steps_per_range
    )
    # Steps visit the t_list nodes except the trailing 0 (highest sigma first).
    visited = sorted(hi_sigmas + lo_sigmas, reverse=True)
    expected_nodes = sorted([float(t) for t in expected[:-1]], reverse=True)
    assert visited == pytest.approx(expected_nodes, rel=1e-6)


def test_student_predictor_steps_per_range_honored():
    """``steps_per_range=[2, 1]`` runs the low segment twice and the high once."""
    predictor = _build_student_predictor(steps_per_range=[2, 1])
    lo_sigmas, hi_sigmas = _record_dispatch_sigmas(predictor)

    batch = make_paired_batch_data((8, 16), (16, 32), batch_size=1)
    predictor.generate_on_batch_no_target(batch.coarse, n_samples=1)

    assert len(hi_sigmas) == 1
    assert len(lo_sigmas) == 2


def test_student_predictor_length_mismatch_raises():
    with pytest.raises(ValueError, match="steps_per_range"):
        _build_student_predictor(steps_per_range=[1, 1, 1])


def test_student_predictor_save_load_roundtrip(tmp_path):
    predictor = _build_student_predictor(steps_per_range=[2, 1])
    ckpt = tmp_path / "student_moe.pt"
    predictor.save(str(ckpt))

    # The single bundled config dispatches on the persisted sampler_type tag and
    # returns a student cascade predictor for a fastgen_cascade bundle.
    loaded = DenoisingMoEBundledConfig(mixture_of_experts_path=str(ckpt)).build()
    assert isinstance(loaded, DenoisingMoEStudentPredictor)

    assert loaded._sigma_ranges == predictor._sigma_ranges
    assert loaded._steps_per_range == predictor._steps_per_range
    for orig, new in zip(predictor._experts, loaded._experts):
        for p_orig, p_new in zip(orig.module.parameters(), new.module.parameters()):
            assert torch.equal(p_orig.cpu(), p_new.cpu())

    # Seeded generation is identical after a save/load round-trip. Both must be
    # in eval mode (the bundled config eval()s the loaded one) so dropout etc.
    # don't diverge the comparison.
    for expert in predictor._experts:
        expert.module.eval()
    batch = make_paired_batch_data((8, 16), (16, 32), batch_size=1)
    torch.manual_seed(0)
    out_orig = predictor.generate_on_batch_no_target(batch.coarse, n_samples=1)
    torch.manual_seed(0)
    out_new = loaded.generate_on_batch_no_target(batch.coarse, n_samples=1)
    for k in out_orig:
        torch.testing.assert_close(out_orig[k], out_new[k])


def test_student_predictor_config_build_and_state_marker(tmp_path):
    """``DenoisingMoEStudentConfig`` assembles from per-expert checkpoints, sorts
    by sigma_min, and the saved bundle carries the fastgen_cascade marker."""

    coarse_shape = (8, 16)
    fine_shape = (16, 32)
    fine_coords = make_fine_coords(fine_shape)
    expert_ckpts = []
    for i in range(2):
        model = _get_diffusion_model(
            coarse_shape=coarse_shape,
            downscale_factor=2,
            full_fine_coords=fine_coords,
            predict_residual=False,
            use_fine_topography=False,
            static_inputs=StaticInputs(fields=[], coords=fine_coords),
        )
        path = tmp_path / f"student_{i}.ckpt"
        torch.save({"model": model.get_state()}, path)
        expert_ckpts.append(str(path))

    # Pass ranges out of order to confirm __post_init__ sorts steps alongside.
    config = DenoisingMoEStudentConfig(
        denoising_expert_configs=[
            DenoisingExpertCheckpointConfig(
                checkpoint_config=CheckpointModelConfig(
                    checkpoint_path=expert_ckpts[0]
                ),
                sigma_min=0.5,
                sigma_max=1.0,
            ),
            DenoisingExpertCheckpointConfig(
                checkpoint_config=CheckpointModelConfig(
                    checkpoint_path=expert_ckpts[1]
                ),
                sigma_min=0.1,
                sigma_max=0.5,
            ),
        ],
        steps_per_range=[1, 2],  # aligned with the given (unsorted) order
    )
    predictor = config.build()
    assert predictor._sigma_ranges == [(0.1, 0.5), (0.5, 1.0)]
    # steps re-ordered to follow the ascending-sigma sort: low-noise gets 2.
    assert predictor._steps_per_range == [2, 1]

    bundle = tmp_path / "bundle.pt"
    predictor.save(str(bundle))
    state = torch.load(str(bundle), map_location="cpu", weights_only=False)
    assert state["sampler_type"] == "fastgen_cascade"
    assert state["steps_per_range"] == [2, 1]


def test_student_predictor_adds_residual_base():
    """With ``predict_residual=True`` the cascade output is post-processed into a
    full field via ``_primary.postprocess_generated`` (base added back)."""
    predictor = _build_student_predictor(steps_per_range=[1, 1], predict_residual=True)
    batch = make_paired_batch_data((8, 16), (16, 32), batch_size=1)
    out = predictor.generate_on_batch_no_target(batch.coarse, n_samples=1)
    assert set(out) == {"x"}
    assert out["x"].shape == (1, 1, 16, 32)
    assert torch.isfinite(out["x"]).all()
