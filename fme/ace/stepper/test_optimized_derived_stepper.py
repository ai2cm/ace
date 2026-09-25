"""Stepper-level tests of optimized derived variables (rho_wright97 in the loss)."""

import dataclasses
import datetime
import unittest.mock

import dacite
import pytest
import torch

import fme
from fme.ace.data_loading.batch_data import BatchData
from fme.ace.stepper.single_module import StepperConfig, TrainStepperConfig
from fme.core.coordinates import DepthCoordinate, LatLonCoordinates
from fme.core.corrector.loss_config import (
    CorrectorLossConfig,
    PreCorrectorOptimizationConfig,
)
from fme.core.corrector.registry import CorrectionSequence
from fme.core.dataset_info import DatasetInfo
from fme.core.loss import StepLossConfig
from fme.core.ocean_eos import (
    boussinesq_pressure,
    interface_to_center_depth,
    wright97_anomaly,
)
from fme.core.optimization import NullOptimization, OptimizationConfig
from fme.core.optimized_derived import OptimizedDerivedVariableConfig
from fme.core.registry.corrector import CorrectorSelector
from fme.core.registry.module import ModuleSelector
from fme.core.spatial_mask_provider import SpatialMaskProvider
from fme.core.spatial_masking import StaticSpatialMaskingConfig
from fme.core.step import SingleModuleStepConfig, StepSelector
from fme.core.step.single_module import SingleModuleStep
from fme.core.testing import trivial_network_and_loss_normalization

DEVICE = fme.get_device()
IMG_SHAPE = (4, 6)
N_LEVELS = 2
NAMES = [f"{v}_{k}" for v in ("so", "thetao") for k in range(N_LEVELS)]
RHO_NAMES = [f"rho_wright97_{k}" for k in range(N_LEVELS)]


class _AddBias(torch.nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1, n_channels, 1, 1))

    def forward(self, x):
        return x + self.bias


def _dataset_info(with_spatial_masks: bool = False) -> DatasetInfo:
    mask = torch.ones(*IMG_SHAPE, N_LEVELS, device=DEVICE)
    mask[0] = 0.0  # a land row
    masks = {f"mask_{k}": mask[..., k] for k in range(N_LEVELS)}
    return DatasetInfo(
        horizontal_coordinates=LatLonCoordinates(
            lat=torch.linspace(-60.0, 60.0, IMG_SHAPE[0], device=DEVICE),
            lon=torch.linspace(0.0, 300.0, IMG_SHAPE[1], device=DEVICE),
        ),
        vertical_coordinate=DepthCoordinate(
            idepth=torch.tensor([0.0, 10.0, 500.0], device=DEVICE), mask=mask
        ),
        spatial_mask_provider=SpatialMaskProvider(
            masks={**masks, "mask_2d": masks["mask_0"]}
        )
        if with_spatial_masks
        else None,
        timestep=datetime.timedelta(days=5),
    )


def _stepper_config(module: torch.nn.Module, names=NAMES) -> StepperConfig:
    return StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(type="prebuilt", config={"module": module}),
                    in_names=names,
                    out_names=names,
                    normalization=trivial_network_and_loss_normalization(names),
                    corrector=CorrectorSelector("ocean_corrector", {}),
                )
            ),
        ),
    )


def _data(n_timesteps: int = 3, seed: int = 0, names=NAMES) -> BatchData:
    data = BatchData.new_for_testing(
        names=names, n_samples=2, n_timesteps=n_timesteps, img_shape=IMG_SHAPE
    )
    g = torch.Generator().manual_seed(seed)
    shape = data.data[NAMES[0]].shape
    for k in range(N_LEVELS):
        for name, lo, span in ((f"so_{k}", 34.0, 2.0), (f"thetao_{k}", 2.0, 20.0)):
            data.data[name].copy_(lo + span * torch.rand(shape, generator=g))
    return data


def _train_stepper(module: torch.nn.Module, names=NAMES, **train_config_kwargs):
    return TrainStepperConfig(**train_config_kwargs).get_train_stepper(
        _stepper_config(module, names), _dataset_info()
    )


def test_train_on_batch_with_rho():
    """Only rho_wright97 carries loss weight, so the parameter update is driven through
    the EOS."""
    torch.manual_seed(0)
    module = _AddBias(len(NAMES))
    stepper = _train_stepper(
        module,
        loss=StepLossConfig(type="MSE", weights={n: 0.0 for n in NAMES}),
        optimized_derived_variables=[OptimizedDerivedVariableConfig(weight=1.0)],
    )
    optimization = OptimizationConfig(lr=1e-2).build(
        modules=stepper.modules, max_epochs=1
    )
    stepped = stepper.train_on_batch(_data(), optimization=optimization)
    loss = stepped.metrics["loss"]
    assert torch.isfinite(loss) and loss > 0
    assert stepped.per_channel_losses is not None
    assert set(stepped.per_channel_losses) == set(NAMES + RHO_NAMES)
    for name in NAMES:
        assert stepped.per_channel_losses[name].loss == 0.0
    for name in RHO_NAMES:
        assert stepped.per_channel_losses[name].loss > 0.0
    # every so and thetao channel moved: the rho gradient reached all four
    (bias,) = [p for p in stepper.modules.parameters()]
    assert (bias.detach().flatten() != 0).all()
    for name in RHO_NAMES:
        assert name not in stepped.gen_data


def test_no_config_is_unchanged():
    """Absent and None configs give the same loss, only output-name channels,
    and a checkpoint state that does not mention the feature."""
    data = _data()
    module = _AddBias(len(NAMES))
    outputs, states = [], []
    for kwargs in (
        {},
        {"optimized_derived_variables": None},
        {"optimized_derived_variables": [OptimizedDerivedVariableConfig()]},
    ):
        torch.manual_seed(0)
        stepper = _train_stepper(module, **kwargs)
        outputs.append(stepper.train_on_batch(data, optimization=NullOptimization()))
        states.append(stepper.get_state())
    absent, none, on = outputs
    torch.testing.assert_close(absent.metrics["loss"], none.metrics["loss"])
    assert absent.per_channel_losses is not None
    assert set(absent.per_channel_losses) == set(NAMES)
    assert set(on.per_channel_losses or {}) == set(NAMES + RHO_NAMES)
    # repr, since the prebuilt module in each config is its own copy
    assert len({repr(state["config"]) for state in states}) == 1
    assert "optimized_derived" not in str(states[2])


def test_train_stepper_config_parses_from_yaml_dict():
    config = dacite.from_dict(
        data_class=TrainStepperConfig,
        data={
            "loss": {"type": "MSE"},
            "optimized_derived_variables": [
                {"name": "rho_wright97", "weight": 0.5, "levels": [0, 1]}
            ],
        },
        config=dacite.Config(strict=True),
    )
    assert config.optimized_derived_variables == [
        OptimizedDerivedVariableConfig(name="rho_wright97", weight=0.5, levels=[0, 1])
    ]
    with pytest.raises(dacite.WrongTypeError):
        dacite.from_dict(
            data_class=TrainStepperConfig,
            data={"optimized_derived_variables": [{"name": "sigma0"}]},
            config=dacite.Config(strict=True),
        )


def test_coupled_ocean_config_passed_to_ocean_build_loss():
    from fme.coupled.stepper import ComponentTrainingConfig, CoupledTrainStepperConfig

    derived = [OptimizedDerivedVariableConfig()]
    config = CoupledTrainStepperConfig(
        n_coupled_steps=1,
        ocean=ComponentTrainingConfig(
            loss=StepLossConfig(), optimized_derived_variables=derived
        ),
        atmosphere=ComponentTrainingConfig(loss=StepLossConfig()),
    )
    stepper = unittest.mock.Mock()
    stepper.n_inner_steps = 2
    config._build_loss(stepper, n_coupled_steps=1)
    stepper.ocean.build_loss.assert_called_once_with(config.ocean.loss, derived)
    stepper.atmosphere.build_loss.assert_called_once_with(config.atmosphere.loss, None)


class _ScaleThetao:
    """A correction that changes thetao: thetao_c = 0.5 * thetao_net + 1."""

    def __call__(self, input_data, gen_data, forcing_data, corrector_state):
        return {
            f"thetao_{k}": 0.5 * gen_data[f"thetao_{k}"] + 1.0 for k in range(N_LEVELS)
        }, corrector_state


def test_train_on_batch_precorrector_rho_from_corrected_thetao():
    """With precorrector_optimization on thetao_, the rho_wright97 losses match a
    stepper whose loss sees the corrected output (no corrector loss), and the
    thetao losses match a stepper with no corrector at all. One forward step, so
    every stepper's network sees the same input."""
    data = _data(n_timesteps=2)
    rho_and_thetao = {}
    for label, correct, corrector_loss in (
        ("precorrector", True, ["thetao_"]),
        ("corrected", True, None),
        ("no_corrector", False, None),
    ):
        torch.manual_seed(0)
        stepper = _train_stepper(
            _AddBias(len(NAMES)),
            loss=StepLossConfig(type="MSE"),
            optimized_derived_variables=[OptimizedDerivedVariableConfig()],
            corrector_loss=None
            if corrector_loss is None
            else CorrectorLossConfig(
                precorrector_optimization=PreCorrectorOptimizationConfig(
                    names_and_prefixes=corrector_loss
                )
            ),
        )
        step = stepper._stepper._step_obj
        assert isinstance(step, SingleModuleStep)
        step._corrector = CorrectionSequence([_ScaleThetao()] if correct else [])
        out = stepper.train_on_batch(data, optimization=NullOptimization())
        assert out.per_channel_losses is not None
        rho_and_thetao[label] = {
            k: v.loss
            for k, v in out.per_channel_losses.items()
            if k.startswith(("rho_wright97_", "thetao_"))
        }
    pre, corrected, none = (
        rho_and_thetao[k] for k in ("precorrector", "corrected", "no_corrector")
    )
    for name in RHO_NAMES:
        torch.testing.assert_close(pre[name], corrected[name])
        assert not torch.allclose(pre[name], none[name])
    for k in range(N_LEVELS):
        name = f"thetao_{k}"
        torch.testing.assert_close(pre[name], none[name])
        assert not torch.allclose(pre[name], corrected[name])


def _expected_rho(data, k: int) -> torch.Tensor:
    """W97 of ``data``'s so_k, thetao_k at the level-centre pressure, NaN off
    the mask; the independent reference for the derived output."""
    vc = _dataset_info().vertical_coordinate
    assert isinstance(vc, DepthCoordinate)
    p = boussinesq_pressure(interface_to_center_depth(vc.idepth.double()))[k]
    S, T = data[f"so_{k}"], data[f"thetao_{k}"]
    rho = wright97_anomaly(S, T, p.to(dtype=S.dtype, device=S.device))
    wet = (vc.mask[..., k] > 0).to(S.device)
    return torch.where(wet & S.isfinite() & T.isfinite(), rho, torch.nan)


def test_train_output_derives_rho_for_aggregators():
    """With the config, TrainOutput's derived gen and target data carry
    rho_wright97_k = W97(so_k, thetao_k, p_k), computed by the loss's own
    OptimizedDerivedVariables."""
    torch.manual_seed(0)
    stepper = _train_stepper(
        _AddBias(len(NAMES)),
        loss=StepLossConfig(type="MSE"),
        optimized_derived_variables=[OptimizedDerivedVariableConfig()],
    )
    assert (
        stepper._derive_func.derived  # type: ignore[attr-defined]
        is stepper._loss_obj.step_loss._derive
    )
    stepped = stepper.train_on_batch(
        _data(), optimization=NullOptimization(), compute_derived_variables=True
    )
    for data in (stepped.gen_data, stepped.target_data):
        for k, name in enumerate(RHO_NAMES):
            expected = _expected_rho(data, k)
            assert expected.isnan().any() and expected.isfinite().any()
            torch.testing.assert_close(data[name], expected, equal_nan=True)
    assert not torch.allclose(
        stepped.gen_data[RHO_NAMES[0]].nan_to_num(),
        stepped.target_data[RHO_NAMES[0]].nan_to_num(),
    )


def test_predict_paired_derives_rho_for_inference():
    """Inline inference (predict_paired) derives rho_wright97_k for prediction
    and reference."""
    torch.manual_seed(0)
    stepper = _train_stepper(
        _AddBias(len(NAMES)),
        loss=StepLossConfig(type="MSE"),
        optimized_derived_variables=[OptimizedDerivedVariableConfig(levels=[1])],
    )
    data = _data()
    ic = data.get_start(stepper._stepper.prognostic_names, stepper.n_ic_timesteps)
    paired, _ = stepper.predict_paired(ic, data, compute_derived_variables=True)
    for side in (paired.prediction, paired.reference):
        assert RHO_NAMES[0] not in side
        torch.testing.assert_close(
            side[RHO_NAMES[1]], _expected_rho(side, 1), equal_nan=True
        )


def test_no_config_derived_outputs_unchanged():
    """No config: derive_func is the vertical coordinate's own, and no
    rho_wright97_* appears in derived train or inference outputs."""
    torch.manual_seed(0)
    stepper = _train_stepper(_AddBias(len(NAMES)), loss=StepLossConfig(type="MSE"))
    assert stepper._stepper.derive_func is stepper._stepper._derive_func
    assert stepper._derive_func is stepper._stepper._derive_func
    data = _data()
    stepped = stepper.train_on_batch(
        data, optimization=NullOptimization(), compute_derived_variables=True
    )
    ic = data.get_start(stepper._stepper.prognostic_names, stepper.n_ic_timesteps)
    paired, _ = stepper.predict_paired(ic, data, compute_derived_variables=True)
    for side in (
        stepped.gen_data,
        stepped.target_data,
        paired.prediction,
        paired.reference,
    ):
        assert not any(n.startswith("rho_wright97") for n in side)


OHC_NAMES = NAMES + ["hfds_total_area"]


class _DropForcingAddBias(_AddBias):
    """Output = the first n_channels input channels (the out_names; the trailing
    forcing channel is dropped) plus a bias."""

    def forward(self, x):
        return x[:, : self.bias.shape[1]] + self.bias


def _ohc_train_stepper(ocean_heat_content_correction: bool):
    """rho-only loss (zero weight on every output channel) and the real
    ocean_corrector with its scaled_temperature heat-content correction and
    precorrector_optimization on thetao_ (as in ocean_rho), or with neither."""
    module = _DropForcingAddBias(len(OHC_NAMES))
    corrector_config = (
        {"ocean_heat_content_correction": {"method": "scaled_temperature"}}
        if ocean_heat_content_correction
        else {}
    )
    in_names = OHC_NAMES + ["sea_surface_fraction"]
    stepper_config = StepperConfig(
        input_masking=StaticSpatialMaskingConfig(
            mask_value=0,
            fill_value=0.0,
            exclude_names_and_prefixes=["sea_surface_fraction"],
        ),
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(type="prebuilt", config={"module": module}),
                    in_names=in_names,
                    out_names=OHC_NAMES,
                    normalization=trivial_network_and_loss_normalization(in_names),
                    corrector=CorrectorSelector("ocean_corrector", corrector_config),
                )
            ),
        ),
    )
    return TrainStepperConfig(
        loss=StepLossConfig(type="MSE", weights={n: 0.0 for n in OHC_NAMES}),
        optimized_derived_variables=[OptimizedDerivedVariableConfig()],
        corrector_loss=CorrectorLossConfig(
            precorrector_optimization=PreCorrectorOptimizationConfig(
                names_and_prefixes=["thetao_"]
            )
        )
        if ocean_heat_content_correction
        else None,
    ).get_train_stepper(stepper_config, _dataset_info(with_spatial_masks=True))


def test_rho_gradient_through_ocean_heat_content_corrector():
    """d(rho loss)/d(bias) with the real scaled_temperature correction differs
    from the correction-off gradient, and reaches hfds_total_area only when the
    correction is on (thetao_c = r * thetao_net, r depends on hfds_total_area)."""
    data = BatchData.new_for_testing(
        names=OHC_NAMES + ["sea_surface_fraction"],
        n_samples=2,
        n_timesteps=2,
        img_shape=IMG_SHAPE,
    )
    base = _data(n_timesteps=2)
    for name in NAMES:
        data.data[name].copy_(base.data[name])
    g = torch.Generator().manual_seed(1)
    data.data["hfds_total_area"].copy_(
        100.0 * torch.randn(data.data["hfds_total_area"].shape, generator=g)
    )
    data.data["sea_surface_fraction"].fill_(1.0)
    for name in OHC_NAMES:  # land row, as real data carries it
        data.data[name][..., 0, :] = float("nan")
    grads = {}
    for on in (True, False):
        torch.manual_seed(0)
        stepper = _ohc_train_stepper(on)
        (bias,) = stepper.modules.parameters()
        captured: list[torch.Tensor] = []
        bias.register_hook(lambda grad: captured.append(grad.detach().clone()))
        out = stepper.train_on_batch(
            data,
            optimization=OptimizationConfig(lr=0.0).build(
                modules=stepper.modules, max_epochs=1
            ),
        )
        assert torch.isfinite(out.metrics["loss"])
        (grad,) = captured
        assert torch.isfinite(grad).all()
        grads[on] = grad.flatten()
    i_hfds = OHC_NAMES.index("hfds_total_area")
    assert grads[True][i_hfds] != 0.0
    assert grads[False][i_hfds] == 0.0
    i_thetao = [OHC_NAMES.index(f"thetao_{k}") for k in range(N_LEVELS)]
    assert not torch.allclose(grads[True][i_thetao], grads[False][i_thetao])


COLUMN_NAMES = NAMES + ["zos"]


def test_train_on_batch_with_pbo():
    """pbo_wright97 alone carries loss weight: its gradient moves the zos, so and
    thetao channels, and TrainOutput's derived gen and target data carry it."""
    torch.manual_seed(0)
    stepper = _train_stepper(
        _AddBias(len(COLUMN_NAMES)),
        names=COLUMN_NAMES,
        loss=StepLossConfig(type="MSE", weights={n: 0.0 for n in COLUMN_NAMES}),
        optimized_derived_variables=[
            OptimizedDerivedVariableConfig(
                name="pbo_wright97", stds={"pbo_wright97": 100.0}
            )
        ],
    )
    optimization = OptimizationConfig(lr=1e-2).build(
        modules=stepper.modules, max_epochs=1
    )
    stepped = stepper.train_on_batch(
        _data(names=COLUMN_NAMES),
        optimization=optimization,
        compute_derived_variables=True,
    )
    assert stepped.per_channel_losses is not None
    assert set(stepped.per_channel_losses) == set(COLUMN_NAMES + ["pbo_wright97"])
    assert stepped.per_channel_losses["pbo_wright97"].loss > 0.0
    (bias,) = [p for p in stepper.modules.parameters()]
    assert (bias.detach().flatten() != 0).all()
    wet = _dataset_info().vertical_coordinate.mask[..., 0] > 0  # type: ignore
    for data in (stepped.gen_data, stepped.target_data):
        pbo = data["pbo_wright97"]
        assert pbo[..., ~wet].isnan().all() and pbo[..., wet].isfinite().all()
