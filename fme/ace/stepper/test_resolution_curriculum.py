import pytest
import torch

from fme.ace.stepper.resolution_curriculum import (
    ResolutionCurriculum,
    TargetResolutionCurriculumConfig,
)


def test_config_validation():
    with pytest.raises(ValueError, match="start_fraction"):
        TargetResolutionCurriculumConfig(start_fraction=0.0, ramp_epochs=5)
    with pytest.raises(ValueError, match="start_fraction"):
        TargetResolutionCurriculumConfig(start_fraction=1.5, ramp_epochs=5)
    with pytest.raises(ValueError, match="ramp_epochs"):
        TargetResolutionCurriculumConfig(start_fraction=0.3, ramp_epochs=0)


def test_cutoff_fraction_schedule():
    c = TargetResolutionCurriculumConfig(start_fraction=0.2, ramp_epochs=10)
    assert c.cutoff_fraction(0) == pytest.approx(0.2)
    assert c.cutoff_fraction(5) == pytest.approx(0.6)
    # ramp complete -> no filtering
    assert c.cutoff_fraction(10) is None
    assert c.cutoff_fraction(20) is None


def _curriculum(start=0.25, ramp=8, nlat=32, nlon=64):
    return ResolutionCurriculum(
        TargetResolutionCurriculumConfig(start_fraction=start, ramp_epochs=ramp),
        nlat=nlat,
        nlon=nlon,
        grid="equiangular",
    )


def test_filter_active_only_during_ramp_and_training():
    cur = _curriculum()
    x = torch.randn(2, 32, 64)
    target = {"PRESsfc": x}

    # before any epoch is set: inactive
    assert not cur.active
    assert torch.equal(cur.filter_target(target)["PRESsfc"], x)

    # during ramp, training: active and modifies the field
    cur.init_for_epoch(0)
    assert cur.active
    assert not torch.equal(cur.filter_target(target)["PRESsfc"], x)

    # eval mode: inactive (validation sees the unfiltered target)
    cur.set_eval()
    assert not cur.active
    assert torch.equal(cur.filter_target(target)["PRESsfc"], x)

    # back to training, but past the ramp: inactive (full resolution)
    cur.set_train()
    cur.init_for_epoch(100)
    assert not cur.active
    assert torch.equal(cur.filter_target(target)["PRESsfc"], x)


def test_non_gridded_variables_pass_through():
    cur = _curriculum()
    cur.init_for_epoch(0)
    scalar = torch.randn(2, 1)
    gridded = torch.randn(2, 32, 64)
    out = cur.filter_target({"global_mean_co2": scalar, "PRESsfc": gridded})
    assert torch.equal(out["global_mean_co2"], scalar)
    assert not torch.equal(out["PRESsfc"], gridded)


def test_cutoff_increases_with_epoch():
    cur = _curriculum(start=0.2, ramp=10, nlat=64, nlon=128)
    target = {"v": _checkerboard_plus_noise(64, 128)}
    cur.init_for_epoch(0)
    early = cur.filter_target(target)["v"].std()
    cur.init_for_epoch(7)
    late = cur.filter_target(target)["v"].std()
    # higher cutoff later -> retains more small-scale variance
    assert late > early


def _checkerboard_plus_noise(nlat, nlon):
    torch.manual_seed(0)
    return torch.randn(nlat, nlon)
