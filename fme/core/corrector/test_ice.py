import datetime

import torch

from fme import get_device
from fme.core.corrector.ice import IceBudgetCorrectionConfig, IceCorrectorConfig
from fme.core.gridded_ops import LatLonOperations

DEVICE = get_device()
IMG_SHAPE = (4, 4)
TIMESTEP = datetime.timedelta(hours=6)


def _build_ice_corrector(corrected_variables):
    config = IceCorrectorConfig(
        budget_correction=IceBudgetCorrectionConfig(
            corrected_variables=corrected_variables
        )
    )
    ops = LatLonOperations(torch.ones(size=IMG_SHAPE))
    return config._build(ops, TIMESTEP)


def _ice_test_data():
    torch.manual_seed(0)
    input_data = {"siconc": torch.rand(IMG_SHAPE, device=DEVICE)}
    gen_data = {
        "siconc": torch.rand(IMG_SHAPE, device=DEVICE),
        "LSRCc": torch.rand(IMG_SHAPE, device=DEVICE) * 1e-6,
        "LSNKc": -torch.rand(IMG_SHAPE, device=DEVICE) * 1e-6,
        "XPRTc": (torch.rand(IMG_SHAPE, device=DEVICE) - 0.5) * 1e-6,
        "other": torch.randn(IMG_SHAPE, device=DEVICE),  # uncorrected field
    }
    return input_data, gen_data


def test_ice_corrector_output_behavior_preserving():
    corrected_variables = {"siconc": ["LSRCc", "LSNKc", "XPRTc"]}
    corrector = _build_ice_corrector(corrected_variables)
    input_data, gen_data = _ice_test_data()
    result = corrector(input_data, gen_data, {}, None)
    # the corrected modified fields match a direct call to the budget config
    direct = IceBudgetCorrectionConfig(corrected_variables=corrected_variables)(
        gen_data, input_data, TIMESTEP.total_seconds()
    )
    for name in ["siconc", "LSRCc", "LSNKc", "XPRTc"]:
        torch.testing.assert_close(result.corrected[name], direct[name])
    # the uncorrected field is carried through unchanged
    torch.testing.assert_close(result.corrected["other"], gen_data["other"])


def test_ice_budget_correction_returns_only_modified():
    corrected_variables = {"siconc": ["LSRCc", "LSNKc", "XPRTc"]}
    corrector = _build_ice_corrector(corrected_variables)
    input_data, gen_data = _ice_test_data()
    result = corrector(input_data, gen_data, {}, None)
    # modified set is the processed prognostic key plus its three budget terms
    assert set(result.modified_names) == {"siconc", "LSRCc", "LSNKc", "XPRTc"}
    assert set(result.diagnostics.delta) == set(result.modified_names)
    for name, delta in result.diagnostics.delta.items():
        torch.testing.assert_close(delta, result.corrected[name] - gen_data[name])
    # the uncorrected field is absent from the set
    assert "other" not in result.modified_names


def test_ice_budget_correction_empty_subset_when_none():
    corrector = _build_ice_corrector(corrected_variables=None)
    input_data, gen_data = _ice_test_data()
    result = corrector(input_data, gen_data, {}, None)
    assert dict(result.diagnostics.delta) == {}
    assert set(result.modified_names) == set()
    for name in gen_data:
        torch.testing.assert_close(result.corrected[name], gen_data[name])
