import pytest
import torch

from .static import StaticInput, StaticInputs


@pytest.mark.parametrize(
    "init_args",
    [
        pytest.param(
            [torch.randn((1, 2, 2))],
            id="3d_data",
        ),
    ],
)
def test_Topography_error_cases(init_args):
    with pytest.raises(ValueError):
        StaticInput(*init_args)


def test_subset():
    full_data_shape = (10, 10)
    data = torch.randn(*full_data_shape)
    topo = StaticInput(data=data)
    lat_slice = slice(2, 6)
    lon_slice = slice(3, 8)
    subset_topo = topo.subset(lat_slice, lon_slice)
    assert torch.allclose(subset_topo.data, data[lat_slice, lon_slice])


def test_StaticInputs_serialize():
    from fme.core.coordinates import LatLonCoordinates

    data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    coords = LatLonCoordinates(
        lat=torch.arange(4, dtype=torch.float32),
        lon=torch.arange(4, dtype=torch.float32),
    )
    topography = StaticInput(data)
    land_frac = StaticInput(data * -1.0)
    static_inputs = StaticInputs([topography, land_frac], coords=coords)
    state = static_inputs.get_state()
    # Verify coords are stored at the top level, not inside each field
    assert "coords" in state
    assert "coords" not in state["fields"][0]
    static_inputs_reconstructed = StaticInputs.from_state(state)
    assert static_inputs_reconstructed[0].data.equal(static_inputs[0].data)
    assert static_inputs_reconstructed[1].data.equal(static_inputs[1].data)


def test_StaticInputs_serialize_backward_compat_with_coords():
    """from_state should silently ignore 'coords' key in fields for old state dicts."""
    from fme.core.coordinates import LatLonCoordinates

    data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    coords = LatLonCoordinates(
        lat=torch.arange(4, dtype=torch.float32),
        lon=torch.arange(4, dtype=torch.float32),
    )
    # Simulate old state dict format that included coords inside fields.
    # from_state should silently ignore extra keys (like 'coords') in field dicts.
    old_state = {
        "fields": [
            {
                "data": data,
                "coords": {
                    "lat": torch.arange(4, dtype=torch.float32),
                    "lon": torch.arange(4, dtype=torch.float32),
                },
            }
        ],
        "coords": coords.get_state(),
    }
    static_inputs = StaticInputs.from_state(old_state)
    assert torch.equal(static_inputs[0].data, data)
