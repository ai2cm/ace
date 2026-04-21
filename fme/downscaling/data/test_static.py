import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.device import get_device

from .static import StaticInput, StaticInputs, _load_coords_from_ds


def _make_coords(n=4):
    from fme.core.coordinates import LatLonCoordinates

    return LatLonCoordinates(
        lat=torch.arange(n, dtype=torch.float32),
        lon=torch.arange(n, dtype=torch.float32),
    )


def test_StaticInput_error_cases():
    data = torch.randn(1, 2, 2)
    with pytest.raises(ValueError):
        StaticInput(data=data)


def test_subset():
    full_data_shape = (10, 10)
    data = torch.randn(*full_data_shape)
    topo = StaticInput(data=data)
    lat_slice = slice(2, 6)
    lon_slice = slice(3, 8)
    subset_topo = topo.subset(lat_slice, lon_slice)
    assert torch.allclose(subset_topo.data, data[lat_slice, lon_slice])


def test_StaticInputs_serialize():
    dim_len = 4
    data = torch.arange(dim_len * dim_len, dtype=torch.float32).reshape(
        dim_len, dim_len
    )
    coords = _make_coords(n=dim_len)
    static_inputs = StaticInputs(
        [StaticInput(data), StaticInput(data * -1.0)], coords=coords
    )
    state = static_inputs.get_state()
    assert "coords" in state
    reconstructed = StaticInputs.from_state(state)
    assert reconstructed[0].data.equal(static_inputs[0].data)
    assert reconstructed[1].data.equal(static_inputs[1].data)
    assert torch.equal(reconstructed.coords.lat, static_inputs.coords.lat)
    assert torch.equal(reconstructed.coords.lon, static_inputs.coords.lon)


def test_StaticInputs_from_state_raises_on_missing_coords():
    data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    with pytest.raises(ValueError, match="No coordinates"):
        StaticInputs.from_state({"fields": [{"data": data}]})


def test_StaticInputs_from_state_legacy_coords_in_fields():
    """from_state handles old format where coords were stored only inside each field."""
    dim_len = 4
    data = torch.arange(dim_len * dim_len, dtype=torch.float32).reshape(
        dim_len, dim_len
    )
    coords = _make_coords(n=dim_len)
    old_state = {
        "fields": [{"data": data, "coords": {"lat": coords.lat, "lon": coords.lon}}],
    }
    result = StaticInputs.from_state(old_state)
    assert torch.equal(result[0].data, data)
    assert torch.equal(result.coords.lat, coords.lat)
    assert torch.equal(result.coords.lon, coords.lon)


def test_from_state_backwards_compatible_has_coords():
    """When state has coords, delegates to from_state."""
    data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    coords = _make_coords()
    state = StaticInputs([StaticInput(data)], coords=coords).get_state()
    result = StaticInputs.from_state_backwards_compatible(
        state=state, static_inputs_config={}
    )
    assert result is not None
    assert torch.equal(result[0].data, data)


def test_from_state_backwards_compatible_no_state_no_config():
    """No static inputs in checkpoint and no config: returns None."""
    result = StaticInputs.from_state_backwards_compatible(
        state={}, static_inputs_config={}
    )
    assert result is None


def test_from_state_backwards_compatible_with_config(tmp_path):
    """Checkpoint with static_inputs_config: loads fields and coords from files."""
    dim_len = 4
    lat = np.linspace(0, 1, dim_len, dtype=np.float32)
    lon = np.linspace(0, 1, dim_len, dtype=np.float32)
    field_data = np.random.rand(dim_len, dim_len).astype(np.float32)
    field_path = str(tmp_path / "field.nc")
    xr.Dataset(
        {"HGTsfc": (["lat", "lon"], field_data)},
        coords={"lat": lat, "lon": lon},
    ).to_netcdf(field_path)
    result = StaticInputs.from_state_backwards_compatible(
        state={},
        static_inputs_config={"HGTsfc": field_path},
    )
    assert result is not None
    assert len(result.fields) == 1


def test_from_state_backwards_compatible_raises_state_and_config():
    """
    Errors if checkpoint state has fields and static_inputs_config is also provided.
    """
    dim_len = 4
    data = torch.arange(dim_len * dim_len, dtype=torch.float32).reshape(
        dim_len, dim_len
    )
    coords = _make_coords(n=dim_len)
    state = StaticInputs([StaticInput(data)], coords=coords).get_state()
    with pytest.raises(ValueError, match="static_inputs_config"):
        StaticInputs.from_state_backwards_compatible(
            state=state,
            static_inputs_config={"HGTsfc": "some/path"},
        )


def test_StaticInput_from_state_places_on_device():
    state = {"data": torch.zeros(4, 4, device="cpu")}
    result = StaticInput.from_state(state)
    assert result.data.device == get_device()


def test_StaticInput_from_state_decouples_memory():
    original = torch.zeros(4, 4, device="cpu")
    result = StaticInput.from_state({"data": original})
    original.fill_(999.0)
    assert result.data.max().item() == 0.0


def test_StaticInputs_from_state_places_on_device():
    coords = _make_coords(n=4)
    data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    state = StaticInputs([StaticInput(data)], coords=coords).get_state()
    result = StaticInputs.from_state(state)
    device = get_device()
    assert result[0].data.device == device
    assert result.coords.lat.device == device
    assert result.coords.lon.device == device


def test_StaticInputs_from_state_decouples_memory():
    coords = _make_coords(n=4)
    data = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    state = StaticInputs([StaticInput(data)], coords=coords).get_state()
    original_lat = state["coords"]["lat"]
    result = StaticInputs.from_state(state)
    original_lat.fill_(999.0)
    assert result.coords.lat.max().item() < 999.0


def test__load_coords_from_ds():
    lat = [0.0, 1.0, 2.0]
    lon = [10.0, 20.0, 30.0, 40.0]
    ds = xr.Dataset(coords={"lat": lat, "lon": lon})

    coords = _load_coords_from_ds(ds)
    assert torch.allclose(coords.lat, torch.tensor(lat, dtype=torch.float32))
    assert torch.allclose(coords.lon, torch.tensor(lon, dtype=torch.float32))

    # expected coord names missing
    ds = xr.Dataset(coords={"x": lon, "y": lat})
    with pytest.raises(ValueError):
        _load_coords_from_ds(ds)
