import numpy as np
import pytest
import torch

from fme.ace.aggregator.inference.data import make_dummy_time
from fme.ace.aggregator.inference.perturbation_response import (
    LatitudeBand,
    PerturbationResponseAggregator,
    PerturbationResponseAggregatorConfig,
    _validate_one_hot,
)
from fme.ace.data_loading.batch_data import PairedData
from fme.core.coordinates import LatLonCoordinates
from fme.core.device import get_device
from fme.core.gridded_ops import LatLonOperations

N_LAT = 4
N_LON = 4
N_TIME = 3


def _coords() -> LatLonCoordinates:
    # One latitude per band: 5N (tropics), 20N (subtropics), 40N (midlat),
    # 70N (outside all bands).
    lat = torch.tensor([5.0, 20.0, 40.0, 70.0], device=get_device())
    lon = torch.tensor([0.0, 90.0, 180.0, 270.0], device=get_device())
    return LatLonCoordinates(lon=lon, lat=lat)


def _ocean_fraction() -> torch.Tensor:
    # Columns 0,1 ocean; columns 2,3 land.
    of = torch.zeros(N_LAT, N_LON, device=get_device())
    of[:, 0:2] = 1.0
    return of


def _response_fields() -> dict[str, torch.Tensor]:
    """Near-surface warming 2 K over land, 1 K over ocean; ~200 hPa warming 2.3 K."""
    land_ocean = torch.full((N_LAT, N_LON), 2.0, device=get_device())
    land_ocean[:, 0:2] = 1.0  # ocean columns warm by 1 K
    fields = {}
    for i in range(8):
        if i == 7:
            fields[f"air_temperature_{i}"] = land_ocean
        elif i == 2:
            fields[f"air_temperature_{i}"] = torch.full(
                (N_LAT, N_LON), 2.3, device=get_device()
            )
        else:
            fields[f"air_temperature_{i}"] = torch.full(
                (N_LAT, N_LON), 1.5, device=get_device()
            )
    return fields


def _paired_data(group_onehot: torch.Tensor) -> PairedData:
    """Baseline members are zero; perturbed members hold the response fields."""
    group_index = group_onehot.argmax(dim=1)
    n_members = group_onehot.shape[0]
    responses = _response_fields()
    prediction = {}
    for name, response in responses.items():
        tensor = torch.zeros(n_members, N_TIME, N_LAT, N_LON, device=get_device())
        for m in range(n_members):
            if group_index[m] == 1:
                tensor[m] = response.unsqueeze(0)  # constant over time
        prediction[name] = tensor
    ocean_fraction = (
        _ocean_fraction().unsqueeze(0).unsqueeze(0).expand(n_members, N_TIME, -1, -1)
    )
    reference = {"ocean_fraction": ocean_fraction}
    return PairedData(
        prediction=prediction,
        reference=reference,
        time=make_dummy_time(n_members, N_TIME),
    )


def _build_aggregator(
    group_onehot: torch.Tensor,
    perturbation_labels=("p4K",),
    config: PerturbationResponseAggregatorConfig | None = None,
) -> PerturbationResponseAggregator:
    return PerturbationResponseAggregator(
        ops=LatLonOperations(torch.ones(N_LAT, N_LON, device=get_device())),
        horizontal_coordinates=_coords(),
        config=config or PerturbationResponseAggregatorConfig(),
        perturbation_labels=list(perturbation_labels),
        group_onehot=group_onehot,
    )


def test_response_diagnostics_recover_known_structure():
    # members: two baseline (group 0), two perturbed (group 1)
    group_onehot = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], device=get_device())
    agg = _build_aggregator(group_onehot)
    agg.record_batch(_paired_data(group_onehot))
    summary = agg.get_summary()
    logs = summary.logs

    assert summary.loss is None

    p = "p4K"
    # Land warms 2 K, ocean 1 K -> ratio 2 in every band.
    for band in ("tropics", "subtropics", "midlatitudes"):
        assert logs[f"{p}/land_warming/{band}"] == pytest.approx(2.0)
        assert logs[f"{p}/ocean_warming/{band}"] == pytest.approx(1.0)
        assert logs[f"{p}/land_ocean_warming_ratio/{band}"] == pytest.approx(2.0)

    # Tropical ocean: surface (air_temperature_7) warms 1 K, ~200 hPa
    # (air_temperature_2) warms 2.3 K -> vertical ratio 2.3.
    assert logs[f"{p}/tropical_ocean_surface_warming"] == pytest.approx(1.0)
    assert logs[f"{p}/tropical_ocean_upper_warming"] == pytest.approx(2.3)
    assert logs[f"{p}/vertical_warming_ratio_tropical_ocean"] == pytest.approx(2.3)

    # Global-mean column profile: uniform fields recover their level value;
    # air_temperature_7 is half land (2) half ocean (1) -> 1.5.
    assert logs[f"{p}/column_warming/air_temperature_2"] == pytest.approx(2.3)
    assert logs[f"{p}/column_warming/air_temperature_7"] == pytest.approx(1.5)


def test_zero_response_gives_zero_warming_and_nan_ratio():
    # A checkpoint with no response: perturbed identical to baseline. Warmings
    # are zero and the ratio is NaN (ill-defined) rather than crashing.
    group_onehot = torch.tensor([[1, 0], [0, 1]], device=get_device())
    data = _paired_data(group_onehot)
    for name in data.prediction:
        data.prediction[name][1] = data.prediction[name][0]
    agg = _build_aggregator(group_onehot)
    agg.record_batch(data)
    logs = agg.get_summary().logs
    assert logs["p4K/land_warming/tropics"] == pytest.approx(0.0)
    assert logs["p4K/ocean_warming/tropics"] == pytest.approx(0.0)
    assert np.isnan(logs["p4K/land_ocean_warming_ratio/tropics"])


def test_accumulates_across_windows():
    group_onehot = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], device=get_device())
    agg = _build_aggregator(group_onehot)
    # two windows with identical constant-in-time data: the time-mean response
    # is unchanged by recording the same window twice.
    agg.record_batch(_paired_data(group_onehot))
    agg.record_batch(_paired_data(group_onehot))
    logs = agg.get_summary().logs
    assert logs["p4K/land_ocean_warming_ratio/tropics"] == pytest.approx(2.0)


def test_more_than_one_perturbation_not_implemented():
    three_groups = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=get_device())
    with pytest.raises(NotImplementedError):
        _validate_one_hot(three_groups)


def test_non_one_hot_rejected():
    bad = torch.tensor([[1, 1], [0, 1]], device=get_device())
    with pytest.raises(ValueError):
        _validate_one_hot(bad)


def test_label_count_must_match_groups():
    group_onehot = torch.tensor([[1, 0], [0, 1]], device=get_device())
    with pytest.raises(ValueError):
        _build_aggregator(group_onehot, perturbation_labels=("a", "b"))


def test_latitude_band_validation():
    with pytest.raises(ValueError):
        LatitudeBand("bad", 30.0, 15.0)


def test_config_rejects_out_of_range_level_index():
    with pytest.raises(ValueError):
        PerturbationResponseAggregatorConfig(
            column_temperature_names=["air_temperature_0", "air_temperature_1"],
            vertical_surface_index=5,
        )


def test_diagnostics_written(tmp_path):
    group_onehot = torch.tensor([[1, 0], [0, 1]], device=get_device())
    agg = PerturbationResponseAggregator(
        ops=LatLonOperations(torch.ones(N_LAT, N_LON, device=get_device())),
        horizontal_coordinates=_coords(),
        config=PerturbationResponseAggregatorConfig(),
        perturbation_labels=["p4K"],
        group_onehot=group_onehot,
        output_dir=str(tmp_path),
        save_diagnostics=True,
    )
    agg.record_batch(_paired_data(group_onehot))
    agg.flush_diagnostics(subdir="epoch_0001")
    out = tmp_path / "epoch_0001" / "perturbation_response_diagnostics.nc"
    assert out.exists()
    import xarray as xr

    ds = xr.open_dataset(out)
    assert "p4K__air_temperature_7" in ds
    # land cells (cols 2,3) warm by 2 K
    np.testing.assert_allclose(ds["p4K__air_temperature_7"].values[:, 2:], 2.0)


def test_response_maps_logged_for_every_field():
    group_onehot = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], device=get_device())
    agg = _build_aggregator(group_onehot)
    agg.record_batch(_paired_data(group_onehot))
    logs = agg.get_summary().logs
    # A 2D response map is logged for every predicted field (here the 8 levels).
    for i in range(8):
        assert f"p4K/response_map/air_temperature_{i}" in logs
    # The scalar diagnostics still co-exist in the same logs dict.
    assert "p4K/land_ocean_warming_ratio/tropics" in logs


def test_response_map_variables_restricts_maps():
    group_onehot = torch.tensor([[1, 0], [0, 1]], device=get_device())
    config = PerturbationResponseAggregatorConfig(
        response_map_variables=["air_temperature_7"]
    )
    agg = _build_aggregator(group_onehot, config=config)
    agg.record_batch(_paired_data(group_onehot))
    logs = agg.get_summary().logs
    map_keys = [k for k in logs if "/response_map/" in k]
    assert map_keys == ["p4K/response_map/air_temperature_7"]
