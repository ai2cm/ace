import torch

from fme.ace.aggregator.inference.data import make_dummy_time
from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.ace.data_loading.perturbation import PerturbationSelector, SSTPerturbation
from fme.ace.data_loading.perturbation_pair import build_perturbation_pair_data
from fme.core.device import get_device
from fme.core.generics.data import SimpleInferenceData

N_IC = 2
N_LAT = 2
N_LON = 2
AMPLITUDE = 4.0


def _perturbation() -> SSTPerturbation:
    return SSTPerturbation(
        sst=[PerturbationSelector(type="constant", config={"amplitude": AMPLITUDE})]
    )


def _ocean_fraction(n_time: int) -> torch.Tensor:
    # column 0 ocean, column 1 land
    of = torch.zeros(N_IC, n_time, N_LAT, N_LON, device=get_device())
    of[:, :, :, 0] = 1.0
    return of


def _initial_condition(include_surface_temperature: bool) -> PrognosticState:
    data = {
        "air_temperature_7": torch.ones(N_IC, 1, N_LAT, N_LON, device=get_device()),
    }
    if include_surface_temperature:
        data["surface_temperature"] = torch.full(
            (N_IC, 1, N_LAT, N_LON), 10.0, device=get_device()
        )
    return PrognosticState(BatchData(data=data, time=make_dummy_time(N_IC, 1)))


def _window(n_time: int = 2) -> BatchData:
    data = {
        "surface_temperature": torch.full(
            (N_IC, n_time, N_LAT, N_LON), 20.0, device=get_device()
        ),
        "ocean_fraction": _ocean_fraction(n_time),
        "air_temperature_7": torch.ones(
            N_IC, n_time, N_LAT, N_LON, device=get_device()
        ),
    }
    return BatchData(data=data, time=make_dummy_time(N_IC, n_time))


def _meshgrid():
    lat = torch.tensor([10.0, 20.0], device=get_device())
    lon = torch.tensor([0.0, 180.0], device=get_device())
    lats, lons = torch.meshgrid(lat, lon, indexing="ij")
    return lats, lons


def _build(full_field: bool, include_surface_temperature: bool = True):
    lats, lons = _meshgrid()
    base = SimpleInferenceData(
        _initial_condition(include_surface_temperature), [_window()]
    )
    return build_perturbation_pair_data(
        initial_condition=base.initial_condition,
        loader=base.loader,
        perturbation=_perturbation(),
        surface_temperature_name="surface_temperature",
        ocean_fraction_name="ocean_fraction",
        lats=lats,
        lons=lons,
        perturb_initial_condition_full_field=full_field,
    )


def test_group_encoding_layout():
    _, group_onehot = _build(full_field=True)
    assert group_onehot.shape == (2 * N_IC, 2)
    expected = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    assert torch.equal(group_onehot.cpu(), expected)


def test_forcing_perturbation_is_ocean_masked():
    data, _ = _build(full_field=True)
    window = next(iter(data.loader))
    sst = window.data["surface_temperature"]
    assert sst.shape[0] == 2 * N_IC
    baseline, perturbed = sst[:N_IC], sst[N_IC:]
    # baseline unchanged everywhere
    assert torch.allclose(baseline, torch.full_like(baseline, 20.0))
    # perturbed: +4 over ocean (col 0), unchanged over land (col 1)
    assert torch.allclose(perturbed[..., 0], torch.full_like(perturbed[..., 0], 24.0))
    assert torch.allclose(perturbed[..., 1], torch.full_like(perturbed[..., 1], 20.0))


def test_ic_perturbation_full_field():
    data, _ = _build(full_field=True)
    ic = data.initial_condition.as_batch_data()
    sst = ic.data["surface_temperature"]
    assert sst.shape[0] == 2 * N_IC
    baseline, perturbed = sst[:N_IC], sst[N_IC:]
    assert torch.allclose(baseline, torch.full_like(baseline, 10.0))
    # whole field perturbed, including land column
    assert torch.allclose(perturbed, torch.full_like(perturbed, 14.0))


def test_ic_perturbation_ocean_masked():
    data, _ = _build(full_field=False)
    ic = data.initial_condition.as_batch_data()
    perturbed = ic.data["surface_temperature"][N_IC:]
    # ocean column perturbed, land column not
    assert torch.allclose(perturbed[..., 0], torch.full_like(perturbed[..., 0], 14.0))
    assert torch.allclose(perturbed[..., 1], torch.full_like(perturbed[..., 1], 10.0))


def test_initial_condition_unperturbed_when_surface_temperature_not_prognostic():
    data, _ = _build(full_field=True, include_surface_temperature=False)
    ic = data.initial_condition.as_batch_data()
    # surface_temperature absent from IC -> IC simply duplicated, unchanged
    air = ic.data["air_temperature_7"]
    assert air.shape[0] == 2 * N_IC
    assert torch.allclose(air, torch.ones_like(air))


def test_other_forcing_variables_are_duplicated_unchanged():
    data, _ = _build(full_field=True)
    window = next(iter(data.loader))
    of = window.data["ocean_fraction"]
    assert of.shape[0] == 2 * N_IC
    # ocean fraction identical across baseline and perturbed members
    assert torch.allclose(of[:N_IC], of[N_IC:])
