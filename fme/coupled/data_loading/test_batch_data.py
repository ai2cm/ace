import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData
from fme.core.random_state import RandomState
from fme.coupled.data_loading.batch_data import CoupledPrognosticState


def _prognostic_state(name: str):
    index = xr.date_range("2000", freq="6h", periods=1, use_cftime=True)
    time = xr.DataArray(index.values[None, :], dims=["sample", "time"])
    return BatchData.new_on_cpu(
        data={name: torch.zeros(1, 1, 4, 8)},
        time=time,
        horizontal_dims=["lat", "lon"],
    ).get_start([name], n_ic_timesteps=1)


def _coupled_state() -> CoupledPrognosticState:
    return CoupledPrognosticState(
        ocean_data=_prognostic_state("sst"),
        atmosphere_data=_prognostic_state("surface_temperature"),
    )


def test_apply_config_seed_seeds_both_components():
    """Coupled inference needs a seed so the stochastic atmosphere replays the
    same noise sequence across runs (e.g. between ablation arms)."""
    state = _coupled_state()
    seeded = state.apply_config_seed(0)
    for component in (seeded.ocean_data, seeded.atmosphere_data):
        stepper_state = component.as_batch_data().stepper_state
        assert stepper_state is not None
        assert stepper_state.random_state is not None
    # the original is unchanged
    assert state.ocean_data.as_batch_data().stepper_state is None
    assert state.atmosphere_data.as_batch_data().stepper_state is None


def test_apply_config_seed_none_leaves_both_components_unseeded():
    state = _coupled_state()
    result = state.apply_config_seed(None)
    assert result.ocean_data is state.ocean_data
    assert result.atmosphere_data is state.atmosphere_data


def test_apply_config_seed_gives_the_components_different_streams():
    """The components are seeded from the configured value but not from the
    *same* stream: two identical stochastic modules on a shared grid would
    otherwise draw bitwise-identical noise fields in the two realms."""
    seeded = _coupled_state().apply_config_seed(0)
    draws = []
    for component in (seeded.ocean_data, seeded.atmosphere_data):
        stepper_state = component.as_batch_data().stepper_state
        assert stepper_state is not None
        random_state = stepper_state.random_state
        assert random_state is not None
        draws.append(torch.randn(4, generator=random_state.generator))
    assert not torch.equal(draws[0], draws[1])


def test_apply_config_seed_defers_to_a_restored_random_state():
    """Precedence is resolved per component, so a half-restored coupled restart
    continues the realm that has a restored generator and seeds the other."""
    restored = RandomState.from_seed(11)
    state = CoupledPrognosticState(
        ocean_data=_prognostic_state("sst"),
        atmosphere_data=_prognostic_state("surface_temperature").with_random_state(
            restored
        ),
    )
    result = state.apply_config_seed(0)
    atmosphere_state = result.atmosphere_data.as_batch_data().stepper_state
    assert atmosphere_state is not None
    assert atmosphere_state.random_state is restored
    # the ocean carried no restored state, so it takes the config seed
    ocean_state = result.ocean_data.as_batch_data().stepper_state
    assert ocean_state is not None
    assert ocean_state.random_state is not None
