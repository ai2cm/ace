import pytest

from .data_loading.test_data_loader import MockCoupledData, create_coupled_data_on_disk


@pytest.fixture(scope="session")
def mock_coupled_data(tmp_path_factory) -> MockCoupledData:
    """Session-scoped fixture that creates a mock coupled dataset on disk that
    can be reused for all tests.

    """
    data_dir = tmp_path_factory.mktemp("coupled_data")
    ocean_names = ["o_exog", "o_prog", "o_sfc"]
    atmos_names = ["a_exog", "a_diag", "a_prog", "a_sfc", "constant_mask"]
    n_forward_times_ocean = 2
    n_forward_times_atmos = 4
    return create_coupled_data_on_disk(
        data_dir,
        n_forward_times_ocean=n_forward_times_ocean,
        n_forward_times_atmosphere=n_forward_times_atmos,
        ocean_names=ocean_names,
        atmosphere_names=atmos_names,
    )
