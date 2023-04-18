import pytest
import metrics

variables = ['variable1', 'variable2', 'variable3']
times = [1, 2, 4, 8, 16]
grid_yts = [45, 90, 180]
grid_xts = [45, 90, 180]

test_parameters = [(variable, time, grid_yt, grid_xt) for variable in variables
                   for time in times for grid_yt in grid_yts for grid_xt in grid_xts]


@pytest.mark.parametrize("variable, time, grid_yt, grid_xt", test_parameters)
def test_weighted_global_mean_bias(variable, time, grid_yt, grid_xt):
    result = metrics.weighted_global_mean_bias(variable, time, grid_yt, grid_xt)
    assert isinstance(result, float), "The result should be a float"


@pytest.mark.parametrize("variable, time, grid_yt, grid_xt", test_parameters)
def test_weighted_time_mean_bias(variable, time, grid_yt, grid_xt):
    result = metrics.weighted_time_mean_bias(variable, time, grid_yt, grid_xt)
    assert isinstance(result, float), "The result should be a float"


@pytest.mark.parametrize("variable, time, grid_yt, grid_xt", test_parameters)
def test_weighted_global_time_rmse(variable, time, grid_yt, grid_xt):
    result = metrics.weighted_global_time_rmse(variable, time, grid_yt, grid_xt)
    assert isinstance(result, float), "The result should be a float"
