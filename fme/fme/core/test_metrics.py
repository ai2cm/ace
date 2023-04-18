import pytest
import metrics
import torch

variables = [1, 2, 4]
times = [1, 2, 4]
grid_yts = [2, 4]
grid_xts = [1, 2, 4]

test_parameters = [(variable, time, grid_yt, grid_xt) for variable in variables
                   for time in times for grid_yt in grid_yts for grid_xt in grid_xts]


@pytest.mark.parametrize("variable, time, grid_yt, grid_xt", test_parameters)
def test_weighted_global_mean_bias(variable, time, grid_yt, grid_xt):
    """Tests the shapes and a couple simple cases of the global mean bias."""
    x = torch.randn(variable, time, grid_yt, grid_xt)
    y = torch.randn(variable, time, grid_yt, grid_xt)
    result = metrics.weighted_global_mean_bias(x, y)
    assert result.shape == (variable,), \
        f"Global mean bias should have shape (variable,)"

    result = metrics.weighted_global_mean_bias(x, x.clone())
    assert torch.all(torch.isclose(result, torch.tensor(0.0))), "Global mean bias between identical tensors should be zero."

    x = torch.zeros(variable, time, grid_yt, grid_xt)
    y = torch.ones(variable, time, grid_yt, grid_xt)
    result = metrics.weighted_global_mean_bias(x, y)
    spherical_area_weights = metrics.spherical_area_weights(grid_yt, grid_xt)
    assert torch.all(torch.isclose(result, spherical_area_weights.mean())), "Global mean bias between zero and one should be the mean of the lat weights."



@pytest.mark.parametrize("variable, time, grid_yt, grid_xt", test_parameters)
def test_weighted_time_mean_bias(variable, time, grid_yt, grid_xt):
    x = torch.randn(variable, time, grid_yt, grid_xt)
    y = torch.randn(variable, time, grid_yt, grid_xt)
    result = metrics.weighted_time_mean_bias(x, y)
    assert result.shape == (variable, time), "Time mean bias should have shape (variable, time)"

    result = metrics.weighted_time_mean_bias(x, x.clone())
    assert torch.all(torch.isclose(result, torch.tensor(0.0))), "Time mean bias between identical tensors should be zero."

    x = torch.zeros(variable, time, grid_yt, grid_xt)
    y = torch.ones(variable, time, grid_yt, grid_xt)
    result = metrics.weighted_time_mean_bias(x, y)
    spherical_area_weights = metrics.spherical_area_weights(grid_yt, grid_xt)
    assert torch.all(torch.isclose(result, spherical_area_weights.mean((-1, -2)))), "Time mean bias between zero and one should be the mean of the lat weights."


@pytest.mark.parametrize("variable, time, grid_yt, grid_xt", test_parameters)
def test_weighted_global_time_rmse(variable, time, grid_yt, grid_xt):
    x = torch.randn(variable, time, grid_yt, grid_xt)
    y = torch.randn(variable, time, grid_yt, grid_xt)
    result = metrics.weighted_global_time_rmse(x, y)
    assert result.shape == (variable,), "Global time RMSE should have shape (variable,)"

    result = metrics.weighted_global_time_rmse(x, x.clone())
    assert torch.all(torch.isclose(result, torch.tensor(0.0))), "Global time RMSE between identical tensors should be zero."

    x = torch.zeros(variable, time, grid_yt, grid_xt)
    y = torch.ones(variable, time, grid_yt, grid_xt)
    result = metrics.weighted_global_time_rmse(x, y)
    spherical_area_weights = metrics.spherical_area_weights(grid_yt, grid_xt)
    expected = torch.sqrt(spherical_area_weights.mean())
    assert torch.all(torch.isclose(result, expected)), f"Global time RMSE between zero and one should be the sqrt(mean) of the lat weights. {result}"


@pytest.mark.parametrize("variable, time, grid_yt, grid_xt", test_parameters)
def test_per_variable_fno_loss(variable, time, grid_yt, grid_xt):
    del time  # unused in this test

    x = torch.randn(variable, grid_yt, grid_xt)
    y = torch.randn(variable, grid_yt, grid_xt)
    result = metrics.per_variable_fno_loss(x, y)
    assert result.shape == (variable,), "Per variable FNO loss should have shape (variable,)"

    result = metrics.per_variable_fno_loss(x, x.clone())
    assert torch.all(torch.isclose(result, torch.tensor(0.0))), "Per variable FNO loss between identical tensors should be zero."