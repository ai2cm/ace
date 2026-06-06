import pytest
import torch

from fme.core.dicts import add_names, add_residual, to_flat_dict, to_nested_dict


def get_cfg_and_args_dicts():
    config_d = {
        "top": 1,
        "seq": [dict(a=1), dict(a=2)],
        "nested": {"k1": 2, "k2": 3, "double_nest": {"k1": 4, "k2": 5}},
    }

    flat_d = {
        "top": 1,
        "seq": [dict(a=1), dict(a=2)],
        "nested.k1": 2,
        "nested.k2": 3,
        "nested.double_nest.k1": 4,
        "nested.double_nest.k2": 5,
    }

    return config_d, flat_d


def test_to_flat_dict():
    config_d, expected = get_cfg_and_args_dicts()
    result = to_flat_dict(config_d)
    assert result == expected


def test_to_nested_dict():
    expected, args_d = get_cfg_and_args_dicts()
    result = to_nested_dict(args_d)
    assert result == expected


def test_flat_dict_round_trip():
    config_d, _ = get_cfg_and_args_dicts()

    args_d = to_flat_dict(config_d)
    result = to_nested_dict(args_d)

    assert result == config_d


def test_add_names():
    input = {"a": 1, "b": 2}
    output = {"a": 3, "b": 4}
    names = ["a"]
    new_output = add_names(input, output, names)
    assert new_output["a"] == 4
    assert new_output["b"] == 4


def test_add_names_passes_with_extra_name():
    input = {"a": 1, "b": 2}
    output = {"a": 3, "b": 4}
    names = ["a", "c"]
    new_output = add_names(input, output, names)
    assert new_output["a"] == 4
    assert new_output["b"] == 4


def test_add_names_raises_key_error_on_missing_input():
    input = {"a": 1}
    output = {"a": 3, "b": 4}
    names = ["b"]
    with pytest.raises(KeyError):
        add_names(input, output, names)


def test_add_residual_full_only():
    inp = {"a": torch.tensor([[[1.0, 3.0]]]), "b": torch.tensor([[[2.0, 4.0]]])}
    tend = {"a": torch.tensor([[[0.1, 0.2]]]), "b": torch.tensor([[[0.3, 0.4]]])}
    result = add_residual(
        inp, tend, full_residual_names=["a", "b"], anomaly_only_residual_names=[]
    )
    torch.testing.assert_close(result["a"], torch.tensor([[[1.1, 3.2]]]))
    torch.testing.assert_close(result["b"], torch.tensor([[[2.3, 4.4]]]))


def test_add_residual_anomaly_only():
    inp = {"a": torch.tensor([[[10.0, 20.0]]])}
    tend = {"a": torch.tensor([[[0.5, 0.5]]])}
    result = add_residual(
        inp, tend, full_residual_names=[], anomaly_only_residual_names=["a"]
    )
    spatial_mean = inp["a"].mean(dim=(-2, -1), keepdim=True)
    expected = (inp["a"] - spatial_mean) + tend["a"]
    torch.testing.assert_close(result["a"], expected)


def test_add_residual_mixed():
    inp = {"a": torch.tensor([[[10.0, 20.0]]]), "b": torch.tensor([[[5.0, 5.0]]])}
    tend = {"a": torch.tensor([[[1.0, 1.0]]]), "b": torch.tensor([[[2.0, 2.0]]])}
    result = add_residual(
        inp, tend, full_residual_names=["b"], anomaly_only_residual_names=["a"]
    )
    # "b" gets full residual
    torch.testing.assert_close(result["b"], torch.tensor([[[7.0, 7.0]]]))
    # "a" gets anomaly-only: spatial_mean of [10, 20] = 15, anomaly = [-5, 5]
    torch.testing.assert_close(result["a"], torch.tensor([[[-4.0, 6.0]]]))


def test_add_residual_no_residual():
    inp = {"a": torch.tensor([[[10.0, 20.0]]])}
    tend = {"a": torch.tensor([[[1.0, 2.0]]])}
    result = add_residual(
        inp, tend, full_residual_names=[], anomaly_only_residual_names=[]
    )
    torch.testing.assert_close(result["a"], tend["a"])
