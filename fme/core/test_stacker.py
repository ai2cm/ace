import pytest
import torch

from .stacker import Stacker, unstack

_PREFIX_MAP = {"a": ["aa_", "ab_"], "b": ["b_", "bb_"], "c": ["ca", "cb"]}


@pytest.mark.parametrize("prefix_map", [_PREFIX_MAP])
@pytest.mark.parametrize(
    "data_names, expected_level_names",
    [
        (
            ["aa_0", "aa_1", "b_00", "b_01", "b", "ca"],
            {"a": ["aa_0", "aa_1"], "b": ["b_00", "b_01"], "c": ["ca"]},
        ),
        (
            ["ab_1", "ab_0", "bb_1", "bb_2", "cb"],
            {"a": ["ab_0", "ab_1"], "b": (ValueError, "Missing level 0"), "c": ["cb"]},
        ),
        (
            ["ab_0", "b_", "cc"],
            {
                "a": ["ab_0"],
                "b": ["b_"],
                "c": (KeyError, "No prefix associated with 'c'"),
            },
        ),
    ],
)
def test_get_all_level_names(prefix_map, data_names, expected_level_names):
    data = {name: torch.zeros(2, 2) for name in data_names}
    stacker = Stacker(prefix_map)
    for standard_name in expected_level_names.keys():
        expected_names = expected_level_names[standard_name]
        if isinstance(expected_names, tuple):
            expected_error, message = expected_names
            with pytest.raises(expected_error, match=message):
                stacker.get_all_level_names(standard_name, data)
        else:
            assert (
                stacker.get_all_level_names(standard_name, data)
                == expected_level_names[standard_name]
            )


@pytest.mark.parametrize("prefix_map", [_PREFIX_MAP])
@pytest.mark.parametrize(
    "data_names, expected_level_names",
    [
        (
            ["aa_0", "aa_1", "b_00", "b_01", "b", "ca"],
            {"a": ["aa_0", "aa_1"], "b": ["b_00", "b_01"], "c": ["ca"]},
        ),
        (
            ["ab_0", "ab_1", "bb_2", "bb_1", "cb"],
            {"a": ["ab_0", "ab_1"], "c": ["cb"]},
        ),
        (
            ["ab_0", "b_", "cc"],
            {"a": ["ab_0"], "b": ["b_"]},
        ),
    ],
)
def test_stack_unstack(prefix_map, data_names, expected_level_names):
    torch.manual_seed(0)
    data = {name: torch.rand(2, 2) for name in data_names}
    stacker = Stacker(prefix_map)
    for standard_name in expected_level_names.keys():
        level_names = expected_level_names[standard_name]
        if len(level_names) == 1:
            expected_stacked = data[level_names[0]].unsqueeze(-1)
        else:
            expected_stacked = torch.stack([data[name] for name in level_names], dim=-1)
        stacked = stacker(standard_name, data)
        assert torch.allclose(stacked, expected_stacked)
        unstacked = unstack(
            stacked, names=stacker.get_all_level_names(standard_name, data)
        )
        for name in level_names:
            assert name in unstacked
            assert torch.allclose(unstacked[name], data[name])


@pytest.mark.parametrize(
    "data_names, expected_prefix_map, expected_level_names",
    [
        (
            ["aa_1", "aa_0", "b_00", "b_01", "b", "ca"],
            {"aa_": ["aa_"], "b_": ["b_"], "b": ["b"], "ca": ["ca"]},
            {"aa_": ["aa_0", "aa_1"], "b_": ["b_00", "b_01"], "b": ["b"], "ca": ["ca"]},
        ),
        (
            ["ab_0", "ab_1", "bb_1", "bb_2", "cb"],
            {"ab_": ["ab_"], "bb_": ["bb_"], "cb": ["cb"]},
            {
                "ab_": ["ab_0", "ab_1"],
                "bb_": (ValueError, "Missing level 0"),
                "cb": ["cb"],
            },
        ),
        (
            ["ab_0", "b_", "cc"],
            {"ab_": ["ab_"], "b_": ["b_"], "cc": ["cc"]},
            {"ab_": ["ab_0"], "b_": ["b_"], "cc": ["cc"]},
        ),
    ],
)
def test_inferred_stacker(data_names, expected_prefix_map, expected_level_names):
    data = {name: torch.rand(2, 2) for name in data_names}
    stacker = Stacker()
    with pytest.raises(RuntimeError, match="hasn't yet been built"):
        stacker("a", data)
    stacker.infer_prefix_map(data.keys())
    with pytest.raises(RuntimeError, match="has already been built"):
        stacker.infer_prefix_map(data.keys())
    assert stacker.prefix_map == expected_prefix_map
    standard_names = stacker.standard_names
    assert standard_names == list(expected_prefix_map.keys())
    for standard_name in stacker.standard_names:
        expected_levels = expected_level_names[standard_name]
        if isinstance(expected_levels, tuple):
            expected_error, message = expected_levels
            with pytest.raises(expected_error, match=message):
                _ = stacker(standard_name, data)
        else:
            stacked = stacker(standard_name, data)
            tensors = [data[name] for name in expected_levels]
            expected_stacked = torch.stack(tensors, dim=-1)
            assert torch.allclose(stacked, expected_stacked)
