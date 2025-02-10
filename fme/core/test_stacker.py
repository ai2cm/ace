import pytest
import torch

from .stacker import Stacker, natural_sort, unstack


@pytest.mark.parametrize(
    "names, sorted_names",
    [
        (
            ["a_1", "b_1", "c_1", "a_2"],
            [
                "a_1",
                "a_2",
                "b_1",
                "c_1",
            ],
        ),
        (
            [
                "a_0",
                "a_1",
                "a_12",
                "a_2",
            ],
            [
                "a_0",
                "a_1",
                "a_2",
                "a_12",
            ],
        ),
        (
            [
                "a_0001",
                "a_0012",
                "a_0002",
            ],
            [
                "a_0001",
                "a_0002",
                "a_0012",
            ],
        ),
        (
            [
                "ab1",
                "aa10",
                "aa2",
            ],
            ["aa2", "aa10", "ab1"],
        ),
    ],
)
def test_natural_sort(names, sorted_names):
    assert natural_sort(names) == sorted_names


_PREFIX_MAPS = [{"a": ["aa_", "ab_"], "b": ["b_", "bb_"], "c": ["ca", "cb"]}]

_DATA_NAMES = [
    ["aa_0", "aa_1", "b_00", "b_01", "b", "ca"],
    ["ab_0", "ab_1", "bb_1", "bb_2", "cb"],
    ["ab_0", "b_", "cc"],
]


@pytest.mark.parametrize(
    "prefix_map, data_names, expected_level_names",
    [
        (
            _PREFIX_MAPS[0],
            _DATA_NAMES[0],
            {"a": ["aa_0", "aa_1"], "b": ["b_00", "b_01"], "c": ["ca"]},
        ),
        (
            _PREFIX_MAPS[0],
            _DATA_NAMES[1],
            {"a": ["ab_0", "ab_1"], "b": ValueError, "c": ["cb"]},
        ),
        (
            _PREFIX_MAPS[0],
            _DATA_NAMES[2],
            {"a": ["ab_0"], "b": ["b_"], "c": KeyError},
        ),
    ],
)
def test_get_all_level_names(prefix_map, data_names, expected_level_names):
    data = {name: torch.zeros(2, 2) for name in data_names}
    stacker = Stacker(prefix_map)
    for standard_name in expected_level_names.keys():
        if isinstance(expected_level_names[standard_name], type):
            with pytest.raises(expected_level_names[standard_name]):
                stacker.get_all_level_names(standard_name, data)
        else:
            assert (
                stacker.get_all_level_names(standard_name, data)
                == expected_level_names[standard_name]
            )


@pytest.mark.parametrize(
    "prefix_map, data_names, expected_level_names",
    [
        (
            _PREFIX_MAPS[0],
            _DATA_NAMES[0],
            {"a": ["aa_0", "aa_1"], "b": ["b_00", "b_01"], "c": ["ca"]},
        ),
        (
            _PREFIX_MAPS[0],
            _DATA_NAMES[1],
            {"a": ["ab_0", "ab_1"], "c": ["cb"]},
        ),
        (
            _PREFIX_MAPS[0],
            _DATA_NAMES[2],
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
