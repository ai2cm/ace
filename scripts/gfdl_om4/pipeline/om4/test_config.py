import pytest

from ..postprocess import PostprocessConfig
from ..randco2.config import StreamConfig as RandCO2StreamConfig
from .config import StreamConfig


@pytest.mark.parametrize("stream_config", [StreamConfig, RandCO2StreamConfig])
def test_drop_outputs_unknown_name_raises(stream_config):
    kelvin_sst = PostprocessConfig("kelvin_sst", {"celsius_sst": "SST"})
    stream_config(
        name="ice",
        store="toy",
        variables=["SST"],
        postprocess=[kelvin_sst],
        drop_outputs=["SST", "SST_degK"],
    )
    with pytest.raises(ValueError, match="drop_outputs \\['sst'\\]"):
        stream_config(
            name="ice",
            store="toy",
            variables=["SST"],
            postprocess=[kelvin_sst],
            drop_outputs=["sst"],
        )


def _sea_ice_stream(sea_ice="sea_ice", **time_options) -> StreamConfig:
    return StreamConfig(
        name="ice",
        store="toy",
        variables=["sea_ice_fraction", "HI", "UI", "VI"],
        renaming={"sea_ice_fraction": "ocean_sea_ice_fraction"},
        full_cell_variables=["sea_ice_fraction"],
        postprocess=[sea_ice],
        **time_options,
    )


@pytest.mark.parametrize(
    "sea_ice",
    ["sea_ice", PostprocessConfig("sea_ice", {"sea_ice_fraction": "sea_ice_fraction"})],
)
def test_sea_ice_on_block_mean_stream_raises(sea_ice):
    with pytest.raises(ValueError, match="sea_ice postprocess to a block-meaned"):
        _sea_ice_stream(sea_ice, time_block_mean_factor=20)


def test_sea_ice_on_subsampled_stream_loads():
    _sea_ice_stream(time_subsample_stride=20)


def _ice_area_stream(**options) -> StreamConfig:
    fields = {
        "name": "ice",
        "store": "toy",
        "variables": ["sea_ice_fraction", "HI", "TS"],
        "renaming": {"sea_ice_fraction": "ocean_sea_ice_fraction"},
        "full_cell_variables": ["sea_ice_fraction"],
        "ice_area_weight": "sea_ice_fraction",
        "ice_area_weighted": ["HI", "TS"],
        "ice_area_products": {"HI": "ocean_sea_ice_thickness_mean"},
    }
    return StreamConfig(**{**fields, **options})


def test_ice_area_config_loads():
    stream = _ice_area_stream()
    assert "ocean_sea_ice_thickness_mean" in stream.output_names()
    weight_only = _ice_area_stream(
        variables=["TS"],
        renaming={},
        full_cell_variables=[],
        ice_area_weighted=["TS"],
        ice_area_products={},
        ice_area_weight_store="weights",
    )
    assert weight_only.output_names() == {"TS"}
    assert weight_only.weight_only() == "sea_ice_fraction"


@pytest.mark.parametrize(
    "options, match",
    [
        ({"ice_area_weighted": ["HI", "HS"]}, "'HS' not in stream"),
        (
            {"ice_area_weighted": ["sea_ice_fraction"], "ice_area_products": {}},
            "is a full_cell_variable",
        ),
        ({"ice_area_weight": None}, "without ice_area_weight"),
        (
            {
                "variables": ["c", "HI", "TS"],
                "renaming": {},
                "full_cell_variables": [],
                "ice_area_weight": "c",
                "ice_area_weighted": ["HI", "TS", "c"],
            },
            "ice_area_weight 'c' is in ice_area_weighted",
        ),
        ({"ice_area_weight_store": "weights"}, "is in variables"),
        (
            {
                "ice_area_weight": None,
                "ice_area_weighted": [],
                "ice_area_products": {},
                "ice_area_weight_store": "weights",
            },
            "ice_area_weight_store without ice_area_weight",
        ),
        ({"ice_area_products": {"SST": "x"}}, "keys \\['SST'\\] not in"),
        ({"ice_area_products": {"HI": "TS"}}, "names \\['TS'\\] collide"),
        (
            {"ice_area_products": {"HI": "v", "TS": "v"}},
            "names \\['v'\\] collide",
        ),
    ],
)
def test_ice_area_config_validation(options, match):
    with pytest.raises(ValueError, match=match):
        _ice_area_stream(**options)


def test_ice_area_weighted_on_block_mean_stream_raises():
    with pytest.raises(ValueError, match="ice_area_weighted on a block-meaned"):
        StreamConfig(
            name="ice",
            store="toy",
            variables=["sea_ice_fraction", "TS"],
            ice_area_weight="sea_ice_fraction",
            ice_area_weighted=["TS"],
            time_block_mean_factor=20,
        )
