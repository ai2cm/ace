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
