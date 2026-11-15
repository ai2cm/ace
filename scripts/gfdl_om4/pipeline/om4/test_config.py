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
