from fcn_mip import schema
import json


def test_model():
    obj = schema.Model(
        architecture="some_arch",
        n_history=0,
        channel_set=schema.ChannelSet.var34,
        grid=schema.Grid.grid_720x1440,
        in_channels=[0, 1],
        out_channels=[0, 1],
    )
    loaded = json.loads(obj.json())
    assert loaded["channel_set"] == obj.channel_set.var34.value
