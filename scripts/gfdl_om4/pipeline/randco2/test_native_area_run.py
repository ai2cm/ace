import pathlib

import pytest

from ..om4.config import load_config as load_om4_config
from .config import load_config

CONFIGS = pathlib.Path(__file__).parents[2] / "configs"

TOY_CONFIG = """
target_grid: F90
weights_url: gs://bucket/weights
output:
  path: gs://bucket/out.zarr
statics:
  store: gs://bucket/ocean_static.zarr
wetmask:
  store: gs://bucket/source.zarr
  variable: SST
stream:
  name: ice_snapshot
  store: gs://bucket/source.zarr
  variables:
    - SST
"""


def test_config_accepts_statics_with_empty_variables(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(TOY_CONFIG)
    config = load_config(str(path))
    assert config.statics.store == "gs://bucket/ocean_static.zarr"
    assert config.statics.variables == []


@pytest.mark.parametrize("resolution", ["1deg", "4deg"])
def test_randco2_configs_set_statics_store(resolution):
    config = load_config(
        str(CONFIGS / f"cm4-like-am4-randco2-sea-surface-{resolution}.yaml"),
        member="member",
    )
    om4 = load_om4_config(str(CONFIGS / "om4-picontrol-1deg-5daily.yaml"))
    assert config.statics.store == om4.statics.store
    assert config.statics.variables == []
