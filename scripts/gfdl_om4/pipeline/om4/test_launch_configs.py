"""The 5-daily 1-degree launch configs write every requested field, with the
per-ice-area SIS2 fields ice-area weighted (no network)."""

import pathlib

import pytest

from ..output_attrs import EASTWARD, NORTHWARD, geographic_component_attrs
from ..postprocess import SEA_ICE_FRACTION, PostprocessConfig
from ..randco2.config import load_config as load_randco2_config
from .config import PipelineConfig, StreamConfig, load_config

CONFIGS = pathlib.Path(__file__).parents[2] / "configs"
MEMBER = "ic_0001"

# Store basename T_bulkice is read from, per launch config.
LAUNCH_CONFIGS = {
    "om4-picontrol-1deg-5daily.yaml": "ice_6hourly.zarr",
    "om4-1pctco2-1deg-5daily.yaml": "T_bulkice_6hourly_snap.zarr",
    "cm4-abrupt-4xCO2-ensemble-ocean-1deg-5daily.yaml": "ice_6hourly_snap.zarr",
}

REQUESTED_FIELDS = (
    "SSS",
    "sisnmass",
    "simass",
    "HS",
    "TSN",
    "TS",
    "T_bulkice",
    "SSH",
    "SALTF",
    "LSRCi",
    "LSRCs",
    "LSNKi",
    "LSNKs",
    "BSNKi",
    "BSNKs",
    "SN2IC",
    "interval_average_sea_ice_fraction",
)

PER_ICE_AREA = (
    "HI",
    "HS",
    "TS",
    "TSN",
    "UI",
    "VI",
    "T1",
    "T2",
    "T3",
    "T4",
    "T_bulkice",
)


def _load(name: str) -> PipelineConfig:
    member = MEMBER if "{member}" in (CONFIGS / name).read_text() else None
    return load_config(str(CONFIGS / name), member=member)


def _written(config: PipelineConfig) -> set[str]:
    names: set[str] = set()
    for stream in config.streams:
        names |= stream.output_names() - set(stream.drop_outputs)
    return names


def _sea_ice_fraction_source(stream: StreamConfig) -> str | None:
    for entry in stream.postprocess:
        selection = PostprocessConfig(entry) if isinstance(entry, str) else entry
        if selection.name == "sea_ice":
            return selection.sources.get("sea_ice_fraction", SEA_ICE_FRACTION)
    return None


@pytest.mark.parametrize("name", LAUNCH_CONFIGS)
def test_launch_config_writes_requested_fields(name):
    written = _written(_load(name))
    expected = {
        *REQUESTED_FIELDS,
        "ocean_sea_ice_thickness_mean",
        "calving_residue",
        "SST_degK",
    }
    assert expected <= written, sorted(expected - written)
    assert "SST" not in written


@pytest.mark.parametrize("name", LAUNCH_CONFIGS)
def test_launch_config_weights_per_ice_area_fields(name):
    streams = _load(name).streams
    weighted = {x for stream in streams for x in stream.ice_area_weighted}
    assert set(PER_ICE_AREA) <= weighted, sorted(set(PER_ICE_AREA) - weighted)
    for stream in streams:
        source = _sea_ice_fraction_source(stream)
        if source is not None:
            assert "HI" in stream.ice_area_weighted, stream.name
            assert stream.ice_area_weight == source, stream.name


@pytest.mark.parametrize("name, basename", LAUNCH_CONFIGS.items())
def test_launch_config_t_bulkice_store(name, basename):
    stores = [
        stream.store
        for stream in _load(name).streams
        if "T_bulkice" in stream.variables
    ]
    assert [store.rstrip("/").rsplit("/", 1)[-1] for store in stores] == [basename]


def test_randco2_config_drops_sst():
    config = load_randco2_config(
        str(CONFIGS / "cm4-like-am4-randco2-sea-surface-1deg.yaml"), member=MEMBER
    )
    assert config.stream.drop_outputs == ["SST"]


# Source attrs of every rotated component in the om4 configs, as read from
# their stores (identical across stores).
ROTATED_SOURCE_ATTRS = {
    "uo": {
        "long_name": "Sea Water X Velocity",
        "standard_name": "sea_water_x_velocity",
    },
    "vo": {
        "long_name": "Sea Water Y Velocity",
        "standard_name": "sea_water_y_velocity",
    },
    "tauuo": {
        "long_name": "Surface Downward X Stress",
        "standard_name": "surface_downward_x_stress",
    },
    "tauvo": {
        "long_name": "Surface Downward Y Stress",
        "standard_name": "surface_downward_y_stress",
    },
    "UI": {"long_name": "ice velocity - x component"},
    "VI": {"long_name": "ice velocity - y component"},
}
ROTATED_LABELS = {
    "uo": ("Sea Water Eastward Velocity", "eastward_sea_water_velocity"),
    "vo": ("Sea Water Northward Velocity", "northward_sea_water_velocity"),
    "tauuo": ("Surface Downward Eastward Stress", "surface_downward_eastward_stress"),
    "tauvo": ("Surface Downward Northward Stress", "surface_downward_northward_stress"),
    "UI": ("ice velocity - eastward component", None),
    "VI": ("ice velocity - northward component", None),
}


def _rotated_pairs() -> set[tuple[str, ...]]:
    pairs = set()
    for path in sorted(CONFIGS.glob("**/*.yaml")):
        if "randco2" in path.name:
            continue
        config = _load(str(path.relative_to(CONFIGS)))
        pairs |= {tuple(pair) for s in config.streams for pair in s.rotated_pairs}
    return pairs


def test_rotated_pairs_relabeled_geographic():
    pairs = _rotated_pairs()
    assert pairs
    for u_name, v_name in pairs:
        for name, direction in ((u_name, EASTWARD), (v_name, NORTHWARD)):
            attrs = geographic_component_attrs(
                ROTATED_SOURCE_ATTRS[name], name, direction
            )
            long_name, standard_name = ROTATED_LABELS[name]
            assert attrs["long_name"] == long_name
            assert attrs.get("standard_name") == standard_name


def test_rotated_component_drops_unknown_standard_name():
    attrs = geographic_component_attrs(
        {"long_name": "zonal flow", "standard_name": "x_flow"}, "u", EASTWARD
    )
    assert attrs == {"long_name": "zonal flow, eastward component"}
