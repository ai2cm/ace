import json
from pydantic import BaseModel
from typing import Literal, List, Union, Optional
import datetime
from fcn_mip.schema import InitialConditionSource


# https://docs.pydantic.dev/usage/types/#discriminated-unions-aka-tagged-unions
class WeatherEventProperties(BaseModel):
    """
    Attributes:
        netcdf: load the initial conditions from this path if given

    """

    name: str
    start_time: Optional[datetime.datetime]
    initial_condition_source: InitialConditionSource = InitialConditionSource.era5
    netcdf: str = ""


class Diagnostic(BaseModel):
    type: str
    function: str = ""
    channels: List[str]
    nbins: int = 10


class Window(BaseModel):
    type: Literal["Window"] = "Window"
    name: str
    lat_min: float = -90
    lat_max: float = 90
    lon_min: float = 0
    lon_max: float = 360
    diagnostics: List[Diagnostic]


class CWBDomain(BaseModel):
    type: Literal["CWBDomain"]
    name: str
    path: str = "/lustre/fsw/sw_climate_fno/nbrenowitz/2023-01-24-cwb-4years.zarr"
    diagnostics: List[Diagnostic]


class MultiPoint(BaseModel):
    type: Literal["MultiPoint"]
    name: str
    lat: List[float]
    lon: List[float]
    diagnostics: List[Diagnostic]


Domain = Union[Window, CWBDomain, MultiPoint]


class WeatherEvent(BaseModel):
    properties: WeatherEventProperties
    domains: List[Domain]


def _read():
    with open("weather_events.json") as f:
        return json.load(f)


def list_():
    events = _read()
    return list(events)


def read(forecast_name: str) -> WeatherEvent:
    weather_events = _read()
    weather_event = weather_events[forecast_name]

    for domain in weather_event["domains"]:
        if domain["name"] == "global":
            domain["type"] = "Window"
            domain["lat_min"] = -90
            domain["lat_max"] = 90
            domain["lon_min"] = 0
            domain["lon_max"] = 360
        elif domain["name"] == "northern_hemisphere":
            domain["lat_min"] = 0
            domain["lat_max"] = 90
            domain["lon_min"] = 0
            domain["lon_max"] = 360
        elif domain["name"] == "southern_hemisphere":
            domain["lat_min"] = -90
            domain["lat_max"] = 0
            domain["lon_min"] = 0
            domain["lon_max"] = 360
        elif domain["name"] == "CWBDomain":
            if len(domain["diagnostics"]) > 1:
                print("CWBDomain only supports one diagnostic")
                domain["diagnostics"] = domain["diagnostics"][0]
    event = WeatherEvent.parse_obj(weather_event)
    return event
