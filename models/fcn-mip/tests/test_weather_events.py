from fcn_mip import weather_events
import pytest


@pytest.mark.parametrize("event", weather_events.list_())
def test_read(event):
    domain = weather_events.read(event)
    print(domain)


def test_parse():
    obj = {
        "properties": {"name": "Globe", "start_time": "2018-01-01 00:00:00"},
        "domains": [
            {
                "name": "global",
                "type": "Window",
                "diagnostics": [
                    {
                        "type": "absolute",
                        "function": "mean",
                        "channels": ["tcwv", "t2m", "u10", "v10"],
                    }
                ],
            }
        ],
    }

    weather_events.WeatherEvent.parse_obj(obj)


def test_parse_cwbdomain():
    obj = {
        "properties": {"name": "Taiwan", "start_time": "2018-10-07 18:00:00"},
        "domains": [
            {
                "name": "Taiwan",
                "type": "CWBDomain",
                "lat_min": 18,
                "lat_max": 30,
                "lon_min": 115,
                "lon_max": 125,
                "diagnostics": [
                    {
                        "type": "raw",
                        "function": "",
                        "channels": [
                            "u10",
                            "v10",
                            "t2m",
                            "sp",
                            "msl",
                            "t850",
                            "u1000",
                            "v1000",
                            "z1000",
                            "u850",
                            "v850",
                            "z850",
                            "u500",
                            "v500",
                            "z500",
                            "t500",
                            "z50",
                            "r500",
                            "r850",
                            "tcwv",
                            "u100m",
                            "v100m",
                            "u250",
                            "v250",
                            "z250",
                            "t250",
                        ],
                    }
                ],
            }
        ],
    }
    weather_events.WeatherEvent.parse_obj(obj)


def test_parse_multipoint():
    obj = {
        "properties": {"name": "EastCoast", "start_time": "2018-01-01 00:00:00"},
        "domains": [
            {
                "name": "Somewhere",
                "type": "MultiPoint",
                "lat": [40, 25, 42, 18, 29, 38],
                "lon": [286, 280, 289, 294, 265, 283],
                "diagnostics": [
                    {
                        "type": "raw",
                        "function": "",
                        "channels": ["tcwv", "t2m", "u10", "v10"],
                    }
                ],
            }
        ],
    }
    weather_events.WeatherEvent.parse_obj(obj)
