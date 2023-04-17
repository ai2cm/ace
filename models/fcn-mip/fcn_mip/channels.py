"""Global table of channels
"""
import dataclasses


@dataclasses.dataclass
class Channel:
    units: str
    standard_name: str

    def add_attributes(self, nc_variable):
        nc_variable.units = self.units
        nc_variable.standard_name = self.standard_name


channels = {
    "u200": Channel("m s^-1", "eastward_wind"),
    "v200": Channel("m s^-1", "northward_wind"),
    "t200": Channel("degK", "air_temperature"),
    "z200": Channel("m^2 s^-2", "geopotential"),
    "u500": Channel("m s^-1", "eastward_wind"),
    "v500": Channel("m s^-1", "northward_wind"),
    "t500": Channel("degK", "air_temperature"),
    "z500": Channel("m^2 s^-2", "geopotential"),
    "u850": Channel("m s^-1", "eastward_wind"),
    "v850": Channel("m s^-1", "northward_wind"),
    "t850": Channel("degK", "air_temperature"),
    "z850": Channel("m^2 s^-2", "geopotential"),
}
