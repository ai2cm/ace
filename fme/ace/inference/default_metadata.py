from fme.core.dataset.data_typing import VariableMetadata

ERA5_VERSION_1 = {
    "DLWRFsfc": {
        "long_name": "Downward LW radiative flux at surface",
        "units": "W/m**2",
    },
    "DPT2m": {"long_name": "2m dewpoint temperature", "units": "K"},
    "DSWRFsfc": {
        "long_name": "Downward SW radiative flux at surface",
        "units": "W/m**2",
    },
    "DSWRFtoa": {"long_name": "Downward SW radiative flux at TOA", "units": "W/m**2"},
    "HGTsfc": {"long_name": "Topography height", "units": "m"},
    "LHTFLsfc": {"long_name": "Latent heat flux", "units": "W/m**2"},
    "PRATEsfc": {"long_name": "Surface precipitation rate", "units": "kg/m**2/s"},
    "PRESsfc": {"long_name": "Surface pressure", "units": "Pa"},
    "Q200": {"long_name": "Specific humidity at 200 hPa", "units": "kg kg**-1"},
    "Q2m": {"long_name": "2m specific humidity", "units": "kg/kg"},
    "Q500": {"long_name": "Specific humidity at 500 hPa", "units": "kg kg**-1"},
    "Q850": {"long_name": "Specific humidity at 850 hPa", "units": "kg kg**-1"},
    "SHTFLsfc": {"long_name": "Sensible heat flux", "units": "W/m**2"},
    "TMP200": {"long_name": "Temperature at 200 hPa", "units": "K"},
    "TMP2m": {"long_name": "2m air temperature", "units": "K"},
    "TMP500": {"long_name": "Temperature at 500 hPa", "units": "K"},
    "TMP850": {"long_name": "Temperature at 850 hPa", "units": "K"},
    "UGRD10m": {"long_name": "10m U component of wind", "units": "m/s"},
    "UGRD200": {"long_name": "U component of wind at 200 hPa", "units": "m s**-1"},
    "UGRD500": {"long_name": "U component of wind at 500 hPa", "units": "m s**-1"},
    "UGRD850": {"long_name": "U component of wind at 850 hPa", "units": "m s**-1"},
    "ULWRFsfc": {"long_name": "Upward LW radiative flux at surface", "units": "W/m**2"},
    "ULWRFtoa": {"long_name": "Upward LW radiative flux at TOA", "units": "W/m**2"},
    "USWRFsfc": {"long_name": "Upward SW radiative flux at surface", "units": "W/m**2"},
    "USWRFtoa": {"long_name": "Upward SW radiative flux at TOA", "units": "W/m**2"},
    "VGRD10m": {"long_name": "10m V component of wind", "units": "m/s"},
    "VGRD200": {"long_name": "V component of wind at 200 hPa", "units": "m s**-1"},
    "VGRD500": {"long_name": "V component of wind at 500 hPa", "units": "m s**-1"},
    "VGRD850": {"long_name": "V component of wind at 850 hPa", "units": "m s**-1"},
    "air_temperature_0": {"long_name": "Temperature level-0", "units": "K"},
    "air_temperature_1": {"long_name": "Temperature level-1", "units": "K"},
    "air_temperature_2": {"long_name": "Temperature level-2", "units": "K"},
    "air_temperature_3": {"long_name": "Temperature level-3", "units": "K"},
    "air_temperature_4": {"long_name": "Temperature level-4", "units": "K"},
    "air_temperature_5": {"long_name": "Temperature level-5", "units": "K"},
    "air_temperature_6": {"long_name": "Temperature level-6", "units": "K"},
    "air_temperature_7": {"long_name": "Temperature level-7", "units": "K"},
    "eastward_wind_0": {"long_name": "U component of wind level-0", "units": "m s**-1"},
    "eastward_wind_1": {"long_name": "U component of wind level-1", "units": "m s**-1"},
    "eastward_wind_2": {"long_name": "U component of wind level-2", "units": "m s**-1"},
    "eastward_wind_3": {"long_name": "U component of wind level-3", "units": "m s**-1"},
    "eastward_wind_4": {"long_name": "U component of wind level-4", "units": "m s**-1"},
    "eastward_wind_5": {"long_name": "U component of wind level-5", "units": "m s**-1"},
    "eastward_wind_6": {"long_name": "U component of wind level-6", "units": "m s**-1"},
    "eastward_wind_7": {"long_name": "U component of wind level-7", "units": "m s**-1"},
    "global_mean_co2": {"long_name": "global mean CO2 concentration", "units": "ppm"},
    "h1000": {"long_name": "Geopotential height at 1000 hPa", "units": "m"},
    "h200": {"long_name": "Geopotential height at 200 hPa", "units": "m"},
    "h250": {"long_name": "Geopotential height at 250 hPa", "units": "m"},
    "h300": {"long_name": "Geopotential height at 300 hPa", "units": "m"},
    "h500": {"long_name": "Geopotential height at 500 hPa", "units": "m"},
    "h700": {"long_name": "Geopotential height at 700 hPa", "units": "m"},
    "h850": {"long_name": "Geopotential height at 850 hPa", "units": "m"},
    "land_fraction": {"long_name": "land fraction", "units": "(0-1)"},
    "northward_wind_0": {
        "long_name": "V component of wind level-0",
        "units": "m s**-1",
    },
    "northward_wind_1": {
        "long_name": "V component of wind level-1",
        "units": "m s**-1",
    },
    "northward_wind_2": {
        "long_name": "V component of wind level-2",
        "units": "m s**-1",
    },
    "northward_wind_3": {
        "long_name": "V component of wind level-3",
        "units": "m s**-1",
    },
    "northward_wind_4": {
        "long_name": "V component of wind level-4",
        "units": "m s**-1",
    },
    "northward_wind_5": {
        "long_name": "V component of wind level-5",
        "units": "m s**-1",
    },
    "northward_wind_6": {
        "long_name": "V component of wind level-6",
        "units": "m s**-1",
    },
    "northward_wind_7": {
        "long_name": "V component of wind level-7",
        "units": "m s**-1",
    },
    "ocean_fraction": {"long_name": "ocean fraction", "units": "(0-1)"},
    "sea_ice_fraction": {"long_name": "sea ice fraction", "units": "(0-1)"},
    "soil_moisture_0": {
        "long_name": "Volumetric soil water layer 1",
        "units": "m**3 m**-3",
    },
    "soil_moisture_1": {
        "long_name": "Volumetric soil water layer 2",
        "units": "m**3 m**-3",
    },
    "soil_moisture_2": {
        "long_name": "Volumetric soil water layer 3",
        "units": "m**3 m**-3",
    },
    "soil_moisture_3": {
        "long_name": "Volumetric soil water layer 4",
        "units": "m**3 m**-3",
    },
    "specific_total_water_0": {
        "long_name": "Specific total water level-0",
        "units": "kg kg**-1",
    },
    "specific_total_water_1": {
        "long_name": "Specific total water level-1",
        "units": "kg kg**-1",
    },
    "specific_total_water_2": {
        "long_name": "Specific total water level-2",
        "units": "kg kg**-1",
    },
    "specific_total_water_3": {
        "long_name": "Specific total water level-3",
        "units": "kg kg**-1",
    },
    "specific_total_water_4": {
        "long_name": "Specific total water level-4",
        "units": "kg kg**-1",
    },
    "specific_total_water_5": {
        "long_name": "Specific total water level-5",
        "units": "kg kg**-1",
    },
    "specific_total_water_6": {
        "long_name": "Specific total water level-6",
        "units": "kg kg**-1",
    },
    "specific_total_water_7": {
        "long_name": "Specific total water level-7",
        "units": "kg kg**-1",
    },
    "surface_temperature": {"long_name": "Skin temperature", "units": "K"},
    "tendency_of_total_water_path_due_to_advection": {
        "long_name": "Tendency of total water path due to advection",
        "units": "kg/m**2/s",
    },
    "total_column_water_vapour": {
        "long_name": "Total column vertically-integrated water vapour",
        "units": "kg m**-2",
    },
}

VERSIONS = {
    "era5_v1": ERA5_VERSION_1,
}


def get_default_variable_metadata(
    version: str = "era5_v1",
) -> dict[str, VariableMetadata]:
    asdict = VERSIONS[version]
    return {key: VariableMetadata(**value) for key, value in asdict.items()}
