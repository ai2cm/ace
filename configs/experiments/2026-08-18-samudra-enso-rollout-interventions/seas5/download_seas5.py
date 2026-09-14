"""Download SEAS5 hindcast monthly-mean SST for the Nino3.4 comparison.

Requires ~/.cdsapirc with a Copernicus CDS key (https://cds.climate.copernicus.eu).
Fetches the 25-member re-forecasts for init months 2002-01..2010-03 (our 99-IC
protocol), leads 1-6 months (the public C3S archive's extent), Nino3.4 box only.
Also fetches 1993-2016 for the lead-dependent model climatology (drift removal).
"""

import os

import cdsapi

OUT = os.environ.get("SEAS5_OUT", "seas5_data")
os.makedirs(OUT, exist_ok=True)
c = cdsapi.Client()

# hindcast years for climatology + verification (1993-2016 = SEAS5 re-forecast set)
for year in range(1993, 2017):
    months = list(range(1, 13)) if year != 2010 else list(range(1, 13))
    dest = f"{OUT}/seas5_sst_{year}.nc"
    if os.path.exists(dest):
        continue
    c.retrieve(
        "seasonal-monthly-single-levels",
        {
            "format": "netcdf",
            "originating_centre": "ecmwf",
            "system": "51",
            "variable": "sea_surface_temperature",
            "product_type": "monthly_mean",
            "year": str(year),
            "month": [f"{m:02d}" for m in months],
            "leadtime_month": ["1", "2", "3", "4", "5", "6"],
            # N/W/S/E, Nino3.4 with a margin
            "area": [6, -171, -6, -119],
        },
        dest,
    )
    print("downloaded", dest)
