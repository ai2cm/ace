import os
from typing import Literal

import xarray as xr

var_names = {
    "coarse": {
        "atmos_8xdaily_inst_coarse": [
            "PWAT_coarse",
        ],
        "atmos_static_coarse": ["zsurf_coarse"],
        "sfc_8xdaily_ave_coarse": [
            "PRATEsfc_coarse",
            "UGRD10m_coarse",
            "VGRD10m_coarse",
        ],
        "sfc_8xdaily_inst_coarse": ["TMPsfc_coarse"],
    },
    "fine": {"atmos_static": ["zsurf"], "sfc_8xdaily_ave": ["PRATEsfc"]},
}

ace_var_names_mapping = {
    "PWAT_coarse": "total_water_path",
    "PRATEsfc_coarse": "PRATEsfc",
    "UGRD10m_coarse": "U10m",
    "VGRD10m_coarse": "V10m",
    "TMPsfc_coarse": "surface_temperature",
    "zsurf": "HGTsfc",
    "zsurf_coarse": "HGTsfc",
}

initial_conditions = [
    "ic_0001",
    "ic_0002",
    "ic_0003",
    "ic_0004",
    "ic_0005",
    "ic_0006",
    "ic_0007",
    "ic_0008",
    "ic_0009",
    "ic_0010",
    "ic_0011",
]

paths = {
    "fine": (
        "gs://vcm-ml-raw-flexible-retention/2023-08-14-C384-reference-ensemble/"
        "regridded-zarrs/gaussian_grid_720_by_1440"
    ),
    "coarse": (
        "gs://vcm-ml-raw-flexible-retention/2023-08-14-C384-reference-ensemble/"
        "regridded-zarrs/gaussian_grid_90_by_180"
    ),
}


outpaths = {"fine": "gaussian_grid_720_by_1440", "coarse": "gaussian_grid_90_by_180"}

ENABLE_COLDLINE = False


def read_dataset(
    resolution: Literal["fine", "coarse"], initial_condition: str
) -> xr.Dataset:
    datasets = []
    for filename, vars in var_names[resolution].items():
        ds = xr.open_zarr(
            os.path.join(paths[resolution], initial_condition, f"{filename}.zarr")
        )
        ds = ds[vars]
        for var in ds:
            del ds[var].encoding["chunks"]
            del ds[var].encoding["preferred_chunks"]
            if var in ace_var_names_mapping:
                ds = ds.rename({var: ace_var_names_mapping[var]})
        datasets.append(ds)

    return xr.merge(datasets)


def main():
    if not ENABLE_COLDLINE:
        raise ValueError(
            "The data for this script have been moved to coldline storage."
            "To re-run this operation, set the ENABLE_COLDLINE variable to True."
            "But be aware that this will incur costs."
        )
    for ic in initial_conditions:
        for resolution in ["coarse", "fine"]:
            print(f"Processing {ic}, '{resolution}'")
            ds = read_dataset(resolution, ic)
            outpath = (
                "gs://vcm-ml-intermediate/"
                "2024-05-09-C384-reference-ensemble-downscaling"
                f"/{outpaths[resolution]}/{ic}.zarr"
            )

            if resolution == "coarse":
                ds = ds.rename(
                    {"grid_yt_coarse": "grid_yt", "grid_xt_coarse": "grid_xt"}
                )

            ds = ds.chunk({"time": 1})
            ds.to_zarr(store=outpath, mode="w")


if __name__ == "__main__":
    main()
