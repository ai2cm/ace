ERA5_AIMIP_ZARR := gs://vcm-ml-intermediate/2025-08-27-era5-1deg-monthly-mean-forcing-1978-2024.zarr
ERA5_AIMIP_START_TIME := 1978-10-01
ERA5_AIMIP_STOP_TIME := 2024-12-31

amip_enso_index:
	python compute_enso_index.py

era5_aimip_enso_index:
	python compute_enso_index.py \
		--sst-dataset $(ERA5_AIMIP_ZARR) \
		--ocean-mask-source $(ERA5_AIMIP_ZARR) \
		--sst-var surface_temperature \
		--lat-dim latitude \
		--lon-dim longitude \
		--ocean-mask-lat-dim latitude \
		--ocean-mask-lon-dim longitude \
		--start-time $(ERA5_AIMIP_START_TIME) \
		--stop-time $(ERA5_AIMIP_STOP_TIME)

.PHONY: amip_enso_index era5_aimip_enso_index
