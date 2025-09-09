
ZENODO_FORCING_URL := https://zenodo.org/api/records/16782373/files/ERA5-0.25deg-monthly-mean-forcing-1978-2024.nc/content
LOCAL_DATA_DIR := ./data
LOCAL_FORCING_FILE := $(LOCAL_DATA_DIR)/ERA5-0.25deg-monthly-mean-forcing-1978-2024.nc
LOCAL_REGRIDDED_FILE := $(LOCAL_DATA_DIR)/ERA5-1deg-monthly-mean-forcing-1978-2024.nc
OUTPUT_ZARR_NAME = 2025-08-27-aimip-era5-1deg-forcing-1978-2024.zarr
LOCAL_OUTPUT_ZARR := $(LOCAL_DATA_DIR)/$(OUTPUT_ZARR_NAME)
GCS_PATH_PROCESSED_FORCING ?= gs://vcm-ml-intermediate/$(OUTPUT_ZARR_NAME)
ENVIRONMENT_NAME=regrid-aimip-forcing

create_environment:
	conda create -n $(ENVIRONMENT_NAME)
	conda install -n $(ENVIRONMENT_NAME) -c conda-forge xesmf dask netCDF4 zarr gcsfs scipy

all: process_aimip_forcing

data_dir:
	mkdir -p $(LOCAL_DATA_DIR)

$(LOCAL_FORCING_FILE): data_dir
	wget -O $(LOCAL_FORCING_FILE) $(ZENODO_FORCING_URL)

$(LOCAL_REGRIDDED_FILE): $(LOCAL_FORCING_FILE)
	python regrid_aimip_forcing.py $(LOCAL_FORCING_FILE) $(LOCAL_REGRIDDED_FILE)

$(LOCAL_OUTPUT_ZARR): $(LOCAL_REGRIDDED_FILE)
	python interpolate_aimip_forcing.py $(LOCAL_REGRIDDED_FILE) $(LOCAL_OUTPUT_ZARR)

upload_processed_aimip_forcing: $(LOCAL_OUTPUT_ZARR)
	gsutil -m cp -r $(LOCAL_OUTPUT_ZARR) $(GCS_PATH_PROCESSED_FORCING)

process_aimip_forcing: upload_processed_aimip_forcing
	@echo "Processed AIMIP forcing file uploaded to $(GCS_PATH_PROCESSED_FORCING)"

clean:
	rm -rf $(LOCAL_DATA_DIR)

.PHONY: all clean upload_processed_aimip_forcing process_aimip_forcing