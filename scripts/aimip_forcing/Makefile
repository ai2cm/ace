PUBLIC_FORCING_FILE_NAME := ERA5-0.25deg-monthly-mean-forcing-1978-2024.nc
LOCAL_DATA_DIR := ./data
LOCAL_OUTPUT_DIR := $(LOCAL_DATA_DIR)/create_public_forcing
LOCAL_PATH_PUBLIC_AIMIP_FORCING = $(LOCAL_OUTPUT_DIR)/$(PUBLIC_FORCING_FILE_NAME)
GCS_PATH_PUBLIC_AIMIP_FORCING := gs://vcm-ml-intermediate/2025-09-05-era5-0.25deg-monthly-mean-forcing-1978-2024/
ZENODO_FORCING_URL := https://zenodo.org/api/records/17065758/files/$(PUBLIC_FORCING_FILE_NAME)/content
LOCAL_FORCING_FILE := $(LOCAL_DATA_DIR)/$(PUBLIC_FORCING_FILE_NAME)
LOCAL_REGRIDDED_FILE := $(LOCAL_DATA_DIR)/ERA5-1deg-monthly-mean-forcing-1978-2024.nc
OUTPUT_ZARR_NAME = 2025-09-09-aimip-era5-1deg-forcing-1978-2024.zarr
LOCAL_OUTPUT_ZARR := $(LOCAL_DATA_DIR)/$(OUTPUT_ZARR_NAME)
GCS_PATH_PROCESSED_FORCING ?= gs://vcm-ml-intermediate/$(OUTPUT_ZARR_NAME)
ENVIRONMENT_NAME=regrid-aimip-forcing
IC_OUTPUT_DIR := $(LOCAL_DATA_DIR)/aimip_ics
GCS_IC_PATH ?= gs://vcm-ml-intermediate/2025-09-12-AIMIP-evaluation-ICs/
ERA5_GCS_DATA ?= gs://vcm-ml-intermediate/2024-06-20-era5-1deg-8layer-1940-2022.zarr
MODIFIED_FORCING_NAME = 2025-09-12-aimip-era5-1deg-forcing-1978-2024-repeat-first.zarr
LOCAL_MODIFIED_FORCING := $(LOCAL_DATA_DIR)/$(MODIFIED_FORCING_NAME)
GCS_PATH_MODIFIED_FORCING ?= gs://vcm-ml-intermediate/$(MODIFIED_FORCING_NAME)

create_environment:
	conda create -n $(ENVIRONMENT_NAME)
	conda install -n $(ENVIRONMENT_NAME) -c conda-forge xesmf dask netCDF4 zarr gcsfs scipy

all: process_aimip_forcing

data_dir:
	mkdir -p $(LOCAL_DATA_DIR)

output_dir: data_dir
	mkdir -p $(LOCAL_OUTPUT_DIR)

create_aimip_forcing_file: output_dir
	python create_public_aimip_forcing.py $(LOCAL_PATH_PUBLIC_AIMIP_FORCING)

upload_aimip_forcing_file: create_aimip_forcing_file
	gsutil cp $(LOCAL_PATH_PUBLIC_AIMIP_FORCING) $(GCS_PATH_PUBLIC_AIMIP_FORCING)

create_aimip_forcing: upload_aimip_forcing_file
	@echo "Public AIMIP forcing file created and uploaded to $(GCS_PATH_PUBLIC_AIMIP_FORCING)

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

$(IC_OUTPUT_DIR)/.done: data_dir
	mkdir -p $(IC_OUTPUT_DIR)
	python create_aimip_ic_datasets.py $(IC_OUTPUT_DIR) \
	  --era5-gcs-data $(ERA5_GCS_DATA)
	touch $(IC_OUTPUT_DIR)/.done

upload_aimip_ics: $(IC_OUTPUT_DIR)/.done
	gsutil -m cp $(IC_OUTPUT_DIR)/*.nc $(GCS_IC_PATH)

create_aimip_ics: upload_aimip_ics
	@echo "AIMIP IC datasets uploaded to $(GCS_IC_PATH)"

$(LOCAL_MODIFIED_FORCING): data_dir
	python prepend_first_timestep_forcing.py $(LOCAL_MODIFIED_FORCING) \
	  --input-forcing-path $(GCS_PATH_PROCESSED_FORCING)

upload_modified_forcing: $(LOCAL_MODIFIED_FORCING)
	gsutil -m cp -r $(LOCAL_MODIFIED_FORCING) $(GCS_PATH_MODIFIED_FORCING)

create_modified_forcing: upload_modified_forcing
	@echo "Modified AIMIP forcing uploaded to $(GCS_PATH_MODIFIED_FORCING)"

create_aimip_evaluation_datasets: create_aimip_ics create_modified_forcing
	@echo "All AIMIP evaluation datasets created"

# ---------------------------------------------------------------------------
# Daily variant (for the daily 1-degree model, e.g. nobgd4ek)
#
# Reuses the shared spatial regrid ($(LOCAL_REGRIDDED_FILE)); the 1-degree
# Gaussian grid is identical for the six-hourly and daily models. Only the time
# step and the initial-condition variable set differ, so this variant just
# reinterpolates the regridded monthly forcing to a daily step and builds ICs
# that include the daily model's near-surface prognostics.
#
# CONFIRM BEFORE RUNNING:
#   - DAILY_ERA5_GCS_DATA: the GCS path of the daily zarr (the training config
#     mounts it as /climate-default/2026-03-19-era5-1deg-8layer-daily-1940-2025.zarr).
#   - The daily calendar hour (assumed 06Z below) and the IC / prepend timestamps
#     must match timestamps that actually exist in that zarr.
# ---------------------------------------------------------------------------
DAILY_ERA5_GCS_DATA ?= gs://vcm-ml-intermediate/2026-03-19-era5-1deg-8layer-daily-1940-2025.zarr
DAILY_START_TIME ?= 1978-10-01T00:00:00
DAILY_END_TIME ?= 2024-12-31T23:59:59
DAILY_OUTPUT_ZARR_NAME = 2026-07-20-aimip-era5-1deg-daily-forcing-1978-2024.zarr
LOCAL_DAILY_OUTPUT_ZARR := $(LOCAL_DATA_DIR)/$(DAILY_OUTPUT_ZARR_NAME)
GCS_PATH_DAILY_FORCING ?= gs://vcm-ml-intermediate/$(DAILY_OUTPUT_ZARR_NAME)
DAILY_IC_OUTPUT_DIR := $(LOCAL_DATA_DIR)/aimip_ics_daily
GCS_DAILY_IC_PATH ?= gs://vcm-ml-intermediate/2026-07-20-AIMIP-evaluation-ICs-daily/
DAILY_IC_TARGET ?= 1978-09-30T06:00:00
DAILY_FORCING_FIRST_STEP ?= 1978-10-01T06:00:00
DAILY_MODIFIED_FORCING_NAME = 2026-07-20-aimip-era5-1deg-daily-forcing-1978-2024-repeat-first.zarr
LOCAL_DAILY_MODIFIED_FORCING := $(LOCAL_DATA_DIR)/$(DAILY_MODIFIED_FORCING_NAME)
GCS_PATH_DAILY_MODIFIED_FORCING ?= gs://vcm-ml-intermediate/$(DAILY_MODIFIED_FORCING_NAME)

$(LOCAL_DAILY_OUTPUT_ZARR): $(LOCAL_REGRIDDED_FILE)
	python interpolate_aimip_forcing.py $(LOCAL_REGRIDDED_FILE) $(LOCAL_DAILY_OUTPUT_ZARR) \
	  --freq 1D \
	  --ace2-era5-gcs-data $(DAILY_ERA5_GCS_DATA) \
	  --start-time $(DAILY_START_TIME) \
	  --end-time $(DAILY_END_TIME) \
	  --extension-start ""

upload_processed_daily_forcing: $(LOCAL_DAILY_OUTPUT_ZARR)
	gsutil -m cp -r $(LOCAL_DAILY_OUTPUT_ZARR) $(GCS_PATH_DAILY_FORCING)

process_daily_forcing: upload_processed_daily_forcing
	@echo "Processed daily AIMIP forcing uploaded to $(GCS_PATH_DAILY_FORCING)"

$(DAILY_IC_OUTPUT_DIR)/.done: data_dir
	mkdir -p $(DAILY_IC_OUTPUT_DIR)
	python create_aimip_ic_datasets.py $(DAILY_IC_OUTPUT_DIR) \
	  --era5-gcs-data $(DAILY_ERA5_GCS_DATA) \
	  --target-timestamp $(DAILY_IC_TARGET) \
	  --include-near-surface
	touch $(DAILY_IC_OUTPUT_DIR)/.done

upload_daily_aimip_ics: $(DAILY_IC_OUTPUT_DIR)/.done
	gsutil -m cp $(DAILY_IC_OUTPUT_DIR)/*.nc $(GCS_DAILY_IC_PATH)

create_daily_aimip_ics: upload_daily_aimip_ics
	@echo "Daily AIMIP IC datasets uploaded to $(GCS_DAILY_IC_PATH)"

$(LOCAL_DAILY_MODIFIED_FORCING): data_dir
	python prepend_first_timestep_forcing.py $(LOCAL_DAILY_MODIFIED_FORCING) \
	  --input-forcing-path $(GCS_PATH_DAILY_FORCING) \
	  --input-timestamp $(DAILY_FORCING_FIRST_STEP) \
	  --output-timestamp $(DAILY_IC_TARGET)

upload_daily_modified_forcing: $(LOCAL_DAILY_MODIFIED_FORCING)
	gsutil -m cp -r $(LOCAL_DAILY_MODIFIED_FORCING) $(GCS_PATH_DAILY_MODIFIED_FORCING)

create_daily_modified_forcing: upload_daily_modified_forcing
	@echo "Modified daily AIMIP forcing uploaded to $(GCS_PATH_DAILY_MODIFIED_FORCING)"

create_daily_aimip_evaluation_datasets: create_daily_aimip_ics create_daily_modified_forcing
	@echo "All daily AIMIP evaluation datasets created"

clean:
	rm -rf $(LOCAL_DATA_DIR)

.PHONY: all clean upload_processed_aimip_forcing process_aimip_forcing \
  create_aimip_forcing_file upload_aimip_forcing_file create_aimip_forcing \
  upload_aimip_ics create_aimip_ics upload_modified_forcing create_modified_forcing \
  create_aimip_evaluation_datasets \
  upload_processed_daily_forcing process_daily_forcing \
  upload_daily_aimip_ics create_daily_aimip_ics \
  upload_daily_modified_forcing create_daily_modified_forcing \
  create_daily_aimip_evaluation_datasets
