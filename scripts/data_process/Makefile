# The dependencies of the scripts below (where not containerized) are installed in the "fv3net" conda environment
# which can be installed using fv3net's Makefile. See
# https://github.com/ai2cm/fv3net/blob/8ed295cf0b8ca49e24ae5d6dd00f57e8b30169ac/Makefile#L310

# Some data has been put to coldline storage making it expensive
# to access.  In order to run processing on that data, set this
# variable to true.
ENABLE_COLDLINE ?= false

RESOLUTION ?= 1deg
LAYERS ?= 8layer

ROOT_BASE_CLIMSST_1DEG = gs://vcm-ml-raw-flexible-retention/2023-07-08-C96-FME-reference-ensemble/regridded-zarrs/gaussian_grid_180_by_360
ROOT_BASE_CLIMSST_4DEG = gs://vcm-ml-raw-flexible-retention/2023-07-08-C96-FME-reference-ensemble/regridded-zarrs/gaussian_grid_45_by_90
ROOT_BASE_AMIP_1DEG = gs://vcm-ml-raw-flexible-retention/2023-10-20-C96-FME-AMIP-ensemble-reference/regridded-zarrs/gaussian_grid_180_by_360
ROOT_BASE_AMIP_4DEG = gs://vcm-ml-raw-flexible-retention/2023-10-20-C96-FME-AMIP-ensemble-reference/regridded-zarrs/gaussian_grid_45_by_90
ROOT_INTERMEDIATE_CLIMSST_1DEG = gs://vcm-ml-intermediate/2023-08-09-vertically-resolved-1deg-fme-ensemble-dataset
ROOT_INTERMEDIATE_CLIMSST_4DEG = gs://vcm-ml-intermediate/2023-08-09-vertically-resolved-4deg-fme-ensemble-dataset
ROOT_INTERMEDIATE_AMIP_1DEG = gs://vcm-ml-intermediate/2023-10-27-vertically-resolved-1deg-fme-amip-ensemble-dataset
ROOT_INTERMEDIATE_AMIP_4DEG = gs://vcm-ml-intermediate/2023-10-27-vertically-resolved-4deg-fme-amip-ensemble-dataset
OUTPUT_DIR_CLIMSST_1DEG = /climate-default/2023-08-11-vertically-resolved-1deg-fme-ensemble-dataset
OUTPUT_DIR_CLIMSST_4DEG = /climate-default/2023-08-11-vertically-resolved-4deg-fme-ensemble-dataset
OUTPUT_DIR_AMIP_1DEG = /climate-default/2023-11-01-vertically-resolved-1deg-fme-amip-ensemble-dataset
OUTPUT_DIR_AMIP_4DEG = /climate-default/2023-11-01-vertically-resolved-4deg-fme-amip-ensemble-dataset
NAME_CLIMSST_1DEG = fv3gfs-ensemble
NAME_CLIMSST_4DEG = fv3gfs-ensemble-4deg
NAME_AMIP_1DEG = fv3gfs-AMIP-ensemble
NAME_AMIP_4DEG = fv3gfs-AMIP-ensemble-4deg
ATMOSPHERE_PROCESSING_IMAGE = us-central1-docker.pkg.dev/vcm-ml/full-model/atmosphere-processing
ATMOSPHERE_PROCESSING_IMAGE_VERSION = v2025.03.1  # Use vYYYY.MM.MICRO (MICRO is zero-indexed)

ifeq ($(RESOLUTION),1deg)
	ROOT_INTERMEDIATE ?= $(ROOT_INTERMEDIATE_CLIMSST_1DEG)
	ROOT_INTERMEDIATE_AMIP ?= $(ROOT_INTERMEDIATE_AMIP_1DEG)
	ROOT_BASE ?= $(ROOT_BASE_CLIMSST_1DEG)
	ROOT_AMIP ?= $(ROOT_BASE_AMIP_1DEG)
	OUTPUT_DIR ?= $(OUTPUT_DIR_CLIMSST_1DEG)
	OUTPUT_DIR_AMIP ?= $(OUTPUT_DIR_AMIP_1DEG)
	NAME ?= $(NAME_CLIMSST_1DEG)
	NAME_AMIP ?= $(NAME_AMIP_1DEG)
else ifeq ($(RESOLUTION),4deg)
	ROOT_INTERMEDIATE ?= $(ROOT_INTERMEDIATE_CLIMSST_4DEG)
	ROOT_INTERMEDIATE_AMIP ?= $(ROOT_INTERMEDIATE_AMIP_4DEG)
	ROOT_BASE ?= $(ROOT_BASE_CLIMSST_4DEG)
	ROOT_AMIP ?= $(ROOT_BASE_AMIP_4DEG)
	OUTPUT_DIR ?= $(OUTPUT_DIR_CLIMSST_4DEG)
	OUTPUT_DIR_AMIP ?= $(OUTPUT_DIR_AMIP_4DEG)
	NAME ?= $(NAME_CLIMSST_4DEG)
	NAME_AMIP ?= $(NAME_AMIP_4DEG)
endif

# the netCDF generation step is done locally usually, so user sets input and output directories
NC_INPUT ?=
NC_OUTPUT ?=

SHIELD_RES ?= c96

.PHONY: shield_AMIP_dataset
shield_AMIP_dataset:
	./compute_dataset.sh \
	--config configs/shield-amip-ensemble-$(SHIELD_RES)-$(RESOLUTION)-8layer.yaml

.PHONY: shield_AMIP_monthly_netcdfs
shield_AMIP_monthly_netcdfs:
	./convert_to_monthly_netcdf_fv3gfs.sh \
	--input-url $(NC_INPUT) \
	--n-ic 2 \
	--output-dir $(NC_OUTPUT) \
	--start-date 1940-01-01 \
	--end-date 2021-12-31

.PHONY: shield_c24_4deg_climSST_dataset
shield_c24_4deg_climSST_dataset:
	./compute_dataset.sh --config configs/shield-c24-ensemble-4deg-8layer.yaml

.PHONY: shield_c24_4deg_climSST_monthly_netcdfs
shield_c24_4deg_climSST_monthly_netcdfs:
	./convert_to_monthly_netcdf_fv3gfs.sh \
	--input-url gs://vcm-ml-intermediate/2024-04-05-vertically-resolved-4deg-c24-shield-fme-ensemble-dataset \
	--n-ic 21 \
	--output-dir $(NC_OUTPUT) \
	--start-date 2021-01-01 \
	--end-date 2030-12-31

shield_c24_4deg_climSST_stats_beaker_dataset:
	./compute_stats.sh --config configs/shield-c24-ensemble-4deg-8layer.yaml

.PHONY: shield_c96_4deg_climSST_dataset
shield_c96_4deg_climSST_dataset:
	./compute_dataset.sh --config configs/shield-c96-4deg-8layer.yaml

.PHONY: shield_c96_4deg_climSST_monthly_netcdfs
shield_c96_4deg_climSST_monthly_netcdfs:
	./convert_to_monthly_netcdf_fv3gfs.sh \
	--input-url gs://vcm-ml-intermediate/2024-04-02-vertically-resolved-4deg-c96-shield-fme-dataset \
	--n-ic 1 \
	--output-dir $(NC_OUTPUT) \
	--start-date 2035-01-01 \
	--end-date 2060-12-31

shield_c96_4deg_climSST_stats_beaker_dataset:
	./compute_stats.sh --config configs/shield-c96-4deg-8layer.yaml

.PHONY: fv3gfs_1deg_climSST_dataset
fv3gfs_1deg_climSST_dataset:
	$(MAKE) fv3gfs_climSST_dataset RESOLUTION=1deg LAYERS=8layer

.PHONY: fv3gfs_climSST_dataset
fv3gfs_climSST_dataset:
	./compute_dataset.sh --config configs/fv3gfs-ensemble-$(RESOLUTION)-$(LAYERS).yaml

.PHONY: fv3gfs_1deg_climSST_monthly_netcdfs
fv3gfs_1deg_climSST_monthly_netcdfs:
	$(MAKE) fv3gfs_climSST_monthly_netcdfs RESOLUTION=1deg

.PHONY: fv3gfs_4deg_climSST_monthly_netcdfs
fv3gfs_4deg_climSST_monthly_netcdfs:
	$(MAKE) fv3gfs_climSST_monthly_netcdfs RESOLUTION=4deg

.PHONY: enable_coldline_check
enable_coldline_check:
	@if [ "$(ENABLE_COLDLINE)" != "true" ]; then \
		echo "Processing target is deprecated due to coldlined data" \
		echo "to run, ENABLE_COLDLINE must be set to true. Exiting."; \
		exit 1; \
	fi

.PHONY: fv3gfs_climSST_monthly_netcdfs
fv3gfs_climSST_monthly_netcdfs:
	./convert_to_monthly_netcdf_fv3gfs.sh \
	--input-url $(ROOT_INTERMEDIATE) \
	--n-ic 11 \
	--output-dir $(OUTPUT_DIR) \
	--start-date 2021-01-01 \
	--end-date 2030-12-31

.PHONY: fv3gfs_1deg_climSST_stats_beaker_dataset
fv3gfs_1deg_climSST_stats_beaker_dataset:
	$(MAKE) fv3gfs_climSST_stats_beaker_dataset RESOLUTION=1deg

fv3gfs_climSST_stats_beaker_dataset: enable_coldline_check
	./compute_stats.sh --config configs/fv3gfs-ensemble-$(RESOLUTION)-$(LAYERS).yaml

# This took around ~10 hours to complete.  The roundtrip_filter adds significant computational time.
# If we plan to do this regularly, paralellizing the script across time using xpartition to
# launch jobs for different time chunks would probably be a good idea.
fv3gfs_climSST_c48_baseline_dataset: enable_coldline_check
	./compute_dataset.sh --config configs/fv3gfs-c48-ensemble-1deg-8layer.yaml

.PHONY: fv3gfs_climSST_c48_baseline_monthly_netcdfs
fv3gfs_climSST_c48_baseline_monthly_netcdfs:
	python convert_to_monthly_netcdf.py \
	--prepend-nans \
	gs://vcm-ml-intermediate/2023-09-01-vertically-resolved-1deg-fme-c48-baseline-dataset/ic_0011.zarr \
	/net/nfs/climate/data/2023-09-12-vertically-resolved-1deg-fme-c48-baseline-dataset-truncated-065/ic_0011 \
	--start-date 2021-01-01 \
	--end-date 2030-12-31

.PHONY: fv3gfs_1deg_AMIP_dataset
fv3gfs_1deg_AMIP_dataset:
	$(MAKE) fv3gfs_AMIP_dataset RESOLUTION=1deg LAYERS=8layer

fv3gfs_AMIP_dataset: enable_coldline_check
	./compute_dataset.sh --config configs/fv3gfs-amip-ensemble-$(RESOLUTION)-$(LAYERS).yaml

.PHONY: fv3gfs_1deg_AMIP_monthly_netcdfs
fv3gfs_1deg_AMIP_monthly_netcdfs:
	$(MAKE) fv3gfs_AMIP_monthly_netcdfs RESOLUTION=1deg

.PHONY: fv3gfs_AMIP_monthly_netcdfs
fv3gfs_AMIP_monthly_netcdfs:
	./convert_to_monthly_netcdf_fv3gfs.sh \
	--input-url $(ROOT_INTERMEDIATE_AMIP) \
	--n-ic 4 \
	--output-dir $(OUTPUT_DIR_AMIP) \
	--start-date 1990-01-01 \
	--end-date 2019-12-31

.PHONY: fv3gfs_1deg_AMIP_stats_beaker_dataset
fv3gfs_1deg_AMIP_stats_beaker_dataset:
	$(MAKE) fv3gfs_AMIP_stats_beaker_dataset RESOLUTION=1deg

fv3gfs_AMIP_stats_beaker_dataset: enable_coldline_check
	./compute_stats.sh --config configs/fv3gfs-amip-ensemble-$(RESOLUTION)-$(LAYERS).yaml

# TODO: Add AMIP baseline C48 dataset processing when available

.PHONY: shield_som_spin_up_c96_dataset
shield_som_c96_spin_up_dataset:
	./compute_dataset.sh --config configs/shield-som-spin-up-c96-1deg-$(LAYERS).yaml

.PHONY: shield_som_ensemble_c96_dataset
shield_som_ensemble_c96_dataset:
	./compute_dataset.sh --config configs/shield-som-ensemble-c96-$(RESOLUTION)-$(LAYERS).yaml

.PHONY: shield_som_abrupt_co2_increase_c96_dataset
shield_som_abrupt_co2_increase_c96_dataset:
	./compute_dataset.sh --config configs/shield-som-abrupt-co2-increase-c96-$(RESOLUTION)-$(LAYERS).yaml

.PHONY: shield_som_increasing_co2_c96_dataset
shield_som_c96_increasing_co2_dataset:
	./compute_dataset.sh --config configs/shield-som-increasing-co2-c96-$(RESOLUTION)-$(LAYERS).yaml

.PHONY: shield_som_c96_radiation_multi_call_dataset
shield_som_c96_radiation_multi_call_dataset:
	./compute_dataset.sh --config configs/shield-som-radiation-multi-call-c96-1deg-$(LAYERS).yaml

.PHONY: shield_som_c24_dataset
shield_som_c24_dataset:
	./compute_dataset.sh --config configs/shield-som-c24-4deg-$(LAYERS).yaml

.PHONY: shield_som_c24_tuned_cdmbgwd_dataset
shield_som_c24_tuned_cdmbgwd_dataset:
	./compute_dataset.sh --config configs/shield-som-c24-tuned-cdmbgwd-4deg-$(LAYERS).yaml

# In total (not including full_field stats), this took 7 hours 15 minutes to
# complete using a single Perlmutter CPU node.

.PHONY: e3smv2_1deg_climSST_dataset
e3smv2_1deg_climSST_dataset:
	sbatch -J 2024-07-10-e3smv2-1deg-testing generate_datasets_e3smv2.sh \
	--input-dir /global/cfs/cdirs/m4492/rebassoo/e3sm_test/post/atm/180x360_gaussian/ts \
	--config configs/e3sm-1deg-8layer.yaml \
	--zarr /global/cfs/cdirs/m4492/fme-preprocess/zarr/2024-07-10-e3smv2-1deg-testing.zarr \
	--output-dir /global/cfs/cdirs/m4492/fme-preprocess/2024-07-10-e3smv2-1deg-testing

.PHONY: era5_1deg_stats_beaker_dataset
era5_1deg_stats_beaker_dataset:
	./compute_stats.sh --config configs/era5-1deg-8layer-1940-2022.yaml

.PHONY: era5_1deg_16layer_stats_beaker_dataset
era5_1deg_16layer_stats_beaker_dataset:
	./compute_stats.sh --config configs/era5-1deg-16layer-1940-2022.yaml

.PHONY: compute_cm4_trial_run_atmosphere_dataset
compute_cm4_trial_run_atmosphere_dataset:
	python -u compute_dataset.py \
	--config configs/pre-industrial-CM4-1deg-8layer-trial-run.yaml \
	--run-directory gs://vcm-ml-raw-flexible-retention/2024-08-10-pre-industrial-CM4-simulation/regridded-zarrs/gaussian_grid_180_by_360/trial-run \
	--output-store gs://vcm-ml-intermediate/2024-09-20-cm4-1deg-8layer-trial-run.zarr

.PHONY: compute_cm4_trial_run_atmosphere_stats
compute_cm4_trial_run_atmosphere_stats:
	python -u get_stats.py configs/pre-industrial-CM4-1deg-8layer-trial-run.yaml 0

.PHONY: cm4_picontrol_atmosphere_1deg_8layer_200yr
cm4_picontrol_atmosphere_1deg_8layer_200yr:
	./compute_dataset.sh --config configs/CM4-piControl-atmosphere-1deg-8layer-200yr.yaml

.PHONY: cm4_picontrol_atmosphere_1deg_8layer_200yr_monthly_netcdfs
cm4_picontrol_atmosphere_1deg_8layer_200yr_monthly_netcdfs:
	./convert_to_monthly_netcdf_fv3gfs.sh \
	--input-url $(NC_INPUT) \
	--n-ic 1 \
	--output-dir $(NC_OUTPUT) \
	--start-date "0151-01-01T06:00:00" \
	--end-date "0351-01-01T00:00:00"

create_ocean_data_proc_env:
	conda create -y -n ocean_data_proc python=3.10 pip yq
	conda install -y -n ocean_data_proc -c conda-forge xesmf
	conda run --no-capture-output -n ocean_data_proc python -m pip install uv
	conda run --no-capture-output -n ocean_data_proc uv pip install -r requirements-ocean.txt
	conda run --no-capture-output -n ocean_data_proc bash ./install_ocean_emulators.sh

.PHONY: cm4_trial_run_ocean_dataset_and_stats
cm4_trial_run_ocean_dataset_and_stats:
	./compute_ocean_dataset.sh --config configs/pre-industrial-CM4-1deg-ocean.yaml

.PHONY: shield_som_c96_abrupt_4xCO2_ensemble_dataset
shield_som_c96_abrupt4xCO2_ensemble_dataset:
	./compute_dataset.sh --config configs/shield-som-abrupt4xCO2-ensemble-c96-1deg-$(LAYERS).yaml

.PHONY: shield_som_c96_idealized_0p1_heating_ensemble_dataset
shield_som_c96_idealized_0p1_heating_ensemble_dataset:
	./compute_dataset.sh --config configs/shield-som-idealized-0p1-heating-ensemble-c96-1deg-$(LAYERS).yaml

.PHONY: shield_som_c96_increasing_co2_with_3d_heating_rate_diagnostics_dataset
shield_som_c96_increasing_co2_with_3d_heating_rate_diagnostics_dataset:
	./compute_dataset.sh --config configs/shield-som-increasing-co2-with-3d-heating-rate-diagnostics-1deg-$(LAYERS).yaml

.PHONY: shield_amip_c96_idealized_0p1_heating_ensemble_dataset
shield_amip_c96_idealized_0p1_heating_ensemble_dataset:
	./compute_dataset.sh --config configs/shield-amip-idealized-0p1-heating-ensemble-c96-1deg-$(LAYERS).yaml

.PHONY: build_atmosphere_processing_image
build_atmosphere_processing_image:
	docker build -f atmosphere.Dockerfile -t $(ATMOSPHERE_PROCESSING_IMAGE):$(ATMOSPHERE_PROCESSING_IMAGE_VERSION) .

.PHONY: push_atmosphere_processing_image
push_atmosphere_processing_image:
	docker push $(ATMOSPHERE_PROCESSING_IMAGE):$(ATMOSPHERE_PROCESSING_IMAGE_VERSION)
