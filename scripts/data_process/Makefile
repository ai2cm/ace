# The dependencies of the scripts below (where not containerized) are installed in the "fv3net" conda environment
# which can be installed using fv3net's Makefile. See
# https://github.com/ai2cm/fv3net/blob/8ed295cf0b8ca49e24ae5d6dd00f57e8b30169ac/Makefile#L310

RESOLUTION ?= "1deg"
LAYERS ?= "8layer"

ROOT_BASE_CLIMSST_1DEG = gs://vcm-ml-raw-flexible-retention/2023-07-08-C96-FME-reference-ensemble/regridded-zarrs/gaussian_grid_180_by_360
ROOT_BASE_CLIMSST_4DEG = gs://vcm-ml-raw-flexible-retention/2023-07-08-C96-FME-reference-ensemble/regridded-zarrs/gaussian_grid_45_by_90
ROOT_BASE_AMIP_1DEG = gs://vcm-ml-raw-flexible-retention/2023-10-20-C96-FME-AMIP-ensemble-reference/regridded-zarrs/gaussian_grid_180_by_360
ROOT_BASE_AMIP_4DEG = gs://vcm-ml-raw-flexible-retention/2023-10-20-C96-FME-AMIP-ensemble-reference/regridded-zarrs/gaussian_grid_45_by_90
ROOT_INTERMEDIATE_CLIMSST_1DEG = gs://vcm-ml-intermediate/2023-08-09-vertically-resolved-1deg-fme-ensemble-dataset
ROOT_INTERMEDIATE_CLIMSST_4DEG = gs://vcm-ml-intermediate/2023-08-09-vertically-resolved-4deg-fme-ensemble-dataset
ROOT_INTERMEDIATE_AMIP_1DEG = gs://vcm-ml-intermediate/2023-10-27-vertically-resolved-1deg-fme-amip-ensemble-dataset
ROOT_INTERMEDIATE_AMIP_4DEG = gs://vcm-ml-intermediate/2023-10-27-vertically-resolved-4deg-fme-amip-ensemble-dataset
OUTPUT_DIR_CLIMSST_1DEG = /net/nfs/climate/data/2023-08-11-vertically-resolved-1deg-fme-ensemble-dataset
OUTPUT_DIR_CLIMSST_4DEG = /net/nfs/climate/data/2023-08-11-vertically-resolved-4deg-fme-ensemble-dataset
OUTPUT_DIR_AMIP_1DEG = /pscratch/sd/b/bhenn/data/2023-11-01-vertically-resolved-1deg-fme-amip-ensemble-dataset
OUTPUT_DIR_AMIP_4DEG = /pscratch/sd/b/bhenn/data/2023-11-01-vertically-resolved-4deg-fme-amip-ensemble-dataset
NAME_CLIMSST_1DEG = "fv3gfs-ensemble"
NAME_CLIMSST_4DEG = "fv3gfs-ensemble-4deg"
NAME_AMIP_1DEG = "fv3gfs-AMIP-ensemble"
NAME_AMIP_4DEG = "fv3gfs-AMIP-ensemble-4deg"

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
else
	$(error RESOLUTION must be 1deg or 4deg)
endif

.PHONY: fv3gfs_1deg_climSST_dataset
fv3gfs_1deg_climSST_dataset:
	$(MAKE) fv3gfs_climSST_dataset RESOLUTION=1deg LAYERS=8layer

.PHONY: fv3gfs_climSST_dataset
fv3gfs_climSST_dataset:
	./compute_dataset.sh \
	--config configs/fv3gfs-ensemble-$(RESOLUTION)-$(LAYERS).yaml

.PHONY: fv3gfs_1deg_climSST_monthly_netcdfs
fv3gfs_1deg_climSST_monthly_netcdfs:
	$(MAKE) fv3gfs_climSST_monthly_netcdfs RESOLUTION=1deg

.PHONY: fv3gfs_climSST_monthly_netcdfs
fv3gfs_climSST_monthly_netcdfs:
	./convert_to_monthly_netcdf_fv3gfs.sh \
	--input-url $(ROOT_INTERMEDIATE) \
	--n-ic 11 \
	--output-dir $(OUTPUT_DIR) \
	--start-date 2021-01-01 \
	--end-date 2030-12-31

.PHONY: fv3gfs_1deg_climSST_residual_stats_beaker_dataset
fv3gfs_1deg_climSST_residual_stats_beaker_dataset:
	$(MAKE) fv3gfs_climSST_residual_stats_beaker_dataset RESOLUTION=1deg

.PHONY: fv3gfs_climSST_residual_stats_beaker_dataset
fv3gfs_climSST_residual_stats_beaker_dataset:
	./generate_beaker_stats_dataset_fv3gfs.sh \
	--input-url $(ROOT_INTERMEDIATE)/ic_0001.zarr \
	--start-date 2021-01-01 \
	--end-date 2030-12-31 \
	--name "${NAME}-ic0001-stats-residual-scaling-all-years-v2" \
	--desc "Coefficients for normalization for 8-level vertically resolved dataset with annually-repeating SSTs, using residual scaling. Includes surface height."

.PHONY: fv3gfs_1deg_climSST_full_field_stats_beaker_dataset
fv3gfs_1deg_climSST_full_field_stats_beaker_dataset:
	$(MAKE) fv3gfs_climSST_full_field_stats_beaker_dataset RESOLUTION=1deg

.PHONY: fv3gfs_climSST_full_field_stats_beaker_dataset
fv3gfs_climSST_full_field_stats_beaker_dataset:
	./generate_beaker_stats_dataset_fv3gfs.sh \
	--input-url $(ROOT_INTERMEDIATE)/ic_0001.zarr \
	--start-date 2021-01-01 \
	--end-date 2030-12-31 \
	--name "${NAME}-ic0001-stats-full-field-scaling-all-years-v2" \
	--desc "Coefficients for normalization for 8-level vertically resolved dataset with annually-repeating SSTs, using full field scaling. Includes surface height." \
	--script-flags "--full-field"

# This took around ~10 hours to complete.  The roundtrip_filter adds significant computational time.
# If we plan to do this regularly, paralellizing the script across time using xpartition to
# launch jobs for different time chunks would probably be a good idea.
.PHONY: fv3gfs_climSST_c48_baseline_dataset
fv3gfs_climSST_c48_baseline_dataset:
	./compute_dataset.sh \
	--config configs/fv3gfs-c48-ensemble-1deg-8layer.yaml

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

.PHONY: fv3gfs_AMIP_dataset
fv3gfs_AMIP_dataset:
	./compute_dataset.sh \
	--config configs/fv3gfs-amip-ensemble-$(RESOLUTION)-$(LAYERS).yaml

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

.PHONY: fv3gfs_AMIP_stats_beaker_dataset
fv3gfs_AMIP_stats_beaker_dataset:
	./generate_beaker_stats_dataset_fv3gfs.sh \
	--input-url $(ROOT_INTERMEDIATE_AMIP)/ic_0001.zarr \
	--start-date 1990-01-01 \
	--end-date 2019-12-31 \
	--name "${NAME_AMIP}-ensemble-ic0001-stats-residual-scaling-all-years-v2" \
	--desc "Coefficients for normalization for 8-level vertically resolved dataset with AMIP SSTs, using residual scaling."

# TODO: Add AMIP baseline C48 dataset processing when available

# In total (not including full_field stats), this took 7 hours 15 minutes to
# complete using a single Perlmutter CPU node.

.PHONY: e3smv2_1deg_climSST_dataset
e3smv2_1deg_climSST_dataset:
	sbatch -J 2023-11-22-generate_datasets_e3smv2 generate_datasets_e3smv2.sh \
	--input-dir /global/cfs/cdirs/e3sm/golaz/E3SM/fme/20230614.v2.LR.F2010/post/atm/180x360_gaussian/ts \
	--time-invariant-input-dir /global/cfs/cdirs/m4331/jpduncan/e3smv2/time_invariant \
	--zarr $SCRATCH/ai2/zarr/2023-11-22-e3smv2-vertically-resolved-1deg-fme-dataset.zarr \
	--output-dir $SCRATCH/ai2/zarr/2023-11-22-e3smv2-vertically-resolved-1deg-fme-dataset.zarr
