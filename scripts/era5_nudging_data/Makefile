IMAGE=us-central1-docker.pkg.dev/vcm-ml/full-model/era5-nudging-data
IMAGE_VERSION=v2026.08.0  # Use vYYYY.MM.MICRO (MICRO is zero-indexed)

TRIAL_START_DATETIME=2016-08-01T00:00:00
TRIAL_END_DATETIME=2016-08-02T00:00:00
TRIAL_NUDGING_DESTINATION=gs://vcm-ml-scratch/spencerc/2026-08-18-ERA5-nudging-data/nudging-files
TRIAL_SURFACE_BOUNDARY_CONDITIONS_DESTINATION=gs://vcm-ml-scratch/spencerc/2026-08-18-ERA5-nudging-data/surface-boundary-conditions.nc
TRIAL_N_WORKERS=2

DYAMOND_START_DATETIME=2016-08-01T00:00:00
DYAMOND_END_DATETIME=2016-09-10T06:00:00
DYAMOND_NUDGING_DESTINATION=gs://vcm-ml-intermediate/2026-08-18-ERA5-nudging-data-2016-08-01-to-2016-09-10/nudging-files
DYAMOND_SURFACE_BOUNDARY_CONDITIONS_DESTINATION=gs://vcm-ml-intermediate/2026-08-18-ERA5-nudging-data-2016-08-01-to-2016-09-10/surface-boundary-conditions.nc
DYAMOND_N_WORKERS=2

.PHONY: build_image
build_image:
	docker build -f Dockerfile -t $(IMAGE):$(IMAGE_VERSION) .

.PHONY: push_image
push_image:
	docker push $(IMAGE):$(IMAGE_VERSION)

.PHONY: submit_trial_nudging
submit_trial_nudging:
	./submit_nudging.sh \
	--start-datetime $(TRIAL_START_DATETIME) \
	--end-datetime $(TRIAL_END_DATETIME) \
	--destination $(TRIAL_NUDGING_DESTINATION) \
	--n-workers $(TRIAL_N_WORKERS)

.PHONY: submit_dyamond_nudging
submit_dyamond_nudging:
	./submit_nudging.sh \
	--start-datetime $(DYAMOND_START_DATETIME) \
	--end-datetime $(DYAMOND_END_DATETIME) \
	--destination $(DYAMOND_NUDGING_DESTINATION) \
	--n-workers $(DYAMOND_N_WORKERS)

.PHONY: submit_trial_surface_boundary_conditions
submit_trial_surface_boundary_conditions:
	./submit_surface_boundary_conditions.sh \
	--start-datetime $(TRIAL_START_DATETIME) \
	--end-datetime $(TRIAL_END_DATETIME) \
	--destination $(TRIAL_SURFACE_BOUNDARY_CONDITIONS_DESTINATION)

.PHONY: submit_dyamond_surface_boundary_conditions
submit_dyamond_surface_boundary_conditions:
	./submit_surface_boundary_conditions.sh \
	--start-datetime $(DYAMOND_START_DATETIME) \
	--end-datetime $(DYAMOND_END_DATETIME) \
	--destination $(DYAMOND_SURFACE_BOUNDARY_CONDITIONS_DESTINATION)

.PHONY: submit_trial
submit_trial: submit_trial_nudging submit_trial_surface_boundary_conditions

.PHONY: submit_dyamond
submit_dyamond: submit_dyamond_nudging submit_dyamond_surface_boundary_conditions
