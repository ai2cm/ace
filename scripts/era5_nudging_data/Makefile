IMAGE=us-central1-docker.pkg.dev/vcm-ml/full-model/era5-nudging-data
IMAGE_VERSION=v2026.07.0  # Use vYYYY.MM.MICRO (MICRO is zero-indexed)

TRIAL_START_DATETIME=2020-01-01T00:00:00
TRIAL_END_DATETIME=2020-01-02T00:00:00
TRIAL_DESTINATION=gs://vcm-ml-scratch/spencerc/2026-08-06-ERA5-nudging-data
TRIAL_N_WORKERS=2

DYAMOND_START_DATETIME=2016-08-01T00:00:00
DYAMOND_END_DATETIME=2016-09-10T06:00:00
DYAMOND_DESTINATION=gs://vcm-ml-intermediate/2026-08-11-ERA5-nudging-data-2016-08-01-to-2016-09-10
DYAMOND_N_WORKERS=2

.PHONY: build_image
build_image:
	docker build -f Dockerfile -t $(IMAGE):$(IMAGE_VERSION) .

.PHONY: push_image
push_image:
	docker push $(IMAGE):$(IMAGE_VERSION)

.PHONY: submit_trial
submit_trial:
	./submit.sh \
	--start-datetime $(TRIAL_START_DATETIME) \
	--end-datetime $(TRIAL_END_DATETIME) \
	--destination $(TRIAL_DESTINATION) \
	--n-workers $(TRIAL_N_WORKERS)

.PHONY: submit_dyamond
submit_dyamond:
	./submit.sh \
	--start-datetime $(DYAMOND_START_DATETIME) \
	--end-datetime $(DYAMOND_END_DATETIME) \
	--destination $(DYAMOND_DESTINATION) \
	--n-workers $(DYAMOND_N_WORKERS)
