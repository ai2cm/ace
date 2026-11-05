LOCAL_ENVIRONMENT ?= gfdl-om4-ingestion

# One worker image for both pipelines (om4.mk, randco2.mk).
VERSION ?= 2026-09-29-gfdl-om4-xrbeam-pipeline
IMAGE_NAME ?= us-central1-docker.pkg.dev/vcm-ml/full-model/gfdl-om4-ingest-dataflow:$(VERSION)

# Durable pipeline inputs (static grid files, regridding weights, face masks)
# live under one dated, immutable GCS prefix.
INPUTS_URL_ROOT ?= gs://vcm-ml-intermediate/2026-07-15-gfdl-om4-pipeline-inputs
HGRID_URL ?= $(INPUTS_URL_ROOT)/statics/ocean_hgrid.nc
WEIGHTS_URL_ROOT ?= $(INPUTS_URL_ROOT)/regridding-weights
FACE_MASKS_URL_ROOT ?= $(INPUTS_URL_ROOT)/face-masks

create_environment:
	conda env create -f environment.yaml
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) pip install -r dataflow-requirements.txt

generate_weights: generate_weights_one_degree generate_weights_four_degree

generate_weights_one_degree:
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.weights \
		--hgrid-url $(HGRID_URL) \
		--target-grid F90 \
		--output-url $(WEIGHTS_URL_ROOT)/om4-tripolar-0.25deg-to-F90

generate_weights_four_degree:
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.weights \
		--hgrid-url $(HGRID_URL) \
		--target-grid F22.5 \
		--output-url $(WEIGHTS_URL_ROOT)/om4-tripolar-0.25deg-to-F22.5

SMOKE_NUM_TIMESTEPS ?= 6

build_dataflow:
	docker build --platform=linux/amd64 -t $(IMAGE_NAME) .
	docker run --rm --entrypoint python $(IMAGE_NAME) -c "import pipeline.om4.run, pipeline.randco2.run, xesmf"

push_dataflow: build_dataflow
	docker push $(IMAGE_NAME)

enter:
	docker run --rm -v $$(pwd):/gfdl_om4 -w /gfdl_om4 --entrypoint "/bin/bash" -it $(IMAGE_NAME)

# The worker image must exist in Artifact Registry before a launch. A missing
# tag otherwise surfaces only as ImagePullBackOff on the workers: each job
# boots a VM, never starts its SDK harness, and so never completes a bundle —
# and throughput-based autoscaling holds the job at one idle worker rather
# than failing it. dataflow and randco2_dataflow depend on it.
IMAGE_REPO = $(firstword $(subst :, ,$(IMAGE_NAME)))
IMAGE_TAG = $(lastword $(subst :, ,$(IMAGE_NAME)))
check_dataflow_image:
	@tags=$$(gcloud artifacts docker tags list $(IMAGE_REPO) --format="value(tag)") || { \
		echo "ERROR: could not list tags of $(IMAGE_REPO) (see gcloud error above)."; \
		exit 1; }; \
	echo "$$tags" | grep -qx "$(IMAGE_TAG)" || { \
		echo "ERROR: $(IMAGE_NAME) is absent from Artifact Registry."; \
		echo "Run 'make push_dataflow' before launching."; \
		exit 1; }

.PHONY: create_environment generate_weights generate_weights_one_degree \
	generate_weights_four_degree build_dataflow push_dataflow enter \
	check_dataflow_image

include om4.mk randco2.mk
