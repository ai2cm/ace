LOCAL_ENVIRONMENT ?= gfdl-om4-ingestion

VERSION ?= 2026-07-09-gfdl-om4-xrbeam-pipeline
IMAGE_NAME ?= us-central1-docker.pkg.dev/vcm-ml/full-model/gfdl-om4-ingest-dataflow:$(VERSION)

HGRID_URL ?= gs://vcm-ml-scratch/jamesd/2024-11-11-static-data/ocean_hgrid.nc
WEIGHTS_VERSION ?= v1
WEIGHTS_URL_ROOT ?= gs://vcm-ml-scratch/jamesd/gfdl-om4-regridding-weights/$(WEIGHTS_VERSION)

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

CONFIG ?= configs/om4-picontrol-1deg.yaml

FACE_MASKS_VERSION ?= v1
FACE_MASKS_URL_ROOT ?= gs://vcm-ml-scratch/jamesd/gfdl-om4-face-masks/$(FACE_MASKS_VERSION)

# Expected flagged surface-face counts are an independent census of the
# source's remap-born zero faces; generation fails if they don't reconcile.
generate_face_masks:
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.face_masks \
		--config $(CONFIG) \
		--stream snapshot_ocean \
		--output-url $(FACE_MASKS_URL_ROOT)/om4-picontrol-2026-06-24 \
		--expected-surface-count-u 16960 \
		--expected-surface-count-v 16694
TEST_NUM_TIMESTEPS ?= 6
TEST_OUTPUT_PATH ?= gs://vcm-ml-scratch/jamesd/gfdl-om4-test/om4-picontrol-1deg-subset.zarr

test_run:
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.run \
		--config $(CONFIG) \
		--num-timesteps $(TEST_NUM_TIMESTEPS) \
		--output-path $(TEST_OUTPUT_PATH) \
		--runner DirectRunner \
		--job_server_timeout 3600

build_dataflow:
	docker build --platform=linux/amd64 -t $(IMAGE_NAME) .
	docker run --rm --entrypoint python $(IMAGE_NAME) -c "import pipeline.run, xesmf"

push_dataflow: build_dataflow
	docker push $(IMAGE_NAME)

enter:
	docker run --rm -v $$(pwd):/gfdl_om4 -w /gfdl_om4 --entrypoint "/bin/bash" -it $(IMAGE_NAME)

dataflow:
	SDK_CONTAINER_IMAGE=$(IMAGE_NAME) \
		conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) \
		./run-dataflow.sh DataflowRunner $(CONFIG)

.PHONY: create_environment generate_weights generate_weights_one_degree \
	generate_weights_four_degree generate_face_masks test_run build_dataflow \
	push_dataflow enter dataflow
