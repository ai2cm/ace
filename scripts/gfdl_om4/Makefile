LOCAL_ENVIRONMENT ?= gfdl-om4-ingestion

VERSION ?= 2026-07-09-gfdl-om4-xrbeam-pipeline
IMAGE_NAME ?= us-central1-docker.pkg.dev/vcm-ml/full-model/gfdl-om4-ingest-dataflow:$(VERSION)

# Durable pipeline inputs (static grid files, regridding weights, face
# masks) live under one dated, immutable GCS prefix. Nothing consumed by a
# production run may live under vcm-ml-scratch (auto-deletion risk).
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

# Face-mask artifacts are per-simulation. Each scans that source's first
# year (the masks are static, and the run-time staleness assertion guards
# the rest of the series) with expected flagged surface-face counts from an
# independent census of the source's remap-born zero faces; generation
# fails if they don't reconcile. Both simulations census identically.
generate_face_masks: generate_face_masks_picontrol generate_face_masks_1pctco2

generate_face_masks_picontrol:
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.face_masks \
		--config configs/om4-picontrol-1deg.yaml \
		--stream snapshot_ocean \
		--output-url $(FACE_MASKS_URL_ROOT)/om4-picontrol-2026-06-19 \
		--start-time 0151-01-06 \
		--end-time 0152-01-01 \
		--expected-surface-count-u 16960 \
		--expected-surface-count-v 16694

generate_face_masks_1pctco2:
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.face_masks \
		--config configs/om4-1pctco2-1deg.yaml \
		--stream snapshot_ocean \
		--output-url $(FACE_MASKS_URL_ROOT)/om4-1pctco2-2026-06-19 \
		--start-time 0001-01-06 \
		--end-time 0002-01-01 \
		--expected-surface-count-u 16960 \
		--expected-surface-count-v 16694

CONFIG ?= configs/om4-picontrol-1deg.yaml

SMOKE_NUM_TIMESTEPS ?= 6
SMOKE_OUTPUT_ROOT ?= gs://vcm-ml-scratch/jamesd/gfdl-om4-test
SMOKE_OUTPUT_PATH = $(SMOKE_OUTPUT_ROOT)/smoke-$(notdir $(basename $(CONFIG))).zarr

# A few timesteps of $(CONFIG) through the DirectRunner into a throwaway
# store: the wetmask conform step must be a no-op (zero conformed cells)
# and the output must open with the config's expected variable set.
smoke_test:
	-gsutil -m -q rm -r $(SMOKE_OUTPUT_PATH) 2> /dev/null
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.run \
		--config $(CONFIG) \
		--num-timesteps $(SMOKE_NUM_TIMESTEPS) \
		--output-path $(SMOKE_OUTPUT_PATH) \
		--max-conformed-cells 0 \
		--runner DirectRunner \
		--job_server_timeout 3600
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.check_output \
		--config $(CONFIG) \
		--output-path $(SMOKE_OUTPUT_PATH) \
		--expected-timesteps $(SMOKE_NUM_TIMESTEPS)

smoke_test_picontrol_1deg:
	$(MAKE) smoke_test CONFIG=configs/om4-picontrol-1deg.yaml

smoke_test_picontrol_4deg:
	$(MAKE) smoke_test CONFIG=configs/om4-picontrol-4deg.yaml

smoke_test_1pctco2_1deg:
	$(MAKE) smoke_test CONFIG=configs/om4-1pctco2-1deg.yaml

smoke_test_1pctco2_4deg:
	$(MAKE) smoke_test CONFIG=configs/om4-1pctco2-4deg.yaml

# A repeat run against an already-written output store must refuse to
# initialize into it (runs a smoke test first if its store is absent).
smoke_test_repeat_fails:
	gsutil -q ls $(SMOKE_OUTPUT_PATH)/zarr.json > /dev/null 2>&1 || $(MAKE) smoke_test
	! conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.run \
		--config $(CONFIG) \
		--num-timesteps $(SMOKE_NUM_TIMESTEPS) \
		--output-path $(SMOKE_OUTPUT_PATH) \
		--runner DirectRunner \
		--job_server_timeout 3600

# The configs derive each simulation's wetmask from its own snapshot store;
# downstream training/analysis assumes the output stores share one mask.
check_wetmask_equivalence:
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.check_wetmask_equivalence \
		configs/om4-picontrol-1deg.yaml \
		configs/om4-1pctco2-1deg.yaml

smoke_tests: check_wetmask_equivalence smoke_test_picontrol_1deg \
	smoke_test_picontrol_4deg smoke_test_1pctco2_1deg \
	smoke_test_1pctco2_4deg smoke_test_repeat_fails

build_dataflow:
	docker build --platform=linux/amd64 -t $(IMAGE_NAME) .
	docker run --rm --entrypoint python $(IMAGE_NAME) -c "import pipeline.run, xesmf"

push_dataflow: build_dataflow
	docker push $(IMAGE_NAME)

enter:
	docker run --rm -v $$(pwd):/gfdl_om4 -w /gfdl_om4 --entrypoint "/bin/bash" -it $(IMAGE_NAME)

# Production sources have a static footprint, so the wetmask conform step
# must be a no-op in production launches too; a source that needs actual
# conforming is a deliberate opt-in (drop the flag by invoking
# run-dataflow.sh directly).
dataflow:
	SDK_CONTAINER_IMAGE=$(IMAGE_NAME) \
		conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) \
		./run-dataflow.sh DataflowRunner $(CONFIG) --max-conformed-cells 0

dataflow_picontrol_1deg:
	$(MAKE) dataflow CONFIG=configs/om4-picontrol-1deg.yaml

dataflow_picontrol_4deg:
	$(MAKE) dataflow CONFIG=configs/om4-picontrol-4deg.yaml

dataflow_1pctco2_1deg:
	$(MAKE) dataflow CONFIG=configs/om4-1pctco2-1deg.yaml

dataflow_1pctco2_4deg:
	$(MAKE) dataflow CONFIG=configs/om4-1pctco2-4deg.yaml

.PHONY: create_environment generate_weights generate_weights_one_degree \
	generate_weights_four_degree generate_face_masks \
	generate_face_masks_picontrol generate_face_masks_1pctco2 smoke_test \
	smoke_test_picontrol_1deg smoke_test_picontrol_4deg \
	smoke_test_1pctco2_1deg smoke_test_1pctco2_4deg \
	smoke_test_repeat_fails check_wetmask_equivalence smoke_tests \
	build_dataflow push_dataflow enter dataflow dataflow_picontrol_1deg \
	dataflow_picontrol_4deg dataflow_1pctco2_1deg dataflow_1pctco2_4deg
