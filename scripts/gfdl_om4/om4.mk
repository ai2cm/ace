# OM4 pipeline (pipeline/om4/): the OM4/CM4 ocean stores, one per config.
# Included by the Makefile; run from scripts/gfdl_om4/ as `make <target>`.

# Face-mask artifacts are per-simulation. Each scans that source's first
# year (the masks are static, and the run-time staleness assertion guards
# the rest of the series) with expected flagged surface-face counts from an
# independent census of the source's remap-born zero faces; generation
# fails if they don't reconcile. Both simulations census identically.
generate_face_masks: generate_face_masks_picontrol generate_face_masks_1pctco2

generate_face_masks_picontrol:
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.om4.face_masks \
		--config configs/om4-picontrol-1deg-5daily.yaml \
		--stream snapshot_ocean \
		--output-url $(FACE_MASKS_URL_ROOT)/om4-picontrol-2026-06-19 \
		--start-time 0151-01-06 \
		--end-time 0152-01-01 \
		--expected-surface-count-u 16960 \
		--expected-surface-count-v 16694

generate_face_masks_1pctco2:
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.om4.face_masks \
		--config configs/om4-1pctco2-1deg-5daily.yaml \
		--stream snapshot_ocean \
		--output-url $(FACE_MASKS_URL_ROOT)/om4-1pctco2-2026-06-19 \
		--start-time 0001-01-06 \
		--end-time 0002-01-01 \
		--expected-surface-count-u 16960 \
		--expected-surface-count-v 16694

CONFIG ?= configs/om4-picontrol-1deg-5daily.yaml

SMOKE_OUTPUT_ROOT ?= gs://vcm-ml-scratch/jamesd/gfdl-om4-test
SMOKE_OUTPUT_PATH = $(SMOKE_OUTPUT_ROOT)/smoke-$(notdir $(basename $(CONFIG))).zarr

# A few timesteps of $(CONFIG) through the DirectRunner into a throwaway
# store: every chunk's footprint must equal the wetmask (the run fails
# otherwise) and the output must open with the config's expected variable
# set.
smoke_test:
	-gsutil -m -q rm -r $(SMOKE_OUTPUT_PATH) 2> /dev/null
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.om4.run \
		--config $(CONFIG) \
		--num-timesteps $(SMOKE_NUM_TIMESTEPS) \
		--output-path $(SMOKE_OUTPUT_PATH) \
		--runner DirectRunner \
		--job_server_timeout 3600
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.om4.check_output \
		--config $(CONFIG) \
		--output-path $(SMOKE_OUTPUT_PATH) \
		--expected-timesteps $(SMOKE_NUM_TIMESTEPS)

smoke_test_picontrol_1deg:
	$(MAKE) smoke_test CONFIG=configs/om4-picontrol-1deg-5daily.yaml

smoke_test_picontrol_4deg:
	$(MAKE) smoke_test CONFIG=configs/om4-picontrol-4deg-5daily.yaml

smoke_test_1pctco2_1deg:
	$(MAKE) smoke_test CONFIG=configs/om4-1pctco2-1deg-5daily.yaml

smoke_test_1pctco2_4deg:
	$(MAKE) smoke_test CONFIG=configs/om4-1pctco2-4deg-5daily.yaml

smoke_test_picontrol_1deg_1daily:
	$(MAKE) smoke_test CONFIG=configs/cm4-piControl-1deg-1daily.yaml

smoke_test_picontrol_4deg_1daily:
	$(MAKE) smoke_test CONFIG=configs/cm4-piControl-4deg-1daily.yaml

smoke_test_1pctco2_1deg_1daily:
	$(MAKE) smoke_test CONFIG=configs/cm4-1pctCO2-1deg-1daily.yaml

smoke_test_1pctco2_4deg_1daily:
	$(MAKE) smoke_test CONFIG=configs/cm4-1pctCO2-4deg-1daily.yaml

# A repeat run against an already-written output store must refuse to
# initialize into it (runs a smoke test first if its store is absent).
smoke_test_repeat_fails:
	gsutil -q ls $(SMOKE_OUTPUT_PATH)/zarr.json > /dev/null 2>&1 || $(MAKE) smoke_test
	! conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.om4.run \
		--config $(CONFIG) \
		--num-timesteps $(SMOKE_NUM_TIMESTEPS) \
		--output-path $(SMOKE_OUTPUT_PATH) \
		--runner DirectRunner \
		--job_server_timeout 3600

# The configs derive each simulation's wetmask from its own snapshot store;
# downstream training/analysis assumes the output stores share one mask.
check_wetmask_equivalence:
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.om4.check_wetmask_equivalence \
		configs/om4-picontrol-1deg-5daily.yaml \
		configs/om4-1pctco2-1deg-5daily.yaml

smoke_tests: check_wetmask_equivalence smoke_test_picontrol_1deg \
	smoke_test_picontrol_4deg smoke_test_1pctco2_1deg \
	smoke_test_1pctco2_4deg smoke_test_repeat_fails

# The 1-daily configs carry the 5-daily wetmask sources verbatim, so this
# guards that pinning rather than the 1-daily streams themselves.
check_wetmask_equivalence_1daily:
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.om4.check_wetmask_equivalence \
		configs/cm4-piControl-1deg-1daily.yaml \
		configs/cm4-1pctCO2-1deg-1daily.yaml

smoke_tests_1daily: check_wetmask_equivalence_1daily \
	smoke_test_picontrol_1deg_1daily smoke_test_picontrol_4deg_1daily \
	smoke_test_1pctco2_1deg_1daily smoke_test_1pctco2_4deg_1daily
	$(MAKE) smoke_test_repeat_fails CONFIG=configs/cm4-piControl-1deg-1daily.yaml

# Sea-ice budget companion stores: 5-daily block means of the SIS2 fluxes
# and ice/snow transport, on the 5-daily stores' time axis. Their wetmask
# sources are the 5-daily configs' verbatim.
check_wetmask_equivalence_budget:
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.om4.check_wetmask_equivalence \
		configs/om4-picontrol-1deg-5daily-budget.yaml \
		configs/om4-1pctco2-1deg-5daily-budget.yaml

smoke_test_picontrol_1deg_budget:
	$(MAKE) smoke_test CONFIG=configs/om4-picontrol-1deg-5daily-budget.yaml

smoke_test_1pctco2_1deg_budget:
	$(MAKE) smoke_test CONFIG=configs/om4-1pctco2-1deg-5daily-budget.yaml

smoke_tests_budget: check_wetmask_equivalence_budget \
	smoke_test_picontrol_1deg_budget smoke_test_1pctco2_1deg_budget

test:
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pytest --confcutdir=. pipeline

dataflow: check_dataflow_image
	SDK_CONTAINER_IMAGE=$(IMAGE_NAME) \
		conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) \
		./run-dataflow.sh om4 DataflowRunner $(CONFIG)

dataflow_picontrol_1deg:
	$(MAKE) dataflow CONFIG=configs/om4-picontrol-1deg-5daily.yaml

dataflow_picontrol_4deg:
	$(MAKE) dataflow CONFIG=configs/om4-picontrol-4deg-5daily.yaml

dataflow_1pctco2_1deg:
	$(MAKE) dataflow CONFIG=configs/om4-1pctco2-1deg-5daily.yaml

dataflow_1pctco2_4deg:
	$(MAKE) dataflow CONFIG=configs/om4-1pctco2-4deg-5daily.yaml

dataflow_picontrol_1deg_1daily:
	$(MAKE) dataflow CONFIG=configs/cm4-piControl-1deg-1daily.yaml

dataflow_picontrol_4deg_1daily:
	$(MAKE) dataflow CONFIG=configs/cm4-piControl-4deg-1daily.yaml

dataflow_1pctco2_1deg_1daily:
	$(MAKE) dataflow CONFIG=configs/cm4-1pctCO2-1deg-1daily.yaml

dataflow_1pctco2_4deg_1daily:
	$(MAKE) dataflow CONFIG=configs/cm4-1pctCO2-4deg-1daily.yaml

dataflow_picontrol_1deg_budget:
	$(MAKE) dataflow CONFIG=configs/om4-picontrol-1deg-5daily-budget.yaml

dataflow_1pctco2_1deg_budget:
	$(MAKE) dataflow CONFIG=configs/om4-1pctco2-1deg-5daily-budget.yaml

.PHONY: generate_face_masks \
	generate_face_masks_picontrol generate_face_masks_1pctco2 smoke_test \
	smoke_test_picontrol_1deg smoke_test_picontrol_4deg \
	smoke_test_1pctco2_1deg smoke_test_1pctco2_4deg \
	smoke_test_repeat_fails check_wetmask_equivalence smoke_tests \
	smoke_test_picontrol_1deg_1daily smoke_test_picontrol_4deg_1daily \
	smoke_test_1pctco2_1deg_1daily smoke_test_1pctco2_4deg_1daily \
	check_wetmask_equivalence_1daily smoke_tests_1daily \
	check_wetmask_equivalence_budget smoke_test_picontrol_1deg_budget \
	smoke_test_1pctco2_1deg_budget smoke_tests_budget test \
	dataflow_picontrol_1deg_budget dataflow_1pctco2_1deg_budget \
	dataflow dataflow_picontrol_1deg \
	dataflow_picontrol_4deg dataflow_1pctco2_1deg dataflow_1pctco2_4deg \
	dataflow_picontrol_1deg_1daily dataflow_picontrol_4deg_1daily \
	dataflow_1pctco2_1deg_1daily dataflow_1pctco2_4deg_1daily
