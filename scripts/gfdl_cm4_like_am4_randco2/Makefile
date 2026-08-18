LOCAL_ENVIRONMENT ?= cm4-randco2-ingestion

VERSION ?= 2026-08-17-cm4-like-am4-randco2-sea-surface
IMAGE_NAME ?= us-central1-docker.pkg.dev/vcm-ml/full-model/cm4-randco2-sea-surface-dataflow:$(VERSION)

# One config for the whole ensemble; the member name fills the {member}
# placeholder in its source and output URLs.
CONFIG ?= configs/cm4-like-am4-randco2-sea-surface-1deg.yaml

# The nine ensemble members, one output store each.
MEMBERS = \
	random-CO2-1xCO2-ic_0001 \
	random-CO2-1xCO2-ic_0002 \
	random-CO2-1xCO2-ic_0003 \
	random-CO2-2xCO2-ic_0001 \
	random-CO2-2xCO2-ic_0002 \
	random-CO2-2xCO2-ic_0003 \
	random-CO2-4xCO2-ic_0001 \
	random-CO2-4xCO2-ic_0002 \
	random-CO2-4xCO2-ic_0003

# The regridding weights this pipeline reads are the published OM4 tripolar
# artifact under the dated, immutable inputs prefix named in $(CONFIG);
# these sources sit on the same native tracer grid, so there is no
# weight-generation step here.

create_environment:
	conda env create -f environment.yaml
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) pip install -r dataflow-requirements.txt

SMOKE_NUM_TIMESTEPS ?= 6
SMOKE_OUTPUT_ROOT ?= gs://vcm-ml-scratch/jamesd/cm4-randco2-sea-surface-test
SMOKE_OUTPUT_PATH = $(SMOKE_OUTPUT_ROOT)/smoke-$(MEMBER).zarr

# A few timesteps of one member through the DirectRunner into a throwaway
# store: every chunk's valid-data footprint must equal the wetmask, the
# full-cell and ocean-relative sea-ice fractions must reconcile, and the
# output must open with the config's expected variable set.
smoke_test:
	$(if $(MEMBER),,$(error set MEMBER to one of: $(MEMBERS)))
	-gsutil -m -q rm -r $(SMOKE_OUTPUT_PATH) 2> /dev/null
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.run \
		--config $(CONFIG) \
		--member $(MEMBER) \
		--num-timesteps $(SMOKE_NUM_TIMESTEPS) \
		--output-path $(SMOKE_OUTPUT_PATH) \
		--runner DirectRunner \
		--job_server_timeout 3600
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.check_output \
		--config $(CONFIG) \
		--member $(MEMBER) \
		--output-path $(SMOKE_OUTPUT_PATH) \
		--expected-timesteps $(SMOKE_NUM_TIMESTEPS)

# Per-member smoke test, e.g. make smoke_test_random-CO2-1xCO2-ic_0001. The
# explicit targets below take precedence over this pattern.
smoke_test_%:
	$(MAKE) smoke_test MEMBER=$*

# A repeat run against an already-written output store must refuse to
# initialize into it (runs a smoke test first if its store is absent).
REPEAT_MEMBER ?= $(firstword $(MEMBERS))
smoke_test_repeat_fails:
	$(MAKE) smoke_test_$(REPEAT_MEMBER)
	! conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.run \
		--config $(CONFIG) \
		--member $(REPEAT_MEMBER) \
		--num-timesteps $(SMOKE_NUM_TIMESTEPS) \
		--output-path $(SMOKE_OUTPUT_ROOT)/smoke-$(REPEAT_MEMBER).zarr \
		--runner DirectRunner \
		--job_server_timeout 3600

# Each member derives its wetmask from its own source store; downstream
# training and analysis assume the nine output stores share one mask.
check_wetmask_equivalence:
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.check_wetmask_equivalence \
		--config $(CONFIG) \
		--members $(MEMBERS)

smoke_tests: check_wetmask_equivalence $(addprefix smoke_test_,$(MEMBERS)) \
	smoke_test_repeat_fails

build_dataflow:
	docker build --platform=linux/amd64 -t $(IMAGE_NAME) .
	docker run --rm --entrypoint python $(IMAGE_NAME) -c "import pipeline.run, xesmf"

push_dataflow: build_dataflow
	docker push $(IMAGE_NAME)

enter:
	docker run --rm -v $$(pwd):/cm4_randco2 -w /cm4_randco2 --entrypoint "/bin/bash" -it $(IMAGE_NAME)

dataflow:
	$(if $(MEMBER),,$(error set MEMBER to one of: $(MEMBERS)))
	SDK_CONTAINER_IMAGE=$(IMAGE_NAME) \
		conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) \
		./run-dataflow.sh DataflowRunner $(CONFIG) --member $(MEMBER)

# Per-member production launch, e.g. make dataflow_random-CO2-1xCO2-ic_0001.
# The explicit dataflow_all below takes precedence over this pattern.
dataflow_%:
	$(MAKE) dataflow MEMBER=$*

# All nine production jobs. Each is an independent Dataflow launch writing
# its own subdirectory of the dated parent prefix in $(CONFIG).
dataflow_all: $(addprefix dataflow_,$(MEMBERS))

# The per-member targets are deliberately absent: make skips implicit-rule
# search for a phony target, which would leave the pattern rules above with
# nothing to match.
.PHONY: create_environment smoke_test smoke_test_repeat_fails \
	check_wetmask_equivalence smoke_tests build_dataflow push_dataflow \
	enter dataflow dataflow_all
