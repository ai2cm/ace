# randco2 pipeline (pipeline/randco2/): the CM4-like-AM4 random-CO2
# sea-surface stores, one per ensemble member. Included by the Makefile; run
# from scripts/gfdl_om4/ as `make randco2_<target>`. Targets and variables
# carry a randco2_/RANDCO2_ prefix so that none collides with om4.mk's
# (CONFIG, SMOKE_OUTPUT_ROOT, smoke_test_%, dataflow_%).

# One config for the whole ensemble; the member name fills the {member}
# placeholder in its source and output URLs.
RANDCO2_CONFIG ?= configs/cm4-like-am4-randco2-sea-surface-1deg.yaml

# The nine ensemble members, one output store each.
RANDCO2_MEMBERS = \
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
# artifact under the dated, immutable inputs prefix named in
# $(RANDCO2_CONFIG); these sources sit on the same native tracer grid, so
# there is no weight-generation step here.

RANDCO2_SMOKE_OUTPUT_ROOT ?= gs://vcm-ml-scratch/jamesd/cm4-randco2-sea-surface-test
RANDCO2_SMOKE_OUTPUT_PATH = $(RANDCO2_SMOKE_OUTPUT_ROOT)/smoke-$(MEMBER).zarr

# A few timesteps of one member through the DirectRunner into a throwaway
# store: every chunk's valid-data footprint must equal the wetmask, the
# full-cell and ocean-relative sea-ice fractions must reconcile, and the
# output must open with the config's expected variable set.
randco2_smoke_test:
	$(if $(MEMBER),,$(error set MEMBER to one of: $(RANDCO2_MEMBERS)))
	-gsutil -m -q rm -r $(RANDCO2_SMOKE_OUTPUT_PATH) 2> /dev/null
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.randco2.run \
		--config $(RANDCO2_CONFIG) \
		--member $(MEMBER) \
		--num-timesteps $(SMOKE_NUM_TIMESTEPS) \
		--output-path $(RANDCO2_SMOKE_OUTPUT_PATH) \
		--runner DirectRunner \
		--job_server_timeout 3600
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.randco2.check_output \
		--config $(RANDCO2_CONFIG) \
		--member $(MEMBER) \
		--output-path $(RANDCO2_SMOKE_OUTPUT_PATH) \
		--expected-timesteps $(SMOKE_NUM_TIMESTEPS)

# Per-member smoke test, e.g. make randco2_smoke_test_random-CO2-1xCO2-ic_0001.
# The explicit randco2_smoke_test_* targets below take precedence over this
# pattern.
randco2_smoke_test_%:
	$(MAKE) randco2_smoke_test MEMBER=$*

# A repeat run against an already-written output store must refuse to
# initialize into it (runs a smoke test first).
RANDCO2_REPEAT_MEMBER ?= $(firstword $(RANDCO2_MEMBERS))
randco2_smoke_test_repeat_fails:
	$(MAKE) randco2_smoke_test_$(RANDCO2_REPEAT_MEMBER)
	! conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.randco2.run \
		--config $(RANDCO2_CONFIG) \
		--member $(RANDCO2_REPEAT_MEMBER) \
		--num-timesteps $(SMOKE_NUM_TIMESTEPS) \
		--output-path $(RANDCO2_SMOKE_OUTPUT_ROOT)/smoke-$(RANDCO2_REPEAT_MEMBER).zarr \
		--runner DirectRunner \
		--job_server_timeout 3600

# The source stores are chunked wider than one timestep along time, so a
# source chunk can straddle an output shard boundary. This runs enough
# timesteps to cross one within a few source chunks, by shrinking the shard
# rather than lengthening the run: with a 10-timestep source chunk and a
# 15-timestep shard, the third chunk starts inside the second shard.
RANDCO2_BOUNDARY_MEMBER ?= $(firstword $(RANDCO2_MEMBERS))
RANDCO2_BOUNDARY_NUM_TIMESTEPS ?= 40
RANDCO2_BOUNDARY_TIME_SHARD_SIZE ?= 15
RANDCO2_BOUNDARY_OUTPUT_PATH = $(RANDCO2_SMOKE_OUTPUT_ROOT)/boundary-$(RANDCO2_BOUNDARY_MEMBER).zarr
randco2_smoke_test_shard_boundary:
	-gsutil -m -q rm -r $(RANDCO2_BOUNDARY_OUTPUT_PATH) 2> /dev/null
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.randco2.run \
		--config $(RANDCO2_CONFIG) \
		--member $(RANDCO2_BOUNDARY_MEMBER) \
		--num-timesteps $(RANDCO2_BOUNDARY_NUM_TIMESTEPS) \
		--time-shard-size $(RANDCO2_BOUNDARY_TIME_SHARD_SIZE) \
		--output-path $(RANDCO2_BOUNDARY_OUTPUT_PATH) \
		--runner DirectRunner \
		--job_server_timeout 3600
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.randco2.check_output \
		--config $(RANDCO2_CONFIG) \
		--member $(RANDCO2_BOUNDARY_MEMBER) \
		--output-path $(RANDCO2_BOUNDARY_OUTPUT_PATH) \
		--expected-timesteps $(RANDCO2_BOUNDARY_NUM_TIMESTEPS)

# Each member derives its wetmask from its own source store; downstream
# training and analysis assume the nine output stores share one mask.
randco2_check_wetmask_equivalence:
	conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) python -m pipeline.randco2.check_wetmask_equivalence \
		--config $(RANDCO2_CONFIG) \
		--members $(RANDCO2_MEMBERS)

randco2_smoke_tests: randco2_check_wetmask_equivalence \
	$(addprefix randco2_smoke_test_,$(RANDCO2_MEMBERS)) \
	randco2_smoke_test_repeat_fails randco2_smoke_test_shard_boundary

randco2_dataflow: check_dataflow_image
	$(if $(MEMBER),,$(error set MEMBER to one of: $(RANDCO2_MEMBERS)))
	SDK_CONTAINER_IMAGE=$(IMAGE_NAME) \
		conda run --no-capture-output -n $(LOCAL_ENVIRONMENT) \
		./run-dataflow.sh randco2 DataflowRunner $(RANDCO2_CONFIG) --member $(MEMBER)

# Per-member production launch, e.g. make randco2_dataflow_random-CO2-1xCO2-ic_0001.
# The explicit randco2_dataflow_all below takes precedence over this pattern.
randco2_dataflow_%:
	$(MAKE) randco2_dataflow MEMBER=$*

# All nine production jobs. Each is an independent Dataflow launch writing
# its own subdirectory of the dated parent prefix in $(RANDCO2_CONFIG).
randco2_dataflow_all: $(addprefix randco2_dataflow_,$(RANDCO2_MEMBERS))

# The 4-degree twin of the stores above: the same targets re-run against
# the F22.5 config, with its own smoke-test scratch root so the 1- and
# 4-degree smoke stores never share a path.
RANDCO2_4DEG_CONFIG = configs/cm4-like-am4-randco2-sea-surface-4deg.yaml
RANDCO2_4DEG_SMOKE_OUTPUT_ROOT ?= $(RANDCO2_SMOKE_OUTPUT_ROOT)-4deg
RANDCO2_4DEG = RANDCO2_CONFIG=$(RANDCO2_4DEG_CONFIG) \
	RANDCO2_SMOKE_OUTPUT_ROOT=$(RANDCO2_4DEG_SMOKE_OUTPUT_ROOT)

randco2_4deg_smoke_tests:
	$(MAKE) randco2_smoke_tests $(RANDCO2_4DEG)

randco2_4deg_dataflow_all:
	$(MAKE) randco2_dataflow_all $(RANDCO2_4DEG)

# The per-member targets are deliberately absent: make skips implicit-rule
# search for a phony target, which would leave the pattern rules above with
# nothing to match.
.PHONY: randco2_smoke_test randco2_smoke_test_repeat_fails \
	randco2_smoke_test_shard_boundary randco2_check_wetmask_equivalence \
	randco2_smoke_tests randco2_dataflow randco2_dataflow_all \
	randco2_4deg_smoke_tests randco2_4deg_dataflow_all
