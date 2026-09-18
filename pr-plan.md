# Argo workflow for coupled dataset processing, with Makefile rules for the coupled configs

`scripts/data_process/create_coupled_datasets.py` runs only under `conda run -n
create_coupled_datasets` on a workstation today. This adds the argo path it has
been missing — a submitting shell entry point, a workflow manifest, and a
container image built from `requirements-coupled.txt` — and routes every coupled
config in `scripts/data_process/configs/` through the `Makefile` in both the
local and the argo flavour.

The three-stage chain inside `create_coupled_datasets.py`
(`compute_coupled_sea_ice -> compute_coupled_ocean -> compute_coupled_atmosphere`,
then stats) stays inside one process and maps to **one** workflow step, not
three: `CoupledDatasetsConfig.write_datasets_and_stats` already skips a stage
whose zarr exists, so a rerun resumes without any workflow-level dependency
graph, and the stages share in-memory intermediates that a step boundary would
force back through GCS.

---

## `scripts/data_process/coupled.Dockerfile` (new)

```dockerfile
FROM python:3.12-slim

# google-cloud-cli + gcloud config set project vcm-ml
#   — identical block to atmosphere.Dockerfile

COPY requirements-coupled.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
```

> **Why a second image.** The workflow cannot reuse
> `atmosphere-processing`: `requirements-coupled.txt` names `bokeh`,
> `h5netcdf`, `h5py` and `pandas`, none of which is in
> `requirements-atmosphere.txt`, and `create_coupled_datasets.py` imports
> `writer_utils` / `merge_stats`, which need them. `requirements-atmosphere.txt`
> is not extended: that would rebuild the atmosphere image for the coupled
> path's benefit.

**Build prerequisite.** The image must exist in the registry before any of the
four submit commands will run. `make build_coupled_processing_image
push_coupled_processing_image` is the maintainer's step and is not run by this
PR.

## `scripts/data_process/create_coupled_datasets_argo_workflow.yaml` (new)

```yaml
metadata:
  generateName: create-coupled-datasets-
spec:
  entrypoint: create-coupled-datasets
  volumes: [gcp-key-secret]                     # same secret mount as compute_dataset_argo_workflow.yaml
  arguments:
    parameters:
    # one per local module create_coupled_datasets.py imports
    - name: create_coupled_datasets_script
    - name: coupled_dataset_utils_script
    - name: get_stats_script
    - name: merge_stats_script
    - name: combine_stats_script
    - name: writer_utils_script
    - name: fs_utils_script
    - name: config
    - name: image                               # default: the COUPLED_PROCESSING_IMAGE tag
    - name: debug
      value: "false"
    - name: subsample
      value: "false"
  templates:
    - name: create-coupled-datasets
      tolerations: [dedicated=highmem-sim-pool:NoSchedule]
      container:
        image: "{{workflow.parameters.image}}"
        resources:                              # matches compute-fme-dataset-individual;
          limits:   {cpu: "30000m", memory: "230Gi"}   # output_writer.n_dask_workers is 32
          requests: {cpu: "30000m", memory: "230Gi"}
        command: ["bash", "-c", "-e"]
        args:
          - |
            # heredoc each *_script parameter to its import name, config to config.yaml
            python -u create_coupled_datasets.py --yaml config.yaml \
              $( [ "{{workflow.parameters.debug}}" = true ] && echo --debug ) \
              $( [ "{{workflow.parameters.subsample}}" = true ] && echo --subsample )
        env: [GOOGLE_APPLICATION_CREDENTIALS, CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE]
        volumeMounts: [/secret/gcp-credentials]
```

> **Heredoc file names are load-bearing.** `create_coupled_datasets.py` imports
> its siblings by module name (`from coupled_dataset_utils import ...`,
> `from get_stats import ...`). Each `*_script` parameter must be written to
> `<module>.py`, not to `script.py` as the atmosphere workflow does for its
> single entry point.

## `scripts/data_process/create_coupled_datasets.sh` (new)

```bash
#!/bin/bash
# Submit create_coupled_datasets.py to argo. Modelled on compute_dataset.sh.
set -e

# --config <path>   required
# --debug           pass --debug through to the script
# --subsample       pass --subsample through to the script
# --image <tag>     override the default container image
# --dry-run         print the argo submit command; submit nothing

output=$(argo submit create_coupled_datasets_argo_workflow.yaml \
    -p create_coupled_datasets_script="$(< create_coupled_datasets.py)" \
    -p coupled_dataset_utils_script="$(< coupled_dataset_utils.py)" \
    -p get_stats_script="$(< get_stats.py)" \
    -p merge_stats_script="$(< merge_stats.py)" \
    -p combine_stats_script="$(< combine_stats.py)" \
    -p writer_utils_script="$(< writer_utils.py)" \
    -p fs_utils_script="$(< fs_utils.py)" \
    -p config="$(< ${CONFIG})" \
    -p debug=${DEBUG} \
    -p subsample=${SUBSAMPLE} \
    -p image=${IMAGE})

echo "Argo job submitted: $(echo "$output" | grep 'Name:' | awk '{print $2}')"
```

Unlike `compute_dataset.sh` there is no `yq` read of the config: the coupled
config has no `runs`/`data_output_directory` fan-out keys, and every output path
is derived inside `CreateCoupledDatasetsConfig` from `version` / `family_name` /
`output_directory`.

## `scripts/data_process/Makefile` (modified)

```makefile
COUPLED_PROCESSING_IMAGE = us-central1-docker.pkg.dev/vcm-ml/full-model/coupled-processing   # NEW
COUPLED_PROCESSING_IMAGE_VERSION = v2026.09.0                                                # NEW

# NEW — one config path per coupled dataset, shared by the local and argo targets
CONFIG_CM4_PICONTROL_COUPLED         = configs/CM4-piControl-coupled-$(RESOLUTION)-200yr.yaml
CONFIG_CM4_1PCTCO2_COUPLED           = configs/CM4-1pctCO2-coupled-$(RESOLUTION)-140yr.yaml
CONFIG_CM4_PICONTROL_COUPLED_1DAILY  = configs/CM4-piControl-coupled-$(RESOLUTION)-1daily-200yr.yaml
CONFIG_CM4_1PCTCO2_COUPLED_1DAILY    = configs/CM4-1pctCO2-coupled-$(RESOLUTION)-1daily-140yr.yaml
CONFIG_E3SMV3_PICONTROL_COUPLED      = configs/E3SMv3-piControl-100yr-coupled.yaml
CONFIG_CM4_LIKE_AM4_RANDCO2_COUPLED  = configs/CM4-like-AM4-random-CO2-ensemble-coupled.yaml

.PHONY: build_coupled_processing_image push_coupled_processing_image           # NEW
build_coupled_processing_image:
	docker build -f coupled.Dockerfile -t $(COUPLED_PROCESSING_IMAGE):$(COUPLED_PROCESSING_IMAGE_VERSION) .
push_coupled_processing_image:
	docker push $(COUPLED_PROCESSING_IMAGE):$(COUPLED_PROCESSING_IMAGE_VERSION)

cm4_piControl_coupled:                       # CHANGED — config path via the variable
	conda run --no-capture-output -n create_coupled_datasets python ./create_coupled_datasets.py --yaml $(CONFIG_CM4_PICONTROL_COUPLED)

cm4_piControl_coupled_argo:                  # NEW — one *_argo sibling per target above
	./create_coupled_datasets.sh --config $(CONFIG_CM4_PICONTROL_COUPLED) --image $(COUPLED_PROCESSING_IMAGE):$(COUPLED_PROCESSING_IMAGE_VERSION)

# …cm4_1pctCO2_coupled{,_argo}, cm4_piControl_coupled_1daily{,_argo},
#   cm4_1pctCO2_coupled_1daily{,_argo}, e3smv3_piControl_100yr_coupled{,_argo},
#   cm4_like_am4_random_co2_ensemble_coupled{,_argo} follow the same two-line pair.

.PHONY: cm4_like_am4_random_co2_ensemble_coupled   # CHANGED — the bare `.PHONY:` above it names nothing
```

Existing local targets keep their names, their recipe shape and their behaviour;
only the literal config path moves into a variable. Nothing on the
`compute_dataset.sh` side is touched.

## `scripts/data_process/README.md` (modified)

One paragraph beside the existing argo paragraph: the coupled targets run either
locally in the `create_coupled_datasets` conda env or on argo via the `_argo`
sibling, and the argo flavour needs `coupled-processing` pushed first.

---

## Tests

## `scripts/data_process/test_data_process_workflows.py` (new)

```python
# Static checks over the shell + yaml pair. No argo, no container, no network.

def test_coupled_workflow_parameters_are_declared():
    # GOAL: every {{workflow.parameters.X}} referenced in
    # create_coupled_datasets_argo_workflow.yaml is declared under
    # spec.arguments.parameters.
    # PARAMETERIZE over both workflow yamls (compute_dataset_argo_workflow.yaml too).
    ...

def test_coupled_submit_script_passes_every_local_import():
    # GOAL: the real failure mode — a module missing inside the container.
    # Parse create_coupled_datasets.py's local (sibling-module) imports with ast,
    # and assert each appears as a `-p <name>_script="$(< <module>.py)"` line in
    # create_coupled_datasets.sh and as a declared workflow parameter.
    ...

def test_coupled_workflow_writes_each_script_to_its_module_name():
    # GOAL: each *_script parameter is heredoc'd to <module>.py in the container
    # args, so the sibling imports resolve.
    ...
```

## `scripts/data_process/test_config.py` (modified)

```python
def test_every_makefile_coupled_config_exists():  # NEW
    # GOAL: every `--yaml configs/...` path in the Makefile's coupled targets
    # resolves on disk, expanded for RESOLUTION in {1deg, 4deg}.
    ...
```

---

## The four submit commands

Printed by the build stage and run by the maintainer on `james-vm`; nothing in
this PR submits them.

```bash
cd scripts/data_process

make cm4_piControl_coupled_1daily_argo RESOLUTION=1deg
make cm4_1pctCO2_coupled_1daily_argo   RESOLUTION=1deg
make cm4_piControl_coupled_1daily_argo RESOLUTION=4deg
make cm4_1pctCO2_coupled_1daily_argo   RESOLUTION=4deg
```

equivalently, without make:

```bash
./create_coupled_datasets.sh --config configs/CM4-piControl-coupled-1deg-1daily-200yr.yaml
./create_coupled_datasets.sh --config configs/CM4-1pctCO2-coupled-1deg-1daily-140yr.yaml
./create_coupled_datasets.sh --config configs/CM4-piControl-coupled-4deg-1daily-200yr.yaml
./create_coupled_datasets.sh --config configs/CM4-1pctCO2-coupled-4deg-1daily-140yr.yaml
```

---

## Open Questions

- Image tag: a fresh `coupled-processing` repository, or a `coupled` tag prefix
  inside the existing `atmosphere-processing` repository?
- `230Gi` / `30000m` copied from `compute-fme-dataset-individual`, or sized
  down — the coupled run's concurrency is `output_writer.n_dask_workers`, 32 in
  the CM4 1-daily configs?
- `_argo` suffix on the new targets, or `_local` on the existing conda ones with
  the bare name becoming the argo path?
