# Argo workflow for coupled dataset processing, with Makefile rules for the coupled configs

`scripts/data_process/create_coupled_datasets.py` runs only under `conda run -n
create_coupled_datasets` on a workstation today. This adds the argo path it has
been missing — a submitting shell entry point, a workflow manifest, and a
container image built from `requirements-coupled.txt` — and routes the coupled
configs in `scripts/data_process/configs/` through the `Makefile`: every one in
the local conda flavour, and the GCS-backed ones in an argo flavour too.

The three-stage chain inside `create_coupled_datasets.py`
(`compute_coupled_sea_ice -> compute_coupled_ocean -> compute_coupled_atmosphere`,
then stats) stays inside one process and maps to **one** workflow step, not
three: `CreateCoupledDatasetsConfig.write_coupled_datasets` — `main`'s entry
point — delegates to `CoupledDatasetsConfig.write_datasets_and_stats`, which
skips a stage whose zarr already exists, and `_merge_stats` / `combine_stats`
skip likewise on an existing output directory. A rerun therefore resumes
without any workflow-level dependency graph, and the stages share in-memory
intermediates that a step boundary would force back through GCS. Resume is by
resubmitting the same workflow; there is no in-workflow retry.

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
four submit commands will run: `make build_coupled_processing_image
push_coupled_processing_image`.

## `scripts/data_process/create_coupled_datasets_argo_workflow.yaml` (new)

```yaml
metadata:
  generateName: create-coupled-datasets-
spec:
  entrypoint: create-coupled-datasets
  volumes: [gcp-key-secret]                     # same secret mount as compute_dataset_argo_workflow.yaml
  arguments:
    parameters:
    # one per module in create_coupled_datasets.py's transitive sibling-import
    # closure — see "The import closure is nine modules" below
    - name: create_coupled_datasets_script
    - name: coupled_dataset_utils_script
    - name: create_window_avg_dataset_script
    - name: time_utils_script
    - name: get_stats_script
    - name: merge_stats_script
    - name: combine_stats_script
    - name: writer_utils_script
    - name: fs_utils_script
    - name: config
    - name: image
      value: "us-central1-docker.pkg.dev/vcm-ml/full-model/coupled-processing:v2026.09.0"
    - name: debug
      value: "false"
    - name: subsample
      value: "false"
  templates:
    - name: create-coupled-datasets
      tolerations: [dedicated=highmem-sim-pool:NoSchedule]
      container:
        image: "{{workflow.parameters.image}}"
        resources:                              # as the heavy compute-fme-dataset-individual
          limits:   {cpu: "30000m", memory: "230Gi"}   # steps; 30 CPU against
          requests: {cpu: "30000m", memory: "230Gi"}   # output_writer.n_dask_workers: 32
        command: ["bash", "-c", "-e"]
        args:
          - |
            # heredoc each *_script parameter to its import name, config to
            # config.yaml, each with a QUOTED delimiter: cat << 'EOF' > <module>.py
            # An unquoted delimiter expands $, ${} and backticks inside the
            # python source, silently corrupting it in-container.
            flags=()
            [ "{{workflow.parameters.debug}}" = "true" ] && flags+=(--debug)
            [ "{{workflow.parameters.subsample}}" = "true" ] && flags+=(--subsample)
            python -u create_coupled_datasets.py --yaml config.yaml "${flags[@]}"
        env: [GOOGLE_APPLICATION_CREDENTIALS, CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE]
        volumeMounts: [/secret/gcp-credentials]
```

> **The import closure is nine modules.** `create_coupled_datasets.py` imports
> its siblings by module name, and those siblings import further siblings:
> `coupled_dataset_utils` imports `create_window_avg_dataset`, which imports
> `time_utils`. Passing only the entry point's own imports leaves the container
> to die on `ModuleNotFoundError`. The closure is `create_coupled_datasets`,
> `coupled_dataset_utils`, `create_window_avg_dataset`, `time_utils`,
> `get_stats`, `merge_stats`, `combine_stats`, `writer_utils`, `fs_utils`.

> **Heredoc file names are load-bearing.** Each `*_script` parameter must be
> written to `<module>.py`, not to `script.py` as the atmosphere workflow does
> for its single entry point, or the sibling imports will not resolve.

## `scripts/data_process/create_coupled_datasets.sh` (new)

```bash
#!/bin/bash
# Submit create_coupled_datasets.py to argo. Modelled on compute_dataset.sh.
set -e

CONFIG=
DEBUG=false
SUBSAMPLE=false
IMAGE=
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)    CONFIG="$2"; shift 2 ;;      # required
        --image)     IMAGE="$2";  shift 2 ;;      # override the workflow default
        --debug)     DEBUG=true;      shift ;;
        --subsample) SUBSAMPLE=true;  shift ;;
        --dry-run)   DRY_RUN=true;    shift ;;    # print the submit command, submit nothing
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done
[ -n "$CONFIG" ] || { echo "--config is required" >&2; exit 1; }

args=(create_coupled_datasets_argo_workflow.yaml
    -p create_coupled_datasets_script="$(< create_coupled_datasets.py)"
    -p coupled_dataset_utils_script="$(< coupled_dataset_utils.py)"
    -p create_window_avg_dataset_script="$(< create_window_avg_dataset.py)"
    -p time_utils_script="$(< time_utils.py)"
    -p get_stats_script="$(< get_stats.py)"
    -p merge_stats_script="$(< merge_stats.py)"
    -p combine_stats_script="$(< combine_stats.py)"
    -p writer_utils_script="$(< writer_utils.py)"
    -p fs_utils_script="$(< fs_utils.py)"
    -p config="$(< "$CONFIG")"
    -p debug="$DEBUG"
    -p subsample="$SUBSAMPLE")
# --image is omitted entirely when unset, so the workflow default applies
[ -n "$IMAGE" ] && args+=(-p image="$IMAGE")

if [ "$DRY_RUN" = true ]; then
    echo "argo submit ${args[*]}"
    exit 0
fi

output=$(argo submit "${args[@]}")
echo "Argo job submitted: $(echo "$output" | grep 'Name:' | awk '{print $2}')"
```

Unlike `compute_dataset.sh` there is no `yq` read of the config: the coupled
config has no `runs`/`data_output_directory` fan-out keys, and every output path
is derived inside `CreateCoupledDatasetsConfig` from `version` / `family_name` /
`output_directory`. `--dry-run` therefore prints the submit command, not the
output stores.

## `scripts/data_process/Makefile` (modified)

Every coupled config keeps its local conda target. **The argo flavour is added
only for the GCS-backed configs** — the four CM4 families, whose
`output_directory` and every input `zarr_path` are under
`gs://vcm-ml-intermediate`, reachable with the `gcp-key` secret the workflow
mounts. `E3SMv3-piControl-100yr-coupled.yaml` writes to
`/pscratch/sd/e/elynnwu/fme-dataset` (NERSC) and
`CM4-like-AM4-random-CO2-ensemble-coupled.yaml` to `/climate-default` (weka);
neither filesystem is mounted in this workflow, so an `_argo` sibling for them
would be broken on arrival. They stay local-only.

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

# --- CM4: local + argo ------------------------------------------------------
cm4_piControl_coupled:                       # CHANGED — config path via the variable
	conda run --no-capture-output -n create_coupled_datasets python ./create_coupled_datasets.py --yaml $(CONFIG_CM4_PICONTROL_COUPLED)

.PHONY: cm4_piControl_coupled_argo           # NEW
cm4_piControl_coupled_argo:
	./create_coupled_datasets.sh --config $(CONFIG_CM4_PICONTROL_COUPLED) --image $(COUPLED_PROCESSING_IMAGE):$(COUPLED_PROCESSING_IMAGE_VERSION)

cm4_1pctCO2_coupled:                         # CHANGED
	conda run --no-capture-output -n create_coupled_datasets python ./create_coupled_datasets.py --yaml $(CONFIG_CM4_1PCTCO2_COUPLED)

.PHONY: cm4_1pctCO2_coupled_argo             # NEW
cm4_1pctCO2_coupled_argo:
	./create_coupled_datasets.sh --config $(CONFIG_CM4_1PCTCO2_COUPLED) --image $(COUPLED_PROCESSING_IMAGE):$(COUPLED_PROCESSING_IMAGE_VERSION)

cm4_piControl_coupled_1daily:                # CHANGED
	conda run --no-capture-output -n create_coupled_datasets python ./create_coupled_datasets.py --yaml $(CONFIG_CM4_PICONTROL_COUPLED_1DAILY)

.PHONY: cm4_piControl_coupled_1daily_argo    # NEW
cm4_piControl_coupled_1daily_argo:
	./create_coupled_datasets.sh --config $(CONFIG_CM4_PICONTROL_COUPLED_1DAILY) --image $(COUPLED_PROCESSING_IMAGE):$(COUPLED_PROCESSING_IMAGE_VERSION)

cm4_1pctCO2_coupled_1daily:                  # CHANGED
	conda run --no-capture-output -n create_coupled_datasets python ./create_coupled_datasets.py --yaml $(CONFIG_CM4_1PCTCO2_COUPLED_1DAILY)

.PHONY: cm4_1pctCO2_coupled_1daily_argo      # NEW
cm4_1pctCO2_coupled_1daily_argo:
	./create_coupled_datasets.sh --config $(CONFIG_CM4_1PCTCO2_COUPLED_1DAILY) --image $(COUPLED_PROCESSING_IMAGE):$(COUPLED_PROCESSING_IMAGE_VERSION)

# --- local only: output stores are not on GCS -------------------------------
e3smv3_piControl_100yr_coupled:              # CHANGED — config path via the variable
	conda run --no-capture-output -n create_coupled_datasets python ./create_coupled_datasets.py --yaml $(CONFIG_E3SMV3_PICONTROL_COUPLED)

.PHONY: cm4_like_am4_random_co2_ensemble_coupled   # CHANGED — the bare `.PHONY:` above it names nothing
cm4_like_am4_random_co2_ensemble_coupled:    # CHANGED — config path via the variable
	conda run --no-capture-output -n create_coupled_datasets python ./create_coupled_datasets.py --yaml $(CONFIG_CM4_LIKE_AM4_RANDCO2_COUPLED)
```

Existing local targets keep their names, their recipe shape and their behaviour;
only the literal config path moves into a variable. Nothing on the
`compute_dataset.sh` side is touched.

## `scripts/data_process/README.md` (modified)

One paragraph beside the existing argo paragraph: the coupled targets run
locally in the `create_coupled_datasets` conda env, the CM4 ones also on argo
via the `_argo` sibling; the argo flavour needs `coupled-processing` pushed
first, and is offered only where the config's stores are on GCS.

---

## Tests

Both files run under the repo's default `fme` environment with `python -m
pytest`, alongside the existing `scripts/data_process` tests, and need neither
the `create_coupled_datasets` env nor any of `requirements-coupled.txt`: they
read the shell, yaml and Makefile as text and parse python with `ast`.

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
    # Walk the TRANSITIVE closure: starting from create_coupled_datasets.py,
    # ast-parse each module, collect imports that resolve to a sibling .py in
    # the same directory, and recurse to a fixpoint. A closure computed from
    # the entry point alone misses create_window_avg_dataset -> time_utils and
    # passes green against a workflow that dies on ModuleNotFoundError.
    # Assert each module in the closure appears as a
    # `-p <module>_script="$(< <module>.py)"` line in create_coupled_datasets.sh
    # and as a declared workflow parameter.
    ...

def test_coupled_workflow_writes_each_script_to_its_module_name():
    # GOAL: each *_script parameter is heredoc'd to <module>.py in the container
    # args, so the sibling imports resolve.
    # HOW: yaml.safe_load the manifest, take the template's container args
    # string, and regex its heredoc openers for
    # (cat << 'EOF' > <file>, {{workflow.parameters.<name>}}) pairs; assert the
    # pairing is <module>.py <- <module>_script for every declared *_script.
    ...

def test_coupled_workflow_heredocs_quote_the_delimiter():
    # GOAL: every heredoc carrying python source uses a QUOTED delimiter
    # (<< 'EOF'), so $, ${} and backticks in the source survive verbatim.
    ...
```

## `scripts/data_process/test_config.py` (modified)

```python
def test_every_makefile_coupled_config_exists():  # NEW
    # GOAL: every `--yaml configs/...` path in the Makefile's coupled targets,
    # and every `--config configs/...` path in its _argo targets, resolves on
    # disk, expanded for RESOLUTION in {1deg, 4deg}.
    # NOTE: "coupled config" here means a create_coupled_datasets.py config —
    # the ones test_config.py classifies by the `coupled_datasets` key. The
    # `*-coupled-ic.yaml` files and e3smv3-coupled-atm-1deg.yaml are other
    # config kinds despite the name, and no Makefile coupled target names them.
    ...

def test_argo_coupled_targets_only_name_gcs_backed_configs():  # NEW
    # GOAL: the workflow mounts only the gcp-key secret, so an _argo target
    # whose config reads or writes a non-gs:// store cannot run.
    ...
```

---

## The four submit commands

Run on a host with `argo` configured, once the coupled image is pushed.

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

## Alternatives considered

- **A `coupled` tag prefix inside the existing `atmosphere-processing`
  repository**, rather than a fresh `coupled-processing` one. Rejected: the
  repository-per-image shape is what `ATMOSPHERE_PROCESSING_IMAGE` and
  `HEALPIX_PROCESSING_IMAGE` already do, and one repository holding two
  unrelated dependency sets makes the version variables ambiguous.
- **Sizing the container below `230Gi` / `30000m`.** Rejected for the first
  cut: `30000m` is 30 CPU against `output_writer.n_dask_workers: 32`, so the
  request is matched to the configs' own concurrency rather than arbitrary, and
  it is the size every heavy step of `compute_dataset_argo_workflow.yaml`
  already requests on the same pool. The coupled job holds atmosphere, ocean
  and sea-ice intermediates concurrently, so sizing down is a measurement to
  make from a first run's usage, not a guess to make ahead of one.
- **`_local` on the existing conda targets, with the bare name becoming the
  argo path.** Rejected: it renames targets that already work and are already
  in use, to no benefit. The `_argo` suffix is purely additive, and leaves
  every existing invocation meaning what it meant.
- **Reusing `atmosphere-processing` by extending
  `requirements-atmosphere.txt`.** Rejected above: it rebuilds the atmosphere
  image for the coupled path's benefit.
- **One workflow step per processing stage.** Rejected above: the stages share
  in-memory intermediates, and the existing skip-on-exists behaviour already
  gives resume without a dependency graph.
