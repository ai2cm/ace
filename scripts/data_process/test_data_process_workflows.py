"""Static checks over the argo submit scripts and their workflow manifests.

Nothing here submits a workflow, builds an image, or touches the network: the
yaml is read as text, the python is parsed with `ast`, and the submit script
runs only with --dry-run.
"""

import ast
import os
import re
import subprocess

import pytest
import yaml

DIRNAME = os.path.abspath(os.path.dirname(__file__))

COUPLED_WORKFLOW = os.path.join(DIRNAME, "create_coupled_datasets_argo_workflow.yaml")
COUPLED_SUBMIT_SCRIPT = os.path.join(DIRNAME, "create_coupled_datasets.sh")
COUPLED_ENTRY_POINTS = [
    os.path.join(DIRNAME, "create_coupled_datasets.py"),
    os.path.join(DIRNAME, "upload_coupled_stats.py"),
]

WORKFLOW_YAMLS = [
    COUPLED_WORKFLOW,
    os.path.join(DIRNAME, "compute_dataset_argo_workflow.yaml"),
]

PARAMETER_REFERENCE = re.compile(r"\{\{workflow\.parameters\.([A-Za-z0-9_]+)\}\}")
HEREDOC_OPENER = re.compile(
    r"cat\s*<<\s*(?P<quote>'?)(?P<delim>[A-Za-z0-9_]+)(?P=quote)\s*>\s*(?P<file>\S+)"
)
SCRIPT_PARAMETER_FLAG = re.compile(r"-p\s+([A-Za-z0-9_]+_script)=")


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _declared_parameters(path: str) -> list[str]:
    manifest = yaml.safe_load(_read(path))
    return [p["name"] for p in manifest["spec"]["arguments"]["parameters"]]


def _container_args(path: str) -> str:
    manifest = yaml.safe_load(_read(path))
    args = []
    for template in manifest["spec"]["templates"]:
        container = template.get("container")
        if container is not None:
            args.extend(container.get("args", []))
    return "\n".join(args)


def _sibling_import_closure(entry_point: str) -> set[str]:
    """Modules reachable from `entry_point` by imports of siblings in its directory.

    A closure computed from the entry point's direct imports alone is not
    enough: `coupled_dataset_utils` imports `create_window_avg_dataset`, which
    imports `time_utils`, and neither appears in the entry point's own imports.
    """
    directory = os.path.dirname(entry_point)
    closure: set[str] = set()
    pending = [os.path.splitext(os.path.basename(entry_point))[0]]
    while pending:
        module = pending.pop()
        if module in closure:
            continue
        closure.add(module)
        tree = ast.parse(_read(os.path.join(directory, module + ".py")))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            for name in names:
                if name is None:
                    continue
                root = name.split(".")[0]
                if os.path.exists(os.path.join(directory, root + ".py")):
                    pending.append(root)
    return closure


@pytest.mark.parametrize("path", WORKFLOW_YAMLS, ids=os.path.basename)
def test_workflow_parameters_are_declared(path):
    declared = set(_declared_parameters(path))
    referenced = set(PARAMETER_REFERENCE.findall(_read(path)))
    assert referenced - declared == set()


@pytest.mark.parametrize("entry_point", COUPLED_ENTRY_POINTS, ids=os.path.basename)
def test_coupled_submit_script_passes_every_local_import(entry_point):
    closure = _sibling_import_closure(entry_point)
    passed = set(SCRIPT_PARAMETER_FLAG.findall(_read(COUPLED_SUBMIT_SCRIPT)))
    declared = set(_declared_parameters(COUPLED_WORKFLOW))
    expected = {module + "_script" for module in closure}
    assert expected - passed == set()
    assert expected - declared == set()


def test_coupled_workflow_writes_each_script_to_its_module_name():
    args = _container_args(COUPLED_WORKFLOW)
    script_parameters = [
        name
        for name in _declared_parameters(COUPLED_WORKFLOW)
        if name.endswith("_script")
    ]
    written = {}
    for opener in HEREDOC_OPENER.finditer(args):
        body_start = opener.end()
        body_end = args.index("\n" + opener.group("delim"), body_start)
        body = args[body_start:body_end]
        for name in PARAMETER_REFERENCE.findall(body):
            written[name] = opener.group("file")
    assert {name: written.get(name) for name in script_parameters} == {
        name: name[: -len("_script")] + ".py" for name in script_parameters
    }


def test_coupled_workflow_heredocs_quote_the_delimiter():
    args = _container_args(COUPLED_WORKFLOW)
    unquoted = [
        opener.group("file")
        for opener in HEREDOC_OPENER.finditer(args)
        if opener.group("quote") != "'"
    ]
    assert unquoted == []


def _coupled_dag_tasks() -> dict[str, dict]:
    manifest = yaml.safe_load(_read(COUPLED_WORKFLOW))
    templates = {t["name"]: t for t in manifest["spec"]["templates"]}
    tasks = templates[manifest["spec"]["entrypoint"]]["dag"]["tasks"]
    return {task["name"]: task for task in tasks}


def _config_argument(task: dict) -> str:
    (parameter,) = task["arguments"]["parameters"]
    assert parameter["name"] == "config"
    return parameter["value"]


def test_coupled_workflow_chains_the_dependent_config():
    tasks = _coupled_dag_tasks()
    assert {name: task["template"] for name, task in tasks.items()} == {
        "create": "create-coupled-datasets",
        "upload": "upload-coupled-stats",
        "create-dependent": "create-coupled-datasets",
        "upload-dependent": "upload-coupled-stats",
    }
    assert {name: task.get("depends") for name, task in tasks.items()} == {
        "create": None,
        "upload": "create",
        "create-dependent": "create",
        "upload-dependent": "create-dependent",
    }
    assert {name: _config_argument(task) for name, task in tasks.items()} == {
        "create": "{{workflow.parameters.config}}",
        "upload": "{{workflow.parameters.config}}",
        "create-dependent": "{{workflow.parameters.dependent_config}}",
        "upload-dependent": "{{workflow.parameters.dependent_config}}",
    }


def test_coupled_workflow_skip_conditions():
    tasks = _coupled_dag_tasks()
    no_stats = [
        "{{workflow.parameters.debug}} == false",
        "{{workflow.parameters.subsample}} == false",
    ]
    run_dependent = "{{workflow.parameters.run_dependent}} == true"
    assert "when" not in tasks["create"]
    assert all(c in tasks["upload"]["when"] for c in no_stats)
    assert run_dependent not in tasks["upload"]["when"]
    assert tasks["create-dependent"]["when"] == run_dependent
    # a bare `depends` is also met by a skipped task, so the dependent upload
    # repeats run_dependent
    assert all(
        c in tasks["upload-dependent"]["when"] for c in no_stats + [run_dependent]
    )


def test_coupled_templates_read_config_from_inputs():
    manifest = yaml.safe_load(_read(COUPLED_WORKFLOW))
    for template in manifest["spec"]["templates"]:
        if "container" not in template:
            continue
        assert template["inputs"]["parameters"] == [{"name": "config"}]
        args = "\n".join(template["container"]["args"])
        assert "{{inputs.parameters.config}}" in args
        assert "{{workflow.parameters.config}}" not in args


def _submit(*flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", COUPLED_SUBMIT_SCRIPT, *flags, "--dry-run"],
        cwd=DIRNAME,
        capture_output=True,
        text=True,
    )


PAIR_FLAGS = (
    "--config",
    "configs/CM4-piControl-coupled-1deg-1daily-200yr.yaml",
    "--dependent-config",
    "configs/CM4-1pctCO2-coupled-1deg-1daily-140yr.yaml",
)


def test_coupled_submit_script_passes_the_dependent_config():
    single = _submit(*PAIR_FLAGS[:2])
    pair = _submit(*PAIR_FLAGS)
    assert single.returncode == 0 and pair.returncode == 0
    assert "run_dependent" not in single.stdout
    assert "dependent_config" not in single.stdout
    assert "-p run_dependent=true" in pair.stdout
    assert "-p dependent_config=# make cm4_1pctCO2_coupled_1daily" in pair.stdout


@pytest.mark.parametrize("flag", ["--debug", "--subsample"])
def test_coupled_submit_script_rejects_dependent_config_with(flag):
    result = _submit(*PAIR_FLAGS, flag)
    assert result.returncode != 0
    assert "--dependent-config cannot be combined" in result.stdout
