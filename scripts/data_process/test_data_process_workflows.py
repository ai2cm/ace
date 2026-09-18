"""Static checks over the argo submit scripts and their workflow manifests.

Nothing here submits a workflow, builds an image, or touches the network: the
shell and yaml are read as text and the python is parsed with `ast`.
"""

import ast
import os
import re

import pytest
import yaml

DIRNAME = os.path.abspath(os.path.dirname(__file__))

COUPLED_WORKFLOW = os.path.join(DIRNAME, "create_coupled_datasets_argo_workflow.yaml")
COUPLED_SUBMIT_SCRIPT = os.path.join(DIRNAME, "create_coupled_datasets.sh")
COUPLED_ENTRY_POINT = os.path.join(DIRNAME, "create_coupled_datasets.py")

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


def test_coupled_submit_script_passes_every_local_import():
    closure = _sibling_import_closure(COUPLED_ENTRY_POINT)
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
