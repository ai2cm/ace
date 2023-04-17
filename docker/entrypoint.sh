#!/bin/bash


vscode () {
    set -e
    wget -nc https://github.com/coder/code-server/releases/download/v4.9.1/code-server_4.9.1_amd64.deb
    dpkg -i code-server_4.9.1_amd64.deb
    code-server --auth password --bind-addr 0.0.0.0:${VSCODE_PORT}
    set +e
}

# needed for some reason in docker images
ldconfig

if [[ -d "$FCN_MIP_SRC" ]]
then
    make -C "$FCN_MIP_SRC" update_submodules
    make -C "$FCN_MIP_SRC" install_local_dependencies
fi

$@
