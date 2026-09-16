#!/bin/bash
# Patch the checkpoint with a decade-specific unaccounted heating, then infer.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CONSTANT_UAH:?Set CONSTANT_UAH to the unaccounted heating in W/m2}"
echo "=== Patching checkpoint with constant_unaccounted_heating=${CONSTANT_UAH} ==="
python "$SCRIPT_DIR/patch-ckpt-energy-corrector-uah.py" /ckpt.tar /ckpt_ec.tar "$CONSTANT_UAH"
echo "=== Running inference ==="
python -I -m fme.ace.inference "$@"
