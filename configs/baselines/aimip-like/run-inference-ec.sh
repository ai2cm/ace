#!/bin/bash
# Patch the checkpoint to add energy corrector, then run inference.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=== Patching checkpoint with energy corrector ==="
python "$SCRIPT_DIR/patch-ckpt-energy-corrector.py" /ckpt.tar /ckpt_ec.tar
echo "=== Running inference ==="
python -I -m fme.ace.inference "$@"
