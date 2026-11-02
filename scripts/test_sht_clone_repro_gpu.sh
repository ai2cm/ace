#!/bin/bash
# Throwaway: run distributed SHT compile tests on GPU/NCCL without and with
# the .clone() calls in fme/core/sht/_distributed.py.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
python -c 'import torch, torch_harmonics; print("torch", torch.__version__, "cuda", torch.version.cuda, "th", torch_harmonics.__version__, "gpus", torch.cuda.device_count())'
nvidia-smi -L

run_matrix() {
  for cfg in "1 2 2" "2 1 2" "2 2 4"; do
    read H W N <<< "$cfg"
    echo "===== [$1] H=$H W=$W"
    FME_DISTRIBUTED_BACKEND=model FME_DISTRIBUTED_H=$H FME_DISTRIBUTED_W=$W \
      torchrun --nproc-per-node "$N" -m pytest -m parallel -q -p no:cacheprovider \
      fme/core/sht/test_distributed_compile.py fme/core/sht/test_repro_stress.py > /tmp/log.txt 2>&1
    echo "exit $?"
    grep -E '^E .*Error|passed|failed' /tmp/log.txt | cut -c1-300 | sort | uniq -c | head -30
    grep -q passed /tmp/log.txt || tail -60 /tmp/log.txt
  done
}

echo "clone() count: $(grep -c 'clone()' fme/core/sht/_distributed.py)"
run_matrix NO_CLONE
sed -i 's/torch.view_as_complex(xs)$/torch.view_as_complex(xs).clone()/' fme/core/sht/_distributed.py
echo "clone() count: $(grep -c 'clone()' fme/core/sht/_distributed.py)"
run_matrix WITH_CLONE
