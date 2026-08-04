#!/bin/bash
# Bind each torchrun local rank to a distinct block of CPU cores.
#
# Polaris compute nodes have 64 hardware threads and 4 A100s. We run one
# torchrun agent per node that spawns NGPUS_PER_NODE local ranks. Without
# binding, every rank sees all 64 cores, so per-process thread/process pools
# (OpenMP, OpenBLAS, torch inductor compile pool, dataloader workers, ...) each
# size themselves to 64 and the node runs out of processes/threads
# (os.fork -> "Resource temporarily unavailable"). Giving each rank its own
# 1/NGPU slice keeps the totals balanced and leaves cores for data loading.
#
# Usage (via torchrun --no-python): set-affinity.sh python -m fme.ace.train CONFIG
LOCAL_RANK=${LOCAL_RANK:-0}
NGPU=${NGPUS_PER_NODE:-4}
TOTAL=$(nproc --all)
PER=$(( TOTAL / NGPU ))
START=$(( LOCAL_RANK * PER ))
END=$(( START + PER - 1 ))

exec taskset -c ${START}-${END} "$@"
