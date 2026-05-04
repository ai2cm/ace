#!/bin/bash
# Benchmark data loading: zarr vs netcdf, varying worker counts.
set -e

ZARR_DIR=./scripts/cmip6_data/data/cmip6-daily-pilot/v0
NC_DIR=./scripts/cmip6_data/data/cmip6-daily-pilot/v0-nc
N_BATCHES=100
SLEEP=0.0

run_one() {
    local label="$1" engine="$2" data_dir="$3" workers="$4"
    echo "=== $label: engine=$engine workers=$workers ==="
    python -c "
import time, torch
from fme.core.distributed.distributed import Distributed
from fme.ace.data_loading.config import DataLoaderConfig
from fme.ace.data_loading.cmip6 import Cmip6DataConfig
from fme.ace.data_loading.getters import get_gridded_data
from fme.ace.requirements import DataRequirements
from fme.core.dataset.schedule import IntSchedule
import dacite

dataset_cfg = Cmip6DataConfig(
    data_dir='$data_dir',
    engine='$engine',
    exclude_source_ids=['AWI-CM-1-1-MR', 'KACE-1-0-G'],
    experiments=['historical', 'ssp245', 'ssp585'],
)
loader_cfg = DataLoaderConfig(
    dataset=dataset_cfg,
    batch_size=4,
    num_data_workers=$workers,
    prefetch_factor=4 if $workers > 0 else None,
)
names = [
    'ts', 'ua1000','ua850','ua700','ua500','ua250','ua100','ua50','ua10',
    'va1000','va850','va700','va500','va250','va100','va50','va10',
    'hus1000','hus850','hus700','hus500','hus250','hus100','hus50','hus10',
    'zg1000','zg850','zg700','zg500','zg250','zg100','zg50','zg10',
    'tas','huss','psl','pr',
    'ta_derived_layer_1000_850','ta_derived_layer_850_700',
    'ta_derived_layer_700_500','ta_derived_layer_500_250',
    'ta_derived_layer_250_100','ta_derived_layer_100_50',
    'ta_derived_layer_50_10',
]
reqs = DataRequirements(names, 2)

with Distributed.context():
    t0 = time.time()
    data = get_gridded_data(loader_cfg, train=True, requirements=reqs)
    init_time = time.time() - t0
    print(f'Init: {init_time:.1f}s')

    loader = data.loader
    load_times = []
    t_start = time.time()
    for i, batch in enumerate(loader):
        if i >= $N_BATCHES:
            break
        load_times.append(time.time() - t_start)
        time.sleep($SLEEP)
        t_start = time.time()

    total = sum(load_times)
    mean = total / len(load_times)
    print(f'Batches: {len(load_times)}')
    print(f'Mean load time: {mean:.4f}s')
    print(f'Total load time: {total:.2f}s')
    print(f'Total wall time: {total + len(load_times)*$SLEEP:.2f}s')
" 2>&1
    echo ""
}

run_one "zarr-0w"    zarr    "$ZARR_DIR" 0
run_one "zarr-4w"    zarr    "$ZARR_DIR" 4
run_one "netcdf-0w"  netcdf4 "$NC_DIR"   0
run_one "netcdf-4w"  netcdf4 "$NC_DIR"   4
run_one "netcdf-8w"  netcdf4 "$NC_DIR"   8
