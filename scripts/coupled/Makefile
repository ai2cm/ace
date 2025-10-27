CM4_piControl_VERSION ?= 2025-08-22

.PHONY: CM4_piControl_coupling_datasets
CM4_piControl_coupling_datasets:
	python create_coupling_datasets.py \
		--yaml configs/CM4-piControl-200yr-interpolate_sst-sst_from_ts.yaml \
		--dataset atmosphere
	python get_stats.py \
		--input_zarr /climate-default/$(CM4_piControl_VERSION)-cm4-piControl-200yr-coupled-interpolate_sst-sst_from_ts-atmosphere.zarr \
		--stats_dir /climate-default/$(CM4_piControl_VERSION)-cm4-piControl-200yr-coupled-interpolate_sst-sst_from_ts-atmosphere-stats \
		--n_dask_workers 64
	python combine_stats.py configs/CM4-piControl-200yr-combine-stats-atmosphere.yaml
	beaker dataset create \
		--name $(CM4_piControl_VERSION)-cm4-piControl-200yr-coupled-stats-atmosphere \
		--workspace ai2/ace \
		/climate-default/$(CM4_piControl_VERSION)-cm4-piControl-200yr-coupled-stats/atmosphere
	python create_coupling_datasets.py --yaml configs/CM4-piControl-200yr-interpolate_sst.yaml
	python get_stats.py \
		--input_zarr /climate-default/$(CM4_piControl_VERSION)-cm4-piControl-200yr-coupled-ocean.zarr \
		--stats_dir /climate-default/$(CM4_piControl_VERSION)-cm4-piControl-200yr-coupled-ocean-stats \
		--n_dask_workers 64
	python combine_stats.py configs/CM4-piControl-200yr-combine-stats-ocean.yaml
	beaker dataset create \
		--name $(CM4_piControl_VERSION)-cm4-piControl-200yr-coupled-stats-ocean \
		--workspace ai2/ace \
		/climate-default/$(CM4_piControl_VERSION)-cm4-piControl-200yr-coupled-stats/ocean
