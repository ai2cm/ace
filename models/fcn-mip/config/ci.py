AFNO_26_MEAN = "stats/global_means.npy"
AFNO_26_SCALE = "stats/global_stds.npy"

ERA5_2018_ZARR = "/root/data/34Vars/out_of_sample/2018.zarr"

MEAN = "/root/models/gfno_26ch_sc3_layers8_tt64/center.npy"
SCALE = "/root/models/gfno_26ch_sc3_layers8_tt64/scale.npy"
TIME_MEAN = "/mount/34vars/stats/time_means.npy"

MODEL_REGISTRY = None
TIME_MEAN_73 = "s3://sw_climate_fno/test_datasets/73var-6hourly/stats/time_means.npy"
CHANNEL_76_DATA = "s3://sw_climate_fno/test_datasets/73var-6hourly"
INITIAL_CONDITION_DIRECTORY = "/mount/34vars"

# DIAGNOSTICS used in tests
TEST_DIAGNOSTICS = ["raw", "ensemble_mean", "ensemble_variance"]