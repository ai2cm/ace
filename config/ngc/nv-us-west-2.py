MODEL_REGISTRY = "s3://sw_climate_fno/nbrenowitz/model_packages"
TIME_MEAN_73 = "s3://sw_climate_fno/test_datasets/73var-6hourly/stats/time_means.npy"
CHANNEL_76_DATA = "s3://sw_climate_fno/test_datasets/73var-6hourly"

MEAN = "s3://sw_climate_fno/nbrenowitz/model_packages/hafno_baseline_26ch_edim512_mlp2/global_means.npy"
SCALE = "s3://sw_climate_fno/nbrenowitz/model_packages/hafno_baseline_26ch_edim512_mlp2/global_stds.npy"

TIME_MEAN = "s3://sw_climate_fno/34Vars/stats/time_means.npy"
INITIAL_CONDITION_DIRECTORY = "s3://sw_climate_fno/34Vars"

# If set will use joblib.Memory to cache steps requiring network access
LOCAL_CACHE = "/tmp/cache"

# DIAGNOSTICS used in tests
TEST_DIAGNOSTICS = ["raw", "ensemble_mean", "ensemble_variance", "skill"]