# Per-channel std of the diffused quantity (residual over linear interp),
# for the 5-channel model (adds air_temperature_at_two_meters to the
# existing 4-channel video PMD models). Builds the VideoDiffusionModelConfig
# inline -- no 5-channel training yaml exists yet -- reusing the exact
# architecture from
# ../../configs/experiments/2026-06-30-video-pmd-per-channel-noise/video_train.yaml,
# with T2m's normalization stats from the toy/compute_norm_stats.py global-
# domain run (mean=279.3386, std=20.2363, train split 2013-2021 only, no
# leakage) and a placeholder training_noise_distribution for T2m (doesn't
# affect the residual/normalization measured here -- only EDM
# preconditioning/loss weighting, which this script's output feeds into).
#
# Run (on Beaker, fme image): FME_FORCE_CPU=1 python3 -m scripts.video_pmd_eval.compute_residual_std_5ch
import math

import torch

from fme.core.dataset.time import TimeSlice
from fme.core.dataset.xarray import XarrayDataConfig
from fme.downscaling.data.config import PairedDataLoaderConfig
from fme.downscaling.data.utils import ClosedInterval
from fme.downscaling.video_models import (
    LogNormalNoiseDistribution,
    NormalizationConfig,
    VideoDiffusionModelConfig,
    _linear_interp_endpoints,
)

SOURCE_PATH = "/climate-default/2026-06-25-temporal-diffusion"
STORE = "2025-07-25-X-SHiELD-AMIP-FME-3h.zarr"
TRAIN_START, TRAIN_STOP = "2013-01-02", "2021-12-31"
N_BATCHES = 40
BATCH_SIZE = 8

OUT_NAMES = [
    "eastward_wind_at_ten_meters",
    "northward_wind_at_ten_meters",
    "PRMSL",
    "PRATEsfc",
    "air_temperature_at_two_meters",
]


def main():
    model_cfg = VideoDiffusionModelConfig(
        out_names=OUT_NAMES,
        n_timesteps=9,
        model_channels=128,
        n_heads=8,
        num_freqs=4,
        backbone="simple",
        noise_embedding_type="positional",
        channel_mult=[1, 2, 2, 2, 2],
        num_blocks=2,
        attention_levels=[3, 4],
        sigma_min=0.002,
        sigma_max=150.0,
        churn=0.0,
        num_diffusion_generation_steps=50,
        log_transform_channels={"PRATEsfc": 86400.0},
        training_noise_distributions={
            name: LogNormalNoiseDistribution(p_mean=-1.0, p_std=1.2)
            for name in OUT_NAMES
        },
        normalization=NormalizationConfig(
            means={
                "eastward_wind_at_ten_meters": -0.036135,
                "northward_wind_at_ten_meters": 0.186555,
                "PRMSL": 1008.281106,
                "PRATEsfc": 0.557112,
                "air_temperature_at_two_meters": 279.3386,
            },
            stds={
                "eastward_wind_at_ten_meters": 5.548626,
                "northward_wind_at_ten_meters": 4.603089,
                "PRMSL": 14.908773,
                "PRATEsfc": 0.871331,
                "air_temperature_at_two_meters": 20.2363,
            },
        ),
    )
    model = model_cfg.build()
    names = model_cfg.out_names

    data_cfg = PairedDataLoaderConfig(
        fine=[XarrayDataConfig(SOURCE_PATH, file_pattern=STORE, engine="zarr",
                               subset=TimeSlice(TRAIN_START, TRAIN_STOP))],
        coarse=[XarrayDataConfig(SOURCE_PATH, file_pattern=STORE, engine="zarr",
                                 subset=TimeSlice(TRAIN_START, TRAIN_STOP))],
        batch_size=BATCH_SIZE, num_data_workers=0, strict_ensemble=False,
        lat_extent=ClosedInterval(-88, 88), lon_extent=ClosedInterval(0, 360),
        n_timesteps=model_cfg.n_timesteps,
    )
    data = data_cfg.build_video(train=True, requirements=model_cfg.data_requirements)

    n_t = model_cfg.n_timesteps
    interior = slice(1, n_t - 1)
    res_sumsq = torch.zeros(len(names))
    clip_sumsq = torch.zeros(len(names))
    count = torch.zeros(len(names))

    for b, batch in enumerate(data.loader):
        if b >= N_BATCHES:
            break
        clip = model._pack_normalized(batch.fine.data)
        baseline = _linear_interp_endpoints(clip)
        residual = (clip - baseline)[:, :, interior]
        clip_i = clip[:, :, interior]
        res_sumsq += (residual ** 2).sum(dim=(0, 2, 3, 4))
        clip_sumsq += (clip_i ** 2).sum(dim=(0, 2, 3, 4))
        count += residual[0, 0].numel() * residual.shape[0]

    res_std = (res_sumsq / count).sqrt()
    clip_std = (clip_sumsq / count).sqrt()

    print(f"\nTRAIN residual diagnostic over {int(count[0].item())} interior "
          f"voxels/channel ({N_BATCHES * BATCH_SIZE} clips)\n")
    print(f"{'channel':<32} {'resid_std':>9} {'clip_std':>9} {'resid/clip':>10} "
          f"{'p_mean*':>8} {'sigma_max*':>10}")
    print("-" * 82)
    for i, name in enumerate(names):
        rs = res_std[i].item()
        cs = clip_std[i].item()
        p_mean = math.log(rs)
        sigma_max = round(rs * 160, 1)
        print(f"{name:<32} {rs:>9.4f} {cs:>9.4f} {rs / cs:>10.3f} "
              f"{p_mean:>8.2f} {sigma_max:>10.1f}")
    print("\n* suggested: sigma_data := resid_std (per channel), p_mean := "
          "ln(resid_std), sigma_max := ~160 x resid_std (EDM ratio).")


if __name__ == "__main__":
    main()
