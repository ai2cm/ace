import torch

from fme.downscaling.data import StaticInputs, get_normalized_topography
from fme.downscaling.models import CheckpointModelConfig

scratch_dir = "/Users/annak/Code/scratch/checkpoints/hiro-ckpt"

topography_path = "gs://vcm-ml-raw-flexible-retention/2025-08-12-X-SHiELD-AMIP-downscaling/regridded-zarrs/gaussian_grid_180_by_360_refined_to_5760_by_11520/control/static.zarr"
checkpoint_path = f"{scratch_dir}/hiro-0.ckpt"

topography = get_normalized_topography(topography_path)

config = CheckpointModelConfig(checkpoint_path=checkpoint_path)
model = config.build()
model.static_inputs = StaticInputs([topography])

new_checkpoint = config._checkpoint
new_checkpoint.update(model=model.get_state())  # type: ignore

torch.save(new_checkpoint, f"{scratch_dir}/HiRO.ckpt")
