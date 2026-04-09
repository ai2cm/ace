import torch

from fme.downscaling.data import StaticInputs, get_normalized_topography
from fme.downscaling.models import CheckpointModelConfig

scratch_dir = "/climate-default/home/annak/scratch/checkpoints/full-field-precip-v1"

topography_path = "/climate-default/2025-09-25-downscaling-data-X-SHiELD-AMIP-downscaling/3km.zarr"
checkpoint_path = f"{scratch_dir}/checkpoints/latest.ckpt"

config = CheckpointModelConfig(
    checkpoint_path=checkpoint_path,
    static_inputs={
        "HGTsfc": topography_path,
    },
    fine_coordinates_path=topography_path,
)
model = config.build()

new_checkpoint = config._checkpoint
new_checkpoint.update(model=model.get_state())  # type: ignore

torch.save(new_checkpoint, f"{scratch_dir}/latest.ckpt")