import torch

from fme.downscaling.data import StaticInputs, get_normalized_topography
from fme.downscaling.models import CheckpointModelConfig

scratch_dir = "/climate-default/home/annak/scratch/checkpoints/hiro"

topography_path = "/climate-default/2025-09-25-downscaling-data-X-SHiELD-AMIP-downscaling/3km.zarr"
checkpoint_path = f"{scratch_dir}/checkpoints/best_histogram_tail.ckpt"

topography = get_normalized_topography(topography_path).to_device()
import pdb; pdb.set_trace()

config = CheckpointModelConfig(checkpoint_path=checkpoint_path)
model = config.build()
model.static_inputs = StaticInputs([topography])

new_checkpoint = config._checkpoint
new_checkpoint.update(model=model.get_state())  # type: ignore

torch.save(new_checkpoint, f"{scratch_dir}/HiRO.ckpt")
