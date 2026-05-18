"""
Convert FastGen distillation checkpoints to ACE format.

Loads the latest .pth checkpoint for each trained distillation method,
hot-swaps the student weights into the teacher model, and saves ACE-format
.ckpt files to /results/.

Outputs:
    /results/dmd2_student.ckpt
    /results/fdistill_student.ckpt
"""

from fme.core.distributed import Distributed
from fme.downscaling.distillation.student_checkpoint import save_student_checkpoint
from fme.downscaling.models import FastgenStudentConfig

TEACHER_CKPT = "/checkpoints/best_histogram_tail.ckpt"
FINE_COORDS = (
    "/climate-default"
    "/2025-09-25-downscaling-data-X-SHiELD-AMIP-downscaling"
    "/3km.zarr"
)
MODELS = [
    (
        "dmd2",
        "/dmd2/0022620.pth",
        1,  # DMD2 is a single-step method
    ),
    (
        "fdistill",
        "/fdistill/0024440.pth",
        4,  # f-distill trained with 4-step Karras schedule
    ),
]

with Distributed.context():
    for name, pth_path, num_steps in MODELS:
        print(f"Converting {name} ({pth_path}, {num_steps} step(s)) ...")
        cfg = FastgenStudentConfig(
            fastgen_checkpoint_path=pth_path,
            teacher_checkpoint_path=TEACHER_CKPT,
            fine_coordinates_path=FINE_COORDS,
        )
        model = cfg.build()
        output_path = f"/results/{name}_student.ckpt"
        save_student_checkpoint(
            student_module=model.module,
            teacher=model,
            path=output_path,
            num_sampling_steps=num_steps,
        )
        print(f"  Saved {output_path}")

print("Done.")
