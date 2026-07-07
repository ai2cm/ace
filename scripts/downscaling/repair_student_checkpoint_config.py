#!/usr/bin/env python
"""
Repair a distilled-student checkpoint whose baked config is the wrong expert's.

Background: ``save_student_checkpoint`` inherits its config from the
``DiffusionModel`` passed as ``teacher_model``. Before the fix in
``fastgen_train.py``, that was always the MoE ``_primary`` (experts[0], the
low-noise expert) regardless of which expert was distilled. So a Student-Hi
checkpoint (expert 1, 256-wide) was saved with expert-0's 128-wide config and
fails to reload with a size mismatch.

The student *weights* are correct; only the baked config (architecture,
sigma_data, normalizers, grid) is the wrong expert's. This rebuilds the
checkpoint from the correct expert's ``DiffusionModel`` state -- taken from the
bundled MoE teacher -- with the student's weights swapped in. No retraining.

The repaired checkpoint keeps the student's raw (training-time) variable names,
matching the sibling per-expert student checkpoints (which are also saved
without applying the MoE rename), so they bundle together consistently.

Usage:
    python repair_student_checkpoint_config.py \\
        --student-ckpt /hi/best_student_tail.ckpt \\
        --teacher-bundle /teacher/bundled_moe_multivariate.ckpt \\
        --expert-index 1 \\
        --output /out/hi_best_student_tail_fixed.ckpt
"""

import argparse

import torch


def _strip_module_prefix(state_dict: dict) -> dict:
    return {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--student-ckpt", required=True)
    parser.add_argument(
        "--teacher-bundle",
        required=True,
        help="Bundled MoE teacher (DenoisingMoEPredictor.save output) whose "
        "experts[expert_index] provides the correct config.",
    )
    parser.add_argument("--expert-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sampler-type",
        default="fastgen",
        help="sampler_type to bake into the repaired config (default fastgen).",
    )
    return parser.parse_args()


def main(
    student_ckpt: str,
    teacher_bundle: str,
    expert_index: int,
    output: str,
    sampler_type: str,
) -> None:
    student = torch.load(student_ckpt, map_location="cpu", weights_only=False)["model"]
    bundle = torch.load(teacher_bundle, map_location="cpu", weights_only=False)
    experts = bundle["experts"]
    if not 0 <= expert_index < len(experts):
        raise ValueError(
            f"expert_index {expert_index} out of range for {len(experts)} experts."
        )
    ref = experts[expert_index]

    # Verify the student's weights actually fit the correct expert's architecture
    # (same key set and shapes after normalizing the DDP 'module.' prefix). This
    # catches a wrong --expert-index or a genuinely incompatible checkpoint before
    # anything is written.
    ref_mod = _strip_module_prefix(ref["module"])
    stu_mod = _strip_module_prefix(student["module"])
    if ref_mod.keys() != stu_mod.keys():
        only_ref = sorted(set(ref_mod) - set(stu_mod))[:5]
        only_stu = sorted(set(stu_mod) - set(ref_mod))[:5]
        raise ValueError(
            "Student weights do not match the reference expert's parameter names. "
            f"e.g. only-in-expert={only_ref} only-in-student={only_stu}"
        )
    mismatched = [k for k in ref_mod if ref_mod[k].shape != stu_mod[k].shape]
    if mismatched:
        raise ValueError(
            f"Student weights have shapes incompatible with expert {expert_index} "
            f"(e.g. {mismatched[:3]}). Wrong --expert-index?"
        )

    # Rebuild from the correct expert's state; swap in the student's weights.
    fixed = dict(ref)
    fixed["module"] = student["module"]
    fixed["config"] = dict(ref["config"])
    fixed["config"]["sampler_type"] = sampler_type

    torch.save({"model": fixed}, output)
    print(
        f"Wrote repaired student checkpoint to {output} "
        f"(config from teacher-bundle expert {expert_index}, "
        f"weights from {student_ckpt}, {len(stu_mod)} tensors)."
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.student_ckpt,
        args.teacher_bundle,
        args.expert_index,
        args.output,
        args.sampler_type,
    )
