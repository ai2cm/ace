import os

from fme.core.cloud import is_local


def validate_output_dir(experiment_dir: str, output_dir: str | None) -> None:
    """Check that ``experiment_dir`` is local when ``output_dir`` is set.

    With a separate ``output_dir``, ``experiment_dir`` holds only the job's
    config, logs and wandb state, which must be written to a local filesystem.
    """
    if output_dir is not None and not is_local(experiment_dir):
        raise ValueError(
            "experiment_dir must be on a local filesystem when output_dir is "
            f"set, got experiment_dir={experiment_dir!r}. Put remote outputs "
            "in output_dir instead."
        )


def resolve_output_dir(experiment_dir: str, output_dir: str | None) -> str:
    """Return the directory receiving data outputs: ``output_dir`` if set,
    otherwise ``experiment_dir``.
    """
    return output_dir if output_dir is not None else experiment_dir


def is_separate_output_dir(experiment_dir: str, output_dir: str | None) -> bool:
    """Return whether ``output_dir`` is set to a directory other than
    ``experiment_dir``, normalizing paths when both are local.
    """
    if output_dir is None:
        return False
    if is_local(experiment_dir) and is_local(output_dir):
        return os.path.normpath(experiment_dir) != os.path.normpath(output_dir)
    return experiment_dir != output_dir
