# SPDX-FileCopyrightText: Copyright (c) 2026 Allen Institute for AI
# SPDX-License-Identifier: Apache-2.0
"""
Generate a teacher validation dataset for distillation training.

Use ``fme.downscaling.inference`` (the standard ACE inference pipeline) to
generate and save the teacher validation outputs.  It supports distributed
generation and writes a zarr store with dimensions
``(time, ensemble, latitude, longitude)`` that is compatible with
``BestStudentCheckpointCallback``.

Example config (save as e.g. ``teacher_val_inference.yaml``)::

    experiment_dir: /path/to/val_dataset_dir
    model:
        checkpoint_path: /path/to/teacher.ckpt
    data:
        coarse:
          - data_path: /path/to/coarse_val.zarr
            engine: zarr
        batch_size: 4
        num_data_workers: 0
    outputs:
      - name: val_dataset
        n_ens: 4
        time_range:
            start_time: "2020-01-01T00:00:00"
            end_time:   "2021-12-31T18:00:00"
    logging:
        log_to_screen: true
        log_to_wandb: false
        log_to_file: true

Run::

    python -m fme.downscaling.inference teacher_val_inference.yaml

The output zarr will be written to
``/path/to/val_dataset_dir/val_dataset.zarr`` and can be passed directly
to ``--val-dataset`` in ``fastgen_train.py``.
"""
