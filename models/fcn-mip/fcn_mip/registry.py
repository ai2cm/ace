"""Create-read-update-delete (CRUD) operations for the FCN model registry

The location of the registry is configured using `config.MODEL_REGISTRY`. Both
s3:// and local paths are supported.

The top-level structure of the registry is like this::

    afno_26ch_v/
    baseline_afno_26/
    gfno_26ch_sc3_layers8_tt64/
    hafno_baseline_26ch_edim512_mlp2/
    modulus_afno_20/
    sfno_73ch/
    tfno_no-patching_lr5e-4_full_epochs/


The name of the model is the folder name. Each of these folders has the
following structure::

    sfno_73ch/about.txt            # optional information (e.g. source path)
    sfno_73ch/global_means.npy
    sfno_73ch/global_stds.npy
    sfno_73ch/weights.tar          # model checkpoint
    sfno_73ch/metadata.json


The `metadata.json` file contains data necessary to use the model for forecasts::

    {
        "architecture": "sfno_73ch",
        "n_history": 0,
        "channel_set": "73var",
        "grid": "721x1440",
        "in_channels": [
            0,
            1
        ],
        "out_channels": [
            0,
            1
        ]
    }

Its schema is provided by the :py:class:`fcn_mip.schema.Model`.

The checkpoint file `weights.tar` should have a dictionary of model weights and
parameters in the `model_state` key. For backwards compatibility with FCN
checkpoints produced as of March 1, 2023 the keys should include prefixed
`module.` prefix. This checkpoint format may change in the future.

"""
import os

from fcn_mip import schema
from fcn_mip import filesystem
from config import MODEL_REGISTRY

SEPERATOR = "/"

METADATA = "metadata.json"


def list_models():
    return [os.path.basename(f) for f in filesystem.ls(MODEL_REGISTRY)]


def get_model_path(name: str):
    return MODEL_REGISTRY + SEPERATOR + name


def get_weight_path(name: str):
    return get_model_path(name) + SEPERATOR + "weights.tar"


def get_scale_path(name: str):
    return get_model_path(name) + SEPERATOR + "global_stds.npy"


def get_center_path(name: str):
    return get_model_path(name) + SEPERATOR + "global_means.npy"


def put_metadata(name: str, metadata: schema.Model):
    path = get_model_path(name)
    metadata_path = path + SEPERATOR + METADATA
    filesystem.pipe(metadata_path, metadata.json())


def get_metadata(name: str) -> schema.Model:
    path = get_model_path(name)
    metadata_path = path + SEPERATOR + METADATA
    with filesystem.open(metadata_path) as f:
        return schema.Model.parse_raw(f.read())
