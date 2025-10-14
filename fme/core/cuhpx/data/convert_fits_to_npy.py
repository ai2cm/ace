import pathlib

import numpy as np
from astropy.io import fits

DIR = pathlib.Path(__file__).parent


def read_ring_weight(path: pathlib.Path):
    with fits.open(path) as inp:
        inp_hdu = inp[1]
        weight = inp_hdu.data.field(0)
        weight = weight.flatten()

    return weight


if __name__ == "__main__":
    files = DIR.glob("weight_ring_n*.fits")
    for file in files:
        weight = read_ring_weight(file)
        np.save(file.with_suffix(".npy"), weight)
