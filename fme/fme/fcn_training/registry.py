# this package is installed in models/FourCastNet
from fourcastnet.networks.afnonet import AFNONet, AFNONetParams

# this package is installed in models/fcn-mip
from networks.geometric_v1.sfnonet import (
    FourierNeuralOperatorNet,
    FourierNeuralOperatorParams,
)


NET_REGISTRY = {
    "afno": (AFNONet, AFNONetParams),  # using short acronym for backwards compatibility
    "FourierNeuralOperatorNet": (FourierNeuralOperatorNet, FourierNeuralOperatorParams),
}
