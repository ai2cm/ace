# this package is installed in models/FourCastNet
from fourcastnet.networks.afnonet import AFNONet

# this package is installed in models/fcn-mip
from networks.geometric_v1.sfnonet import FourierNeuralOperatorNet


NET_REGISTRY = {
    "afno": AFNONet,  # using short acronym for backwards compatibility
    "FourierNeuralOperatorNet": FourierNeuralOperatorNet,
}
