from fme.ace.data_loading.dataloader import GenericTorchDataLoader
from fme.coupled.data_loading.batch_data import CoupledBatchData


class CoupledDataLoader(GenericTorchDataLoader[CoupledBatchData]):
    pass
