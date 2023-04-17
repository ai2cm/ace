from functools import partial

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

# preprocessor we need too
from networks.geometric.preprocessor import Preprocessor2D

class MultiStepWrapper(nn.Module):
    def __init__(self, params, model_handle):
        super(MultiStepWrapper, self).__init__()
        self.preprocessor = Preprocessor2D(params)
        self.model = model_handle(params)
        self.checkpointing = params.checkpointing if hasattr(params, "checkpointing") else False

        # collect parameters for history
        self.n_future = params.n_future

    def _forward_train(self, inp):
        result = []
        inpt = inp
        for _ in range(self.n_future + 1):
            
            # compute output
            pred = self.model(inpt)

            if self.checkpointing:
                pred = checkpoint(lambda x: x, pred)

            result.append(pred)
            
            # postprocess: this steps removes the grid
            inpt = self.preprocessor.append_history(inpt, pred)
            
            # add back the grid
            inpt, _ = self.preprocessor(inpt)
            
        # concat the tensors along channel dim to be compatible with flattened target
        result = torch.cat(result, dim=1)
        
        return result

    def _forward_eval(self, inp):
        return self.model(inp)

    def forward(self, inp):

        if self.training:
            return self._forward_train(inp)
        else:
            return self._forward_eval(inp)

    
def get_model(params):

    model_handle = None

    if params.nettype == 'afno':
        
        if 'model_parallel_size' in params and params.model_parallel_size > 1:
            # model_handle = partial(DistributedAFNONet,
            #                        input_is_matmul_parallel=params.split_data_channels,
            #                        output_is_matmul_parallel=params.split_data_channels,
            #                        use_complex_kernels=True)
            raise NotImplementedError(f"Error, net type {params.nettype} not implemented")
        else:
            from networks.geometric.afnonet import AdaptiveFourierNeuralOperatorNet
            model_handle = partial(AdaptiveFourierNeuralOperatorNet, use_complex_kernels=True)

    elif params.nettype == 'fno' or params.nettype == 'hybrid':
        from networks.geometric.fnonet import FourierNeuralOperatorNet
        model_handle = partial(FourierNeuralOperatorNet, use_complex_kernels=True)

    elif params.nettype == 'hfno' or params.nettype == 'hafno':
        # use the Helmholtz decomposition
        from networks.geometric.hfnonet import FourierNeuralOperatorNet
        model_handle = partial(FourierNeuralOperatorNet, use_complex_kernels=True)

    elif params.nettype == 'encdec':
        # use the Helmholtz decomposition
        from networks.geometric.encdecnet import FourierNeuralOperatorNet
        model_handle = partial(FourierNeuralOperatorNet, use_complex_kernels=True)

    else:
         raise NotImplementedError(f"Error, net type {params.nettype} not implemented")

    # wrap into Multi-Step if requested
    if params.n_future > 0:
        model = MultiStepWrapper(params, model_handle)
    else:
        model = model_handle(params)


    return model
        
         
