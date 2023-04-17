import torch

class GraphCastWrapper(torch.nn.Module):
    def __init__(self, model, dtype):
        super().__init__()
        self.model = model
        self.dtype = dtype
        
    def forward(self, x):
        x = x.to(self.dtype)
        y = self.model(x)
        return y
