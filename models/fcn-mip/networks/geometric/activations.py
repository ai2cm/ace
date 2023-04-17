import torch
from torch import nn as nn

# @torch.jit.script
# def cart_relu(z: torch.Tensor) -> torch.Tensor:
#     z = nn.functional.relu(z)
#     return z

# @torch.jit.script
# def real_relu(z: torch.Tensor) -> torch.Tensor:
#     z[..., 0] = nn.functional.relu(z[..., 0])
#     return z

# @torch.jit.script
# def mod_relu(z: torch.Tensor, b: float = 1.) -> torch.Tensor:
#     z = torch.view_as_complex(z)
#     zabs = torch.sqrt(torch.square(z.real) + torch.square(z.imag))
#     out = nn.functional.relu(zabs + 1.0) * torch.exp(1.j * z.angle())
#     # zabs = torch.sqrt(torch.square(z.real.float()) + torch.square(z.imag.float()))
#     # z = nn.functional.relu(zabs + b) * torch.exp(1.j * z.angle())
#     z = torch.view_as_real(z)
#     # zabs = torch.sqrt(torch.square(z[..., 1]) + torch.square(z[..., 0]))
#     # zmod = torch.atan2(z[..., 1], z[..., 0]) 
#     # z[..., 1] = nn.functional.relu(zabs + b) * torch.sin(zmod)
#     # z[..., 0] = nn.functional.relu(zabs + b) * torch.cos(zmod)
#     return z

class ComplexReLU(nn.Module):
    def __init__(self, negative_slope=0., mode="cartesian", bias_shape=None):
        super(ComplexReLU, self).__init__()
        
        # store parameters
        self.mode = mode
        if self.mode in ["modulus", "halfplane"]:
            if bias_shape is not None:
                self.bias = nn.Parameter(torch.ones(bias_shape, dtype=torch.float32))
            else:
                self.bias = nn.Parameter(torch.ones((1), dtype=torch.float32))
        else:
            self.bias = 0

        self.negative_slope = negative_slope
        self.act = nn.LeakyReLU(negative_slope = negative_slope)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.mode == "cartesian":
            zr = torch.view_as_real(z)
            za = self.act(zr)
            out = torch.view_as_complex(za)
        elif self.mode == "modulus":
            zabs = torch.sqrt(torch.square(z.real) + torch.square(z.imag))
            out = self.act(zabs - self.bias) * torch.exp(1.j * z.angle())
        elif self.mode == "halfplane":
            # bias is an angle parameter in this case
            modified_angle = torch.angle(z) - self.bias
            condition = torch.logical_and( (0. <= modified_angle), (modified_angle < torch.pi/2.) )
            out = torch.where(condition, z, self.negative_slope * z)
        elif self.mode == "real":
            zr = torch.view_as_real(z)
            outr = zr.clone()
            outr[..., 0] = self.act(zr[..., 0])
            out = torch.view_as_complex(outr)
        else:
            # identity
            out = z
            
        return out

class ComplexSiLU(nn.Module):
    def __init__(self, mode="cartesian", bias_shape=None):
        super(ComplexSiLU, self).__init__()
        
        # store parameters
        self.mode = mode
        if self.mode in ["modulus", "halfplane"]:
            if bias_shape is not None:
                self.bias = nn.Parameter(torch.ones(bias_shape, dtype=torch.float32))
            else:
                self.bias = nn.Parameter(torch.ones((1), dtype=torch.float32))
        else:
            self.bias = 0

        self.act = nn.SiLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.mode == "cartesian":
            zr = torch.view_as_real(z)
            za = self.act(zr)
            out = torch.view_as_complex(za)
        elif self.mode == "modulus":
            zabs = torch.sqrt(torch.square(z.real) + torch.square(z.imag))
            out = self.act(zabs - self.bias) * torch.exp(1.j * z.angle())
        elif self.mode == "real":
            zr = torch.view_as_real(z)
            outr = zr.clone()
            outr[..., 0] = self.act(zr[..., 0])
            out = torch.view_as_complex(outr)
        else:
            # identity
            out = z
            
        return out

    
class ComplexActivation(nn.Module):
    def __init__(self, activation, mode="cartesian", bias_shape=None):
        super(ComplexActivation, self).__init__()

        # store parameters
        self.mode = mode
        if self.mode == "modulus":
            if bias_shape is not None:
                self.bias = nn.Parameter(torch.zeros(bias_shape, dtype=torch.float32))
            else:
                self.bias = nn.Parameter(torch.zeros((1), dtype=torch.float32))
                
        # real valued activation
        self.act = activation
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.mode == "cartesian":
            zr = torch.view_as_real(z)
            za = self.act(zr)
            out = torch.view_as_complex(za)
        elif self.mode == "modulus":
            zabs = torch.sqrt(torch.square(z.real) + torch.square(z.imag)) 
            out = self.act(zabs + self.bias) * torch.exp(1.j * z.angle())
        else:
            # identity
            out = z 
            
        return out