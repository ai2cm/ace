import torch
from modulus.graph import Graph
from modulus.key import Key
from modulus.models.arch import FuncArch, Arch
from typing import List

# ensure torch.rand() is deterministic
_ = torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# disable tf32 for accuracy
torch.backends.cuda.matmul.allow_tf32 = False


def validate_func_arch_net(
    ref_net: Arch,
    deriv_keys: List[Key],
    validate_with_dict_forward: bool,
):
    """
    Using double precision for testing.
    """
    if validate_with_dict_forward:
        ref_net.forward = ref_net._dict_forward
    ref_graph = (
        Graph(
            [
                ref_net.make_node("ref_net", jit=False),
            ],
            ref_net.input_keys,
            deriv_keys + ref_net.output_keys,
            func_arch=False,
        )
        .double()
        .to(device)
    )

    ft_net = FuncArch(arch=ref_net, deriv_keys=deriv_keys).double().to(device)

    # check result
    batch_size = 20
    in_vars = {
        v.name: torch.rand(
            [batch_size, v.size], device=device, dtype=torch.double
        ).requires_grad_()
        for v in ref_net.input_keys
    }
    ft_out = ft_net(in_vars)
    ref_out = ref_graph(in_vars)
    for k in ref_out.keys():
        assert torch.allclose(ref_out[k], ft_out[k])

    return ft_net
