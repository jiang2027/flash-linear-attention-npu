import torch
import torch_npu 
import fla
from fla.ops.triton.triton_core.causal_conv1d import causal_conv1d_bwd_impl


x = torch.randn((1, 1024, 2048), dtype=torch.bfloat16)
dy = torch.randn((1, 8, 1024, 128), dtype=torch.bfloat16)
weight = torch.randn((4, 2048), dtype=torch.bfloat16)
H = 8
out = causal_conv1d_bwd_impl(x,dy,H,weight=weight)

print(out)