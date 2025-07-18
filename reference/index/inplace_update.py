import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x, idx, value):
        x[idx] = value
        return x

def get_inputs():
    x = torch.zeros(10000, 1024)
    idx = torch.randint(0, 10000, (2048,))
    value = torch.randn(2048, 1024)
    return [x, idx, value]

def get_init_inputs():
    return []
