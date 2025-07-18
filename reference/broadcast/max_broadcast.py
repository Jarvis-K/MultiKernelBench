import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.maximum(a, b)

def get_inputs():
    a = torch.randn(16, 1, 1024)
    b = torch.randn(1, 256, 1)
    return [a, b]

def get_init_inputs():
    return []