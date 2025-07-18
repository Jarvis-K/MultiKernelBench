import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, bias):
        return x + bias

def get_inputs():
    x = torch.randn(4, 2048)
    bias = torch.tensor(3.1415)
    return [x, bias]

def get_init_inputs():
    return []

