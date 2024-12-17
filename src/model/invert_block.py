import torch
from torch import nn


class InvertBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super(InvertBlock, self).__init__()
        self._log_abs = lambda x: torch.log(torch.abs(x))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def reverse(self, *args, **kwargs):
        raise NotImplementedError
