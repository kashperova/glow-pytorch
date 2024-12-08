import torch
from torch import nn, Tensor


class FlowBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FlowBlock, self).__init__()
        self._log_abs = lambda x: torch.log(torch.abs(x))

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        raise NotImplementedError

    def reverse(self, x: Tensor):
        raise NotImplementedError
