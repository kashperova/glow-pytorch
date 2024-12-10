import torch
from torch import nn, Tensor
from torch.nn import functional as F


class ZeroConv2d(nn.Module):
    """
    Zero convolutional layer

    as mentioned in the paper, when last conv layer
    is initialized with zeros, affine coupling firstly
    performs only an identity transformation.
    It's better for stable training.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super(ZeroConv2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        # padding to not change shape
        out = F.pad(x, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 2)
        return out
