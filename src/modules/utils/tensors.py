import math

import torch
from torch import Tensor


def log_abs(x: Tensor) -> Tensor:
    return torch.log(torch.abs(x))


def squeeze(x: Tensor, factor: int = 2) -> Tensor:
    b_size, ch, height, width = x.shape
    x = x.view(b_size, ch, height // factor, factor, width // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.contiguous().view(b_size, ch * factor**2, height // factor, width // factor)
    return x


def reverse_squeeze(x: Tensor, factor: int = 2) -> Tensor:
    b_size, ch, height, width = x.shape
    x = x.view(b_size, ch // factor**2, factor, factor, height, width)
    x = x.permute(0, 1, 4, 2, 5, 3)
    x = x.contiguous().view(b_size, ch // factor**2, height * factor, width * factor)
    return x


def dequantize(x: Tensor, n_bins: int = 256) -> Tensor:
    x = x * 255
    n_bits = math.log(n_bins, 2)

    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))

    x = x / n_bins - 0.5
    x = x + torch.rand_like(x) / n_bins
    return x
