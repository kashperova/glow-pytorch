from math import log, pi

import torch
from torch import Tensor


def gaussian_log_density(x: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    return -0.5 * log(2 * pi) - log_std - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_std)


def sample_from_gaussian(eps: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    return mean + torch.exp(log_std) * eps
