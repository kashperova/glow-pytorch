import torch
from torch import nn, Tensor

from model.flow_block import FlowBlock


class ActNorm(FlowBlock):
    """
    invertible normalization  layer with params
    that initialized on the 1st forward pass;
    after 1st batch, params become trainable

    attrs
    ----------
    scale: nn.Parameter
        scales input along channels

    bias: nn.Parameter
        shifts input along channels

    initialized: bool
        indicates whether params were initialized
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, in_ch, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, in_ch, 1, 1))
        self.initialized = False

    def __init_params(self, x: Tensor):
        mean = x.mean(dim=[0, 2, 3], keepdim=True)
        std = x.std(dim=[0, 2, 3], keepdim=True)
        self.bias.data.copy_(-mean)
        self.scale.data.copy_(1 / (std + 1e-6))
        self.initialized = True

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if not self.initialized:
            self.__init_params(x)

        _, _, h, w = x.shape

        log_abs = torch.log(torch.abs(self.scale))
        log_det = h * w * torch.sum(log_abs)

        return x * self.scale + self.bias, log_det

    def reverse(self, x: Tensor) -> Tensor:
        return (x - self.bias) / self.scale
