import torch
from torch import Tensor

from model.affine_coupling.net import NN
from model.invert_block import InvertBlock
from modules.utils.tensors import log_abs


class AffineCoupling(InvertBlock):
    """
    Affine coupling layer is reverse function
    that split input tensor on 2 parts,
    performs affine transformation on 1st part,
    using the 2nd part as input;
    the 2nd part remains unchanged in final concatenation,
    but it's used to parameterize the transformation.

    attrs
    ----------
    net: NN
        simple net for nonlinear mapping;
        as last layer use initially zero convolution
        to perform identity transformation.
    """

    def __init__(self, in_ch: int, hidden_ch: int):
        super(AffineCoupling, self).__init__()
        self.net = NN(in_ch, hidden_ch)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x_a, x_b = x.chunk(2, dim=1)
        log_s, t = self.net(x_b)
        s = torch.exp(log_s)
        log_det = torch.sum(log_abs(s))
        y_a = x_a * s + t
        y_b = x_b
        return torch.concat([y_a, y_b], dim=1), log_det

    def reverse(self, y: Tensor) -> Tensor:
        y_a, y_b = y.chunk(2, dim=1)
        log_s, t = self.net(y_b)
        s = torch.exp(log_s)
        x_a = (y_a - t) / s
        x_b = y_b
        return torch.concat([x_a, x_b], dim=1)
