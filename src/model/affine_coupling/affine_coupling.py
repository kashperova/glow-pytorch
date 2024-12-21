import torch
from torch import Tensor
from torch.nn import functional as F

from model.affine_coupling.net import NN
from model.invert_block import InvertBlock


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
        super().__init__()
        self.net = NN(in_ch, hidden_ch)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x_a, x_b = x.chunk(2, dim=1)
        log_s, t = self.net(x_a)
        s = F.sigmoid(log_s + 2)
        log_det = torch.sum(torch.log(s).view(x.shape[0], -1), 1)
        y_b = (x_b + t) * s
        y_a = x_a
        return torch.concat([y_a, y_b], dim=1), log_det

    def reverse(self, y: Tensor) -> Tensor:
        y_a, y_b = y.chunk(2, dim=1)
        log_s, t = self.net(y_a)
        s = F.sigmoid(log_s + 2)
        x_b = y_b / s - t
        x_a = y_a
        return torch.concat([x_a, x_b], dim=1)
