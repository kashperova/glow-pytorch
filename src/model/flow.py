from torch import Tensor, nn

from model.actnorm import ActNorm
from model.affine_coupling import AffineCoupling
from model.invert_block import InvertBlock
from model.invert_conv import InvertConv


class Flow(InvertBlock):
    """
    One Flow step that consists of an act norm step,
    followed by an invertible 1 × 1 convolution,
    followed by an affine transformation.

    attrs
    ----------
    layers: nn.Sequential
        sequence of flow step components
    """

    def __init__(self, in_ch: int, coupling_hidden_ch: int):
        super().__init__(in_ch)
        self.layers = nn.Sequential(
            ActNorm(in_ch=in_ch),
            InvertConv(in_ch=in_ch),
            AffineCoupling(in_ch=in_ch, hidden_ch=coupling_hidden_ch),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        log_det_jacob = 0
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_jacob = log_det_jacob + log_det

        return x, log_det_jacob

    def reverse(self, x: Tensor) -> Tensor:
        for layer in self.layers[::-1]:
            x = layer.reverse(x)

        return x
