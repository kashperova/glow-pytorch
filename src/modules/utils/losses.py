from math import log

from torch import Tensor, nn


class GlowLoss(nn.Module):
    def __init__(self, n_bins: int):
        super().__init__()
        self.n_bins = n_bins

    def forward(
        self, outputs: tuple[list[Tensor], Tensor, Tensor], inputs: Tensor
    ) -> Tensor:
        z_list, log_det, log_p = outputs
        n_pixel = inputs.numel() // inputs.shape[0]
        loss = -log(self.n_bins) * n_pixel
        loss = loss + log_det + log_p

        return (-loss / (log(2) * n_pixel)).mean()
