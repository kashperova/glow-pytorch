from typing import Optional

import torch
from torch import Tensor

from model.affine_coupling.zero_conv import ZeroConv2d
from model.flow import Flow
from model.invert_block import InvertBlock
from modules.utils.gaussians import gaussian_log_density, sample_from_gaussian
from modules.utils.tensors import reverse_squeeze, squeeze


class FlowBlock(InvertBlock):
    """
    Flow block that performs several flow transformations;
    For multiscale architecture, squeeze and split applied
    to input tensor in original paper (as well as in Real-NVP);

    Notes: !!! z and x dimensionality can be different
    due to squeeze and split operations, but for each flow level (that's invertible)
    the condition of elements' total number equality (probability density)
    between x and z is satisfied (required for CoV theorem).

    attrs
    ----------
    n_flows: int
        number of flows in block

    squeeze_factor: int
        channel increase scaling factor

    split: bool
        split is applied after flow steps, if attr = True

    flows: list
        flow modules
    """

    def __init__(
        self,
        in_ch: int,
        n_flows: int,
        coupling_hidden_ch: int = 512,
        squeeze_factor: int = 2,
        split: bool = True,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.squeeze_factor = squeeze_factor
        self.n_flows = n_flows
        self.split = split
        if self.split:
            self.prior = ZeroConv2d(
                in_ch=in_ch * squeeze_factor, out_ch=in_ch * squeeze_factor**2
            )
        else:
            self.prior = ZeroConv2d(
                in_ch=in_ch * squeeze_factor * 2, out_ch=in_ch * 2 * squeeze_factor**2
            )

        self.flows: list[Flow] = []
        for _ in range(n_flows):
            self.flows.append(
                Flow(
                    in_ch=in_ch * squeeze_factor**2,
                    coupling_hidden_ch=coupling_hidden_ch,
                )
            )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        out = squeeze(x, factor=self.squeeze_factor)
        log_det_jacob = 0.0
        batch_size = x.shape[0]

        for flow in self.flows:
            out, log_det = flow(out)
            log_det_jacob += log_det

        if self.split:
            # split out on 2 parts
            out, z_new = out.chunk(2, dim=1)
            log_p = self.__get_prob_density(out, batch_size)
        else:
            # for the last level prior distribution
            # is standard gaussian (mean=0, std=1)
            zero = torch.zeros_like(out)
            log_p = self.__get_prob_density(zero, batch_size)
            z_new = out

        return out, log_det_jacob, log_p, z_new

    def reverse(
        self,
        out: Tensor,
        eps: Tensor,
        reconstruct: Optional[bool] = False,
    ) -> Tensor:
        if reconstruct:
            if self.split:
                # eps is z_new
                out = torch.cat([out, eps], dim=1)
        else:
            out = self.__sampling(out=out, eps=eps)

        for flow in self.flows[::-1]:
            out = flow.reverse(out)

        return reverse_squeeze(out, factor=self.squeeze_factor)

    def __get_prob_density(self, out: Tensor, batch_size: int) -> Tensor:
        mean, log_std = self.prior(out).chunk(2, dim=1)
        log_p = gaussian_log_density(out, mean=mean, log_std=log_std)
        log_p = log_p.view(batch_size, -1).sum(1)
        return log_p

    def __sampling(self, out: Tensor, eps: Tensor) -> Tensor:
        if self.split:
            mean, log_std = self.prior(out).chunk(2, dim=1)
            z = sample_from_gaussian(eps, mean, log_std)
            return torch.cat([out, z], dim=1)

        # Glow last level
        zero = torch.zeros_like(out)
        mean, log_sd = self.prior(zero).chunk(2, dim=1)
        return sample_from_gaussian(eps, mean, log_sd)
