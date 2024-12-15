from torch import Tensor

from model.flow_block import FlowBlock
from model.invert_block import InvertBlock


class Glow(InvertBlock):
    """
    Glow model that contains L blocks with K flows in each;
    Each block to L-1 performs squeeze, K flow steps and split operations;
    Only last Lth block performs squeeze and K flow steps (w/o split)
    (Figure 2b in original paper).

    z has hierarchical structure (to reduce computational and memory recourses);
    At each level, half of the data dimension is excluded from the current processing
    and stored separately in z_list (more details in section 3.6, Real-NVP paper);
    For this purpose, the split operation is used.

    n_flows: int
        K in original paper
    num_blocks: int
        L in original paper
    """

    def __init__(
        self,
        in_ch: int,
        n_flows: int,
        num_blocks: int,
        coupling_hidden_ch: int = 512,
        squeeze_factor: int = 2,
    ):
        super(Glow, self).__init__()
        self.n_flows = n_flows
        self.num_blocks = num_blocks
        self.squeeze_factor = squeeze_factor
        self.blocks: list[FlowBlock] = []

        for _ in range(self.num_blocks - 1):
            self.blocks.append(
                FlowBlock(
                    in_ch=in_ch,
                    n_flows=n_flows,
                    coupling_hidden_ch=coupling_hidden_ch,
                    squeeze_factor=squeeze_factor,
                )
            )
            in_ch *= 2  # required for squeeze

        self.blocks.append(
            FlowBlock(
                in_ch=in_ch,
                n_flows=n_flows,
                coupling_hidden_ch=coupling_hidden_ch,
                squeeze_factor=squeeze_factor,
                split=False,
            )
        )

    def forward(self, x: Tensor) -> tuple[list[Tensor], float, float]:
        log_det_jacob = 0.0
        log_p_total = 0.0
        z_list = []
        for block in self.blocks:
            x, log_det, log_p, z_new = block(x)
            log_det_jacob += log_det
            log_p_total += log_p
            z_list.append(z_new)

        return z_list, log_det_jacob, log_p_total

    def reverse(self, z_list: list[Tensor], reconstruct: bool = False) -> Tensor:
        # last Glow level (flow step w/o split)
        x = self.blocks[-1].reverse(
            out=z_list[-1], eps=z_list[-1], reconstruct=reconstruct
        )

        for i, block in enumerate(self.blocks[::-1][1:]):
            x = block.reverse(out=x, eps=z_list[-(i + 1)], reconstruct=reconstruct)

        return x
