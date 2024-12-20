from torch import Tensor, nn

from model.affine_coupling.zero_conv import ZeroConv2d


class NN(nn.Module):
    """
    Simple neural net for affine coupling layer;
    Returns log scale and translation factor.

    In original paper net had 3 conv layers,
    where 2 hidden layers have ReLU activation
    and 512 channels;

    The 1st and last convolutions are 3 × 3,
    while the center conv is 1 × 1,
    since both its input and output have a large number of channels,
    in contrast with the first and last convolution.
    """

    def __init__(self, in_ch: int, hidden_ch: int):
        super(NN, self).__init__()
        conv1 = nn.Conv2d(in_ch // 2, hidden_ch, 3, padding=1)
        conv2 = nn.Conv2d(hidden_ch, hidden_ch, 1)
        self.net = nn.Sequential(
            conv1,
            nn.ReLU(inplace=True),
            conv2,
            nn.ReLU(inplace=True),
            ZeroConv2d(hidden_ch, in_ch),
        )
        self.__init_conv()

    def __init_conv(self):
        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        log_s, t = self.net(x).chunk(2, dim=1)
        return log_s, t
