import torch
from torch import Tensor

from model.glow import Glow


def get_z_list(
    glow: Glow,
    image_height: int,
    image_width: int,
    batch_size: int,
    temperature: float = 0.7,
) -> list[Tensor]:
    z_list = []
    for i, block in enumerate(glow.blocks):
        # because of squeeze
        image_height //= glow.squeeze_factor
        image_width //= glow.squeeze_factor

        # because of split
        num_ch = (
            block.in_ch * glow.squeeze_factor
            if i != glow.num_blocks - 1
            else block.in_ch * glow.squeeze_factor * 2
        )
        z_i = torch.randn(batch_size, num_ch, image_height, image_width) * temperature
        z_list.append(z_i)

    return z_list
