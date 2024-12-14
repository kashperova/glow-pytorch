from torch import Tensor


def squeeze(x: Tensor, factor: int = 2) -> Tensor:
    b_size, ch, height, width = x.shape
    x = x.view(b_size, ch, height // factor, factor, width // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.contiguous().view(b_size, ch * factor**2, height // factor, width // factor)
    return x


def reverse_squeeze(x: Tensor, factor: int = 2) -> Tensor:
    b_size, ch, height, width = x.shape
    x = x.view(b_size, ch // factor**2, factor, factor, height, width)
    x = x.permute(0, 1, 4, 2, 5, 3)
    x = x.contiguous().view(b_size, ch // factor**2, height * factor, width * factor)
    return x
