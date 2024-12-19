import numpy as np
import torch
from scipy import linalg
from torch import Tensor, nn
from torch.nn import functional as F

from model.invert_block import InvertBlock
from modules.utils.tensors import log_abs


class InvertConv(InvertBlock):
    """
    invertible conv layer where weights
    are obtained by LU decomposition;
    so determinant's calculation has not cubic,
    but linear complexity.

    attrs (trainable)
    ----------
    ut_matrix: nn.Parameter
        upper triangular matrix (U) after LU decomposition;
        contains elements only above the diagonal.

    lt_matrix: nn.Parameter
        lower triangular matrix (L) after LU decomposition;
        contains elements only below the diagonal.

    s_vector: nn.Parameter
        contains log of U diagonal abs values;
    """

    def __init__(self, in_ch: int):
        super().__init__()
        weight_matrix = np.random.randn(in_ch, in_ch)
        q, _ = linalg.qr(weight_matrix)
        perm_matrix, lt_matrix, ut_matrix = linalg.lu(q.astype(np.float32))

        perm_matrix = torch.from_numpy(perm_matrix)
        lt_matrix = torch.from_numpy(lt_matrix)
        ut_matrix = torch.from_numpy(ut_matrix)

        self.register_buffer("perm_matrix", perm_matrix)

        self.ut_matrix = nn.Parameter(ut_matrix.triu(1))
        u_mask = torch.triu(torch.ones_like(self.ut_matrix), 1)
        self.register_buffer("u_mask", u_mask)

        self.lt_matrix = nn.Parameter(lt_matrix)
        l_mask = u_mask.T
        # lt_matrix is only lower triangular part of L matrix without diagonal elements,
        # so we need to store diagonal in buffer for W reconstruction
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.register_buffer("l_mask", l_mask)

        s_diag = torch.diag(ut_matrix)
        self.s_vector = nn.Parameter(log_abs(s_diag + 1e-8))
        self.register_buffer("s_sign", torch.sign(s_diag))

    def get_weights(self) -> Tensor:
        # reconstruct L and U decomposition matrices
        l_matrix = self.lt_matrix * self.l_mask + self.l_eye
        u_matrix = self.ut_matrix * self.u_mask + torch.diag(
            self.s_sign * torch.exp(self.s_vector)
        )
        weights = self.perm_matrix @ l_matrix @ u_matrix

        return weights

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        _, _, h, w = x.shape
        weights = self.get_weights()
        log_det = h * w * torch.sum(self.s_vector)
        out = F.conv2d(x, weights.unsqueeze(2).unsqueeze(3))
        return out, log_det

    def reverse(self, x: Tensor) -> Tensor:
        weights = self.get_weights()
        # since W is an orthogonal matrix, W^(-1) = W.T
        out = F.conv2d(x, weights.inverse().unsqueeze(2).unsqueeze(3))
        return out
