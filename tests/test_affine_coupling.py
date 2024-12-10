import pytest
import torch

from model.affine_coupling import AffineCoupling


class TestAffineCoupling:
    in_ch: int = 3
    hidden_ch: int = 512
    batch_size: int = 16
    image_size: int = 32

    @pytest.fixture
    def input_batch(self):
        return torch.randn(
            self.batch_size, self.in_ch, self.image_size, self.image_size
        )

    @pytest.fixture
    def affine_coupling(self):
        return AffineCoupling(in_ch=self.in_ch, hidden_ch=self.hidden_ch)

    def test_forward(self, affine_coupling, input_batch):
        out, _ = affine_coupling(input_batch)
        assert out.shape == input_batch.shape, f"out shape != input shape"

    def test_reverse(self, affine_coupling, input_batch):
        out, _ = affine_coupling(input_batch)
        reverse_out = affine_coupling.reverse(out)

        assert torch.allclose(
            input_batch, reverse_out, atol=1e-5
        ), f"batch after reverse != input batch."

    def test_log_det(self, affine_coupling, input_batch):
        torch.manual_seed(42)
        out, log_det = affine_coupling(input_batch)

        assert torch.allclose(
            input_batch, out, atol=1e-6
        ), "Identity behavior check failed"
        assert log_det == 0, "log det ~= zero for identity transformation"
