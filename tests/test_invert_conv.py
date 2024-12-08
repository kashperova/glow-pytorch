import pytest
import torch

from model.invert_conv import InvertConv


class TestInvertConv:
    in_ch: int = 3
    batch_size: int = 16
    image_size: int = 32

    @pytest.fixture
    def input_batch(self):
        return torch.randn(
            self.batch_size, self.in_ch, self.image_size, self.image_size
        )

    @pytest.fixture
    def invert_conv(self):
        return InvertConv(in_ch=self.in_ch)

    def test_forward(self, invert_conv, input_batch):
        out, _ = invert_conv(input_batch)
        assert out.shape == input_batch.shape, "out shape != input shape"

    def test_reverse(self, invert_conv, input_batch):
        out, _ = invert_conv(input_batch)
        reverse_out = invert_conv.reverse(out)
        assert torch.allclose(
            input_batch, reverse_out, atol=1e-5
        ), "shape after reverse != input shape."

    def test_log_det(self, invert_conv, input_batch):
        _, _, h, w = input_batch.shape
        out, log_det = invert_conv(input_batch)
        weights = invert_conv.get_weights()
        expected = h * w * torch.log(torch.det(weights))
        assert torch.allclose(
            log_det, expected, atol=1e-4
        ), f"log det not equal to expected value."

    def test_get_weights(self, invert_conv, input_batch):
        torch.manual_seed(42)
        weights = torch.linalg.qr(torch.rand(self.in_ch, self.in_ch))[0]
        _, _ = invert_conv(input_batch)
        reconstructed = invert_conv.get_weights()
        assert torch.allclose(
            weights, reconstructed, atol=1e-4
        ), f"weights after LU decomposition is not equal to expected value."
