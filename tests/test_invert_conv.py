import torch
from fixtures.config import TestConfig


class TestInvertConv:
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
        ), "log det not equal to expected value."

    def test_get_weights(self, invert_conv, input_batch):
        torch.manual_seed(TestConfig.seed)
        weights = torch.linalg.qr(torch.rand(TestConfig.in_ch, TestConfig.in_ch))[0]
        _, _ = invert_conv(input_batch)
        reconstructed = invert_conv.get_weights()
        assert torch.allclose(
            weights, reconstructed, atol=1e-3
        ), "weights after LU decomposition is not equal to expected value."
