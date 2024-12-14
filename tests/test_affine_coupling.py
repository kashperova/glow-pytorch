import torch

from fixtures.config import TestConfig


class TestAffineCoupling:
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
        torch.manual_seed(TestConfig.seed)
        out, log_det = affine_coupling(input_batch)

        assert torch.allclose(
            input_batch, out, atol=1e-6
        ), "Identity behavior check failed"
        assert log_det == 0, "log det ~= zero for identity transformation"
