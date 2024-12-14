import torch

from model.actnorm import ActNorm
from model.affine_coupling import AffineCoupling
from model.invert_conv import InvertConv


class TestFlow:
    def test_forward(self, flow, input_batch):
        out, _ = flow(input_batch)
        assert out.shape == input_batch.shape, f"out shape != input shape"
        assert len(flow.layers) == 3, f"flow block contains 3 layers"
        assert isinstance(flow.layers[0], ActNorm), f"act norm should be the 1st layer"
        assert isinstance(
            flow.layers[1], InvertConv
        ), f"invert conv should be the 2nd layer"
        assert isinstance(
            flow.layers[2], AffineCoupling
        ), f"coupling should be the last layer"

    def test_reverse(self, flow, input_batch):
        out, _ = flow(input_batch)
        reverse_out = flow.reverse(out)

        assert reverse_out.shape == input_batch.shape, f"reverse shape != input shape"
        assert torch.allclose(
            input_batch, reverse_out, atol=1e-5
        ), f"batch after reverse != input batch."

    def test_log_det(self, flow, input_batch):
        out, test_log_det = flow(input_batch)

        expected_log_det = 0
        x = input_batch
        for layer in flow.layers:
            x, log_det = layer(x)
            expected_log_det += log_det

        assert (
            test_log_det == expected_log_det
        ), f"due to composition, final log det equal to sum of flow step components log det"
