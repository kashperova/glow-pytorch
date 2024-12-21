import torch

from model.actnorm import ActNorm
from model.affine_coupling import AffineCoupling
from model.invert_conv import InvertConv


class TestFlow:
    def test_forward(self, flow, flow_input_batch):
        out, _ = flow(flow_input_batch)
        assert out.shape == flow_input_batch.shape, "out shape != input shape"
        assert len(flow.layers) == 3, "flow block contains 3 layers"
        assert isinstance(flow.layers[0], ActNorm), "act norm should be the 1st layer"
        assert isinstance(
            flow.layers[1], InvertConv
        ), "invert conv should be the 2nd layer"
        assert isinstance(
            flow.layers[2], AffineCoupling
        ), "coupling should be the last layer"

    def test_reverse(self, flow, flow_input_batch):
        out, _ = flow(flow_input_batch)
        reverse_out = flow.reverse(out)

        assert (
            reverse_out.shape == flow_input_batch.shape
        ), "reverse shape != input shape"
        assert torch.allclose(
            flow_input_batch, reverse_out, atol=1e-5
        ), "batch after reverse != input batch."

    def test_log_det(self, flow, flow_input_batch):
        out, test_log_det = flow(flow_input_batch)

        expected_log_det = 0
        x = flow_input_batch
        for layer in flow.layers:
            x, log_det = layer(x)
            expected_log_det = expected_log_det + log_det

        assert torch.equal(
            test_log_det, expected_log_det
        ), "due to composition, final log det equal to sum of flow step components log det"
