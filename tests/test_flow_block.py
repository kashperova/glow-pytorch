import torch

from modules.utils.tensors import squeeze


class TestFlowBlock:
    def test_forward(self, flow_block, last_flow_block, input_batch):
        squeezed = squeeze(input_batch)
        squeezed_split, _ = squeezed.chunk(2, dim=1)
        out, _, _, _ = flow_block(input_batch)
        assert (
            out.shape == squeezed_split.shape
        ), f"out shape {out.shape} != input squeezed and split shape {squeezed_split.shape}"

        # test last flow block (w/o split)
        out, _, _, _ = last_flow_block(input_batch)
        assert out.shape == squeezed.shape, "out shape != input squeezed shape"

    def test_reconstruct(self, flow_block, last_flow_block, input_batch):
        out, _, _, _ = flow_block(input_batch)
        x = flow_block.reverse(out, eps=out, reconstruct=True)
        assert x.shape == input_batch.shape, "reconstructed shape != input shape"

        # test last flow block (w/o split)
        out, _, _, _ = last_flow_block(input_batch)
        x = last_flow_block.reverse(out, eps=out, reconstruct=True)
        assert x.shape == input_batch.shape, "reconstructed shape != input shape"

    def test_sampling(self, flow_block, last_flow_block, input_batch):
        out, _, _, _ = flow_block(input_batch)
        s = flow_block.reverse(out, eps=out, reconstruct=False)
        assert s.shape == input_batch.shape, "sampled shape != input shape"

        # test last flow block (w/o split)
        out, _, _, _ = last_flow_block(input_batch)
        s = last_flow_block.reverse(out, eps=out, reconstruct=False)
        assert s.shape == input_batch.shape, "sampled shape != input shape"

    def test_log_det(self, flow_block, last_flow_block, input_batch):
        out, test_log_det, log_p, z_new = flow_block(input_batch)
        x = squeeze(input_batch)

        expected_log_det = 0
        for f in flow_block.flows:
            x, log_det = f(x)
            expected_log_det = expected_log_det + log_det

        assert torch.equal(
            test_log_det, expected_log_det
        ), "due to composition, final log det equal to sum of each flow log det"

    def test_log_p(self, flow_block, input_batch):
        _, _, log_p, _ = flow_block(input_batch)

        assert log_p.ndim == 1, f"expected log_p to be 1D, got {log_p.ndim}D"
        assert log_p.shape[0] == input_batch.shape[0], "log_p batch size mismatch"
        assert torch.all(torch.isfinite(log_p)), "log_p contains infinite values"
