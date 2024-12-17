import torch
from torch import Tensor


class TestGlow:
    def test_forward(self, glow, input_batch):
        z_list, log_det_jacob, log_p_total = glow(input_batch)

        assert len(z_list) > 0, "z_list is empty"
        assert all(
            isinstance(z, Tensor) for z in z_list
        ), "all z_list elements should be tensors"
        assert len(z_list) == glow.num_blocks, "len(z_list) != num_blocks"
        assert log_det_jacob != 0.0, "log_det_jacob should be not zero"

    def test_reconstruct(self, glow, input_batch):
        z_list, _, _ = glow(input_batch)
        x = glow.reverse(z_list, reconstruct=True)
        assert input_batch.shape == x.shape, "reconstruct shape != input shape"
        assert torch.allclose(
            input_batch, x, atol=1e-4
        ), "batch after reverse != input batch."

    def test_sampling(self, glow, z_sample, input_batch):
        z_list, _, _ = glow(input_batch)
        for z1, z2 in zip(z_list, z_sample):
            assert z1.shape == z2.shape, "get_z_list contains errors"

        x = glow.reverse(z_sample, reconstruct=False)
        assert input_batch.shape == x.shape, "sampled shape != input shape"
