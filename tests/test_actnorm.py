import torch


class TestActNorm:
    def test_init(self, act_norm, input_batch):
        assert not act_norm.initialized, "initialized before the 1st forward pass"
        act_norm(input_batch)
        assert act_norm.initialized, "init flag not set after 1st forward pass."
        assert not torch.allclose(
            act_norm.scale, torch.ones_like(act_norm.scale)
        ), "scale equals to ones."
        assert not torch.allclose(
            act_norm.bias, torch.zeros_like(act_norm.bias)
        ), "bias equals to zeros."

    def test_forward(self, act_norm, input_batch):
        out, _ = act_norm(input_batch)
        assert out.shape == input_batch.shape, "out shape != input shape"

    def test_reverse(self, act_norm, input_batch):
        out, _ = act_norm(input_batch)
        reverse_out = act_norm.reverse(out)
        assert torch.allclose(
            input_batch, reverse_out, atol=1e-5
        ), "batch after reverse != input batch."

    def test_norm_mean(self, act_norm, input_batch):
        out, _ = act_norm(input_batch)
        # compute mean per channel
        mean = out.mean(dim=[0, 2, 3])
        assert torch.allclose(
            mean, torch.zeros_like(mean), atol=1e-3
        ), f"channels means after norm should be ~0, got {mean.tolist()}"

    def test_norm_std(self, act_norm, input_batch):
        out, _ = act_norm(input_batch)
        # compute std per channel
        std = out.std(dim=[0, 2, 3])
        assert torch.allclose(
            std, torch.ones_like(std), atol=1e-5
        ), f"channels stds after norm should be ~1, got {std.tolist()}"

    def test_log_det(self, act_norm, input_batch):
        out, log_det = act_norm(input_batch)
        _, _, h, w = input_batch.shape
        expected = h * w * torch.sum(torch.log(torch.abs(act_norm.scale)))
        assert log_det == expected, "log det not equal to expected value."
