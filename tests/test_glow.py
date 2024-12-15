from torch import Tensor


class TestGlow:
    def test_forward(self, glow, input_batch):
        z_list, log_det_jacob, log_p_total = glow(input_batch)

        assert len(z_list) > 0, f"z_list is empty"
        assert all(
            isinstance(z, Tensor) for z in z_list
        ), f"all z_list elements should be tensors"
        assert len(z_list) == glow.num_blocks, f"len(z_list) != num_blocks"
        assert log_det_jacob != 0.0, f"log_det_jacob should be not zero"

    def test_reverse(self, glow, input_batch):
        pass
