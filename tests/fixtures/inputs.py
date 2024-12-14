import pytest
import torch

from fixtures.config import TestConfig


@pytest.fixture
def input_batch():
    return torch.randn(
        TestConfig.batch_size,
        TestConfig.in_ch,
        TestConfig.image_size,
        TestConfig.image_size,
    )
