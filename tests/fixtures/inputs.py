import pytest
import torch
from fixtures.config import TestConfig

from modules.utils.sampling import get_z_list


@pytest.fixture(scope="module")
def input_batch():
    return torch.randn(
        TestConfig.batch_size,
        TestConfig.in_ch,
        TestConfig.image_size,
        TestConfig.image_size,
    )


@pytest.fixture(scope="function")
def z_sample(glow):
    return get_z_list(
        glow=glow,
        image_width=TestConfig.image_size,
        image_height=TestConfig.image_size,
        batch_size=TestConfig.batch_size,
    )
