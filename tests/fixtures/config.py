import random

import pytest
import torch


class TestConfig:
    seed: int = 42

    in_ch: int = 3
    batch_size: int = 16
    image_size: int = 32
    coupling_hidden_ch: int = 512


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(TestConfig.seed)
    torch.cuda.manual_seed(TestConfig.seed)
    random.seed(TestConfig.seed)
    yield
