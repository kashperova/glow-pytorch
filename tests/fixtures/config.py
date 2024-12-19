import random

import numpy as np
import pytest
import torch


class TestConfig:
    seed: int = 42

    in_ch: int = 3
    batch_size: int = 16
    image_size: int = 32
    n_flows: int = 4
    num_blocks: int = 3
    squeeze_factor: int = 2
    coupling_hidden_ch: int = 512


def set_test_seed():
    torch.manual_seed(TestConfig.seed)
    torch.cuda.manual_seed(TestConfig.seed)
    random.seed(TestConfig.seed)
    np.random.seed(TestConfig.seed)


@pytest.fixture(autouse=True)
def set_seed():
    set_test_seed()
    yield
