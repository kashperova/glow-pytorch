import random

import pytest
import torch


@pytest.fixture(autouse=True)
def set_seed():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    yield
