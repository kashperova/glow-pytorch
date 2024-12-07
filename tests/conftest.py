import pytest
import torch


@pytest.fixture(autouse=True)
def set_torch_seed():
    seed = 42
    torch.manual_seed(seed)
