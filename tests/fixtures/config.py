import random
from typing import Any

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

    dataset_size: int = 100
    n_bins: int = 256
    trainer_config: dict[str, Any] = {
        "train_test_split": 0.8,
        "train_batch_size": batch_size,
        "test_batch_size": batch_size,
        "n_bins": n_bins,
        "sampling_steps": 5,
        "n_epochs": 2,
        "n_samples": 4,
        "log_dir": "./logs",
        "run_name": "test_run",
        "log_steps": 10,
        "save_dir": "./checkpoints",
        "samples_dir": "./samples",
        "image_size": image_size,
    }


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(TestConfig.seed)
    torch.cuda.manual_seed(TestConfig.seed)
    random.seed(TestConfig.seed)
    yield
