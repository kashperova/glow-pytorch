from unittest.mock import MagicMock

import pytest
import torch
from fixtures.config import TestConfig
from omegaconf import OmegaConf
from torch.optim import Adam

from modules.trainer import Trainer
from modules.utils.losses import GlowLoss
from modules.utils.train import SizedDataset


class MockDataset(SizedDataset):
    def __init__(self, data_size: int, data_shape: tuple):
        self.data = torch.randn(data_size, *data_shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.fixture(scope="session")
def trainer(glow):
    dataset = MockDataset(
        TestConfig.dataset_size,
        (TestConfig.in_ch, TestConfig.image_size, TestConfig.image_size),
    )
    loss_func = GlowLoss(n_bins=TestConfig.n_bins)
    optimizer = Adam(glow.parameters(), lr=1e-3)
    hydra_cfg = OmegaConf.create({"trainer": TestConfig.trainer_config})
    lr_scheduler = MagicMock()
    device = torch.device("cpu")

    trainer = Trainer(
        model=glow,
        config=hydra_cfg,
        dataset=dataset,
        loss_func=loss_func,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
    )

    return trainer
