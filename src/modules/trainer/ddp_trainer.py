from typing import Callable

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from model.glow import Glow
from modules.dataset.ddp import DDP
from modules.trainer.trainer import Trainer
from modules.utils.train import SizedDataset


class DDPTrainer(Trainer):
    def __init__(
        self,
        model: Glow,
        config: DictConfig,
        dataset: SizedDataset,
        loss_func: Callable,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        device: torch.device,
    ):
        super().__init__(
            model=model,
            config=config,
            dataset=dataset,
            loss_func=loss_func,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
        )
        self.ddp: DDP = instantiate(
            config=self.config.ddp,
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
        )

    def train_epoch(self, epoch: int) -> float:
        train_loss = super().train_epoch(epoch)

        loss_tensor = torch.tensor(train_loss, device=self.ddp.rank)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

        return loss_tensor.item() / len(self.train_dataset)

    @torch.inference_mode()
    def test_epoch(self) -> float:
        test_loss = super().test_epoch()

        loss_tensor = torch.tensor(test_loss, device=self.ddp.rank)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

        return loss_tensor.item() / len(self.test_dataset)

    def train(self):
        with self.ddp:
            self.model = self.ddp.get_model(self.model)
            self.train_loader = self.ddp.get_train_loader()
            self.test_loader = self.ddp.get_test_loader()

            for i in tqdm(range(self.train_config.epochs)):
                self.ddp.set_train_epoch(i)
                self.train_epoch(i)
                self.test_epoch()
