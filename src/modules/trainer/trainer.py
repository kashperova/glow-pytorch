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
from modules.utils.tensors import dequantize
from modules.utils.train import SizedDataset, train_test_split


class Trainer:
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
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device

        self.config = config
        self.train_config = config.trainer

        self.train_dataset, self.test_dataset = train_test_split(
            dataset, self.train_config.train_test_split
        )
        self.ddp: DDP = instantiate(
            config=self.config.ddp,
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
        )

    def train_epoch(self) -> float:
        train_loss = 0.0
        for images, _ in self.ddp.get_train_loader():
            images = dequantize(images)
            images = images.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            loss = self.loss_func(outputs, images)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        loss_tensor = torch.tensor(train_loss, device=self.ddp.rank)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

        return loss_tensor.item() / len(self.train_dataset)

    @torch.inference_mode()
    def test_epoch(self) -> float:
        test_loss = 0.0
        for images, _ in self.ddp.get_test_loader():
            images = dequantize(images)
            images = images.to(self.device)
            outputs = self.model(images)
            test_loss += self.loss_func(outputs, images).item()

        loss_tensor = torch.tensor(test_loss, device=self.ddp.rank)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

        return loss_tensor.item() / len(self.test_dataset)

    def train(self):
        with self.ddp:
            self.model = self.ddp.get_model(self.model)

            for i in tqdm(range(self.train_config.epochs)):
                self.ddp.set_train_epoch(i)
                train_loss = self.train_epoch()
                test_loss = self.test_epoch()
                print(f"Train loss: {train_loss}, Test loss: {test_loss}", flush=True)
