import os
from typing import Callable

import torch
from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm import tqdm

from model.glow import Glow
from modules.utils.sampling import get_z_list
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
        self.data_config = config.dataset

        self.train_dataset, self.test_dataset = train_test_split(
            dataset, self.train_config.train_test_split
        )
        self.train_loader = None
        self.test_loader = None

        self.z_list = get_z_list(
            glow=self.model,
            image_height=self.data_config.image_size,
            image_width=self.data_config.image_size,
            batch_size=self.train_config.n_samples,
        )

        self.z_list = [z_i.to(self.device) for z_i in self.z_list]

    def train_epoch(self) -> float:
        run_train_loss = 0.0
        for i, (images, _) in enumerate(self.train_loader):
            images = dequantize(images)
            images = images.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_func(outputs, images)
            loss.backward()
            self.optimizer.step()
            run_train_loss += loss.item()

            if i % self.train_config.sampling_iters == 0:
                self.save_sample(label=f"iter_{i}")

        return run_train_loss

    @torch.inference_mode()
    def test_epoch(self) -> float:
        run_test_loss = 0.0
        for images, _ in self.test_loader:
            images = dequantize(images)
            images = images.to(self.device)
            outputs = self.model(images)
            run_test_loss += self.loss_func(outputs, images).item()

        return run_test_loss

    def train(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.data_config.batch_size,
            num_workers=4,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.data_config.batch_size, num_workers=4
        )

        self.model = nn.DataParallel(self.model).to(self.device)

        with torch.no_grad():
            images = dequantize(next(iter(self.test_loader)))
            self.model.module(images)

        for i in tqdm(range(self.train_config.epochs)):
            train_loss = self.train_epoch()
            train_loss /= len(self.train_dataset)

            test_loss = self.test_epoch()
            test_loss /= len(self.test_dataset)

            self.save_checkpoint(epoch=i)

    def save_checkpoint(self, epoch: int):
        torch.save(
            self.model.state_dict(), f"{self.train_config.save_dir}/model_{epoch}.pt"
        )
        torch.save(
            self.optimizer.state_dict(),
            f"{self.train_config.save_dir}/optimizer_{epoch}.pt",
        )

    @torch.inference_mode()
    def save_sample(self, label: str):
        # todo: change to logging (tensorboard)
        if not os.path.exists(self.train_config.samples_dir):
            os.makedirs(self.train_config.samples_dir)

        utils.save_image(
            self.model.reverse(self.z_list).cpu().data,
            f"{self.train_config.samples_dir}/{label}.png",
            normalize=True,
            nrow=10,
            value_range=(-0.5, 0.5),
        )
