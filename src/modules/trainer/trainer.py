import logging
import os
from typing import Callable

import torch
from omegaconf import DictConfig
from PIL import Image
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm import tqdm

from model.glow import Glow
from modules.logger import TensorboardLogger
from modules.utils.sampling import get_z_list
from modules.utils.tensors import dequantize
from modules.utils.train import SizedDataset, train_test_split

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        self.train_loader = None
        self.test_loader = None
        self.logger = TensorboardLogger(
            log_dir=self.train_config.log_dir,
            run_name=self.train_config.run_name,
            log_steps=self.train_config.log_steps,
        )

        self.z_list = get_z_list(
            glow=self.model,
            image_height=self.train_config.image_size,
            image_width=self.train_config.image_size,
            batch_size=self.train_config.n_samples,
        )

        self.z_list = [z_i.to(self.device) for z_i in self.z_list]

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        run_train_loss, n_iters = 0.0, 0
        for i, images in enumerate(self.train_loader):
            images = dequantize(images, n_bins=self.train_config.n_bins)
            images = images.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_func(outputs, images)
            loss.backward()
            self.optimizer.step()
            run_train_loss += loss.item()
            n_iters += 1

            if i % self.train_config.sampling_steps == 0 and i != 0:
                self.log_samples(step=epoch + i)
                avg_loss = run_train_loss / n_iters
                self.logger.log_train_loss(loss=avg_loss, step=epoch + i)
                logger.info(f"Train avg loss: {avg_loss}")

        return run_train_loss

    @torch.inference_mode()
    def test_epoch(self) -> float:
        self.model.eval()
        run_test_loss = 0.0
        for images in self.test_loader:
            images = dequantize(images, self.train_config.n_bins)
            images = images.to(self.device)
            outputs = self.model(images)
            run_test_loss += self.loss_func(outputs, images).item()

        return run_test_loss

    def train(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_config.train_batch_size,
            num_workers=4,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.train_config.test_batch_size,
            num_workers=4,
        )

        self.model = nn.DataParallel(self.model).to(self.device)

        with torch.no_grad():
            images = dequantize(
                next(iter(self.test_loader)), n_bins=self.train_config.n_bins
            )
            images = images.to(self.device)
            self.model.module(images)

        for i in tqdm(range(self.train_config.n_epochs)):
            train_loss = self.train_epoch(i)
            train_loss /= len(self.train_dataset)

            test_loss = self.test_epoch()
            test_loss /= len(self.test_dataset)

            self.logger.log_test_loss(loss=test_loss, epoch=i + 1)
            self.lr_scheduler.step(test_loss)

            self.save_checkpoint(epoch=i)

    def save_checkpoint(self, epoch: int):
        if not os.path.exists(self.train_config.save_dir):
            os.makedirs(self.train_config.save_dir, exist_ok=True)

        torch.save(
            self.model.state_dict(), f"{self.train_config.save_dir}/model_{epoch}.bin"
        )
        torch.save(
            self.optimizer.state_dict(),
            f"{self.train_config.save_dir}/optimizer_{epoch}.bin",
        )

    @torch.inference_mode()
    def log_samples(self, step: int, save_png: bool = True):
        data = self.model.module.reverse(self.z_list).cpu().data
        grid = utils.make_grid(data, nrow=5, normalize=True, value_range=(-0.5, 0.5))
        self.logger.log_images(grid=grid, step=step)

        if save_png:
            if not os.path.exists(self.train_config.samples_dir):
                os.makedirs(self.train_config.samples_dir)

            np_array = (
                grid.mul(255)
                .add_(0.5)
                .clamp_(0, 255)
                .permute(1, 2, 0)
                .to("cpu", torch.uint8)
                .numpy()
            )
            im = Image.fromarray(np_array)
            im.save(f"{self.train_config.samples_dir}/{step}.png")
