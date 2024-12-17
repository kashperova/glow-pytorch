import os
from contextlib import ContextDecorator

import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as TorchDDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from modules.utils.train import SizedDataset


class DDP(ContextDecorator):
    """ """

    def __init__(
        self,
        rank: int,
        world_size: int,
        train_batch_size: int,
        test_batch_size: int,
        train_dataset: SizedDataset,
        test_dataset: SizedDataset,
        pin_memory: bool = False,
        num_workers: int = 0,
        master_host: str = "localhost",
        master_port: str = "12355",
        backend: str = "nccl",
    ):
        self.rank = rank
        self.world_size = world_size
        self.datasets = {
            "train": (train_dataset, train_batch_size),
            "test": (test_dataset, test_batch_size),
        }
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.__loaders: dict[str, DataLoader] = {}

        self.__master_host = master_host
        self.__master_port = master_port
        self.__backend = backend

    def __enter__(self) -> "DDP":
        os.environ["MASTER_HOST"] = self.__master_host
        os.environ["MASTER_PORT"] = self.__master_port
        dist.init_process_group(
            self.__backend, rank=self.rank, world_size=self.world_size
        )

        for partition, (dataset, batch_size) in self.datasets.items():
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )
            self.__loaders[partition] = DataLoader(
                dataset,
                batch_size=batch_size,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
                drop_last=False,
                shuffle=False,
                sampler=sampler,
            )
        return self

    def __exit__(self, exc_type, exc_val, trace) -> bool:
        dist.destroy_process_group()
        return False

    def set_train_epoch(self, epoch: int):
        self.__loaders["train"].sampler.set_epoch(epoch)

    def get_train_loader(self) -> DataLoader:
        return self.__loaders["train"]

    def get_test_loader(self) -> DataLoader:
        return self.__loaders["test"]

    def get_model(self, model: nn.Module) -> TorchDDP:
        model = model().to(self.rank)
        return TorchDDP(
            model,
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=True,
        )
