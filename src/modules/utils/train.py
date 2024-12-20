import random
from typing import TypeVar

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split

T = TypeVar("T")


class SizedDataset(Dataset[T]):
    def __len__(self) -> int:
        ...


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train_test_split(dataset: SizedDataset, split: int) -> tuple[Subset, Subset]:
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, val_size])
    return train_data, test_data
