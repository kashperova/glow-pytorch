from typing import Optional

from torchvision import datasets

from modules.utils.train import SizedDataset


class DogsDataset(SizedDataset):
    def __init__(self, root: str, transform: Optional):
        self.dataset = datasets.OxfordIIITPet(
            root=root,
            split="trainval",
            target_types="category",
            download=True,
            transform=transform,
        )
        self.dog_indices = [
            i for i, label in enumerate(self.dataset._labels) if label == 2
        ]

    def __len__(self):
        return len(self.dog_indices)

    def __getitem__(self, idx: int):
        dog_idx = self.dog_indices[idx]
        image, label = self.dataset[dog_idx]
        return image, label
