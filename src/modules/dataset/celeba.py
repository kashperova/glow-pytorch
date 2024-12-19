import os
from typing import Optional

from natsort import natsorted
from PIL import Image

from modules.utils.train import SizedDataset


class CelebaDataset(SizedDataset):
    def __init__(self, root: str, transform: Optional):
        files = os.listdir(root)
        self.root = root
        self.transform = transform
        self.files = natsorted(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = os.path.join(self.root, self.files[idx])
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image
