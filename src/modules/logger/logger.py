import os

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, log_dir: str, run_name: str, log_steps: int):
        log_dir = os.path.join(log_dir, run_name)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.log_dir = log_dir
        self.log_steps = log_steps
        self.writer = SummaryWriter(log_dir=log_dir)

    def __del__(self):
        self.writer.flush()
        self.writer.close()

    def log_train_loss(self, loss: float, step: int):
        if step % self.log_steps == 0:
            self.writer.add_scalar("Loss/train", loss, step)

    def log_images(self, grid: Tensor, step: int):
        self.writer.add_image(tag="samples", img_tensor=grid, global_step=step)
