import os

import hydra
import torch
import torch.multiprocessing as mp
from hydra.utils import instantiate
from omegaconf import OmegaConf

from modules.trainer import Trainer
from modules.utils.train import set_seed

OmegaConf.register_new_resolver("world_size", lambda: torch.cuda.device_count())


@hydra.main(
    version_base=None,
    config_path=os.environ["HYDRA_CONFIGS_PATH"],
    config_name=os.environ["TRAIN_CONFIG"],
)
def train(cfg):
    set_seed(cfg.trainer.seed)

    device = torch.device(f"cuda:{cfg.ddp.rank}")

    model = instantiate(cfg.model).to(device)
    loss_func = instantiate(cfg.loss_func).to(device)
    dataset = instantiate(cfg.dataset)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(cfg.optimizer, params=trainable_params)
    lr_scheduler = instantiate(cfg.lr_scheduler, optimizer=optimizer)

    trainer = Trainer(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=cfg,
        device=device,
        dataset=dataset,
    )

    mp.spawn(trainer.train, nprocs=cfg.ddp.world_size, join=True)


if __name__ == "__main__":
    train()
