import os

import hydra
import torch
import torch.multiprocessing as mp
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from modules.trainer import DDPTrainer, Trainer
from modules.utils.train import set_seed

OmegaConf.register_new_resolver("world_size", lambda: torch.cuda.device_count())


def train(cfg: DictConfig):
    set_seed(cfg.trainer.seed)

    if cfg.trainer.use_ddp:
        device = torch.device(f"cuda:{cfg.ddp.rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = instantiate(cfg.model).to(device)
    loss_func = instantiate(cfg.loss_func, n_bins=cfg.trainer.n_bins).to(device)
    dataset = instantiate(cfg.dataset)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(cfg.optimizer, params=trainable_params)
    lr_scheduler = instantiate(cfg.lr_scheduler, optimizer=optimizer)

    trainer_cls = DDPTrainer if cfg.trainer.use_ddp else Trainer

    trainer = trainer_cls(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=cfg,
        device=device,
        dataset=dataset,
    )

    if cfg.trainer.use_ddp:
        mp.spawn(trainer.train, nprocs=cfg.ddp.world_size, join=True)
    else:
        trainer.train()


if __name__ == "__main__":
    config_path = os.environ["HYDRA_CONFIGS_PATH"]
    config_name = os.environ["TRAIN_CONFIG"]
    hydra.initialize(config_path=config_path, version_base=None)
    config = hydra.compose(config_name=config_name)
    train(config)
