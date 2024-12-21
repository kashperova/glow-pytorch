from unittest.mock import MagicMock, patch

from torch import nn
from torch.utils.data import DataLoader


class TestTrainer:
    def test_init(self, trainer):
        assert trainer.model is not None
        assert trainer.train_loader is None
        assert trainer.test_loader is None
        assert trainer.logger is not None
        assert len(trainer.z_list) == trainer.model.num_blocks
        assert trainer.z_list[0].shape[0] == trainer.train_config.n_samples

    def test_train_epoch(self, trainer):
        trainer.train_loader = DataLoader(
            trainer.train_dataset, batch_size=trainer.train_config.train_batch_size
        )
        loss = trainer.train_epoch(epoch=1)
        assert isinstance(loss, float)

    def test_test_epoch(self, trainer):
        trainer.test_loader = DataLoader(
            trainer.test_dataset, batch_size=trainer.train_config.test_batch_size
        )
        loss = trainer.test_epoch()
        assert isinstance(loss, float)

    @patch("modules.trainer.trainer.os.makedirs")
    @patch("modules.trainer.trainer.torch.save")
    def test_save_checkpoint(self, mock_torch_save, mock_makedirs, trainer):
        trainer.save_checkpoint(epoch=1)
        mock_makedirs.assert_called_with(trainer.train_config.save_dir, exist_ok=True)
        assert mock_torch_save.call_count == 2

    @patch("PIL.Image.Image.save")
    def test_log_samples(self, mock_image_save, trainer):
        trainer.model = nn.DataParallel(trainer.model).to(trainer.device)
        trainer.logger = MagicMock()

        trainer.log_samples(step=1)
        mock_image_save.assert_called_with(f"{trainer.train_config.samples_dir}/1.png")
        trainer.logger.log_images.assert_called_once_with(
            grid=trainer.logger.log_images.call_args.kwargs["grid"], step=1
        )

    def test_train(self, trainer):
        trainer.train_epoch = MagicMock(return_value=0.45)
        trainer.test_epoch = MagicMock(return_value=0.5)
        trainer.lr_scheduler.step = MagicMock()
        trainer.save_checkpoint = MagicMock()

        trainer.train()

        assert trainer.train_epoch.call_count == trainer.train_config.n_epochs
        assert trainer.test_epoch.call_count == trainer.train_config.n_epochs
        trainer.lr_scheduler.step.assert_called()
        trainer.save_checkpoint.assert_called()
