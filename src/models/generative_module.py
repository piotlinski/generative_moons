import torch
from lightning import LightningModule
from lightning.pytorch.utilities import types
from torchmetrics import MeanMetric, MeanSquaredError, MinMetric


class GenerativeLitModule(LightningModule):
    """LightningModule for training a generative model."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        """
        :param model: generative model to train
        :param loss: loss function to use
        :param optimizer: optimizer used for training
        :param scheduler: learning rate scheduler used for training
        """
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["model", "loss"])

        self.model = model

        self.loss_fn = loss

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_mse_best = MinMetric()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.

        :param x: input tensor
        :param y: conditional tensor
        :return: output of the model
        """
        return self.model(x=x, y=y.unsqueeze(-1))

    def on_train_start(self):
        """Lightning hook that is called when training begins."""
        self.val_loss.reset()
        self.val_mse.reset()
        self.val_mse_best.reset()

    def model_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Perform a single model step on a batch of data.

        :param batch: input data
        :return: loss value
        """
        x, y = batch
        x_hat = self(x, y)

        loss = self.loss_fn(x_hat, x)

        return loss, x_hat, x, y

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """LightningModule training step.

        :param batch: input data
        :param batch_idx: index of the batch
        :return: loss value
        """
        loss, x_hat, x, y = self.model_step(batch)

        z = {}
        if isinstance(x_hat, tuple):
            x_hat, z = x_hat

        if isinstance(loss, tuple):
            loss, components = loss
            for key, value in components.items():
                self.log(f"train/{key}", value, on_step=False, on_epoch=True, prog_bar=False)

        self.train_loss(loss)
        self.train_mse(x_hat, x)

        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=False)

        return {
            "loss": loss,
            "z": z,
            "x": x,
            "y": y,
            "x_hat": x_hat,
        }

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """LightningModule validation step.

        :param batch: input data
        :param batch_idx: index of the batch
        """
        loss, x_hat, x, y = self.model_step(batch)

        z = {}
        if isinstance(x_hat, tuple):
            x_hat, z = x_hat

        if isinstance(loss, tuple):
            loss, components = loss
            for key, value in components.items():
                self.log(f"val/{key}", value, on_step=False, on_epoch=True, prog_bar=False)

        self.val_loss(loss)
        self.val_mse(x_hat, x)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=False)

        return {
            "loss": loss,
            "z": z,
            "x": x,
            "y": y,
            "x_hat": x_hat,
        }

    def on_validation_epoch_end(self):
        """Update the best validation metric."""
        mse = self.val_mse.compute()
        self.val_mse_best(mse)
        self.log("val/mse_best", self.val_mse_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """LightningModule test step.

        :param batch: input data
        :param batch_idx: index of the batch
        """
        loss, x_hat, x, y = self.model_step(batch)

        if isinstance(x_hat, tuple):
            x_hat, z = x_hat

        if isinstance(loss, tuple):
            loss, components = loss
            for key, value in components.items():
                self.log(f"test/{key}", value, on_step=False, on_epoch=True, prog_bar=False)

        self.test_loss(loss)
        self.test_mse(x_hat, x)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self) -> types.OptimizerLRScheduler:
        """Setup optimizer and LR scheduler."""
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
