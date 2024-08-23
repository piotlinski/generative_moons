from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import torch
import wandb
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only

from src.utils.visualize import visualize_data, visualize_latents


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, list):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self._log = log
        self._log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self._log, log_freq=self._log_freq, log_graph=True)


class VisualizationCallback(Callback):
    """Callback to store samples and outputs for visualization."""

    def __init__(self, log_every_n_epochs: int):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

        self.ready = True
        self.outputs = defaultdict(list)

    def on_sanity_check_start(self, trainer: Trainer, pl_module: LightningModule):
        self.ready = False

    def on_sanity_check_end(self, trainer: Trainer, pl_module: LightningModule):
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    def on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        stage: str = "train",
    ):
        """Store inputs and outputs for visualization."""
        if self.ready and trainer.current_epoch % self.log_every_n_epochs == 0:
            self.outputs[stage].append(outputs)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Store train examples."""
        self.on_batch_end(trainer, pl_module, outputs, batch, batch_idx, stage="train")

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ):
        """Store validation examples."""
        self.on_batch_end(trainer, pl_module, outputs, batch, batch_idx, stage="val")

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule, stage: str = "train"):
        """Cleanup cached batch and outputs."""
        if self.ready:
            self.outputs[stage] = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Cleanup cached train batch and outputs."""
        self.on_epoch_end(trainer, pl_module, "train")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Cleanup cached validation batch and outputs."""
        self.on_epoch_end(trainer, pl_module, "val")


class LogLatentsHistogram(VisualizationCallback):
    """Logs batch latents histogram to wandb."""

    @torch.no_grad()
    def parse_latents(
        self, z: dict[str, torch.Tensor], prefix: str = ""
    ) -> dict[str, wandb.Histogram]:
        """Parse latents for logging."""
        logged_latents = {}
        for name, latent in z.items():
            logged_latents[f"{prefix}{name}"] = latent.cpu().float()
        return {key: wandb.Histogram(value) for key, value in logged_latents.items()}

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule, stage: str = "train"):
        """Prepare visualization at epoch end."""
        if self.ready and trainer.current_epoch % self.log_every_n_epochs == 0:
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            z = {
                key: torch.cat([d["z"][key] for d in self.outputs[stage]], dim=0)
                for key in self.outputs[stage][0]["z"]
            }
            experiment.log(self.parse_latents(z, prefix=f"{stage}/"))

        super().on_epoch_end(trainer, pl_module, stage)


class VisualizeReconstruction(VisualizationCallback):
    """Visualize reconstructed points."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gt_done = {
            "train": False,
            "val": False,
        }

    @torch.no_grad()
    def scatterplot(self, x: torch.Tensor, y: torch.Tensor, prefix: str) -> wandb.Image:
        """Create scatterplot for visualization."""
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        fig = visualize_data(x, y, title=f"{prefix} points")
        return fig

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule, stage: str = "train"):
        """Prepare reconsutrcions visualization."""
        if self.ready and trainer.current_epoch % self.log_every_n_epochs == 0:
            logger = get_wandb_logger(trainer=trainer)
            x = torch.cat([output["x"] for output in self.outputs[stage]], dim=0)
            y = torch.cat([output["y"] for output in self.outputs[stage]], dim=0)
            x_hat = torch.cat([output["x_hat"] for output in self.outputs[stage]], dim=0)
            experiment = logger.experiment

            if not self.gt_done[stage]:
                experiment.log(
                    {f"{stage}/gt_points": self.scatterplot(x, y, prefix=f"GT {stage}")}
                )
                self.gt_done[stage] = True

            experiment.log(
                {
                    f"{stage}/reconstructed_points": self.scatterplot(
                        x_hat, y, prefix=f"Reconstructed {stage}"
                    )
                }
            )

        super().on_epoch_end(trainer, pl_module, stage)


class VisualizeLatentSpace(VisualizationCallback):
    """Visualize latent space."""

    def __init__(
        self, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state

    @torch.no_grad()
    def latent_scatterplot(
        self, z: torch.Tensor, y: torch.Tensor, train_z: torch.Tensor, prefix: str
    ) -> wandb.Image:
        """Create scatterplot for visualization."""
        z = z.cpu().numpy()
        y = y.cpu().numpy()
        train_z = train_z.cpu().numpy()
        fig = visualize_latents(
            z,
            y,
            title=f"{prefix} latent space",
            train_z=train_z,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
        )
        return fig

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule, stage: str = "train"):
        """Prepare latent space visualization."""
        if self.ready and trainer.current_epoch % self.log_every_n_epochs == 0:
            logger = get_wandb_logger(trainer=trainer)
            z = torch.cat([output["z"]["z"] for output in self.outputs[stage]], dim=0)
            y = torch.cat([output["y"] for output in self.outputs[stage]], dim=0)
            train_z = torch.cat([output["z"]["z"] for output in self.outputs["train"]], dim=0)
            experiment = logger.experiment

            experiment.log(
                {
                    f"{stage}/latent_space": self.latent_scatterplot(
                        z, y, train_z, prefix=f"{stage}"
                    )
                }
            )

        super().on_epoch_end(trainer, pl_module, stage)
