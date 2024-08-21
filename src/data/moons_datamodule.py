from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning import LightningDataModule
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split


class MoonsDataModule(LightningDataModule):
    """DataModule for the Moons dataset."""

    def __init__(
        self,
        n_samples: int = 1000,
        noise: float = 0.1,
        random_state: int = 42,
        data_dir: str = "data/",
        train_val_test_split: tuple[float, float, float] = (0.6, 0.2, 0.2),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.batch_size_per_device = batch_size

    @property
    def moons_npz(self) -> Path:
        """Return the path to the moons data file."""
        return Path(self.hparams.data_dir).joinpath(
            f"moons_{self.hparams.n_samples}_{self.hparams.noise}_{self.hparams.random_state}.npz"
        )

    def prepare_data(self):
        """Prepare the moons data for training."""
        X, y = make_moons(
            n_samples=self.hparams.n_samples,
            noise=self.hparams.noise,
            random_state=self.hparams.random_state,
        )
        with self.moons_npz.open("wb") as f:
            np.savez(f, X=X, y=y)

    def setup(self, stage: str | None = None):
        """Load and split the moons data."""
        if not self.data_train and not self.data_val and not self.data_test:
            data = np.load(self.moons_npz)
            X, y = data["X"], data["y"]

            dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).int())

            n_samples = len(dataset)
            lengths = [int(split * n_samples) for split in self.hparams.train_val_test_split]

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=lengths,
                generator=torch.Generator().manual_seed(self.hparams.random_state),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create the train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return the test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
