from collections.abc import Sequence
from typing import Optional, Union

import torch
from lightning.pytorch import LightningDataModule
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import v2


class LitOxfordIIITPet(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.list_transforms = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.CenterCrop(self.patch_size),
                v2.Resize(self.patch_size),
                v2.ToTensor(),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def prepare_data(self):
        OxfordIIITPet(
            self.data_dir,
            split="trainval",
            download=True,
        )
        OxfordIIITPet(
            self.data_dir,
            split="test",
            download=True,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self.train_dataset, self.val_dataset = data.random_split(
                OxfordIIITPet(
                    self.data_dir,
                    split="trainval",
                    transform=self.list_transforms,
                ),
                [0.8, 0.2],
                generator=torch.Generator().manual_seed(42),
            )

        if stage == "test":
            self.test_dataset = OxfordIIITPet(
                self.data_dir,
                split="test",
                transform=self.list_transforms,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, list[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, list[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
