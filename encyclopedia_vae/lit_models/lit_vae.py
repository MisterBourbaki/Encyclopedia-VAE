# import os

import lightning.pytorch as pl
import torch

# import torchvision.utils as vutils
from torch import optim

from encyclopedia_vae.models.vanilla_vae import VanillaVAE


class LitVAE(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: list = [32, 64, 128, 256, 512],
        mid_inflate: int = 2,
        mid_dim: int = 512 * 4,
        lr: float = 0.005,
        weight_decay: float = 0.0,
        scheduler_gamma: float = 0.95,
        kld_weight: float = 0.00025,
    ) -> None:
        super().__init__()

        self.model = VanillaVAE(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            mid_inflate=mid_inflate,
            mid_dim=mid_dim,
        )
        self.kld_weight = kld_weight
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, input: torch.tensor) -> torch.tensor:
        return self.model(input)

    def training_step(self, batch, batch_idx):
        real_img, _ = batch

        results = self.forward(real_img)
        train_loss = self.model.loss(
            results,
            kld_weight=self.kld_weight,
        )

        self.log_dict(
            {key: val.item() for key, val in train_loss.items()}, prog_bar=True
        )

        return train_loss["loss"]

    def validation_step(self, batch, batch_idx):
        real_img, _ = batch
        results = self.forward(real_img)
        val_loss = self.model.loss(results, kld_weight=1.0)

        self.log_dict(
            {f"val_{key}": val.item() for key, val in val_loss.items()}, prog_bar=True
        )

    # def on_validation_epoch_end(self) -> None:
    #     self.sample_images()

    # def sample_images(self):
    #     # Get sample reconstruction image
    #     test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
    #     test_input = test_input.to(self.curr_device)
    #     test_label = test_label.to(self.curr_device)

    #     #         test_input, test_label = batch
    #     recons = self.model.generate(test_input, labels=test_label)
    #     vutils.save_image(
    #         recons.data,
    #         os.path.join(
    #             self.logger.log_dir,
    #             "Reconstructions",
    #             f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png",
    #         ),
    #         normalize=True,
    #         nrow=12,
    #     )

    #     try:
    #         samples = self.model.sample(144, self.curr_device, labels=test_label)
    #         vutils.save_image(
    #             samples.cpu().data,
    #             os.path.join(
    #                 self.logger.log_dir,
    #                 "Samples",
    #                 f"{self.logger.name}_Epoch_{self.current_epoch}.png",
    #             ),
    #             normalize=True,
    #             nrow=12,
    #         )
    #     except Warning:
    #         pass

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
