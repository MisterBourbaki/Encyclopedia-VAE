import torch
from torch import nn

from encyclopedia_vae.models.vanilla_vae import VanillaVAE
from encyclopedia_vae.types_helpers import ForwardReturn


class ConditionalVAE(VanillaVAE):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        latent_dim: int,
        hidden_dims: list = [32, 64, 128, 256, 512],
        img_size: int = 64,
    ) -> None:
        super().__init__(
            in_channels=in_channels + 1,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            latent_dim_dec=latent_dim + num_classes,
        )

        self.img_size = img_size

        self.embed_class = nn.Linear(num_classes, img_size * img_size)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, input: torch.tensor, labels: torch.tensor) -> ForwardReturn:
        y = labels.float()
        embedded_class = self.embed_class(y)
        embedded_class = embedded_class.view(
            -1, self.img_size, self.img_size
        ).unsqueeze(1)
        embedded_input = self.embed_data(input)

        x = torch.cat([embedded_input, embedded_class], dim=1)
        latents = self.encode(x)
        mu, log_var, _ = latents.values()
        z = self.reparametrize(mu, log_var)

        z = torch.cat([z, y], dim=1)
        return ForwardReturn(
            output=self.decode(z), input=input, encoded=latents, latents=z
        )

    def sample(self, num_samples: int, labels: torch.tensor) -> torch.tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (torch.tensor)
        """
        y = labels.float()
        z = torch.randn(num_samples, self.latent_dim)

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.tensor, labels: torch.tensor) -> torch.tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.tensor) [B x C x H x W]
        :return: (torch.tensor) [B x C x H x W]
        """

        return self.forward(x, labels)["output"]
