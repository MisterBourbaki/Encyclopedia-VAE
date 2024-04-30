import torch
from einops.layers.torch import Rearrange
from torch import nn

from encyclopedia_vae.losses import loss_function
from encyclopedia_vae.modules import (
    build_encoder,
    build_full_decoder,
)
from encyclopedia_vae.types_helpers import EncoderReturn, ForwardReturn, LossReturn


class VanillaVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        latent_dim_dec: int = None,
        hidden_dims: list = [32, 64, 128, 256, 512],
        mid_inflate: int = 2,
        mid_dim: int = 512 * 4,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        if not latent_dim_dec:
            latent_dim_dec = latent_dim

        self.encoder = build_encoder(in_channels=in_channels, hidden_dims=hidden_dims)
        self.flatten = Rearrange("B C H W -> B (C H W)")
        self.fc_mu = nn.Linear(mid_dim, latent_dim)
        self.fc_var = nn.Linear(mid_dim, latent_dim)

        self.full_decoder = build_full_decoder(
            latent_dim=latent_dim_dec,
            hidden_dims=hidden_dims[::-1],
            mid_inflate=mid_inflate,
            mid_dim=mid_dim,
        )

    def encode(self, input: torch.tensor) -> EncoderReturn:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (torch.tensor) Input tensor to encoder [N x C x H x W]
        :return: (torch.tensor) List of latent codes
        """
        result = self.encoder(input)
        result = self.flatten(result)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return EncoderReturn(mu=mu, log_var=log_var, pre_latents=result)

    def decode(self, z: torch.tensor) -> torch.tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (torch.tensor) [B x D]
        :return: (torch.tensor) [B x C x H x W]
        """
        result = self.full_decoder(z)
        return result

    def reparametrize(self, mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (torch.tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (torch.tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (torch.tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.tensor) -> ForwardReturn:
        latents = self.encode(input)
        mu, log_var, _ = latents.values()
        z = self.reparametrize(mu, log_var)
        return ForwardReturn(
            output=self.decode(z), input=input, encoded=latents, latents=z
        )

    def loss(self, output_model: ForwardReturn, kld_weight: float) -> LossReturn:
        """Computes the VAE loss function.

        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        return loss_function(output_model, kld_weight)

    def sample(self, num_samples: int) -> torch.tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (torch.tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.tensor) -> torch.tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.tensor) [B x C x H x W]
        :return: (torch.tensor) [B x C x H x W]
        """

        return self.forward(x)["output"]
