from typing import TypedDict

import torch
from einops.layers.torch import Rearrange
from torch import nn

from encyclopedia_vae.losses import loss_function
from encyclopedia_vae.models import BaseVAE
from encyclopedia_vae.modules import (
    build_encoder,
    build_full_decoder,
)


class EncoderReturn(TypedDict):
    mu: torch.tensor
    log_var: torch.tensor
    pre_latents: torch.tensor


class ForwardReturn(TypedDict):
    output: torch.tensor
    input: torch.tensor
    latents: EncoderReturn


class VanillaVAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: list = [32, 64, 128, 256, 512],
        **kwargs,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = build_encoder(in_channels=in_channels, hidden_dims=hidden_dims)
        self.flatten = Rearrange("B C H W -> B (C H W)", H=2, W=2)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        # self.reshape = Rearrange("B (C H W) -> B C H W", C=hidden_dims[-1], H=2, W=2)
        # self.decoder = build_core_decoder(hidden_dims=hidden_dims)
        # self.final_layer = create_final_layer(last_dim=hidden_dims[0])
        self.full_decoder = build_full_decoder(
            latent_dim=latent_dim, hidden_dims=hidden_dims[::-1]
        )

    def encode(self, input: torch.tensor) -> list[torch.tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (torch.tensor) Input tensor to encoder [N x C x H x W]
        :return: (torch.tensor) List of latent codes
        """
        result = self.encoder(input)
        result = self.flatten(result)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        # return [mu, log_var]
        return EncoderReturn(mu=mu, log_var=log_var, pre_latents=result)

    def decode(self, z: torch.tensor) -> torch.tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (torch.tensor) [B x D]
        :return: (torch.tensor) [B x C x H x W]
        """
        # result = self.decoder_input(z)
        # result = self.reshape(result)
        # result = self.decoder(result)
        # result = self.final_layer(result)
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

    def forward(self, input: torch.tensor, **kwargs) -> list[torch.tensor]:
        latents = self.encode(input)
        mu, log_var, _ = latents.values()
        z = self.reparametrize(mu, log_var)
        # return [self.decode(z), input, mu, log_var]
        return ForwardReturn(output=self.decode(z), input=input, latents=latents)

    def loss_function_(self, input, recons, mu, log_var, kld_weight, **kwargs) -> dict:
        """Computes the VAE loss function.

        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        return loss_function(input, recons, mu, log_var, kld_weight)

    def sample(self, num_samples: int, current_device: int, **kwargs) -> torch.tensor:
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

    def generate(self, x: torch.tensor, **kwargs) -> torch.tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.tensor) [B x C x H x W]
        :return: (torch.tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
