"""Module for the implementation of the Vanilla VAE model."""

import torch
from einops.layers.torch import Rearrange
from torch import nn

from encyclopedia_vae.losses import loss_function
from encyclopedia_vae.modules import (
    build_encoder,
    build_full_decoder,
)
from encyclopedia_vae.types_helpers import OutputEncoder, OutputLoss, OutputModel


class VanillaVAE(nn.Module):
    """Implementation of the classical VAE.

    Attributes
    ----------
    latent_dim: int
        dimension of the latent space
    encoder: nn.Module
        implements the Encoder part
    full_decoder: nn.Module
        implements the Decoder part
    flatten: nn.Module
        einops Rearrange layer to ease the pre process of the latent work
    fc_mu: nn.Module
        linear Module which "learns" the mean vector of the latents distribution
    fc_var: nn.Module
        linear Module which "learns" the log-var vector of the latents distribution
    """

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

    def encode(self, input: torch.tensor) -> OutputEncoder:
        """Encode the input and returns the latent codes.

        Parameters
        ----------
        input: torch.tensor
            Input tensor to encoder [N x C x H x W]

        Returns
        -------
        OutputEncoder
            contains mu, log_var and pre-latent tensors.
        """
        result = self.encoder(input)
        result = self.flatten(result)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return OutputEncoder(mu=mu, log_var=log_var, pre_latents=result)

    def decode(self, z: torch.tensor) -> torch.tensor:
        """Decode the given latent codes onto the image space.

        Parameters
        ----------
        z: torch.tensor
            tensor of latents, of shape [B x D]

        Returns
        -------
        torch.tensor
            tensor of shape[B x C x H x W]
        """
        result = self.full_decoder(z)
        return result

    def reparametrize(self, mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:
        """Implement the reparameterization trick.

        The reparametrization trick to sample from N(mu, var) from N(0,1).

        Parameters
        ----------
        mu: torch.tensor
            Mean of the latent Gaussian [B x D]
        logvar: torch.tensor
            Standard deviation of the latent Gaussian [B x D]

        Returns
        -------
        torch.tensor
            tensor of shape [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.tensor) -> OutputModel:
        """Forward function.

        Parameters
        ----------
        input : torch.tensor
            tensor of shape [B C H W]

        Returns
        -------
        OutputModel
            contains output, input, various latents tensors.
        """
        latents = self.encode(input)
        mu, log_var, _ = latents.values()
        z = self.reparametrize(mu, log_var)
        return OutputModel(
            output=self.decode(z), input=input, encoded=latents, latents=z
        )

    def loss(self, output_model: OutputModel, kld_weight: float) -> OutputLoss:
        """Compute the VAE loss function.

        Wraps the loss_function defined in the losses module.

        Parameters
        ----------
        output_model: OutputModel
            the output of the full model.
        kld_weight: float
            the weight for the KLD loss term.
        """
        return loss_function(output_model, kld_weight)

    def sample(self, num_samples: int) -> torch.tensor:
        """Sample images from the latent space.

        Parameters
        ----------
        num_samples: int
            Number of samples

        Returns
        -------
        torch.tensor
        """
        z = torch.randn(num_samples, self.latent_dim)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.tensor) -> torch.tensor:
        """Return the reconstructed image from tensor x.

        Parameters
        ----------
        x: torch.tensor
            input tensor of shape [B x C x H x W]

        Returns
        -------
        torch.tensor
            generated tensor from the model, of shape [B x C x H x W]
        """
        return self.forward(x)["output"]
