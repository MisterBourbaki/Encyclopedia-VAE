"""Module for the BetaTC VAE implementation."""

import torch
from torch import nn

from encyclopedia_vae.losses import loss_function_betatc
from encyclopedia_vae.models.vanilla_vae import VanillaVAE
from encyclopedia_vae.modules import (
    build_encoder,
)
from encyclopedia_vae.types_helpers import OutputEncoder, OutputModel


class BetaTCVAE(VanillaVAE):
    """Implementation of BetaTC VAE model.

    Subclass the bare VAE model that is VanillaVAE.

    Attributes
    ----------
    beta: int
        beta arg, by default 4.
    anneal_steps: int, optional
        number of anneal steps, by default 200

    Methods
    -------
    loss(output_model, kld_weight)
        the loss function for the BetaVAE model.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: list = [32, 32, 32, 32],
        mid_inflate: int = 4,
        mid_dim: int = 256 * 2,
        anneal_steps: int = 200,
        alpha: float = 1.0,
        beta: float = 6.0,
        gamma: float = 1.0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            mid_inflate=mid_inflate,
            mid_dim=mid_dim,
        )

        self.num_iter = 0
        self.anneal_steps = anneal_steps

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.encoder = build_encoder(
            in_channels=in_channels, hidden_dims=hidden_dims, kernel_size=4
        )

        self.fc = nn.Linear(hidden_dims[-1] * 16, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

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

        result = self.fc(result)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return OutputEncoder(mu=mu, log_var=log_var, pre_latents=result)

    def loss(self, output_model: OutputModel, kld_weight) -> dict:
        """Loss function.

        Wraps the encyclopedia_vae.losses.loss_function_betatc.

        Parameters
        ----------
        output_model : OutputModel
            _description_
        kld_weight : float
            _description_

        Returns
        -------
        OutputLoss
            _description_
        """
        loss = loss_function_betatc(
            output_model=output_model,
            kld_weight=kld_weight,
            num_iter=self.num_iter,
            anneal_steps=self.anneal_steps,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )
        self.num_iter += 1

        return loss
