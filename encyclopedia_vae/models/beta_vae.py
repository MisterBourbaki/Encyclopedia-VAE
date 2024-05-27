"""Module for the BetaVAE implementation."""

from encyclopedia_vae.losses import loss_function_beta
from encyclopedia_vae.models.vanilla_vae import VanillaVAE
from encyclopedia_vae.types_helpers import OutputLoss, OutputModel


class BetaVAE(VanillaVAE):
    """Implementation of BetaVAE model.

    Subclass the bare VAE model that is VanillaVAE.

    Attributes
    ----------
    beta: int
        beta arg, by default 4.

    Methods
    -------
    loss(output_model, kld_weight)
        the loss function for the BetaVAE model.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: list = [32, 64, 128, 256, 512],
        beta: int = 4,
        gamma: float = 1000.0,
        max_capacity: int = 25,
        capacity_stop_iter: int = 1e5,
        loss_type: str = "B",
    ) -> None:
        super().__init__(
            in_channels=in_channels, latent_dim=latent_dim, hidden_dims=hidden_dims
        )

        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.capa_max = max_capacity
        self.capa_stop_iter = capacity_stop_iter
        self.num_iter = 0

    def loss(self, output_model: OutputModel, kld_weight: float) -> OutputLoss:
        """Loss function.

        Wraps the encyclopedia_vae.losses.loss_function_beta.

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
        loss = loss_function_beta(
            output_model=output_model,
            kld_weight=kld_weight,
            num_iter=self.num_iter,
            capa_max=self.capa_max,
            capa_stop_iter=self.capa_stop_iter,
            loss_type=self.loss_type,
            beta=self.beta,
            gamma=self.gamma,
        )
        self.num_iter += 1
        return loss
