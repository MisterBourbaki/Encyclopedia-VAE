from encyclopedia_vae.losses import loss_function_beta
from encyclopedia_vae.models.vanilla_vae import VanillaVAE
from encyclopedia_vae.types_helpers import ForwardReturn, LossReturn


class BetaVAE(VanillaVAE):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: list = [32, 64, 128, 256, 512],
        beta: int = 4,
        gamma: float = 1000.0,
        max_capacity: int = 25,
        Capacity_max_iter: int = 1e5,
        loss_type: str = "B",
    ) -> None:
        super().__init__(
            in_channels=in_channels, latent_dim=latent_dim, hidden_dims=hidden_dims
        )

        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = max_capacity
        self.C_stop_iter = Capacity_max_iter
        self.num_iter = 0

    def loss(self, output_model: ForwardReturn, kld_weight: float) -> LossReturn:
        loss = loss_function_beta(
            output_model=output_model,
            kld_weight=kld_weight,
            num_iter=self.num_iter,
            C_max=self.C_max,
            C_stop_iter=self.C_stop_iter,
            loss_type=self.loss_type,
            beta=self.beta,
            gamma=self.gamma,
        )
        self.num_iter += 1
        return loss
