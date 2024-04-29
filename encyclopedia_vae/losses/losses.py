import torch
from rich import print as pprint
from torch.nn import functional as F

from encyclopedia_vae.types import ForwardReturn, LossReturn


def loss_function(output_model: ForwardReturn, kld_weight, **kwargs) -> LossReturn:
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    :param args:
    :param kwargs:
    :return:
    """

    # kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
    recons_loss = F.mse_loss(output_model["output"], output_model["input"])
    pprint(f"In loss function, MSE is {recons_loss}")

    kld_loss = compute_kld_loss(
        output_model["latents"]["mu"], output_model["latents"]["log_var"]
    )
    pprint(f"In loss function, KLD is {kld_loss}")
    loss = recons_loss + kld_weight * kld_loss
    return LossReturn(loss=loss, reconstruction_loss=recons_loss, kld_loss=-kld_loss)


def compute_kld_loss(mu: torch.tensor, log_var: torch.tensor) -> torch.tensor:
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
    )

    return kld_loss
