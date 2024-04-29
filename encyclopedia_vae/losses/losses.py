import torch
from torch.nn import functional as F

from encyclopedia_vae.types import ForwardReturn, LossReturn


def loss_function(output_model: ForwardReturn, kld_weight) -> LossReturn:
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    :param args:
    :param kwargs:
    :return:
    """

    recons_loss = F.mse_loss(output_model["output"], output_model["input"])

    kld_loss = compute_kld_loss(
        output_model["latents"]["mu"], output_model["latents"]["log_var"]
    )
    loss = recons_loss + kld_weight * kld_loss
    return LossReturn(loss=loss, reconstruction_loss=recons_loss, kld_loss=-kld_loss)


def compute_kld_loss(mu: torch.tensor, log_var: torch.tensor) -> torch.tensor:
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
    )

    return kld_loss


def loss_function_beta(
    output_model: ForwardReturn,
    kld_weight,
    num_iter,
    C_max,
    C_stop_iter,
    loss_type,
    beta,
    gamma,
) -> dict:
    recons_loss = F.mse_loss(output_model["output"], output_model["input"])

    kld_loss = compute_kld_loss(
        output_model["latents"]["mu"], output_model["latents"]["log_var"]
    )

    if loss_type == "H":  # https://openreview.net/forum?id=Sy2fzU9gl
        loss = recons_loss + beta * kld_weight * kld_loss
    elif loss_type == "B":  # https://arxiv.org/pdf/1804.03599.pdf
        C = torch.clamp(C_max / C_stop_iter * num_iter, 0, C_max)
        loss = recons_loss + gamma * kld_weight * (kld_loss - C).abs()
    else:
        raise ValueError("Undefined loss type.")

    return LossReturn(loss=loss, reconstruction_loss=recons_loss, kld_loss=-kld_loss)
