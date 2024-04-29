import math

import torch
from einops import rearrange
from torch.nn import functional as F

from encyclopedia_vae.types_helpers import ForwardReturn, LossBetaTCReturn, LossReturn


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
        output_model["encoded"]["mu"], output_model["encoded"]["log_var"]
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
        output_model["encoded"]["mu"], output_model["encoded"]["log_var"]
    )

    if loss_type == "H":  # https://openreview.net/forum?id=Sy2fzU9gl
        loss = recons_loss + beta * kld_weight * kld_loss
    elif loss_type == "B":  # https://arxiv.org/pdf/1804.03599.pdf
        C = torch.clamp(C_max / C_stop_iter * num_iter, 0, C_max)
        loss = recons_loss + gamma * kld_weight * (kld_loss - C).abs()
    else:
        raise ValueError("Undefined loss type.")

    return LossReturn(loss=loss, reconstruction_loss=recons_loss, kld_loss=-kld_loss)


def log_density_gaussian(x: torch.tensor, mu: torch.tensor, logvar: torch.tensor):
    """
    Computes the log pdf of the Gaussian with parameters mu and logvar at x
    :param x: (torch.tensor) Point at whichGaussian PDF is to be evaluated
    :param mu: (torch.tensor) Mean of the Gaussian distribution
    :param logvar: (torch.tensor) Log variance of the Gaussian distribution
    :return:
    """
    norm = -0.5 * (math.log(2 * math.pi) + logvar)
    log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
    return log_density


def loss_function_betatc(
    output_model: ForwardReturn,
    kld_weight,
    num_iter,
    anneal_steps,
    alpha,
    beta,
    gamma,
) -> dict:
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    :param args:
    :param kwargs:
    :return:
    """
    mu, log_var, _ = output_model["encoded"].values()
    z = output_model["latents"]
    weight = 1  # kwargs['M_N']  # Account for the minibatch samples from the dataset

    recons_loss = F.mse_loss(
        output_model["output"], output_model["input"], reduction="sum"
    )

    log_q_zx = log_density_gaussian(z, mu, log_var).sum(dim=1)

    zeros = torch.zeros_like(z)
    log_p_z = log_density_gaussian(z, zeros, zeros).sum(dim=1)

    batch_size, _ = z.shape
    mat_log_q_z = log_density_gaussian(
        rearrange(z, "B D -> B 1 D"),
        rearrange(mu, "B D -> 1 B D"),
        rearrange(log_var, "B D -> 1 B D"),
    )

    # Reference
    # [1] https://github.com/YannDubs/disentangling-vae/blob/535bbd2e9aeb5a200663a4f82f1d34e084c4ba8d/disvae/utils/math.py#L54
    dataset_size = (1 / kld_weight) * batch_size  # dataset size
    strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))
    importance_weights = torch.full((batch_size, batch_size), 1 / (batch_size - 1))
    importance_weights.view(-1)[::batch_size] = 1 / dataset_size
    importance_weights.view(-1)[1::batch_size] = strat_weight
    importance_weights[batch_size - 2, 0] = strat_weight
    log_importance_weights = importance_weights.log()

    mat_log_q_z += rearrange(log_importance_weights, "B1 B2 -> B1 B2 1")

    log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
    log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

    mi_loss = (log_q_zx - log_q_z).mean()
    tc_loss = (log_q_z - log_prod_q_z).mean()
    kld_loss = (log_prod_q_z - log_p_z).mean()

    num_iter += 1
    anneal_rate = min(0 + 1 * num_iter / anneal_steps, 1)

    loss = (
        recons_loss / batch_size
        + alpha * mi_loss
        + weight * (beta * tc_loss + anneal_rate * gamma * kld_loss)
    )

    return LossBetaTCReturn(
        loss=loss,
        reconstruction_loss=recons_loss,
        kld_loss=-kld_loss,
        tc_loss=tc_loss,
        mi_loss=mi_loss,
    )
