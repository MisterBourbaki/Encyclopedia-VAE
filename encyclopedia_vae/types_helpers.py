"""Module to enforce coherent type hints in the library.

Contains few TypedDict classes:
- OutputEncoder
- OutputModel
OutputLoss

"""

from typing import TypedDict

import torch


class OutputEncoder(TypedDict):
    """TypedDict for normalizing outputs of all Encoder."""

    mu: torch.tensor
    log_var: torch.tensor
    pre_latents: torch.tensor


class OutputModel(TypedDict):
    """TypedDict for normalizing outputs of all VAE models."""

    output: torch.tensor
    input: torch.tensor
    encoded: OutputEncoder
    latents: torch.tensor


class OutputLoss(TypedDict):
    """TypedDict for normalizing outputs of all losses."""

    loss: torch.tensor
    reconstruction_loss: torch.tensor
    kld_loss: torch.tensor


class OutputLossBetaTC(OutputLoss):
    """Subclass of OutputLoss to take special care of BetaTC loss."""

    tc_loss: torch.tensor
    mi_loss: torch.tensor
