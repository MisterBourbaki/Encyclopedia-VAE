from typing import TypedDict

import torch


class EncoderReturn(TypedDict):
    mu: torch.tensor
    log_var: torch.tensor
    pre_latents: torch.tensor


class ForwardReturn(TypedDict):
    output: torch.tensor
    input: torch.tensor
    encoded: EncoderReturn
    latents: torch.tensor


class LossReturn(TypedDict):
    loss: torch.tensor
    reconstruction_loss: torch.tensor
    kld_loss: torch.tensor


class LossBetaTCReturn(TypedDict):
    loss: torch.tensor
    reconstruction_loss: torch.tensor
    kld_loss: torch.tensor
    tc_loss: torch.tensor
    mi_loss: torch.tensor
