from typing import TypedDict

import torch


class EncoderReturn(TypedDict):
    mu: torch.tensor
    log_var: torch.tensor
    pre_latents: torch.tensor


class ForwardReturn(TypedDict):
    output: torch.tensor
    input: torch.tensor
    latents: EncoderReturn


class LossReturn(TypedDict):
    loss: torch.tensor
    reconstruction_loss: torch.tensor
    kld_loss: torch.tensor
