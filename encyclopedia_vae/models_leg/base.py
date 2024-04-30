from abc import abstractmethod
from typing import Any

import torch
from torch import nn


class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, input: torch.tensor) -> list[torch.tensor]:
        raise NotImplementedError

    def decode(self, input: torch.tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> torch.tensor:
        raise NotImplementedError

    def generate(self, x: torch.tensor, **kwargs) -> torch.tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: torch.tensor) -> torch.tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> torch.tensor:
        pass
