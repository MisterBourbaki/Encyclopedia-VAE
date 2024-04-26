import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        h_dim,
        hidden_dims: list[int] = [32, 64, 128, 256, 512],
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 2,
    ):
        modules = []

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

    def forward(self, input: torch.tensor) -> torch.tensor:
        return self.encoder(input)
