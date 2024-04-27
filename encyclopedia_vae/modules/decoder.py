import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_dims: list[int] = [32, 64, 128, 256, 512],
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 2,
        output_padding: int = 1,
    ):
        super().__init__()
        layers = []
        hidden_dims = hidden_dims[::-1]

        for i in range(len(hidden_dims) - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*layers)

    def forward(self, input: torch.tensor) -> torch.tensor:
        return self.decoder(input)
