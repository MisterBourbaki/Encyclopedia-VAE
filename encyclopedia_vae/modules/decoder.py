from einops.layers.torch import Rearrange
from torch import nn


def build_core_decoder(
    hidden_dims: list[int] = [512, 256, 128, 64, 32],  # [32, 64, 128, 256, 512],
    kernel_size: int = 3,
    padding: int = 1,
    stride: int = 2,
    output_padding: int = 1,
):
    _layers = []
    for i in range(len(hidden_dims) - 1):
        _layers.append(
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

    return nn.Sequential(*_layers)


def build_full_decoder(
    latent_dim: int,
    hidden_dims: list[int] = [512, 256, 128, 64, 32],  # [32, 64, 128, 256, 512],
    kernel_size: int = 3,
    padding: int = 1,
    stride: int = 2,
    output_padding: int = 1,
    mid_dim: int = 512 * 4,
    mid_inflate: int = 2,
):
    module = nn.Sequential(
        nn.Linear(latent_dim, mid_dim),
        Rearrange(
            "B (C H W) -> B C H W", C=hidden_dims[0], H=mid_inflate, W=mid_inflate
        ),
        build_core_decoder(
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            output_padding=output_padding,
        ),
        create_final_layer(last_dim=hidden_dims[-1]),
    )

    return module


def create_final_layer(
    last_dim,
    kernel_size: int = 3,
    padding: int = 1,
    stride: int = 2,
    output_padding: int = 1,
):
    mod = nn.Sequential(
        nn.ConvTranspose2d(
            last_dim,
            last_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        ),
        nn.BatchNorm2d(last_dim),
        nn.LeakyReLU(),
        nn.Conv2d(last_dim, out_channels=3, kernel_size=3, padding=1),
        nn.Tanh(),
    )

    return mod
