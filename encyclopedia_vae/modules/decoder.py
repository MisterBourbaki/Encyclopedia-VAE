"""Module containing functions to generate various Decoders."""

from einops.layers.torch import Rearrange
from torch import nn


def build_core_decoder(
    hidden_dims: list[int] = [512, 256, 128, 64, 32],  # [32, 64, 128, 256, 512],
    kernel_size: int = 3,
    padding: int = 1,
    stride: int = 2,
    output_padding: int = 1,
) -> nn.Sequential:
    """Return a Sequential, 2D (transpose) Convolutional Module.

    Parameters
    ----------
    hidden_dims : list[int], optional
       list of the number of channels of the intermediate,
       "hidden" convolution layers, by default [512, 256, 128, 64, 32]
    kernel_size : int, optional
        the kernel size of all convolutional layers, by default 3
    padding : int, optional
        the padding of all convolutional layers, by default 1
    stride : int, optional
        the stride of all convolutional layers, by default 2
    output_padding : int, optional
        the output_padding of all convolutional layers, by default 1

    Returns
    -------
    nn.Sequential
    """
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
) -> nn.Sequential:
    """Return a Squential Module containing the full Decoder module.

    The generated Decoder Module is made of three parts:
        * an expanding/reshaping layer ;
        * a core Decoder, as built by build_core_decoder ;
        * a final layer, as built by build_final_layer.

    Parameters
    ----------
    latent_dim : int
        _description_
    hidden_dims : list[int], optional
        _description_, by default [512, 256, 128, 64, 32]
    kernel_size : int, optional
        _description_, by default 3
    padding : int, optional
        _description_, by default 1
    stride : int, optional
        _description_, by default 2
    output_padding : int, optional
        _description_, by default 1
    mid_dim : int, optional
        _description_, by default 512*4
    mid_inflate : int, optional
        _description_, by default 2

    Returns
    -------
    nn.Sequential
        _description_
    """
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
) -> nn.Sequential:
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
