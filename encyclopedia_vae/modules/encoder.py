"""Module containing the function generating various Encoder."""

from torch import nn


def build_encoder(
    in_channels: int,
    hidden_dims: list[int] = [32, 64, 128, 256, 512],
    kernel_size: int = 3,
    padding: int = 1,
    stride: int = 2,
) -> nn.Sequential:
    """Return a 2d-convolutional, sequential Encoder.

    Parameters
    ----------
    in_channels : int
        number of channels in the input
    hidden_dims : list[int], optional
        list of the number of channels of the intermediate,
        "hidden" convolution layers, by default [32, 64, 128, 256, 512]
    kernel_size : int, optional
        the kernel size of all convolutional layers, by default 3
    padding : int, optional
        the padding of all convolutional layers, by default 1
    stride : int, optional
        the stride of all convolutional layers, by default 2

    Returns
    -------
    nn.Sequential
        the sequence of convolutional layers
    """
    _layers = []

    for h_dim in hidden_dims:
        _layers.append(
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

    return nn.Sequential(*_layers)
