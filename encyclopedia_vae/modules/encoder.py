from torch import nn


def build_encoder(
    in_channels,
    hidden_dims: list[int] = [32, 64, 128, 256, 512],
    kernel_size: int = 3,
    padding: int = 1,
    stride: int = 2,
):
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
