from torch import nn


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
