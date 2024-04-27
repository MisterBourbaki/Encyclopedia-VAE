from typing import Union

import torch
from torch import nn
from torch.nn import functional as F

from encyclopedia_vae.models import BaseVAE

from ..modules import ResidualLayer, VectorQuantizer


class VQVAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        num_embeddings: int,
        hidden_dims: list = None,
        beta: float = 0.25,
        img_size: int = 64,
        **kwargs,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
            )
        )

        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1),
                nn.LeakyReLU(),
            )
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, self.beta)

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    embedding_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1
                ),
                nn.LeakyReLU(),
            )
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1], out_channels=3, kernel_size=4, stride=2, padding=1
                ),
                nn.Tanh(),
            )
        )

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: torch.tensor) -> list[torch.tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (torch.tensor) Input tensor to encoder [N x C x H x W]
        :return: (torch.tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z: torch.tensor) -> torch.tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (torch.tensor) [B x D x H x W]
        :return: (torch.tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, input: torch.tensor, **kwargs) -> list[torch.tensor]:
        encoding = self.encode(input)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss, "VQ_Loss": vq_loss}

    def sample(
        self, num_samples: int, current_device: Union[int, str], **kwargs
    ) -> torch.tensor:
        raise Warning("VQVAE sampler is not implemented.")

    def generate(self, x: torch.tensor, **kwargs) -> torch.tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.tensor) [B x C x H x W]
        :return: (torch.tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
