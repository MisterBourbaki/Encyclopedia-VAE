import torch
from torch import nn
from torch.nn import functional as F

from models import BaseVAE


class MIWAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: list = None,
        num_samples: int = 5,
        num_estimates: int = 5,
        **kwargs,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.num_samples = num_samples  # K
        self.num_estimates = num_estimates  # M

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, input: torch.tensor) -> list[torch.tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (torch.tensor) Input tensor to encoder [N x C x H x W]
        :return: (torch.tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: torch.tensor) -> torch.tensor:
        """
        Maps the given latent codes of S samples
        onto the image space.
        :param z: (torch.tensor) [B x S x D]
        :return: (torch.tensor) [B x S x C x H x W]
        """
        B, M, S, D = z.size()
        z = z.contiguous().view(-1, self.latent_dim)  # [BMS x D]
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)  # [BMS x C x H x W ]
        result = result.view(
            [B, M, S, result.size(-3), result.size(-2), result.size(-1)]
        )  # [B x M x S x C x H x W]
        return result

    def reparameterize(self, mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:
        """
        :param mu: (torch.tensor) Mean of the latent Gaussian
        :param logvar: (torch.tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.tensor, **kwargs) -> list[torch.tensor]:
        mu, log_var = self.encode(input)
        mu = mu.repeat(self.num_estimates, self.num_samples, 1, 1).permute(
            2, 0, 1, 3
        )  # [B x M x S x D]
        log_var = log_var.repeat(self.num_estimates, self.num_samples, 1, 1).permute(
            2, 0, 1, 3
        )  # [B x M x S x D]
        z = self.reparameterize(mu, log_var)  # [B x M x S x D]
        eps = (z - mu) / log_var  # Prior samples
        return [self.decode(z), input, mu, log_var, z, eps]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        args[4]
        args[5]

        input = input.repeat(self.num_estimates, self.num_samples, 1, 1, 1, 1).permute(
            2, 0, 1, 3, 4, 5
        )  # [B x M x S x C x H x W]

        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset

        log_p_x_z = (
            ((recons - input) ** 2).flatten(3).mean(-1)
        )  # Reconstruction Loss # [B x M x S]

        kld_loss = -0.5 * torch.sum(
            1 + log_var - mu**2 - log_var.exp(), dim=3
        )  # [B x M x S]
        # Get importance weights
        log_weight = log_p_x_z + kld_weight * kld_loss  # .detach().data

        # Rescale the weights (along the sample dim) to lie in [0, 1] and sum to 1
        weight = F.softmax(log_weight, dim=-1)  # [B x M x S]

        loss = torch.mean(
            torch.mean(torch.sum(weight * log_weight, dim=-1), dim=-2), dim=0
        )

        return {
            "loss": loss,
            "Reconstruction_Loss": log_p_x_z.mean(),
            "KLD": -kld_loss.mean(),
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> torch.tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (torch.tensor)
        """
        z = torch.randn(num_samples, 1, 1, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z).squeeze()
        return samples

    def generate(self, x: torch.tensor, **kwargs) -> torch.tensor:
        """
        Given an input image x, returns the reconstructed image.
        Returns only the first reconstructed sample
        :param x: (torch.tensor) [B x C x H x W]
        :return: (torch.tensor) [B x C x H x W]
        """

        return self.forward(x)[0][:, 0, 0, :]
