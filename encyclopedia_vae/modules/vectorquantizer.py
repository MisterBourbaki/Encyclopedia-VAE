import torch
from torch import nn
from torch.nn import functional


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.codebook = nn.Embedding(self.K, self.D)
        self.codebook.weight.data.uniform_(-1 / self.K, 1 / self.K)

    @property
    def resolution(self) -> torch.Tensor:
        """Compute the resolution of the Vector Quantizer.

        The resolution is the log2 of the number of embeddings divided by
        the dimension of the embedding. This is the same as the bitrate by dimension.

        Returns
        -------
        torch.Tensor
            the log2 of the number of embedding divided by the dimension.
        """
        return torch.log2(self.K) / self.D

    def encode(self, latents: torch.Tensor) -> torch.Tensor:
        """Encode the latents by nearest neighboors.

        Parameters
        ----------
        latents : torch.Tensor
            should be channel last!

        Returns
        -------
        torch.Tensor
            tensor holding the coding indices for latents.
        """
        encodings_shape = latents.shape[:-1]
        flat_latents = latents.view(-1, self.D)

        dist = (
            torch.sum(flat_latents**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(flat_latents, self.codebook.weight.t())
        )

        encoding_inds = torch.argmin(dist, dim=1)
        return encoding_inds.view(encodings_shape)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode the given indices into vectors, using the codebook.

        Parameters
        ----------
        indices : torch.Tensor
            tensor holding indices as integers

        Returns
        -------
        torch.Tensor
            a channel last tensor
        """
        return self.codebook(indices)

    def forward(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quantized_latents = self.decode(self.encode(latents))

        commitment_loss = functional.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = functional.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        quantized_latents = latents + (quantized_latents - latents).detach()
        return quantized_latents, vq_loss
