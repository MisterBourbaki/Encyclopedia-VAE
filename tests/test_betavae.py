import unittest

import torch
from torchsummary import summary

from encyclopedia_vae.models.beta_vae import BetaVAE


class TestVAE(unittest.TestCase):
    def setUp(self) -> None:
        # self.model2 = VAE(3, 10)
        self.model = BetaVAE(3, 10, loss_type="H")

    def test_summary(self):
        print(summary(self.model, (3, 64, 64), device="cpu"))
        # print(summary(self.model2, (3, 64, 64), device='cpu'))

    def test_forward(self):
        x = torch.randn(16, 3, 64, 64)
        y = self.model(x)
        print("Model Output size:", y["output"].size())
        # print("Model2 Output size:", self.model2(x)[0].size())

    def test_loss(self):
        x = torch.randn(16, 3, 64, 64)

        result = self.model(x)
        loss = self.model.loss(result, kld_weight=0.005)
        print(loss)


if __name__ == "__main__":
    unittest.main()
