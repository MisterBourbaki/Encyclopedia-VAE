import unittest

import torch

from encyclopedia_vae.models.cvae import ConditionalVAE


class TestCVAE(unittest.TestCase):
    def setUp(self) -> None:
        # self.model2 = VAE(3, 10)
        self.model = ConditionalVAE(in_channels=3, num_classes=40, latent_dim=10)

    # def test_summary(self):
    #     pprint(summary(self.model, (3, 64, 64), device="cpu"))

    def test_forward(self):
        x = torch.randn(16, 3, 64, 64)
        c = torch.randn(16, 40)
        y = self.model(x, c)
        print("Model Output size:", y["output"].size())
        # print("Model2 Output size:", self.model2(x)[0].size())

    def test_loss(self):
        x = torch.randn(16, 3, 64, 64)
        c = torch.randn(16, 40)
        result = self.model(x, labels=c)
        loss = self.model.loss(result, kld_weight=0.005)
        print(loss)


if __name__ == "__main__":
    unittest.main()
