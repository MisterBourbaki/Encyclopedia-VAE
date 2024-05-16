from lightning.pytorch.cli import LightningCLI

from encyclopedia_vae.data.lit_oxfordiitpet import LitOxfordIIITPet
from encyclopedia_vae.lit_models.lit_vae import LitVAE


def cli_main():
    cli = LightningCLI(LitVAE, LitOxfordIIITPet)  # noqa F841


if __name__ == "__main__":
    cli_main()
