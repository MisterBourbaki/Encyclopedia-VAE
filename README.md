# A modular, easy to use Encyclopedia of VAE

** STILL WORK IN PROGRESS **

This project aims at working from the repo Pytorch-VAE and upgrading it (a lot).

## Features:

* Packaging using modern tool like Poetry.
* Start from the "vanilla" VAE model, and building upon it to get all the other ones: by changing the loss, some layers... The code is then far more easy to maintain than the original one.
* Use the full power of Lightning tools, like LightningModule and especially the nice CLI.
* Possibility to import any LightningDataModule to your project.
