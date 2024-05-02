[Version 0.1.0]

* Fork from Pytorch-VAE
* **REFACTO**
    - Aimed at packaging the code properly thanks to Poetry, with in particular a proper dependencies management (which the original repo sadly lacks).
    - Create dedicated Encoder and Decoder Modules, as they are the core of all VAE-type models.
    - Better imports internally.
    -  Using Ruff as a linter and formatter.
