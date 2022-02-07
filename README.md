# ResMLP_pytorch
PyTorch implementation of ResMLP network described in [ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/abs/2105.03404).

## Requirements
Install the following dependencies using `pip` in your python environment.
| Library | Tested version | Required |
| -------|-------|-------|
| **torch** | v1.10.0+cu111 | Yes|
| **torchvision** | v0.11.1+cu111 | Yes|
| **einops** | v0.4.0 | Yes|
| **timm** | v0.5.4 | Yes|
| **jupyter** | v1.0.0| Yes|

If you run the notebook from [Colab](https://colab.research.google.com/) you don't need to install any library as they are already present on the system.

## Usage
Once you have downloaded the project and made sure you have all the dependencies installed, simply run the following command (on terminal) inside the project directory:
```sh
jupyter notebook test_model.ipynb
```

## License
Licensed under the term of [GNU GPL v3.0](LICENSE).