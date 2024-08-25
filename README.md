# generative_moons

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

This repository presents a research on applying generative models to a synthetic moons dataset ([sklearn.datasets.make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)). The experiments utilize a Variational Autoencoder (VAE) model, which is trained on the moons dataset and then used to generate new samples.

The models are implemented in PyTorch and the experiments are conducted using the PyTorch Lightning framework, featuring Hydra for configuration management. .he repository contains a Jupyter notebook that demonstrates the training and evaluation of the VAE model, as well as analysis of the results.

## Installation

#### Conda

```bash
# clone project
git clone https://github.com/piotlinski/generative_moons
cd generative_moons

# create conda environment and install dependencies
conda env create -f environment.yaml -n generative_moons

# activate conda environment
conda activate generative_moons
```

## How to run

Train a sanity-check autoencoder model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
