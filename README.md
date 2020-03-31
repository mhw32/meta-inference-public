# meta-inference-public
A PyTorch implementation of "Meta-Amortized Variational Inference and Learning" (https://arxiv.org/abs/1902.01950)

## Abstract
Despite the recent success in probabilistic modeling and their applications, generative models trained using traditional inference techniques struggle to adapt to new distributions, even when the target distribution may be closely related to the ones seen during training. In this work, we present a doubly-amortized variational inference procedure as a way to address this challenge. By sharing computation across not only a set of query inputs, but also a set of different, related probabilistic models, we learn transferable latent representations that generalize across several related distributions. In particular, given a set of distributions over images, we find the learned representations to transfer to different data transformations. We empirically demonstrate the effectiveness of our method by introducing the MetaVAE, and show that it significantly outperforms baselines on downstream image classification tasks on MNIST (10-50%) and NORB (10-35%).

## Setup Instructions
Tested with Python 3 and PyTorch 1.0. Install a conda environment and install the necessary libraries.
```
conda create -n metavae python=3 anaconda
conda activate metavae
conda install pytorch torchvision -c pytorch
pip install tqdm, dotmap, sklearn
```
This repository is organized as a package. For every fresh terminal, you should source the path.
```
source init_env.sh
```
Download the NORB dataset from https://cs.nyu.edu/~yann/research/norb/. We have a preprocessing script to save NORB images as numpy files. Note that the file contains a `NUMPY_ROOT` variable you should change to your own directory.
```
cd scripts/utils
python preprocess_norb.py
```

## Experiment Instructions

### Transformation Invariance Experiments

### Compiled Inference Experiments

### Exponential Family Experiments

