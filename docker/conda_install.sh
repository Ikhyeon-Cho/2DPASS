#!/bin/bash

# Install additional packages in the active conda environment
conda install -n $1 \
    numpy=1.25 \
    tqdm \
    pyyaml \
    tensorboard \
    numba \
    easydict \
    pyquaternion \
    pytorch-scatter \
    -y -c conda-forge -c pyg

# Activate the conda environment to ensure pip installs to the correct environment
eval "$(conda shell.bash hook)"
conda activate $1

# Install pip packages
pip install spconv-cu113
pip install pytorch-lightning==1.3.8 torchmetrics==0.5
pip install setuptools==59.5.0