# Nimbus

The Nimbus repo contains code for training, validation and application of a machine learning model that classifies cells into marker positive/negative for arbitrary markers and different imaging platforms.

Disclaimer: The environment does not support Apple Silicon chips (M1, M2) currently.

## Installation instructions

Clone the repository

`git clone https://github.com/angelolab/Nimbus.git`


Make a conda environment for Nimbus and activate it

`conda create -n Nimbus python==3.10`

`conda activate Nimbus`

Install CUDA libraries if you have a NVIDIA GPU available 

`conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`

Install the package and all depedencies in the conda environment

`python -m pip install -e Nimbus`

Install tensorflow-metal if you have an Apple Silicon GPU

`python -m pip install tensorflow-metal`

Navigate to the example notebooks and start jupyter

`cd Nimbus/templates`

`jupyter notebook`


