# Nimbus

The Nimbus repo contains code for training, validation and application of a machine learning model that classifies cells into marker positive/negative for arbitrary markers and different imaging platforms.

Disclaimer: The environment does not support Apple Silicon chips (M1, M2) currently.

## Installation instructions

Clone the repository

`git clone https://github.com/angelolab/cell_classification.git`


Make a conda environment for Nimbus and activate it

`conda create -n Nimbus python==3.10`

`conda activate Nimbus`


Install the package and all depedencies in the conda environment

`python -m pip install -e cell_classification`


Navigate to the example notebooks and start jupyter

`cd cell_classification/templates`

`jupyter notebook`


