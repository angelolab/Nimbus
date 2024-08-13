# Nimbus

The Nimbus repo contains code for training and validation of a machine learning model that classifies cells into marker positive/negative for arbitrary markers and different imaging platforms.

The code for using the model and running inference on your own data can be found here: [Nimbus-Inference](https://github.com/angelolab/Nimbus-Inference). Code for generating the figures in the paper can be found here: [Publication plots](https://github.com/angelolab/publications/tree/main/2024-Rumberger_Greenwald_etal_Nimbus).

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

## Citation

```bash
@article{rum2024nimbus,
  title={Automated classification of cellular expression in multiplexed imaging data with Nimbus},
  author={Rumberger, J. Lorenz and Greenwald, Noah F. and Ranek, Jolene S. and Boonrat, Potchara and Walker, Cameron and Franzen, Jannik and Varra, Sricharan Reddy and Kong, Alex and Sowers, Cameron and Liu, Candace C. and Averbukh, Inna and Piyadasa, Hadeesha and Vanguri, Rami and Nederlof, Iris and Wang, Xuefei Julie and Van Valen, David and Kok, Marleen and Hollman, Travis J. and Kainmueller, Dagmar and Angelo, Michael},
  journal={bioRxiv},
  pages={2024--05},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
