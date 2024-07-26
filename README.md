# STD Model

This repository contains the code for the paper SpatioTemporal Diffusion with Koopman operator for short-term time-series prediction. Feel free to star this repository and cite our papers if you find it useful for your research. You can find the citation details below

STD model is used for short-term high-dimension multi-step time series prediction based on the Koopman operator. In addition to its prediction capabilities, our STD method can also be used as a post-processing tool for refining other time-series models to improve multi-step prediction performance effectively.

<!-- ![N-BEATS Architecture](nbeats.png) -->

## Preparation
Install pytorch and other necessary dependencies.

`pip install -r requirements.txt`

## Repository Structure

#### Model
PyTorch implementation of STD can be found in `models/STD.py`

#### Datasets
The loaders for each dataset used in the paper are in `datasets/*.py`
#### Demo
The linear system experiment is shown in `Linear_demo.ipynb`.
#### Experiments
Experiments to reproduce the paper results are located in `experiments/*`, 
where each experiment package contains `<data>/<data>_exp.py`. If you want to reproduce the results, 
you can run the corresponding experiment package. For example, to reproduce the results for Lorenz experiment, 
you can run the following command:

`python ./experiments/lorenz/lorenz_exp.py`

The refined experiments for ETS model can be reproduced by running the following command:

`python ./experiments/lorenz/lorenz_exp_refined.py  --refine --refine_model RDE`

The run commands for reproducing the paper results are listed in `run.sh`.

#### Results
The `results` directory contains the pre-predictions of each model on each dataset and refinement results by STD model. The run commands for reproducing the paper and SI figures are listed in `plot.sh`.
## Citation

## Contact
If you have any questions or want to use the code, please contact:
- suliangyu0917@stu.xjtu.edu.cn
