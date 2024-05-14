# STD Model

This repository contains the code for the paper SpatioTemporal Diffusion with Koopman operator for short-term time-series prediction. Feel free to star this repository and cite our papers if you find it useful for your research. You can find the citation details below

STD model is model for Spatiotemporal information transformation to multi-step time series prediction.

<!-- ![N-BEATS Architecture](nbeats.png) -->

## Repository Structure

#### Model
PyTorch implementation of STD can be found in `models/STD.py`

#### Datasets
The loaders for each dataset used in the paper are in `datasets/*.py`

#### Experiments
Experiments to reproduce the paper results are located in `experiments/*`, 
where each experiment package contains `data/data_exp.py`. If you want to reproduce the results, 
you can run the corresponding experiment package. For example, to reproduce the results for Lorenz experiment, 
you can run the following command:

`python ./experiments/lorenz/lorenz_exp.py`

The refined experiments can be reproduced by running the following command:

`python ./experiments/lorenz/lorenz_exp_refined.py  --refine --refine_model='STD'`
#### Results
The `results` directory contains the pre-predictions of each model on each dataset.
