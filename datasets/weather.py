# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Tourism Dataset
"""
import gin
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import patoolib
from common.settings import DATASETS_PATH

# DATASET_URL = 'https://robjhyndman.com/data/27-3-Athanasopoulos1.zip'

DATASET_PATH = os.path.join(DATASETS_PATH, 'weather')
# DATASET_FILE_PATH = os.path.join(DATASET_PATH, url_file_name(DATASET_URL))


# @dataclass()
# class OzoneMeta:
#     seasonal_patterns = ['min']
#     horizon = 5
#     frequency = 6

# @dataclass()
class weatherDataset:

    @staticmethod
    def load(training: bool = True, downsample = True, step : int=6) :
        """
        Load Tourism dataset from cache.

        :param training: Load training part if training is True, test part otherwise.
        """
        data = pd.read_csv(os.path.join(DATASET_PATH,'weather3.csv'),header=0,index_col=0)
        data.index = pd.to_datetime(data.index)
        # print(data.columns())
        # data.set_index('date')
        if downsample:
            data = data.resample('12h').mean()
        return data




 

