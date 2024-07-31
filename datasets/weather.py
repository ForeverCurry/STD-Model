import os
from dataclasses import dataclass

import pandas as pd
from common.settings import DATASETS_PATH

DATASET_PATH = os.path.join(DATASETS_PATH, 'weather')

# @dataclass()
class weatherDataset:
    @staticmethod
    def load(downsample = True) :
        """
        Load Weather dataset.

        :param training: Load training part if training is True, test part otherwise.
        """
        data = pd.read_csv(os.path.join(DATASET_PATH,'weather3.csv'),header=0,index_col=0)
        data.index = pd.to_datetime(data.index)
        if downsample:
            data = data.resample('12h').mean()
        return data




 

