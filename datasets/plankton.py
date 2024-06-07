import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import patoolib
from common.settings import DATASETS_PATH



DATASET_PATH = os.path.join(DATASETS_PATH, 'plankton')


@dataclass()
class miningMeta:
    seasonal_patterns = ['min']
    horizon = 5
    frequency = 6

@dataclass()
class planktonDataset:

    @staticmethod
    def load(training: bool = True, step : int=6) :
        """
        Load Plankton dataset.

        :param training: Load training part if training is True, test part otherwise.
        """
        data = pd.read_excel(os.path.join(DATASET_PATH,'plankton.xlsx'))
        return data


