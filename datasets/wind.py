"""
wind Dataset
"""

import os
from dataclasses import dataclass
import pandas as pd
from common.settings import DATASETS_PATH

DATASET_PATH = os.path.join(DATASETS_PATH, 'wind')

@dataclass()
class WindDataset:
    @staticmethod
    def load():
        """
        Load wind dataset.
        """
        data = pd.read_csv(os.path.join(DATASET_PATH,'winddata.csv'),
                          header=0, delimiter=",")
        return data


 

