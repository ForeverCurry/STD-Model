import os
from dataclasses import dataclass
import pandas as pd
from common.settings import DATASETS_PATH

DATASET_PATH = os.path.join(DATASETS_PATH, 'plankton')

@dataclass()
class planktonDataset:

    @staticmethod
    def load() :
        """
        Load Plankton dataset.

        :param training: Load training part if training is True, test part otherwise.
        """
        data = pd.read_excel(os.path.join(DATASET_PATH,'plankton.xlsx'))
        return data


