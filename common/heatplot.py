import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from common.settings import DATASETS_PATH
import os
sns.set_theme()
DATASET_PATH = os.path.join(DATASETS_PATH, 'Neural')
data = pd.read_csv(os.path.join(DATASET_PATH,'humanBrain_coef.csv'),sep=',',header=0,)

# fig, axs = plt.subplots(figsize=(9, 9), nrows=1, ncols=1)
sns.heatmap(data,center=0, cmap="vlag")
plt.show()