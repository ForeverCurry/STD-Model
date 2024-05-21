import os
import sys
sys.path.append(os.path.abspath(''))
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt

sns.set_style('white')
cmap = sns.diverging_palette(230, 20, as_cmap=True)
fig, axes = plt.subplots(2, 2, figsize=(20, 10),sharey=True,sharex=True)
y_label = []
x_label = []
index = [22,35,39,45]

for i in range(90):
    if i != 20:
        x_label.append("")
    else:
        x_label.append(rf"$\uparrow$")
for i in range(13):
    y_label.append(fr"$a_{{{i}}}$")
for i, id in enumerate(index):
    coef = pd.read_csv(f'./temp/A_{id}.csv',header=0)
    sns.heatmap(coef.iloc[:,:90], cmap=cmap, center=0, robust=True, ax=axes[i//2,i%2])
    axes[i//2,i%2].set_yticks(ticks=np.arange(1,14),labels = y_label, rotation='horizontal',fontsize=12)
    axes[i//2,i%2].set_xticks(ticks=np.arange(90),labels = x_label, fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()
fig.savefig(os.path.join('./png/coef_main.pdf'), format="pdf", bbox_inches="tight", transparent=True)
    
