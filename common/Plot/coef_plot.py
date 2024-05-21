import os
import sys
sys.path.append(os.path.abspath(''))
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt

cmap = sns.diverging_palette(230, 20, as_cmap=True)
fig, axes = plt.subplots(6, 5, figsize=(20, 10),sharey=True,sharex=True)
y_label = []
x_label = []
for i in range(90):
    if i != 20:
        x_label.append("")
    else:
        x_label.append(rf"$\uparrow$")
for i in range(13):
    y_label.append(fr"$a_{{{i}}}$")
for i in range(20,50):
    coef = pd.read_csv(f'./temp/A_{i}.csv',header=0)
    sns.heatmap(coef.iloc[:,:90], cmap=cmap, center=0, robust=True, ax=axes[(i-20)//5,(i-20)%5])
    axes[(i-20)//5,(i-20)%5].set_yticks(ticks=np.arange(1,14),labels = y_label, rotation='horizontal',fontsize=8)
    axes[(i-20)//5,(i-20)%5].set_xticks(ticks=np.arange(90),labels = x_label, fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()
fig.savefig(os.path.join('./png/coef_plot.pdf'), format="pdf", bbox_inches="tight", transparent=True)
    
