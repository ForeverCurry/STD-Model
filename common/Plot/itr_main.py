import os
import sys
sys.path.append(os.path.abspath(''))
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt
from common.Plot.plot import PlotMeta

fig, axes = plt.subplots(2, 4, figsize=(20, 10),sharex=True)
input_size = 27
output_size = 12
index = [39,43]
itrs = [[10, 100, 200, 320],[10,20,50,80]]
label=['STD pridictions', 'True values', 'STD fitting curve', 'Noisy input']
y = pd.read_csv('./results/Lorenz/STD_Lorenz.csv')
for i, id in enumerate(index):
    df = pd.read_csv(f'./temp/{id}_data.csv')
    y_true = y.iloc[2*id,:]
    for j, itr in enumerate(itrs[i]):
        axes[i,j].set_title(f"{itr*50}th iteration", fontdict=PlotMeta.title_font)
        sns.lineplot(x=np.arange(0,28),y=y_true.iloc[:input_size+1],ax=axes[i,j],linewidth=PlotMeta.linewidth,
                 color=PlotMeta.color['tr'],label=label[3],marker='.',markersize=PlotMeta.markersize)
        sns.lineplot(x=np.arange(27,39),y=y_true.iloc[-output_size:],ax=axes[i,j],linewidth=PlotMeta.linewidth, 
                    color=PlotMeta.color['gt'],label=label[1],marker='.',markersize=PlotMeta.markersize)
        sns.lineplot(x=np.arange(27,39),y=df.iloc[itr,-output_size:],linewidth=PlotMeta.linewidth,ax=axes[i,j],alpha=0.6,
                    color=PlotMeta.color['std'],label=label[0],marker='.',markersize=PlotMeta.markersize)
        sns.lineplot(x=np.arange(0,28),y=df.iloc[itr,:28],linewidth=PlotMeta.linewidth,alpha=0.6,
                    ax=axes[i,j],color=PlotMeta.color['std_fit'],label=label[2],marker='.',markersize=PlotMeta.markersize)
        axes[i,j].axvspan(0,27, color='blue', alpha=0.05)
        axes[i,j].get_legend().set_visible(False)
        axes[i,j].set_xlim(xmin=0,xmax=input_size+output_size-1)
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=20)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.show()
fig.savefig(r"./png/itr_main.pdf", format="pdf", bbox_inches="tight", transparent=True)
