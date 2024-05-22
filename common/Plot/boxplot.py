import os
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
from common.settings import RESULT_PATH
from common.Plot.plot import PlotMeta
names =['length', 'noisy']
methods = [ 'STD','RDE','MVE','Theta','ETS','Arima','ARNN',]
nrmses = [[],[]]
test_size=30
# load data
for i, name in enumerate(names):
    for j, method in enumerate(methods):
        if i==0 and j==6:  ## ARNN has no length-dependent results
            pass
        else:
            nrmse = pd.read_csv(f'./results/Lorenz/robust_{name}_{method}.csv',header=0).iloc[-test_size:,:]
            nrmse_log = nrmse.apply(np.log)
            nrmse_log = nrmse_log.melt(value_name='nrmse',var_name=f'{name}')
            nrmse_log['model'] = f'{method}'
            nrmses[i].append(nrmse_log)
length_df = pd.concat(nrmses[0], ignore_index=True)
noisy_df =  pd.concat(nrmses[1], ignore_index=True)
data = [length_df,noisy_df]
# Plot boxplot
fig, axes = plt.subplots(2, 1, figsize=(20,10))
plt.sca(axes[0])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.sca(axes[1])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
for k, ax in enumerate(axes):
    
    axes[k].set_ylabel("NRMSE(Log-scale)",fontdict=PlotMeta.tick_font)
    axes[k].set_title(f'Performance comparison under various {names[k]}.',fontdict=PlotMeta.title_font)
    sns.boxplot(data=data[k],x=f'{names[k]}', y='nrmse', hue='model', gap=0.1, palette="Set2",ax=axes[k])
    axes[k].legend()
    axes[k].get_legend().set_visible(False)
axes[0].set_xlabel("Input length",fontdict=PlotMeta.title_font)
axes[1].set_xlabel("Noise level",fontdict=PlotMeta.title_font)
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(methods), bbox_to_anchor=(0.5, 0.05),prop={'size':20,'family':'Serif'})

# 显示图形
plt.tight_layout(rect=(0,0.1,1,0.95))
# fig.savefig(fr"D:\ML\Time_series\mymodel\png\robust_{name2}.png",transparent=True)
fig.savefig(r"./png/robust.pdf", format="pdf", bbox_inches="tight",transparent=True)
plt.show()
