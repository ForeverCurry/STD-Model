import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from common.losses import nrmse_loss
from common.Plot.plot import PlotMeta
from common.settings import RESULT_PATH

# define the plot information
title_font = {'family': 'serif', 'color':  'black'}
subtitle_font = {'family': 'serif', 'color':  'black', 'size': 14}

sns.set_style('white')
sns.set_palette('Set1')

fig, axes = plt.subplots(3, 2, figsize=(10, 10))

# define the data information
data = 'Lorenz'
input_size = 27
output_size = 12
id = 45

#Load data from CSV files into dataframes
theta = pd.read_csv(os.path.join(RESULT_PATH,rf'{data}\theta_ref_{data}.csv'))
ets = pd.read_csv(os.path.join(RESULT_PATH,rf'{data}\ETS_ref_{data}.csv'))
arima = pd.read_csv(os.path.join(RESULT_PATH,rf'{data}\ARIMA_ref_{data}.csv'))
mve = pd.read_csv(os.path.join(RESULT_PATH,rf'{data}\MVE_ref_{data}.csv'))
arnn = pd.read_csv(os.path.join(RESULT_PATH,rf'{data}\ARNN_ref_{data}.csv'))
std = pd.read_csv(os.path.join(RESULT_PATH,rf'{data}\STD_{data}.csv'))
RDE = pd.read_csv(os.path.join(RESULT_PATH,rf'{data}\RDE_ref_{data}.csv'))
dataset = [ets, arima, theta, mve, arnn,RDE]

label = ['ETS', 'ARIMA', 'Theta', 'MVE', 'ARNN','RDE']

y_true = std.iloc[2*id,:]
for k,result in enumerate(dataset):
    sns.lineplot(x=np.arange(input_size,input_size+output_size),y=y_true.iloc[-output_size:],color=PlotMeta.color['gt'],
                ax=axes[k//2,k%2],linewidth=PlotMeta.linewidth,label='Groudtruth',marker='.', markersize=PlotMeta.markersize)    
    sns.lineplot(x=np.arange(0,input_size+1),y=y_true.iloc[:input_size+1],color=PlotMeta.color['tr'],
                ax=axes[k//2,k%2],linewidth=PlotMeta.linewidth,marker='.',markersize=PlotMeta.markersize)
    sns.lineplot(x=np.arange(input_size,input_size+output_size),y=result.iloc[2*id,-output_size:],color=PlotMeta.color['other'],
                ax=axes[k//2,k%2],linewidth=PlotMeta.linewidth,label=label[k],marker='.',markersize=PlotMeta.markersize)
    sns.lineplot(x=np.arange(input_size,input_size+output_size),y=result.iloc[2*id+1,-output_size:],color=PlotMeta.color['std'],
                ax=axes[k//2,k%2],linewidth=PlotMeta.linewidth,label=f'{label[k]}+STD',marker='.',markersize=PlotMeta.markersize,alpha=0.6)
    ref_nrmse = nrmse_loss(result.iloc[2*id+1,-output_size:].to_numpy(),y_true[input_size:input_size+output_size].to_numpy())
    nrmse = nrmse_loss(result.iloc[2*id,-output_size:].to_numpy(),y_true[input_size:input_size+output_size].to_numpy())

    axes[k//2,k%2].set_title(f'{label[k]} NRMSE:{nrmse:.2f} and refined NRMSE:{ref_nrmse:.2f}',fontdict=subtitle_font)
    axes[k//2,k%2].yaxis.set_visible(False)
    axes[k//2,k%2].get_legend().set_visible(False)
    axes[k//2,k%2].set_xlim(xmin=0,xmax=input_size+output_size-1)
    axes[k//2,k%2].axvspan(0,input_size, color='blue', alpha=0.05)
handles, labels = axes[0,0].get_legend_handles_labels() 
fig.legend(handles, ['True values', 'Base', '+STD'], loc='lower center', ncol=5, prop={'size': 18, 'family': 'Serif'}, fancybox=True, shadow=True)
plt.tight_layout(w_pad=1,h_pad=1, rect=[0, 0.05, 1, 1])
plt.show()
fig.savefig(rf'.\png\refined_maintext.pdf', format="pdf", bbox_inches="tight",transparent=True)
