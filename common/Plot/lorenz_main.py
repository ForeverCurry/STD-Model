import os
import sys
sys.path.append(os.path.abspath(''))
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt
from common.losses import nrmse_loss, pearson
from common.settings import RESULT_PATH
from common.Plot.plot import PlotMeta


# Define the basic plot information

ticker2_font = {'family': 'Serif', 'color':  'black', 'size': 16, 
                'bbox':dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5),
                'fontstyle': 'italic'}

#Load data
MVE_result = pd.read_csv(os.path.join(RESULT_PATH,'Lorenz\MVE_ref_Lorenz.csv'))
STD_result = pd.read_csv(os.path.join(RESULT_PATH,'Lorenz\STD_Lorenz.csv'))
RDE_result = pd.read_csv(os.path.join(RESULT_PATH,'Lorenz\RDE_ref_Lorenz.csv'))
arnn_result = pd.read_csv(os.path.join(RESULT_PATH,'Lorenz\ARNN_ref_Lorenz.csv'))

# Define the data information
input_size= 27
output_size=12
index=[22,39,43]


plt.close()
fig, axes = plt.subplots(3, 4, figsize=(16, 9))

axes[0,3].set_title(f'STD',fontdict=PlotMeta.title_font)
axes[0,2].set_title(f'RDE',fontdict=PlotMeta.title_font)
axes[0,1].set_title(f'ARNN',fontdict=PlotMeta.title_font)
axes[0,0].set_title(f'MVE',fontdict=PlotMeta.title_font)


def plot(data,ax_row,ax_col, id,start, end):
    sns.lineplot(x=np.arange(input_size,input_size+output_size),y=data.iloc[id,start:end],
                 marker='.', markersize=PlotMeta.markersize,alpha=0.6,
                ax=axes[ax_row,ax_col], linewidth=PlotMeta.linewidth,color=PlotMeta.color['std'],label='Predictions')
    pcc = pearson(data.iloc[id,start:end].to_numpy(),y_true[input_size:input_size+output_size].to_numpy())
    nrmse = nrmse_loss(data.iloc[id,start:end].to_numpy(),y_true[input_size:input_size+output_size].to_numpy())
    axes[ax_row,ax_col].text(x=0.1,y=.9,ha='left', va='top',transform=axes[ax_row,ax_col].transAxes,
                             s=f'PCC:{pcc:.3f}\nNRMSE:{nrmse:.2f}',fontdict=ticker2_font)
    axes[ax_row,ax_col].get_legend().set_visible(False)
    
for i, id in enumerate(index):
    y_true = STD_result.iloc[2*id,:]
    for j in range(4):
        sns.lineplot(x=np.arange(0,input_size+1),y=y_true[:input_size+1],marker='.',markersize=PlotMeta.markersize,
                    ax=axes[i,j],linewidth=PlotMeta.linewidth,color=PlotMeta.color['tr'],label='Noisy input')
        sns.lineplot(x=np.arange(input_size,input_size+output_size),y=y_true[input_size:input_size+output_size],
                    ax=axes[i,j],linewidth=PlotMeta.linewidth,color=PlotMeta.color['gt'],label='True values',marker='.',markersize=PlotMeta.markersize)
    # plot prediction
    plot(MVE_result,i,0,2*id,input_size,input_size+output_size)
    plot(arnn_result,i,1,2*id,input_size,input_size+output_size)
    plot(RDE_result,i,2,2*id,input_size,input_size+output_size)
    plot(STD_result,i,3,2*id+1,input_size,input_size+output_size)
# Set the axis
for m, axs in enumerate(axes):
    for j, ax in enumerate(axs):  
        ax.axvspan(0,27, color='blue', alpha=0.05)
        ax.set_xticks([])
        ax.yaxis.set_visible(False)
        ax.set_yticks([])
        ax.set_xlim(xmin=0, xmax=input_size+output_size-1)
        
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, prop={'size':20,'family':'Serif'})
fig.tight_layout(rect=[0, 0.08, 1, 1])
plt.show()
fig.savefig(r"./png/lorenz_result.pdf", format="pdf", bbox_inches="tight", transparent=True)
