import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from common.ops import main_plot, plot_sub, con
from common.settings import RESULT_PATH
from dataclasses import dataclass

@dataclass
class PlotMeta:
    color= {'std':'#f16c23',
            'gt':'#1b7c3d',
            'tr':'#2b6a99'}
    label = ['STD Predictions', 
                   'True Values']
    title_font = {'family': 'Serif', 'color':  'black', 'size': 14}
    tick_font = {'family': 'Serif', 'color':  'black', 'size': 6}
    linewidth = 1

def plot_result(paths:list, input_size:int, step:int, test_size:int, 
                index:list, titles:list, save_path:str):
    # Function to plot the results of the model
    # paths: list of file paths containing the result data
    # input_size: size of the input data
    # step: pridiction step size
    # test_size: size of the test data
    # index: list containing indices for plotting
    # save_path: path to save the plot
    rows = len(paths)
    cols = np.shape(index)[1]
    ### construct the grid
    gs = gridspec.GridSpec(rows*2, cols)
    axes = []
    for i in range(rows):
        axes.append(plt.subplot(gs[2*i, :]))
        for j in range(cols):
            axes.append(plt.subplot(gs[2*i+1, j:j+1]))
            
    ### Plot
    for i, path in enumerate(paths):
        data = pd.read_csv(os.path.join(RESULT_PATH,path))
        pre = data.iloc[-2*test_size+1,-step:].squeeze()
        ground_truth = data.iloc[-2*test_size,-step:].squeeze()
        for k in range(1, test_size):
            pre = np.concatenate((pre,data.iloc[-2*(test_size-k)+1, -step:].squeeze()),axis=None)
            ground_truth = np.concatenate((ground_truth,data.iloc[-2*(test_size-k), -step:].squeeze()),axis=None)
        main_plot(pre, PlotMeta.label[0], PlotMeta.color['std'], axes[i*(cols+1)])
        main_plot(ground_truth, PlotMeta.label[1], PlotMeta.color['gt'], axes[i*(cols+1)])
        axes[i*(cols+1)].set_title(titles[i], fontdict=PlotMeta.title_font)
        for j in range(cols):
            plot_sub(data, -2*(test_size-index[i][j]+1),input_size, step, 
                     PlotMeta.linewidth,axes[j+1+i*(cols+1)])
            con(axes[j+1+i*(cols+1)], axes[i+i*cols], (index[i][j]-1)*step, step)
    ### show and save the plot

    fig = plt.gcf()
    fig.set_size_inches(16, 5*rows)
    
    plt.tight_layout(h_pad=0.1,w_pad=0.1)
    plt.show()
    fig.savefig(save_path, format="pdf", bbox_inches="tight") 
