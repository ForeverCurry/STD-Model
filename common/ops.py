# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Commonly used functions.
"""
import os
import numpy as np
from scipy.linalg import hankel
import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import convolve
import seaborn as sns
from matplotlib.patches import ConnectionPatch
sns.set_style()
os.environ['CUDA_VISIBLE_DEVICE'] = '0'


def hankel_to_scalar(x):
    r, c = x.shape
    x = t.flip(x,dims=(1,))
    y = t.zeros(r+c-1,device=x.device)
    for j in range(r+c-1):
        yi = t.diagonal(x, offset=-r+1+j)
        y[-j-1] = yi.mean()
    return y

def plot_sub(data, index, input_size, step, linewidth, ax):
    sub1_pre = data.iloc[index+1,-step:].values.squeeze()
    groud1_te =data.iloc[index,-step:].values.squeeze()
    groud1_tr = data.iloc[index,:input_size+1].values.squeeze()
    sns.lineplot(x=np.arange(input_size,input_size+step),y=sub1_pre,ax=ax,linewidth=linewidth, 
                 label='STD Prediction',color='#f16c23',marker='.')
    sns.lineplot(x=np.arange(input_size,input_size+step),y=groud1_te,ax=ax,linewidth=linewidth,
                 label='True values',color='#1b7c3d',marker='.')
    sns.lineplot(x=np.arange(0,input_size+1),y=groud1_tr,linewidth=linewidth,ax=ax,
                 color='#2b6a99',marker='.')
    ax.axvspan(0,input_size, color='blue', alpha=0.05)
    ax.legend(prop={'size':8,'family':'Serif'})
    ax.set_xticks([])
    ax.set_xlim(xmin=0)
    # ax.set_ylabel(f"$y^t$",fontdict=ticker_font, rotation='horizontal')
    ax.set_yticks([])
    
def main_plot(data, label,color,ax):
    sns.lineplot(x=np.arange(len(data)),y=data,alpha=0.8,ax=ax,linewidth=2,
                label=label,color=color,marker='.')
    ax.set_xlim(xmin=0,xmax =len(data))
    
def con(ax1,ax2, point, step):
    ymin1, ymax1 = ax1.get_ylim()
    xmin1, xmax1 = ax1.get_xlim()
    ymin2, ymax2 = ax2.get_ylim()
    con = ConnectionPatch(xyB=(xmin1,ymax1), 
                        coordsB=ax1.transData,
                        xyA=(point, ymin2), 
                        coordsA=ax2.transData,
                        linestyle='--',
                        color='#e11e2d',
                        linewidth = 1.5)
    ax1.add_artist(con)
    con = ConnectionPatch(xyB=(xmax1,ymax1), 
                      coordsB=ax1.transData,
                      xyA=(point+step-1, ymin2), 
                      coordsA=ax2.transData,
                      linestyle='--',
                      color='#e11e2d',
                      linewidth = 1.5)
    ax1.add_artist(con)
    ax2.axvline(x=point, color='#e11e2d', linestyle='--',linewidth = 1.5)
    ax2.axvline(x=point+step-1, color='#e11e2d', linestyle='--',linewidth = 1.5)
    
def hank(y_true, input_size, output_size, refine:bool=False):
    '''
    generate hankel matrix and mask
    '''
    hankel_y = hankel(c=y_true[:output_size+1],
                    r=y_true[output_size:])
    if refine:
        mask = np.ones_like(hankel_y)
    else:
        mask = hankel(c=np.ones(input_size))[:output_size+1, :]
    part_y = mask*hankel_y
    return mask, part_y

def soft_threshold(x, threshold):
    """
    Soft threshold function
    """
    return t.sign(x) * t.relu(t.abs(x) - threshold)

def initialize_weights(model):
    """
    Initialize the weights of the model's modules based on their type.

    :param model: the input model

    The function iterates through each module in the model and initializes the weights based on the module type.
    For Conv2d modules, it initializes the weights using Xavier normal initialization and sets the bias to 0.3.
    For Linear modules, it initializes the weights using normal initialization and sets the bias to zeros.
    For BatchNorm2d modules, it sets the weights to 1 and the bias to zeros.

    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):  # Check if the module is a Conv2d layer
            m.weight.data = nn.init.xavier_normal_(m.weight.data)  # Initialize weights using xavier_normal
            if m.bias is not None:  # Check if the module has a bias
                m.bias.data = nn.init.constant_(m.bias.data, 0.3)  # Set bias to 0.3
        elif isinstance(m, nn.Linear):  # Check if the module is a Linear layer
            m.weight.data = nn.init.normal_(m.weight.data)  # Initialize weights using normal distribution
            if m.bias is not None:  # Check if the module has a bias
                nn.init.zeros_(m.bias.data)  # Set bias to zeros
        elif isinstance(m, nn.BatchNorm2d):  # Check if the module is a BatchNorm2d layer
            m.weight.data.fill_(1)  # Set weights to 1
            m.bias.data.zero_()  # Set bias to zeros
