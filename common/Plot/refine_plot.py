import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from common.losses import nrmse_loss
from common.Plot.plot import PlotMeta
from common.settings import RESULT_PATH
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d','--data', type=str, default='Lorenz', help='dataset name')
parser.add_argument('-i', '--input_size', type=int, default=12, help='input size')
parser.add_argument('-o', '--output_size', type=int, default=1, help='output size')
parser.add_argument('-id', '--index', action='append', type=int,required=True, help='index of test data')
args = parser.parse_args()

title_font = {'family': 'serif', 'color':  'black'}
subtitle_font = {'family': 'serif', 'color':  'black', 'size': 10}

sns.set_style('white')
sns.set_palette('Set1')

def refine_plot(data, input_size, output_size, index:list, linewidth:float=1.5):
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

    fig, axes = plt.subplots(6, 3, figsize=(15, 30))

    for k,result in enumerate(dataset):
        for i, id in enumerate(index):
            y_true = std.iloc[2*id,:]
            sns.lineplot(x=np.arange(input_size,input_size+output_size),y=y_true.iloc[-output_size:],
                        ax=axes[k,i],linewidth=linewidth,label='Groudtruth',marker='.',color=PlotMeta.color['gt'])
            sns.lineplot(x=np.arange(0,input_size+1),y=y_true.iloc[:input_size+1],
                        ax=axes[k,i],linewidth=linewidth,marker='.',color=PlotMeta.color['tr'])
            sns.lineplot(x=np.arange(input_size,input_size+output_size),y=result.iloc[2*id,-output_size:],
                        ax=axes[k,i],label=label[k],marker='.',color=PlotMeta.color['other'])
            sns.lineplot(x=np.arange(input_size,input_size+output_size),y=result.iloc[2*id+1,-output_size:],
                        ax=axes[k,i],label=f'{label[k]}+STD',marker='.',color=PlotMeta.color['std'])
            ref_nrmse = nrmse_loss(result.iloc[2*id+1,-output_size:].to_numpy(),y_true[input_size:input_size+output_size].to_numpy())
            nrmse = nrmse_loss(result.iloc[2*id,-output_size:].to_numpy(),y_true[input_size:input_size+output_size].to_numpy())
        
            axes[k,i].set_title(f'{label[k]} NRMSE:{nrmse:.2f} and refined NRMSE:{ref_nrmse:.2f}',fontdict=subtitle_font)
            axes[k,i].yaxis.set_visible(False)
            axes[k,i].legend(prop={'size':6,'family':'Serif'},
                        fancybox=True, shadow=True, ncol=3)
            axes[k,i].set_xlim(xmin=0,xmax=input_size+output_size-1)
            axes[k,i].axvspan(0,input_size, color='blue', alpha=0.05)
    fig.suptitle(f'Refined Results on {data} dataset', fontdict=title_font,size=16)
    plt.tight_layout(w_pad=2, h_pad=5)
    plt.show()

if __name__ == '__main__':
    print(args.index)
    refine_plot(args.data, args.input_size, args.output_size, args.index)
