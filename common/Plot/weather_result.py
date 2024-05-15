import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
from common.ops import plot_sub, main_plot, con
# print(os.getcwd())
# def data_construct():
# sns.set_palette('Set1')
# palette = sns.color_palette('Set1')
color = {'std':'#f16c23',
         'gt':'#1b7c3d',
         'tr':'#2b6a99'}
label = ['STD Predictions', 'True Values']
title_font = {'family': 'Serif', 'color':  'black', 'size': 14}
tick_font = {'family': 'Serif', 'color':  'black', 'size': 6}
sns.set_style('white')
data_path = r'D:\ML\Time_series\STD_model\results\weather' 
weather = pd.read_csv(os.path.join(data_path,rf'STD_weather.csv'))

linewidth = 1
# fig, axes = plt.subplots(1,1, figsize=(16, 5))
# fig, axes = plt.subplots(2,1, sharex=True, figsize=(16, 10))
gs = gridspec.GridSpec(2, 3)
axes1= plt.subplot(gs[0,:])
axes2 = plt.subplot(gs[1,:1])
axes3= plt.subplot(gs[1,1:2])
axes4= plt.subplot(gs[1,2:3])


# ax = fig.add_subplot(31,sharey=True)
axes = [axes1,axes2,axes3,axes4]

step = 6
input_size = 14
linewidth = 1

plot_sub(weather,46,input_size,step, linewidth,axes[1])
plot_sub(weather,70,input_size,step, linewidth,axes[2])
plot_sub(weather,90,input_size,step, linewidth,axes[3])

pre = weather.iloc[39,-step:].values.squeeze()
ground_truthbold = weather.iloc[38,-step:].squeeze()
for i in range(20,49):
    pre = np.concatenate((pre,weather.iloc[2*i+1,-step:].values.squeeze()),axis=None)
    ground_truthbold = np.concatenate((ground_truthbold,weather.iloc[2*i,-step:].squeeze()),axis=None)

pcc = np.corrcoef(pre,ground_truthbold)[0][1]

data = {'STD Prediction':pre,'Ground truth':ground_truthbold}

axes[0].set_title(f"Operative temperature (PCC:{pcc:.3f})",fontdict=title_font)

main_plot(pre,label[0],color['std'],axes[0])
main_plot(ground_truthbold,label[1],color['gt'],axes[0])

axes[0].legend(prop={'size':8,'family':'Serif'})

# draw bottom connecting line

con(axes[1],axes[0],24,step)
con(axes[2],axes[0],96,step)
con(axes[3],axes[0],156,step)

fig = plt.gcf()

fig.set_size_inches(16, 5)

plt.tight_layout(pad=0.5,h_pad=0.2)
plt.show()
# fig.savefig(r'D:\ML\Time_series\mymodel\png\weather.pdf', format="pdf", bbox_inches="tight")
# fig.savefig(r'D:\ML\Time_series\mymodel\png\weather.png',transparent=True)