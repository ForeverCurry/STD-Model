import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
from common.ops import main_plot, plot_sub, con
print(os.getcwd())
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
data_path = r'D:\ML\Time_series\STD_model\results' 
plankton4 = pd.read_csv(os.path.join(data_path,rf'.\N4\STD_N4.csv'))
plankton11 = pd.read_csv(os.path.join(data_path,rf'.\N12\STD_N12.csv'))
linewidth = 1
# fig, axes = plt.subplots(1,1, figsize=(16, 5))
# fig, axes = plt.subplots(2,1, sharex=True, figsize=(16, 10))
gs = gridspec.GridSpec(4, 3)
axes2 = plt.subplot(gs[1,:1])
axes3= plt.subplot(gs[1,1:2])
axes4= plt.subplot(gs[1,2:3])
axes1= plt.subplot(gs[0,:])
axes6 = plt.subplot(gs[3,:1])
axes7= plt.subplot(gs[3,1:2])
axes8= plt.subplot(gs[3,2:3])
axes5= plt.subplot(gs[2,:])
# ax = fig.add_subplot(31,sharey=True)
axes = [axes1,axes2,axes3,axes4,axes5,axes6,axes7,axes8]
# axes[1].set_xlabel('Time',fontdict=title_font)
# axes[0].set_ylabel('',fontdict=title_font)
# axes[1].set_ylabel('',fontdict=title_font)
start = 0
stop = 24
step = 4
input_size = 10
linewidth = 1
pre_4 = plankton4.iloc[39,-step:].values.squeeze()
ground_truth4 = plankton4.iloc[38,-step:].squeeze()
pre_11 = plankton11.iloc[39,-step:].values.squeeze()
ground_truth11 = plankton11.iloc[38,-step:].squeeze()
# axes.set_xticks(ticks=np.arange(0,300,10),labels = np.arange(0,150,5), fontdict=tick_font)
plot_sub(plankton4,48,input_size,step,linewidth,axes[1])
plot_sub(plankton4,60,input_size,step,linewidth,axes[2])
plot_sub(plankton4,90,input_size,step,linewidth,axes[3])
plot_sub(plankton11,44,input_size,step,linewidth,axes[5])
plot_sub(plankton11,66,input_size,step,linewidth,axes[6])
plot_sub(plankton11,90,input_size,step,linewidth,axes[7])
for i in range(20,50):
    pre_4 = np.concatenate((pre_4,plankton4.iloc[2*i+1,-step:].values.squeeze()),axis=None)
    ground_truth4 = np.concatenate((ground_truth4,plankton4.iloc[2*i,-step:].squeeze()),axis=None)
    pre_11 = np.concatenate((pre_11,plankton11.iloc[2*i+1,-step:].values.squeeze()),axis=None)
    ground_truth11 = np.concatenate((ground_truth11,plankton11.iloc[2*i,-step:].squeeze()),axis=None)
pcc4 = np.corrcoef(pre_4,ground_truth4)[0][1]
pcc11 = np.corrcoef(pre_11,ground_truth11)[0][1]
data_4 = {'STD Prediction':pre_4,'Ground truth':ground_truth4}
data_11 = {'STD Prediction':pre_11,'Ground truth':ground_truth11}
axes[0].set_title(f"plankton N4 (PCC:{pcc4:.3f})",fontdict=title_font)
axes[4].set_title(f"plankton N12 (PCC:{pcc11:.3f})",fontdict=title_font)

# axes[3].set_title(f"Blood oxygen level-dependent (Pearson correlation:{pcc:.3f})",fontdict=title_font)

main_plot(pre_4,label[0],color['std'],axes[0])
main_plot(ground_truth4,label[1],color['gt'],axes[0])
main_plot(pre_11,label[0],color['std'],axes[4])
main_plot(ground_truth11,label[1],color['gt'],axes[4])
# sns.lineplot(x=np.arange(len(ground_truth4)),y=ground_truth4,alpha=0.8,ax=axes[0],linewidth=2,
#              label='True values',color='#1b7c3d',marker='.')
# sns.lineplot(data=data_11,alpha=0.8,ax=axes[4],linewidth=2)

axes[0].legend(prop={'size':8,'family':'Serif'})
axes[4].legend(prop={'size':8,'family':'Serif'})
# draw bottom connecting line
# x = r * np.cos(np.pi / 180 * theta1) + center[0]
# y = r * np.sin(np.pi / 180 * theta1) + center[1]

# Extract the top-left coordinate
# top_left = (bbox.x0, bbox.y1)

con(axes[1],axes[0],20,step)
con(axes[2],axes[0],44,step)
con(axes[3],axes[0],104,step)
con(axes[5],axes[4],12,step)
con(axes[6],axes[4],56,step)
con(axes[7],axes[4],104,step)
fig = plt.gcf()
# fig.add_artist(con)
fig.set_size_inches(16, 10)
# con.set_color('red')

# con.set_linewidth(4)
# sns.lineplot(data=data_,legend='auto',alpha=0.8,ax=axes[3])
plt.tight_layout(pad=0.5,h_pad=0.2)
plt.show()
# fig.savefig(r'D:\ML\Time_series\mymodel\png\plankton.pdf', format="pdf", bbox_inches="tight")
# fig.savefig(r'D:\ML\Time_series\mymodel\png\plankton.png',transparent=True)