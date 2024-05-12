import matplotlib.pyplot as plt
# plt.style.use(['seaborn-v0_8-paper'])
from matplotlib import ticker
import seaborn as sns
sns.set_theme(style="whitegrid")
import pandas as pd
import numpy as np

color = ['#4995C6','#ADDB88','#B9181A']
color_list = color*7
title_font = {'family': 'Serif', 'color':  'black', 'size': 18}
ticker_font = {'family': 'Serif', 'color':  'black', 'size': 14}
# for i in range(3):
#     c= color[i]
#     color_list.append([c]*7)
color_list = np.asarray(color_list)
color_flat = color_list.ravel()
# columns = ['Lorenz','Nagoya','Fukushima', 'Osaka',  'Picophytoplankton', 'Nanophytoplankton', 'BOLD']
columns = ['Lorenz','N4', 'N12','Fukushima', 'Osaka', 'Weather']
# dict1 = {'Theta(w/ STD)':[ 2.936, 1.648, 1.667, 1.365,1.247,  1.410, 1.099],
#          'ETS(w/ STD)':[   2.496,  1.618, 1.528, 1.313,   1.231,  1.389, 1.102],
#         'Arima(w/ STD)':[2.286, 1.658, 1.596, 1.323, 1.303, 1.276, 1.062],  
#         }
# dict2 = {'Theta(w/o STD)':[ 3.069, 1.696,  1.737, 1.391, 1.286, 1.527, 1.305],
#          'ETS(w/o STD)':[   2.636, 1.640,  1.548, 1.362, 1.333, 1.499, 1.176],  
#         'Arima(w/o STD)':[2.446, 1.677,  1.654, 1.403, 1.419, 1.344, 1.079],   
        # }
dict1 = {'Theta(w/ STD)':[ 2.915, 1.247, 1.410, 1.388, 1.718, 0.801],
         'ETS(w/ STD)':[ 2.737, 1.231, 1.389, 1.361, 1.543, 0.801],
        'Arima(w/ STD)':[3.273, 1.303, 1.276, 1.400, 1.646, 0.817],
        'ARNN(w/ STD)':[4.131, 2.609, 2.485, 2.005, 2.796, 1.034],
        'RDE(w/ STD)':[0.664, 2.053, 2.181, 1.864, 1.791, 1.136]} 
        
dict2 = {'Theta(w/o STD)':[3.278, 1.286, 1.527, 1.391, 1.737, 1.011],
         'ETS(w/o STD)': [3.019, 1.333, 1.499, 1.362, 1.548, 1.007],  
        'Arima(w/o STD)':[3.677,1.419, 1.344, 1.403, 1.654, 1.042 ],
        'ARNN(w/o STD)':[4.160, 2.901, 3.071, 2.069, 2.904, 1.23],
        'RDE(w/o STD)':[1.043 ,2.057 ,2.422 ,1.901 ,1.837 ,1.299]}
fig, axes = plt.subplots(5, 1, figsize=(16, 10),sharex=True)
for i in range(5):
    model = np.ravel([[list(dict1.keys())[i]]*6,[list(dict2.keys())[i]]*6])
    nrmse = np.ravel([list(dict1.values())[i],list(dict2.values())[i]])
    data = pd.DataFrame(data={'model':model,
                              'nrmse':nrmse,
                              'data_name':np.ravel(columns*2)})
    sns.lineplot(data=data,x='data_name',y='nrmse',hue='model',
                 palette="Set1",ax=axes[i],marker='*',markersize=10)
    axes[i].legend(prop={'size':10,'family':'Serif'})
    axes[i].set_ylabel(f'NRMSE',fontdict=ticker_font)
# data1 = np.ravel(list(dict1.values()))
# data2 = np.ravel(list(dict2.values()))
# data_name = columns*3
# model1_name = np.ravel([[model]*6 for model in list(dict1.keys())])
# model2_name = np.ravel([[model]*6 for model in list(dict2.keys())])
# data1= pd.DataFrame(data={'data':data1,'model':model1_name,'dataset':data_name})
# data2= pd.DataFrame(data={'data':data2,'model':model2_name,'dataset':data_name})


# ax = sns.lineplot(data=data1,x='dataset',y='data',hue='model',palette="Set2")
# ax = sns.lineplot(data=data2,x='dataset',y='data',hue='model',palette="Set2",alpha=0.3)
# ax.bar_label(ax.containers[0], fontsize=8,fmt=lambda x: f'{x*100:.1f} %')
# ax.bar_label(ax.containers[1], fontsize=8,fmt=lambda x: f'{x*100:.1f} %')
# ax.bar_label(ax.containers[2], fontsize=8,fmt=lambda x: f'{x*100:.1f} %')
# axes.set_ylabel(f'NRMSE',fontdict=title_font,)
# ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1,decimals=1))
axes[4].set_xlabel('',fontdict=title_font)
# ax.set_ylim(1,1.75)
axes[4].xaxis.set_ticklabels(labels = columns,fontdict=ticker_font)
# ax.set_yticklabels(fontsize=20)

# Construct arrays for the anchor positions of the 16 bars.
# _x = dict.keys()
# _y = columns
# _x = np.arange(3)
# _y = np.arange(7)
# _xx, _yy = np.meshgrid(_x, _y)
# x, y = _xx.ravel(), _yy.ravel()
# z = np.ravel(list(dict.values()),order='F')
# zpos = np.zeros_like(z)
# # z=x+y
# ax.bar3d(x, y, zpos,0.5,0.5,z,color=color_flat,shade=True,alpha=0.8)
# ax.set_xticks([0.25,1.25,2.25])
# ax.set_xticklabels(list(dict.keys()),fontdict=title_font)
# ax.set_yticks(np.arange(1,8))
# ax.set_yticklabels(columns,fontdict=title_font)
# ax.zaxis.set_major_formatter(ticker.PercentFormatter(xmax=1,decimals=1))
# ax.grid(axis='y')
plt.tight_layout()
plt.show()

fig.savefig(r'D:\ML\Time_series\mymodel\png\refine_bar.pdf', format="pdf", bbox_inches="tight")
# fig.savefig(r'D:\ML\Time_series\mymodel\png\refine_bar.png', transparent=True)
