import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#Define font style for plot title
title_font = {'family': 'serif', 'color':  'black', 'size': 20}
sns.set_style('white')
sns.set_palette('Set1')
data_path = r'.\results' 
linewidth = [1.5,2,1,1,1]

#Load data from CSV files into dataframes
mve_wind1 = pd.read_csv(os.path.join(data_path,rf'Osaka\MVE_ref_Osaka.csv'))
arnn_wind1 = pd.read_csv(os.path.join(data_path,rf'Osaka\ARNN_ref_Osaka.csv'))
lstd_wind1 = pd.read_csv(os.path.join(data_path,rf'Osaka\STD_Osaka.csv'))
RDE_wind1 = pd.read_csv(os.path.join(data_path,rf'Osaka\RDE_ref_Osaka.csv'))
wind1 = [lstd_wind1, mve_wind1, arnn_wind1,RDE_wind1 ]

mve_wind2 = pd.read_csv(os.path.join(data_path,rf'Fukushima\MVE_ref_Fukushima.csv'))
arnn_wind2 = pd.read_csv(os.path.join(data_path,rf'Fukushima\ARNN_ref_Fukushima.csv'))
lstd_wind2 = pd.read_csv(os.path.join(data_path,rf'Fukushima\STD_Fukushima.csv'))
RDE_wind2 = pd.read_csv(os.path.join(data_path,rf'Fukushima\RDE_ref_Fukushima.csv'))
wind2 = [lstd_wind2, mve_wind2, arnn_wind2,RDE_wind2 ]

mve_N4 = pd.read_csv(os.path.join(data_path,rf'N4\MVE_ref_N4.csv'))
arnn_N4 = pd.read_csv(os.path.join(data_path,rf'N4\ARNN_ref_N4.csv'))
lstd_N4 = pd.read_csv(os.path.join(data_path,rf'N4\STD_N4.csv'))
RDE_N4 = pd.read_csv(os.path.join(data_path,rf'N4\RDE_ref_N4.csv'))
N4 = [lstd_N4,mve_N4,arnn_N4,RDE_N4 ]

mve_N12 = pd.read_csv(os.path.join(data_path,rf'N12\MVE_ref_N12.csv'))
arnn_N12 = pd.read_csv(os.path.join(data_path,rf'N12\ARNN_ref_N12.csv'))
lstd_N12 = pd.read_csv(os.path.join(data_path,rf'N12\STD_N12.csv')) 
RDE_N12 = pd.read_csv(os.path.join(data_path,rf'N12\RDE_ref_N12.csv'))
N12 = [lstd_N12,mve_N12,arnn_N12,RDE_N12 ]

mve_weather = pd.read_csv(os.path.join(data_path,rf'weather\MVE_ref_weather.csv'))
arnn_weather = pd.read_csv(os.path.join(data_path,rf'weather\ARNN_ref_weather.csv'))
lstd_weather = pd.read_csv(os.path.join(data_path,rf'weather\STD_weather.csv'))
RDE_weather = pd.read_csv(os.path.join(data_path,rf'weather\RDE_ref_weather.csv'))
weather = [lstd_weather,mve_weather,arnn_weather,RDE_weather]

mve = [mve_weather,mve_N12,mve_N4,mve_wind1,mve_wind2]
arnn = [arnn_weather,arnn_N12,arnn_N4,arnn_wind1,arnn_wind2]
rde = [RDE_weather,RDE_N12,RDE_N4,RDE_wind1,RDE_wind2]
std = [lstd_weather,lstd_N12,lstd_N4,lstd_wind1,lstd_wind2]

# Define the index and input/output size for each dataset
index = [34,33,43,51,71]
input_size = [14,10,10,48,48]
output_size = [6,4,4,12,12]

# Create subplots for the plot
plt.close()
gs = gridspec.GridSpec(2, 6)
axes1 = plt.subplot(gs[0,:2])
axes2= plt.subplot(gs[0,2:4])
axes3= plt.subplot(gs[0,4:6])
axes4= plt.subplot(gs[1,1:3])
axes5= plt.subplot(gs[1,3:5])

axes = [axes1,axes2,axes3,axes4,axes5]

axes1.set_title(f'Weather',fontdict=title_font)
axes2.set_title(f'N4',fontdict=title_font)
axes3.set_title(f'N12',fontdict=title_font)
axes4.set_title(f'Fukushima',fontdict=title_font)
axes5.set_title(f'Osaka',fontdict=title_font)

y_true_weather = lstd_weather.iloc[2*34,:]
y_true_n3 = lstd_N4.iloc[2*33,:]
y_true_n11 = lstd_N12.iloc[2*43,:]
y_true_wind1 = lstd_wind1.iloc[2*51,:]
y_true_wind2 =  lstd_wind2.iloc[2*71,:]
y_true = [y_true_weather,y_true_n3,y_true_n11,y_true_wind1,y_true_wind2]

label = ['STD','MVE','ARNN','RDE']
linestyle = ['solid','dashed','dotted','dashdot']
alphas=(0.6,0.6,0.6,1)
for k,results in enumerate((weather,N4,N12,wind1,wind2)):

    sns.lineplot(x=np.arange(input_size[k],input_size[k]+output_size[k]),y=y_true[k].iloc[-output_size[k]:],
                    ax=axes[k//3*3+k%3],linewidth=linewidth[0],label='Groudtruth',marker='.')
    sns.lineplot(x=np.arange(0,input_size[k]+1),y=y_true[k].iloc[:input_size[k]+1],ax=axes[k//3*3+k%3],
                    linewidth=linewidth[0],marker='.')

    for r,result in enumerate(results):
        if r==0:
            sns.lineplot(x=np.arange(input_size[k],input_size[k]+output_size[k]),y=result.iloc[2*index[k]+1,-output_size[k]:],
                     linewidth=linewidth[r+1],ax=axes[k//3*3+k%3],label=label[r],marker='.',alpha=alphas[r])
        else:
            sns.lineplot(x=np.arange(input_size[k],input_size[k]+output_size[k]),y=result.iloc[2*index[k],-output_size[k]:],
                    ax=axes[k//3*3+k%3],linewidth=linewidth[r+1],label=label[r],alpha=alphas[r],marker='.')
    axes[k//3*3+k%3].yaxis.set_visible(False)
    axes[k//3*3+k%3].legend(prop={'size':7,'family':'Serif'},
                            fancybox=True, shadow=True,
                            ncol=2)
    axes[k//3*3+k%3].set_xlim(xmin=0,xmax=input_size[k]+output_size[k]-1)
    axes[k//3*3+k%3].axvspan(0,input_size[k], color='blue', alpha=0.05)
fig = plt.gcf()
fig.set_size_inches(16, 6)
plt.tight_layout()
plt.show()
fig.savefig(r'.\png\result_real.pdf', format="pdf", bbox_inches="tight")



