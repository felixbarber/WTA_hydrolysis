import scipy
import skimage
import sys
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from numpy import matlib
import seaborn as sns
import math
import os
import scipy.optimize
import scipy.io as sio
from scipy import stats
import scipy.ndimage

# import h5py
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
from skimage.morphology import disk
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
import image_toolkit as imkit

# norm_cond='LB' # Setting this here since this is the standard normalization condition
norm_cond = 0  # Setting this here since this is the standard normalization condition
data_loc = None
spots = False
comp_times = None

expt_ids = [['/260527_Tun_yocH_induction', '/260521_bAH26', '/260603_yfp_pads']]
data_loc = [['data_ssd3', 'data_ssd3', 'data_ssd3']]
group_id = 'pYocH_mVenus'
pert = 'Tunicamycin added'
norm_cond = 0  # Starting condition. Normally don't need this since the typical SC is LB.
spots = False
celltypes=['WT', r'$\Delta lytE$']
conds=['LB', 'Tun']
label_conds=['LB', '1h Tunicamycin']
plot_var='Average mVenus'

output_dir = './outputs/compiled_data/staining_plots/'
hue = 'Celltype'
label = 'mVenus'

for ind in range(len(expt_ids)):
    temp_df1 = pd.DataFrame()
    for expt_id in expt_ids[ind]:
        path = '/Volumes/' + data_loc[ind][expt_ids[ind].index(expt_id)] + '/Barber_Lab/data/'
        temp_path = path + expt_id + expt_id + '_condition_parameters.pkl'
        with open(temp_path, 'rb') as input:
            expt_vals = pickle.load(input)
        data_dir = "./outputs" + expt_id
        temp_df = pd.read_pickle(data_dir + expt_id)
        # temp_df['Celltype']=expt_vals['Celltype']
        temp_df['expt'] = expt_vals['expt']
        temp_df['Date'] = expt_vals['Date']
        if not (norm_cond == 0):
            temp_df['Normalized Average outline ' + label] = temp_df['Average outline ' + label] / \
                                                             temp_df[temp_df.Condition == norm_cond][
                                                                 'Average outline ' + label].mean()
            temp_df['Normalized Average ' + label] = temp_df['Average ' + label] / \
                                                     temp_df[temp_df.Condition == norm_cond][
                                                         'Average ' + label].mean()
        if expt_ids[ind].index(expt_id) == [0]:
            temp_df1 = pd.DataFrame(columns=temp_df.columns)
        temp_df1 = temp_df1.append(temp_df)


    if ind == 0:
        df = pd.DataFrame(columns=temp_df1.columns)
    df = df.append(temp_df1)

# Now we add columns for growth condition and celltype
# print(df.Condition.values[:10])
temp_celltypes=[celltypes[np.nonzero([celltype in temp_cond for celltype in celltypes])[0][0]] for temp_cond in df.Condition.values]
df['Celltype']=temp_celltypes
temp_conds=[label_conds[np.nonzero([cond in temp_cond for cond in conds])[0][0]] for temp_cond in df.Condition.values]
df['Growth Condition']=temp_conds
# print(temp_celltypes[:10], df.Condition.values[:10], temp_conds[:10])

# Now we make violin plots of average cell intensity grouped by cell type and by tunicamycin condition, normalized to LB.
fig = plt.figure(figsize=[2.5, 2])
sns.set(font_scale=0.9)
sns.set_style("ticks")
ax = plt.subplot(1, 1, 1)
sns.barplot(data=df, x='Growth Condition', y=plot_var,order=label_conds,hue='Celltype',hue_order=celltypes)
plt.legend(loc=[0.0,0.7], title='Celltype')
fig.savefig(output_dir + group_id + '_' + group_id + '_barplot_av.pdf', bbox_inches='tight')
plt.clf()

fig = plt.figure(figsize=[2.5, 2])
sns.set(font_scale=0.9)
sns.set_style("ticks")
ax = plt.subplot(1, 1, 1)
sns.violinplot(data=df, x='Growth Condition', y=plot_var,order=label_conds,hue='Celltype',hue_order=celltypes,inner='quart')
plt.legend(loc=[0.0,0.57], title='Celltype')
fig.savefig(output_dir + group_id + '_' + group_id + '_violinplot_av.pdf', bbox_inches='tight')
plt.clf()

with open(output_dir + group_id + '_' + group_id + '.txt', 'w') as f:
    print(df.groupby('Condition').describe(), file=f)  # Python 3.x
    outs_anova = []
    cond_vals=df.Condition.unique()
    print(plot_var, file=f)
    samples = [np.asarray(df[df['Condition'] == val][plot_var]) for val in cond_vals]
    outs_anova.append(scipy.stats.f_oneway(*samples, axis=0))
    print(cond_vals, file=f)
    if outs_anova[-1][1] < 0.01:
        print(scipy.stats.tukey_hsd(*samples), file=f)
    else:
        print('not significant', file=f)

