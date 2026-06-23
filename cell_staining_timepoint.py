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
sns.set_style("white")
from skimage.morphology import disk
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
import image_toolkit as imkit

#base_path = '/mnt/d/Documents_D/Rojas_lab/data/'
# base_path = '/Users/felixbarber/Documents/Rojas_lab/data/'
# base_path = '/Volumes/data_ssd1/Rojas_Lab/data/'
base_path = '/Volumes/data_ssd3/Barber_Lab/data/'
thresh_peak=900
rescale=False  # Means that we rescale according to mean rather than minimum value. Accounts for things being lower intensity in HADa vs EDADA staining
min_thresh=0.0

# Note: Scenes_remove starts counting from 1.

expt_id, date = '/260617_Tun_Spo0A_act', '6/17/26'
conds=["bBL40_LB", "bBL40_Tun", "bBL41_LB", "bBL41_Tun","bBL42_LB", "bBL42_Tun"]
conditions = ['PspoIIG-GFP LB', 'PspoIIG-GFP Tun', 'PabrB-YFP LB', 'PabrB-YFP Tun', 'PspacC-GFP LB', 'PspacC-GFP Tun', ]
time_labels = ['0 min', '60 min','0 min', '60 min','0 min', '60 min']
num_scenes = [10, 9,10,10,7,7]
scenes_remove = [[],[],[],[],[],[]]  # scenes to remove from consideration
channels, HADA_channel, im_shape = [2], 2, [2304, 2304]
label = 'Reporter'
thresh = 0  # threshold average outline fluorescence to filter out debris

# expt_id, date = '/260603_yfp_pads', '6/3/26'
# conds=["27_LB", "27_tun", "26a_LB", "26a_tun"]
# conditions = ['WT LB','WT Tun', r'$\Delta lytE$ LB', r'$\Delta lytE$ Tun']
# time_labels = ['0 min', '60 min', '0 min', '60 min']
# num_scenes = [16, 16, 14, 14]
# scenes_remove = [[],[],[],[]]  # scenes to remove from consideration
# channels, HADA_channel, im_shape = [2], 2, [2304, 2304]
# label = 'mVenus'
# thresh = 0  # threshold average outline fluorescence to filter out debris

# expt_id, date = '/260527_Tun_yocH_induction', '5/27/26'
# conds=["bBL27_LB", "bAH26_LB", "bAH26_Tun"]
# conditions = ['WT LB', r'$\Delta lytE$ LB', r'$\Delta lytE$ Tun']
# time_labels = ['0 min', '60 min']
# num_scenes = [9, 10, 10]
# scenes_remove = [[],[],[]]  # scenes to remove from consideration
# channels, HADA_channel, im_shape = [2], 2, [2304, 2304]
# label = 'mVenus'
# thresh = 0  # threshold average outline fluorescence to filter out debris


# expt_id, date = '/260415_bBL27_Tun_induction', '4/15/26'
# conds=['LB', '1h_Tun']
# conditions = ['LB', '1h Tunicamycin treatment']
# time_labels = ['0 min', '60 min']
# num_scenes = [10, 10]
# scenes_remove = [[],[]]  # scenes to remove from consideration
# channels, HADA_channel, im_shape = [2], 2, [2304, 2304]
# label = 'mVenus'
# thresh = 0  # threshold average outline fluorescence to filter out debris

# expt_id, date = '/260521_bAH26', '5/21/26'
# conds=['LB', 'Tun']
# conditions = [r'$\Delta lytE$ LB', r'$\Delta lytE$ Tun']
# time_labels = ['0 min', '60 min']
# num_scenes = [10, 14]
# scenes_remove = [[],[]]  # scenes to remove from consideration
# channels, HADA_channel, im_shape = [2], 2, [2304, 2304]
# label = 'mVenus'
# thresh = 0  # threshold average outline fluorescence to filter out debris

out_dir="./outputs"+expt_id
pval=0.01

if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    os.mkdir(out_dir+'/images')

expt_vals_ls=[]
for i0 in range(len(conds)):
    expt_vals_ls.append({'expt_id':expt_id, 'channels':channels,'im_shape':im_shape,
           'base_path':base_path, 'num_scenes':num_scenes[i0], 'cond':conds[i0], 'HADA_channel':HADA_channel, 'date':date,
           'excl_scenes':scenes_remove[i0],'thresh_peak':thresh_peak, 'min_thresh':min_thresh, 'rescale':rescale})

if not os.path.exists(out_dir+expt_id):
    dfs=[]
    for i0 in range(len(expt_vals_ls)):
        print(i0)
        # expt_vals_ls[i0]['median_vals']=imkit.timepoint_bkgd_fluorescence_calculation(expt_vals_ls[i0])
        # dfs.append(imkit.timepoint_import_data_outline(expt_vals_ls[i0]))
        temp_out,temp_ims=imkit.timepoint_import_data_outline_smart_bkgd(expt_vals_ls[i0])
        for i1 in range(len(temp_ims)):
            fig=plt.figure(figsize=[8,8])
            plt.imshow(temp_ims[i1])
            plt.axis('off')
            fig.savefig(out_dir+'/images'+expt_vals_ls[i0]['expt_id']+'_'+expt_vals_ls[i0]['cond']+'_s{0}.png'.format(i1),dpi=300,bbox_inches='tight')
            plt.clf()
        dfs.append(temp_out)
        del temp_ims, temp_out
        dfs[i0]['Condition']=conditions[i0]
        dfs[i0]['Date']=expt_vals_ls[i0]['date']
        
    for i0 in range(len(dfs)):  
        temp_df=dfs[i0].rename(columns={'Average F{0}'.format(expt_vals_ls[i0]['HADA_channel']):'Average '+label,
                            'Integrated F{0}'.format(expt_vals_ls[i0]['HADA_channel']):'Integrated '+label,
                                       'Average outline F{0}'.format(expt_vals_ls[i0]['HADA_channel']):'Average outline '+label,
                            'Integrated outline F{0}'.format(expt_vals_ls[i0]['HADA_channel']):'Integrated outline '+label,
                            'CV outline F{0}'.format(expt_vals_ls[i0]['HADA_channel']):'CV outline '+label,
                            'SD outline F{0}'.format(expt_vals_ls[i0]['HADA_channel']):'SD outline '+label,
                            'Skew outline F{0}'.format(expt_vals_ls[i0]['HADA_channel']):'Skew outline '+label})
        if i0==0:
            df=temp_df.copy()
        else:
            df = df.append(temp_df[[obj for obj in df.columns]])
    df.to_pickle(out_dir+expt_id)
else:
    df=pd.read_pickle(out_dir+expt_id)

df = df[df['Average outline '+label] > thresh]  # filtering out debris that isn't fluorescent.


def iter_plotting(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel,plot_points=True,showfliers=True, temp_hue=None):
    # fig=plt.figure(figsize=[6,4])
    fig = plt.figure()
    ax=plt.subplot(1,1,1)
    if temp_hue is None:
        plot=sns.boxplot(x=temp_iter,y=temp_var,data=df,fliersize=0,showfliers=showfliers)
        if plot_points:
            plot=sns.stripplot(x=temp_iter,y=temp_var,dodge=True,data=df,color='k',edgecolor='black',linewidth=1,alpha=0.1)
    else:
        plot = sns.boxplot(x=temp_iter, y=temp_var, data=df, fliersize=0, showfliers=showfliers, hue=temp_hue)
        if plot_points:
            plot = sns.stripplot(x=temp_iter, y=temp_var, dodge=True, data=df, color='k', edgecolor='black',
                                 linewidth=1, alpha=0.1, hue=temp_hue)
    plot.set(xlabel=temp_xlabel,ylabel=temp_ylab)
    plt.xticks(rotation = 90)
    
    print('Statistical significances for ', temp_var)
    temp_iter_vals=df[temp_iter].unique()
    print(temp_iter_vals)
    for i0 in range(len(temp_iter_vals)):
        cond1=temp_iter_vals[i0]
        for cond2 in temp_iter_vals[i0:]:
            if cond1!=cond2:
                x1=df[df[temp_iter]==cond1][temp_var]
                x2=df[df[temp_iter]==cond2][temp_var]
                print('Condition 1: '+cond1+',','Condition 2: '+cond2+':', scipy.stats.ttest_ind(x2, x1, axis=0, equal_var=False, nan_policy='propagate')[1] < pval)

    return fig,ax
print(df.Scene[:10])
# Now we plot the results and save the figures

sys.stdout = open(out_dir+expt_id+'_outputs.txt', "w")

# HADA average length density of puncta
temp_var, temp_out_name, temp_ylab='Cell spots/length', 'cell_spots_length', r'Cellular puncta/$\mu$m'
temp_iter,temp_xlabel='Condition','Condition'
fig,ax=iter_plotting(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel,showfliers=False)
# ax.set_ylim(ymax=1000)
fig.savefig(out_dir+expt_id+'_'+temp_out_name+'.png',dpi=300,bbox_inches='tight')
plt.clf()

# HADA average length density of wall puncta
temp_var, temp_out_name, temp_ylab='Wall spots/length', 'wall_spots_length', r'Peripheral puncta/$\mu$m'
temp_iter,temp_xlabel='Condition','Condition'
fig,ax=iter_plotting(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel,showfliers=False)
# ax.set_ylim(ymax=1000)
fig.savefig(out_dir+expt_id+'_'+temp_out_name+'.png',dpi=300,bbox_inches='tight')
plt.clf()



# HADA average number of puncta
temp_var, temp_out_name, temp_ylab='Cell spots', 'cell_spots', 'Cellular puncta'
temp_iter,temp_xlabel='Condition','Condition'
fig,ax=iter_plotting(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel,showfliers=False)
# ax.set_ylim(ymax=1000)
fig.savefig(out_dir+expt_id+'_'+temp_out_name+'.png',dpi=300,bbox_inches='tight')
plt.clf()

# HADA average number of puncta
temp_var, temp_out_name, temp_ylab='Wall spots', 'wall_spots', 'Peripheral puncta'
temp_iter,temp_xlabel='Condition','Condition'
fig,ax=iter_plotting(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel,showfliers=False)
# ax.set_ylim(ymax=1000)
fig.savefig(out_dir+expt_id+'_'+temp_out_name+'.png',dpi=300,bbox_inches='tight')
plt.clf()

# HADA outline average values
temp_var, temp_out_name, temp_ylab='Average outline '+label, 'av_outline_fl', 'Average outline fluorescence'
temp_iter,temp_xlabel='Condition','Condition'
fig,ax=iter_plotting(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel,showfliers=False)
# ax.set_ylim(ymax=1000)
fig.savefig(out_dir+expt_id+'_'+temp_out_name+'.png',dpi=300,bbox_inches='tight')
plt.clf()

# HADA outline average values
temp_var, temp_out_name, temp_ylab='Average '+label, 'av_area_fl', 'Average area fluorescence'
temp_iter,temp_xlabel='Condition','Condition'
fig,ax=iter_plotting(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel,showfliers=False)
# ax.set_ylim(ymax=1000)
fig.savefig(out_dir+expt_id+'_'+temp_out_name+'.png',dpi=300,bbox_inches='tight')
plt.clf()

# HADA outline average values
temp_var, temp_out_name, temp_ylab='Average outline '+label, 'av_outline_fl', 'Average outline fluorescence'
temp_iter,temp_xlabel='Condition','Condition'
fig,ax=iter_plotting(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel,plot_points=False,showfliers=False)
# ax.set_ylim(ymax=1000)
fig.savefig(out_dir+expt_id+'_'+temp_out_name+'_no_data.png',dpi=300,bbox_inches='tight')
plt.clf()

# HADA outline average values
temp_var, temp_out_name, temp_ylab='Average outline '+label, 'av_outline_fl', 'FDAA Fluorescence'
temp_iter,temp_xlabel='Condition',''
excl='5 min 0.5ug/mL Tun v2'
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
fig,ax=iter_plotting(df[df.Condition!=excl], temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel,plot_points=False)
# print(df[df.Condition!='5 min 0.5ug/mL Tun v2'].Condition.unique())
ax.set_xticklabels(time_labels)
# ax.set_ylim(top=4000)
# ax.set_ylim(top=2000)
fig.savefig(out_dir+expt_id+'_'+temp_out_name+'_excl_repeat.eps',dpi=300,bbox_inches='tight')
plt.clf()


# HADA area average values
temp_var, temp_out_name, temp_ylab='Average '+label, 'av_area_fl', 'Average area fluorescence'
temp_iter,temp_xlabel='Condition','Condition'
fig,ax=iter_plotting(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel,plot_points=False,showfliers=False)
# ax.set_ylim(ymax=1000)
fig.savefig(out_dir+expt_id+'_'+temp_out_name+'_no_data.png',dpi=300,bbox_inches='tight')
plt.clf()

# HADA outline integrated values
temp_var, temp_out_name, temp_ylab='Integrated outline '+label, 'int_outline_fl', 'Integrated outline fluorescence'
temp_iter,temp_xlabel='Condition','Condition'
fig,ax=iter_plotting(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel)
fig.savefig(out_dir+expt_id+'_'+temp_out_name+'.png',dpi=300,bbox_inches='tight')
plt.clf()

# HADA outline integrated values
temp_var, temp_out_name, temp_ylab='Average outline '+label, 'av_outline_fl', 'Average outline fluorescence'
temp_iter,temp_xlabel='Condition','Condition'
fig,ax=iter_plotting(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel,temp_hue='Scene')
plt.legend(loc=[1.05,0.0])
fig.savefig(out_dir+expt_id+'_'+temp_out_name+'_by_scene.png',dpi=300,bbox_inches='tight')
plt.clf()

# HADA outline integrated values
temp_var, temp_out_name, temp_ylab='Integrated outline '+label, 'int_outline_fl', 'Integrated outline fluorescence'
temp_iter,temp_xlabel='Condition','Condition'
fig,ax=iter_plotting(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel,temp_hue='Scene')
plt.legend(loc=[1.05,0.0])
fig.savefig(out_dir+expt_id+'_'+temp_out_name+'_by_scene.png',dpi=300,bbox_inches='tight')
plt.clf()

# HADA area integrated values
temp_var, temp_out_name, temp_ylab='Integrated '+label, 'int_area_fl', 'Integrated area fluorescence'
temp_iter,temp_xlabel='Condition','Condition'
fig,ax=iter_plotting(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel)
fig.savefig(out_dir+expt_id+'_'+temp_out_name+'.png',dpi=300,bbox_inches='tight')
plt.clf()

# HADA outline CV values
temp_var, temp_out_name, temp_ylab='CV outline '+label, 'cv_outline_fl', 'CV outline fluorescence'
temp_iter,temp_xlabel='Condition','Condition'
fig,ax=iter_plotting(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel)
fig.savefig(out_dir+expt_id+'_'+temp_out_name+'.png',dpi=300,bbox_inches='tight')
plt.clf()

# HADA outline SD values
temp_var, temp_out_name, temp_ylab='SD outline '+label, 'sd_outline_fl', 'SD outline fluorescence'
temp_iter,temp_xlabel='Condition','Condition'
fig,ax=iter_plotting(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel)
fig.savefig(out_dir+expt_id+'_'+temp_out_name+'.png',dpi=300,bbox_inches='tight')
plt.clf()

# HADA outline CV values
temp_var, temp_out_name, temp_ylab='CV outline '+label, 'cv_outline_fl', 'CV outline fluorescence'
temp_iter,temp_xlabel='Condition','Condition'
fig,ax=iter_plotting(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel,plot_points=False,showfliers=False)
plt.ylim(ymax=2.5)
fig.savefig(out_dir+expt_id+'_'+temp_out_name+'_nopoints.png',dpi=300,bbox_inches='tight')
plt.clf()
sys.stdout.close()
