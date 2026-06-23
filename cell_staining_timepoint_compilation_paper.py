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
norm_cond=0 # Setting this here since this is the standard normalization condition
data_loc=None
# expt_ids = [['/241011_bFB69_Tun_WGA', '/241008_bFB69_Tun_WGA']]
# group_id = 'WGA_ponA_comp'
# pert = 'Tunicamycin added'
# plotting_timepoints=[[-5, 5]]
spots=False
comp_times=None

expt_ids = [['/260415_bBL27_Tun_induction', '/260521_bAH26']]
data_loc=[['data_ssd3', 'data_ssd3']]
group_id = 'pYocH_mVenus'
pert = 'Tunicamycin added'
norm_cond='LB' # Starting condition. Normally don't need this since the typical SC is LB.
plotting_timepoints=[[0,60]]
spots=False


output_dir = './outputs/compiled_data/staining_plots/'
hue='Celltype'
label='mVenus'

# This assumes that cell_staining_timepoint.py has already been run
# df = pd.DataFrame()
for ind in range(len(expt_ids)):
    temp_df1 = pd.DataFrame()
    for expt_id in expt_ids[ind]:
        path = '/Volumes/'+data_loc[ind][expt_ids[ind].index(expt_id)]+'/Barber_Lab/data/'
        temp_path = path+expt_id+expt_id+'_condition_parameters.pkl'
        with open(temp_path, 'rb') as input:
            expt_vals=pickle.load(input)
        data_dir="./outputs"+expt_id
        temp_df=pd.read_pickle(data_dir+expt_id)
        temp_df['Celltype']=expt_vals['Celltype']
        temp_df['expt']=expt_vals['expt']
        temp_df['Date']=expt_vals['Date']
        if not(norm_cond==0):
            temp_df['Normalized Average outline '+label]=temp_df['Average outline '+label]/temp_df[temp_df.Condition==norm_cond]['Average outline '+label].mean()
            temp_df['Normalized Average ' + label] = temp_df['Average ' + label] / \
                                                             temp_df[temp_df.Condition == norm_cond][
                                                                 'Average ' + label].mean()
        if expt_ids[ind].index(expt_id)==[0]:
            temp_df1=pd.DataFrame(columns=temp_df.columns)
        temp_df1=temp_df1.append(temp_df)
    # Adding time as a variable
    timepoint_search_terms = ['LB', '0 min', '5 min', '10 min', '15 min', '20 min', '25 min', '30 min', '35 min', '40 min', '45 min', '50 min', '55 min', '60 min', 'Xylose induced', '120 min', '1h']
    # timepoints = [-5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, -5]
    timepoints = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 0, 125, 60]
    out = []
    for cond in temp_df1.Condition:
        print(cond)
        temp=np.nonzero([temp_cond in cond for temp_cond in timepoint_search_terms])
        if len(temp[0])>1:
            temp1 = np.argmax([len(timepoint_search_terms[ind]) for ind in temp[0]])
            out.append(timepoints[temp[0][temp1]])
        else:
            out.append(timepoints[temp[0][0]])
    temp_df1['Time']=out
    print(temp_df1.expt.unique(),temp_df1.Time.unique())
    temp_true=[temp in plotting_timepoints[ind] for temp in temp_df1.Time]
    temp_df1=temp_df1[temp_true]
    if ind==0:
        df=pd.DataFrame(columns=temp_df1.columns)
    df=df.append(temp_df1)
# print(df[['Condition','Time']].sample(10))

def iter_plotting_v1(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel, temp_hue=None,plot_points=True):
    fig=plt.figure(figsize=[8,4])
    ax=plt.subplot(1,1,1)
    plot=sns.boxplot(x=temp_iter,y=temp_var,data=df,fliersize=0, hue=temp_hue)

    if plot_points:
        plot=sns.stripplot(x=temp_iter,y=temp_var,dodge=True,data=df,edgecolor='black',linewidth=1,alpha=0.2, hue=temp_hue)
    plot.set(xlabel=temp_xlabel,ylabel=temp_ylab)
    # plt.xticks(rotation = 90)
    plt.legend()
    # print('Statistical significances for ', temp_var)
    # temp_iter_vals=df[temp_iter].unique()
    # print(temp_iter_vals)
    # for i0 in range(len(temp_iter_vals)):
        # cond1=temp_iter_vals[i0]
        # for cond2 in temp_iter_vals[i0:]:
            # if cond1!=cond2:
                # x1=df[df[temp_iter]==cond1][temp_var]
                # x2=df[df[temp_iter]==cond2][temp_var]
                # print('Condition 1: '+cond1+',','Condition 2: '+cond2+':', scipy.stats.ttest_ind(x2, x1, axis=0, equal_var=False, nan_policy='propagate')[1]<pval)
    
    return fig,ax

def iter_plotting_v2(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel, temp_hue=None,plot_points=True):
    fig=plt.figure(figsize=[8,4])
    ax=plt.subplot(1,1,1)
    # plot=sns.pointplot(x=temp_iter,y=temp_var,data=df,edgecolor='black',linewidth=1,alpha=0.2, hue='Celltype',capsize=0.2)
    plot=sns.lineplot(x=temp_iter,y=temp_var,data=df, hue='Celltype',estimator=np.mean, err_style='bars', markers=True,err_kws={'capsize':5.0})
    plt.vlines(0,ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],label=pert,linestyle='--',colors='k')
    plot.set(xlabel=temp_xlabel,ylabel=temp_ylab)
    # plt.legend()
    plt.legend()
    # print('Statistical significances for ', temp_var)
    # temp_iter_vals=df[temp_iter].unique()
    # print(temp_iter_vals)
    # for i0 in range(len(temp_iter_vals)):
        # cond1=temp_iter_vals[i0]
        # for cond2 in temp_iter_vals[i0:]:
            # if cond1!=cond2:
                # x1=df[df[temp_iter]==cond1][temp_var]
                # x2=df[df[temp_iter]==cond2][temp_var]
                # print('Condition 1: '+cond1+',','Condition 2: '+cond2+':', scipy.stats.ttest_ind(x2, x1, axis=0, equal_var=False, nan_policy='propagate')[1]<pval)
    
    return fig,ax



# Now we plot the results and save the figures
# These are iterable
temp_vars=['Average outline '+label, 'Average outline '+label, 'Average '+label, 'CV outline '+label, 'SD outline '+label, 'Skew outline '+label]
# temp_vars=['Average outline '+label, 'Average outline '+label, 'Average '+label, 'Normalized Average outline '+label]
temp_out_names=['av_outline_fl', 'av_outline_fl_labs', 'av_area_fl', 'CV_outline_fl', 'SD_outline_fl', 'Skew_outline_fl']
temp_ylabs=['Fluorescence', 'FDAA fluorescence', 'Area fluorescence', 'CV Fluorescence', 'SD fluorescence', 'Skew fluroescence']
if norm_cond!=0:
    temp_vars=temp_vars+['Normalized Average outline '+label]
    temp_out_names=temp_out_names+[ 'normed_av_outline_fl']
    temp_ylabs=temp_ylabs+['FDAA staining (norm)']
if spots:
    temp_vars.append('Cell spots/length')
    temp_out_names.append('cell_spots_length')
    temp_ylabs.append(r'Puncta/$\mu m$')
    temp_vars.append('Wall spots/length')
    temp_out_names.append('wall_spots_length')
    temp_ylabs.append(r'Puncta/$\mu m$')

# These are fixed
# temp_xlabel='Condition'
# temp_iter='Condition'
# pval=0.01
#
# for i0 in range(len(temp_vars)):
    # temp_var, temp_out_name, temp_ylab=temp_vars[i0], temp_out_names[i0], temp_ylabs[i0]
    # fig,ax=iter_plotting_v1(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel, temp_hue=hue)
    # fig.savefig(output_dir+group_id+'_'+temp_out_name+'.png',dpi=300,bbox_inches='tight')
    # plt.clf()

print(df.head())
temp_xlabel='Time (min)'
temp_iter='Time'
pval=0.01
print('I got here')
for i0 in range(len(temp_vars)):
    temp_var, temp_out_name, temp_ylab=temp_vars[i0], temp_out_names[i0], temp_ylabs[i0]
    fig,ax=iter_plotting_v2(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel, temp_hue=hue)
    fig.savefig(output_dir+group_id+'_'+temp_out_name+'_timeplot.png',dpi=300,bbox_inches='tight')
    plt.clf()
    print('Finished plot {0}'.format(i0), temp_var)
# HADA outline average values
temp_var, temp_out_name, temp_ylab='Average outline '+label, 'av_outline_fl_labels', 'FDAA Fluorescence'
temp_iter,temp_xlabel='Condition',''
excl='5 min 0.5ug/mL Tun v2'
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
fig,ax=iter_plotting_v1(df[df.Condition!=excl], temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel,plot_points=False, temp_hue=hue)
# print(df[df.Condition!='5 min 0.5ug/mL Tun v2'].Condition.unique())
# ax.set_xticklabels(time_labels)
# ax.set_ylim(top=4000)
ax.set_ylim(top=2000)
fig.savefig(output_dir+group_id+'_'+temp_out_name+'_excl_repeat.eps',dpi=300,bbox_inches='tight')
plt.clf()

# Now we generate paper quality figures

def iter_plotting_v3(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel, temp_hue=None,plot_points=False):
    fig=plt.figure(figsize=[2.5,2])
    sns.set(font_scale=0.9)
    sns.set_style("ticks")
    ax=plt.subplot(1,1,1)
    hue_order=list(df[temp_hue].unique())
    plot = sns.lineplot(x=temp_iter, y=temp_var, data=df, hue=temp_hue, estimator=np.mean, err_style='bars',errorbar=('ci',95),
                        markers=True, err_kws={'capsize': 5.0}, hue_order=hue_order, linewidth=2.0)
    if plot_points:
        temp_df = df.groupby(['Celltype', 'Time', 'expt']).mean()
        plot = sns.stripplot(x=temp_iter, y=temp_var, data=temp_df, edgecolor='black', linewidth=1, hue=temp_hue,
                             hue_order=hue_order, native_scale=True, alpha=0.3,legend=False)
    plt.vlines(0,ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],linestyle='-',colors='k',lw=0.5)
    plot.set(xlabel=temp_xlabel,ylabel=temp_ylab)
    ax.get_legend().remove()
    # plt.legend()
    # plt.legend(loc=[1.05, 0.2])
    return fig,ax



temp_xlabel='Time (min)'
temp_iter='Time'
pval=0.01
sns.set_style("ticks")
print('hi', df.columns)
print('I got here')


for i0 in range(len(temp_vars)):
    temp_var, temp_out_name, temp_ylab=temp_vars[i0], temp_out_names[i0], temp_ylabs[i0]
    fig,ax=iter_plotting_v3(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel, temp_hue=hue)
    fig.savefig(output_dir+group_id+'_'+temp_out_name+'_timeplot_compact.eps',dpi=300,bbox_inches='tight')
    fig.savefig(output_dir + group_id + '_' + temp_out_name + '_timeplot_compact.png', dpi=300, bbox_inches='tight')
    # plt.legend()
    # plt.legend(loc=[0.6, 0.6])
    plt.legend(loc=[0.01, 0.7])
    fig.savefig(output_dir + group_id + '_' + temp_out_name + '_timeplot_compact_legend.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir + group_id + '_' + temp_out_name + '_timeplot_compact_legend.eps', dpi=300,
                bbox_inches='tight')
    plt.clf()
    print('Finished plot {0}'.format(i0))

for i0 in range(len(temp_vars)):
    temp_var, temp_out_name, temp_ylab=temp_vars[i0], temp_out_names[i0], temp_ylabs[i0]
    fig,ax=iter_plotting_v3(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel, temp_hue=hue, plot_points=True)
    fig.savefig(output_dir+group_id+'_'+temp_out_name+'_timeplot_compact_plotpoints.eps',dpi=300,bbox_inches='tight')
    fig.savefig(output_dir + group_id + '_' + temp_out_name + '_timeplot_compact_plotpoints.png', dpi=300, bbox_inches='tight')
    # plt.legend()
    # plt.legend(loc=[0.6, 0.6])
    # plt.legend(loc=[0.4, 0.55])
    plt.legend(loc=[0.5, 0.4])
    # plt.legend(loc=[0.01, 0.7])
    fig.savefig(output_dir + group_id + '_' + temp_out_name + '_timeplot_compact_legend_plotpoints.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir + group_id + '_' + temp_out_name + '_timeplot_compact_legend_plotpoints.pdf', dpi=300,
                bbox_inches='tight')
    plt.clf()
    print('Finished plot {0}'.format(i0))

for i0 in range(len(temp_vars)):
    temp_var, temp_out_name, temp_ylab=temp_vars[i0], temp_out_names[i0], temp_ylabs[i0]
    fig,ax=iter_plotting_v3(df, temp_var,temp_out_name,temp_ylab,temp_iter,temp_xlabel, temp_hue='Date')
    fig.savefig(output_dir+group_id+'_'+temp_out_name+'_timeplot_compact.eps',dpi=300,bbox_inches='tight')
    # plt.legend()
    plt.legend(loc=[0.6, 0.6])
    fig.savefig(output_dir + group_id + '_' + temp_out_name + '_timeplot_compact_legend_expt.png', dpi=300, bbox_inches='tight')
    plt.clf()
    print('Finished plot {0}'.format(i0))

# Now we make violin plots of average cell intensity grouped by cell type and by tunicamycin condition, normalized to LB.
fig=plt.figure(figsize=[2.5,2])
sns.set(font_scale=0.9)
sns.set_style("ticks")
ax=plt.subplot(1,1,1)
sns.violinplot(data=df,x='Celltype', y='Normalized Average mVenus', hue = 'Condition')
plt.legend(loc=[1.0,0.0],title='Tunicamycin treatment')
fig.savefig(output_dir + group_id + '_' + temp_out_name + '_violinplot_av_norm.pdf',bbox_inches='tight')
plt.clf()

fig=plt.figure(figsize=[2.5,2])
sns.set(font_scale=0.9)
sns.set_style("ticks")
ax=plt.subplot(1,1,1)
sns.violinplot(data=df,x='Celltype', y='Average mVenus', hue = 'Condition')
plt.legend(loc=[1.0,0.0],title='Tunicamycin treatment')
fig.savefig(output_dir + group_id + '_' + temp_out_name + '_violinplot_av.pdf',bbox_inches='tight')
plt.clf()

with open(output_dir + group_id + '_' + temp_out_name +'.txt', 'w') as f:
    print(df.groupby('Celltype').describe(), file=f)  # Python 3.x
    if not(comp_times is None):
        for time in comp_times:
            for type in df.Celltype[df.Time==time].unique():
                for type1 in df.Celltype[df.Time == time].unique():
                    if type!=type1:
                        print('Comparing celltypes ',type, ' and ', type1, file=f)
                        print('Time ={0}'.format(time), file=f)
                        xv=np.asarray(df[(df.Celltype==type)*(df.Time==time)]['Cell spots/length'])
                        yv=np.asarray(df[(df.Celltype==type1)*(df.Time==time)]['Cell spots/length'])
                        # print(xv, yv, file=f)
                        print(type, len(xv), file=f)
                        print(type1, len(yv), file=f)
                        print(scipy.stats.ttest_ind(xv,yv), file=f)
