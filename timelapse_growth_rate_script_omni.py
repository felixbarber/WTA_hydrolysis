import scipy
import sys
from scipy import io
import numpy as np
import numpy.matlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import pandas as pd
from numpy import matlib
import seaborn as sns
import math
import os
import scipy.optimize
import scipy.io as sio
from scipy import stats
import scipy.ndimage

sns.set(font_scale=1.5)
sns.set_style("whitegrid")

# This script can be applied to generate a set of basic plots for an experiment with two, sequential antibiotic shocks.
# It is specifically designed so that it allows for different timesteps in different parts of the experiment.

remove_non_growing_cells = True
remove_fliers = True

########################################################################################################
# User inputs
########################################################################################################

expt_id = '/260217_bFB292_IPTG_Mg'
tsteps = 286
dt = 20.0*np.ones(tsteps-1) # time interval between timepoints in seconds
labels = ['1mM IPTG added', r'MgCl$_2$ added']
t_pert = [5*60.0, 65*60.0]
max_time_truncation = 140*60.0
scene_nums = 4
max_width = 4.0
min_width = 0.5
min_length = 2.0
perform_ttest=False
base_path='/Volumes/data_ssd2/Barber_Lab/data'
data_path='/Volumes/data_ssd2/Barber_Lab/data'


# Generic code to be run.
initial_time_plotting = 15  # number of minutes for which to plot the growth rate values of each scene
thresh = 0.00005  # threshold for the average growth rate below which we preclude cells from analysis. This is set very
# low to make sure that cells are actually not growing at all.
window = 2 # number of time points on either side with which to calculate the local slope
flier_thresh = 2.0 # threshold for number of iqrs away from median growth rate beyond which we preclude cells from analysis.
w_flier_thresh = 2.0 # threshold for cell width relative rate of change beyond which we preclude cells from analysis.
tvec=np.cumsum(dt)
tvec=np.insert(tvec,0,0.0)
max_tstep_truncation = len(tvec) # this is the point at which the analysis truncates if you want it to be before the
# end of the dataset
# filtering to only include contigs of so many timepoints or more.
cutoff=50

# making file structures
if not os.path.exists('./outputs'+expt_id):
    os.mkdir('./outputs'+expt_id)
linestyles = ['-','-.','--']
sys.stdout = open('./outputs'+expt_id+expt_id+'_outputs.txt', "w")
# Loading the data (same as antibiotic_growth_inhibition_tester_combined_shock_v2.py)
temp = []
temp1 = []
temp2 = []
temp3 = []
for i0 in range(1, scene_nums+1):
    temp_name = expt_id+'_s{:03d}'.format(i0)
    temp_path = data_path+expt_id+temp_name+'_1_a'+temp_name+'_BT_felix.mat'
    data=scipy.io.loadmat(temp_path)
    temp.append(np.asarray(data['lcell']))
    temp1.append(np.asarray(data['wcell']))
    temp2.append(np.asarray(data['sacell']))
    temp3.append(i0*np.ones(data['lcell'].shape[0]))
lcell = np.concatenate(temp,axis=0)
wcell = np.concatenate(temp1,axis=0)
sacell = np.concatenate(temp2,axis=0)
scene_num_tracker = np.concatenate(temp3,axis=0)
# filtering the data based on dimensions
sacell=sacell[:, :max_tstep_truncation]
lcell=lcell[:, :max_tstep_truncation]
wcell=wcell[:, :max_tstep_truncation]
sacell[np.isnan(lcell)]=np.nan # making sure that the surface area doesn't include fixed objects that have been
# filtered successfully within lcell and wcell.
print(lcell.shape,wcell.shape,sacell.shape, max_tstep_truncation)
time=tvec[:max_tstep_truncation]


###################################################################
cutoff_time = len(time)-1
time_truncated = time[:cutoff_time]
print("cutoff time = ", cutoff_time)

for i0 in range(lcell.shape[0]): # for each cell trace
    temp=(~np.isnan(lcell[i0,:])).astype(int) # 1 for cell, 0 for nan
    inds1=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==1) # beginning of each trace
    inds2=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==-1) # end point of each trace
    lens=(inds2[0]-inds1[0]).tolist() # gives the length of each contig
    for i1 in range(len(lens)):
        if lens[i1]<cutoff:
            lcell[i0,inds1[0][i1]:inds1[0][i1]+lens[i1]]=np.nan
            wcell[i0,inds1[0][i1]:inds1[0][i1]+lens[i1]]=np.nan
            sacell[i0,inds1[0][i1]:inds1[0][i1]+lens[i1]]=np.nan

# fitting to find the surface area growth rate at each point.

sgr_sa = np.zeros(sacell.shape)
sgr_sa[:, :] = np.nan
sgr_l = np.zeros(lcell.shape)
sgr_l[:, :] = np.nan
sgr_w = np.zeros(wcell.shape)
sgr_w[:, :] = np.nan
window = 5
window_sa = 15 # note that the larger window here all but ensures that the measured SA growth rate will be less
# time-sensitive to changes in growth rate.
for i0 in range(sacell.shape[0]):  # number of cells
    for i1 in range(sacell.shape[1]):  # timepoints
        if ~np.isnan(lcell[i0, i1]):  # if this timepoint is not a nan value
            # for specific length growth rate
            temp_x = time[max([i1 - window, 0]):min([i1 + window, len(time)])] # window of length 5 for all regions poss
            temp_y = lcell[i0, max([i1 - window, 0]):min([i1 + window, len(time)])]
            temp_x = temp_x[~np.isnan(temp_y)] # select only points that have actual data
            temp_y = temp_y[~np.isnan(temp_y)]
            temp_vals = scipy.stats.linregress(temp_x, temp_y)
            sgr_l[i0, i1] = temp_vals[0] / np.nanmean(temp_y)
            # for specific width growth rate
            temp_x = time[
                max([i1 - window, 0]):min([i1 + window, len(time)])]  # window of length 5 for all regions poss
            temp_y = wcell[i0, max([i1 - window, 0]):min([i1 + window, len(time)])]
            temp_x = temp_x[~np.isnan(temp_y)]  # select only points that have actual data
            temp_y = temp_y[~np.isnan(temp_y)]
            temp_vals = scipy.stats.linregress(temp_x, temp_y)
            sgr_w[i0, i1] = temp_vals[0] / np.nanmean(temp_y)
            # for specific SA growth rate
            temp_x = time[max([i1 - window_sa, 0]):min([i1 + window_sa, len(time)])]
            temp_y = sacell[i0, max([i1 - window_sa, 0]):min([i1 + window_sa, len(time)])]
            temp_x = temp_x[~np.isnan(temp_y)]
            temp_y = temp_y[~np.isnan(temp_y)]
            temp_vals = scipy.stats.linregress(temp_x, temp_y)
            sgr_sa[i0, i1] = temp_vals[0] / np.nanmean(temp_y)


###################################################################

# Filtering to remove cells that simply don't grow throughout the whole timelapse
if remove_non_growing_cells:
    filt_inds=np.nonzero(np.nanmean(sgr_l,axis=1)<thresh)
    sgr_l[filt_inds,:]=np.nan
    sgr_w[filt_inds, :] = np.nan
    wcell[filt_inds,:]=np.nan
    sgr_sa[filt_inds,:]=np.nan
    lcell[filt_inds,:]=np.nan

# Filtering to remove cells that change their width too fast
if remove_fliers:
    sgrmed = np.nanmedian(sgr_w,axis=0)
    sgrstd = scipy.stats.iqr(sgr_w,axis=0,nan_policy='omit')
    fliers = np.absolute(sgr_w-np.tile(sgrmed,[sgr_w.shape[0],1]))/np.tile(sgrstd,[sgr_w.shape[0],1])>w_flier_thresh
    flier_inds = np.nonzero(np.amax(fliers, axis=1))
    sgr_w[flier_inds, :] = np.nan
    sgr_l[flier_inds, :] = np.nan
    sgr_w[flier_inds, :] = np.nan
    wcell[flier_inds, :] = np.nan
    sgr_sa[flier_inds, :] = np.nan
    lcell[flier_inds, :] = np.nan
    # sgr_w[np.nonzero(fliers)]=np.nan
    # sgr_l[np.nonzero(fliers)] = np.nan
    # sgr_w[np.nonzero(fliers)] = np.nan
    # wcell[np.nonzero(fliers)] = np.nan
    # sgr_sa[np.nonzero(fliers)] = np.nan
    # lcell[np.nonzero(fliers)] = np.nan

    sgrmed = np.nanmedian(sgr_l,axis=0)
    sgrstd = scipy.stats.iqr(sgr_l,axis=0,nan_policy='omit')
    fliers = np.absolute(sgr_l-np.tile(sgrmed,[sgr_l.shape[0],1]))/np.tile(sgrstd,[sgr_l.shape[0],1])>flier_thresh
    flier_inds = np.nonzero(np.amax(fliers, axis=1))
    sgr_w[flier_inds, :] = np.nan
    sgr_l[flier_inds, :] = np.nan
    sgr_w[flier_inds, :] = np.nan
    wcell[flier_inds, :] = np.nan
    sgr_sa[flier_inds, :] = np.nan
    lcell[flier_inds, :] = np.nan

###################################################################
# plotting the cell length growth rate traces
fig=plt.figure(figsize=[8, 5])
xv=time
for i0 in range(lcell.shape[0]):
    yv = lcell[i0, :]
    print(yv)
    plt.plot(xv,yv,alpha=0.4)
ax=plt.gca()
for i0 in range(len(t_pert)):
    plt.vlines(t_pert[i0],ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],label=labels[i0],linestyle=linestyles[i0])
plt.legend(loc=[1.02,0.2])
plt.ylabel(r'Length ($\mu m$)')
plt.xlabel('Time (s)')
fig.savefig('./outputs'+expt_id+expt_id+'_cell_lengths.png',dpi=150,bbox_inches='tight')
plt.clf()


# Plotting the sgr on a nice time axis
sgr=np.zeros(sgr_l.shape)
sgr[:,:]=sgr_l[:,:]
sgr*=3600.0
xv=time/60.0
fig=plt.figure(figsize=[8,5])
for i0 in range(sgr.shape[0]):
    temp=(~np.isnan(sgr[i0,:])).astype(int)
    inds1=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==1)[0] # beginning of each trace
    inds2=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==-1)[0] # end point of each trace
    for ind in range(len(inds1)):
        yv=sgr[i0,inds1[ind]:inds2[ind]]
        xv1 = xv[inds1[ind]:inds2[ind]]
        plt.xlim(left=0.0,right=np.amax(xv))
        plt.plot(xv1,yv,alpha=0.2,color='b')
yv=np.nanmedian(sgr,axis=0)
err=np.nanstd(sgr,axis=0)/np.sqrt(np.sum(~np.isnan(sgr),axis=0))
plt.fill_between(xv,yv-err,yv+err,alpha=0.5,color='r')
plt.plot(xv,yv,label= 'Smoothed median',color='k',lw=3.0)
# plt.ylim(ymin=0)
ax=plt.gca()
for i0 in range(len(t_pert)):
    plt.vlines(t_pert[i0]/60.0,ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],label=labels[i0],linestyle=linestyles[i0])
plt.legend()
plt.xlabel('Time (min)')
plt.ylabel(r'$\frac{1}{l}\frac{dl}{dt}$ ($h^{-1}$)')
fig.savefig('./outputs/'+expt_id+expt_id+'_sgr_truncated.png',bbox_inches='tight',dpi=150)
plt.clf()



# Plotting the sgr on its original axis
sgr=np.zeros(sgr_l.shape)
sgr[:,:]=sgr_l[:,:]

# Filtering to remove cells that simply don't grow throughout the whole timelapse
if remove_non_growing_cells:
    filt_inds=np.nonzero(np.nanmean(sgr,axis=1)<thresh)
    sgr[filt_inds, :] = np.nan

sgrmed = np.nanmedian(sgr,axis=0)
sgrstd = scipy.stats.iqr(sgr,axis=0,nan_policy='omit')


xv=time
fig=plt.figure(figsize=[8,5])
for i0 in range(sgr.shape[0]):
    temp=(~np.isnan(sgr[i0,:])).astype(int)
    inds1=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==1)[0] # beginning of each trace
    inds2=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==-1)[0] # end point of each trace
    for ind in range(len(inds1)):
        yv=sgr[i0,inds1[ind]:inds2[ind]]
        xv1 = xv[inds1[ind]:inds2[ind]]
        plt.xlim(left=0.0,right=np.amax(xv))
        plt.plot(xv1,yv,alpha=0.2,color='b')
yv=np.nanmedian(sgr,axis=0)
err=np.nanstd(sgr,axis=0)/np.sqrt(np.sum(~np.isnan(sgr),axis=0))
plt.fill_between(xv,yv-err,yv+err,alpha=0.5,color='r')
plt.plot(xv,yv,label= 'Smoothed median',color='k',lw=3.0)
plt.ylim(ymin=0)
ax=plt.gca()
for i0 in range(len(t_pert)):
    plt.vlines(t_pert[i0],ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],label=labels[i0],linestyle=linestyles[i0])
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel(r'$\frac{1}{l}\frac{dl}{dt}$ ($s^{-1}$)')
# plt.title('Growth rate response to antibiotic perturbation')
fig.savefig('./outputs/'+expt_id+expt_id+'_sgr.png',bbox_inches='tight',dpi=150)
plt.xlim(xmax=xv[cutoff_time])
fig.savefig('./outputs/'+expt_id+expt_id+'_sgr_truncated_original.png',bbox_inches='tight',dpi=150)
plt.show()
plt.clf()

np.save('./outputs/'+expt_id+expt_id+'_sgr_l.npy',sgr)
print("DIMENSIONS", sgr.shape, scene_num_tracker.shape)
np.save('./outputs/'+expt_id+expt_id+'_scene_nums.npy',scene_num_tracker)
np.save('./outputs/'+expt_id+expt_id+'_time.npy',time)
np.save('./outputs/'+expt_id+expt_id+'_width.npy',wcell)

# Now we just briefly plot the initial growth rate by chamber number for the first 10 mins
temp_cols=["Scene","Initial Growth Rate"]
temp_df=pd.DataFrame(columns=temp_cols)
temp_cutoff = np.nonzero(time > initial_time_plotting * 60)[0][0]
temp_lcutoff=np.nonzero(time>2*60)[0][0]
print(temp_cutoff)
for cell in range(sgr.shape[0]):
    temp_gr=np.nanmean(sgr[cell,temp_lcutoff:temp_cutoff])
    temp_df1=pd.DataFrame(columns=temp_cols, data=[[str(int(scene_num_tracker[cell])),temp_gr]])
    temp_df=pd.concat([temp_df,temp_df1])
print(temp_df.Scene.unique())
fig=plt.figure(figsize=[8,5])
temp_df=temp_df[[~np.isnan(obj) for obj in temp_df["Initial Growth Rate"]]]
sns.boxplot(data=temp_df,x="Scene",y="Initial Growth Rate")
fig.savefig('./outputs/'+expt_id+expt_id+'_sgr_initial_by_scene.png',bbox_inches='tight',dpi=150)
plt.clf()


# Plotting a hexbin version of specific cell growth rates

xv=time
fig=plt.figure(figsize=[8,5])
yv_out = np.array([])
for i0 in range(sgr.shape[0]):
    yv=sgr[i0,:]
    xv1 = xv[:]
    if i0 ==0:
        yv_out=yv
        xv_out=xv1
    else:
        yv_out=np.concatenate([yv,yv_out])
        xv_out = np.concatenate([xv1, xv_out])
plt.xlim(left=0.0,right=np.amax(xv_out))
plt.hexbin(xv_out,yv_out,cmap ='plasma')
yv=scipy.ndimage.gaussian_filter(np.nanmedian(sgr,axis=0),sigma=1)
plt.plot(xv,yv,label= 'Smoothed median',color='g',lw=3.0)
# plt.ylim(ymin=np.nanmedian(sgrmed)-4*np.nanmedian(sgrstd),ymax=np.nanmedian(sgrmed)+6*np.nanmedian(sgrstd))
ax=plt.gca()
for i0 in range(len(t_pert)):
    plt.vlines(t_pert[i0],ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],label=labels[i0],linestyle=linestyles[i0],color='y')
plt.legend(loc=[0.5,0.6])
plt.xlabel('Time (s)')
plt.ylabel(r'$\frac{1}{l}\frac{dl}{dt}$ ($s^{-1}$)')
fig.savefig('./outputs/'+expt_id+expt_id+'_sgr_hexbin.png',bbox_inches='tight',dpi=150)
plt.clf()

###################################################################
# plotting the traces of cell widths

fig=plt.figure(figsize=[8,5])
xv=time
for i0 in range(lcell.shape[0]):
    yv=wcell[i0,:]
    plt.plot(xv,yv,alpha=0.4)
ax=plt.gca()
for i0 in range(len(t_pert)):
    plt.vlines(t_pert[i0],ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],label=labels[i0],linestyle=linestyles[i0])
plt.legend(loc=[1.02,0.2])
plt.ylabel(r'Width ($\mu m$)')
plt.xlabel('Time (s)')
fig.savefig('./outputs'+expt_id+expt_id+'_cell_widths.png',dpi=150,bbox_inches='tight')
plt.clf()

# plotting a nicer set of traces of cell widths

wmed = np.nanmedian(wcell,axis=0)
wstd = np.nanstd(wcell,axis=0)

xv=time[:]
fig=plt.figure(figsize=[8,5])
for i0 in range(wcell.shape[0]):
    temp=(~np.isnan(wcell[i0,:])).astype(int)
    inds1=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==1)[0] # beginning of each trace
    inds2=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==-1)[0] # end point of each trace
    for ind in range(len(inds1)):
        yv=wcell[i0,inds1[ind]:inds2[ind]]
        xv1 = xv[inds1[ind]:inds2[ind]]
        plt.xlim(left=0.0,right=np.amax(xv))
        plt.plot((xv1-t_pert[0])/60.0,yv,alpha=0.1,color='b')
plt.xlim(left=-t_pert[0]/60.0,right=np.amax(xv)/60.0)
plt.plot((xv-t_pert[0])/60.0,wmed,label= 'median',color='r',lw=3.0)
plt.fill_between((xv-t_pert[0])/60.0,wmed-wstd,wmed+wstd,alpha=0.2,color='r',lw=3.0)
# plt.ylim(ymin=np.nanmedian(sgrmed)-0.5*np.nanmedian(sgrstd),ymax=np.nanmedian(sgrmed)+1.0*np.nanmedian(sgrstd))
ax=plt.gca()
for i0 in range(len(t_pert)):
    plt.vlines((t_pert[i0]-t_pert[0])/60.0,ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],label=labels[i0],linestyle=linestyles[i0])
plt.legend(loc=[1.02,0.2])
plt.xlabel('Time (min)')
plt.ylabel(r'Cell width ($\mu m$)')
# plt.title('Cell width in response to antibiotic perturbation')
fig.savefig('./outputs/'+expt_id+expt_id+'_widths_dist_v2.png',bbox_inches='tight',dpi=150)
plt.clf()


###################################################################
# plotting the traces of cell SA

fig=plt.figure(figsize=[8,5])
xv=time
for i0 in range(lcell.shape[0]):
    yv=sacell[i0,:]
    plt.plot(xv,yv,alpha=0.4)
ax=plt.gca()
for i0 in range(len(t_pert)):
    plt.vlines(t_pert[i0],ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],label=labels[i0],linestyle=linestyles[i0])
plt.legend(loc=[1.02,0.2])
# plt.ylim(ymin=0.0,ymax=np.nanmedian(sacell.flatten())+4*np.nanmedian(sacell.flatten()))
plt.ylabel(r'Surface Area ($\mu m^2$)')
plt.xlabel('Time (s)')
fig.savefig('./outputs'+expt_id+expt_id+'_cell_sa.png',dpi=150,bbox_inches='tight')
plt.clf()

###################################################################
sgrmed = np.nanmedian(sgr,axis=0)
sgrstd = np.nanstd(sgr,axis=0)
print('Perturbation index', int(t_pert[0]/dt[0]))
print('Cell length growth rate values')
temp=np.nanmean(sgr[:,:int(t_pert[0]/dt[0])],axis=1)*60**2
max_trunc=int(t_pert[0]/dt[0])+int(30*60.0/dt[0]) # gives the required time delay before calculating the final growth rate
print('Pre shock growth rate median cell average', np.around(np.nanmedian(temp),4))
print('Pre shock growth rate SD cell average', np.around(np.nanstd(temp),4))
print('Pre shock growth rate SEM', np.around(np.nanstd(temp)/np.sqrt(np.sum(~np.isnan(temp))),4))
temp=np.nanmean(sgr[:,max_trunc:],axis=1)*60**2
print('Post shock growth rate median cell average', np.around(np.nanmedian(temp),4))
print('Post shock growth rate SD cell average', np.around(np.nanstd(temp),4))
print('Post shock growth rate SEM', np.around(np.nanstd(temp)/np.sqrt(np.sum(~np.isnan(temp))),4))

###################################################################
# Plotting the SA SGR
flier_thresh=2.0
cutoff_thresh=0.0001
sgr=np.zeros(sgr_sa.shape)
sgr[:,:]=sgr_sa[:,:]

# Filtering to remove cells that simply don't grow throughout the whole timelapse
if remove_non_growing_cells:
    filt_inds=np.nonzero(np.nanmean(sgr,axis=1)<thresh)
    sgr[filt_inds,:]=np.nan
sgrmed = np.nanmedian(sgr,axis=0)
sgrstd = scipy.stats.iqr(sgr,axis=0,nan_policy='omit')

xv=time
fig=plt.figure(figsize=[8,5])
for i0 in range(sgr.shape[0]):
    temp=(~np.isnan(sgr[i0,:])).astype(int)
    inds1=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==1)[0] # beginning of each trace
    inds2=np.nonzero(np.diff(np.concatenate([[0],temp, [0]]))==-1)[0] # end point of each trace
    for ind in range(len(inds1)):
        yv=sgr[i0,inds1[ind]:inds2[ind]]
        xv1 = xv[inds1[ind]:inds2[ind]]
        plt.xlim(left=0.0,right=np.amax(xv))
        plt.plot(xv1,yv,alpha=0.2,color='b')
yv=np.nanmedian(sgr,axis=0)
err=np.nanstd(sgr,axis=0)/np.sqrt(np.sum(~np.isnan(sgr),axis=0))
plt.ylim(ymin=0)
plt.fill_between(xv,yv-err,yv+err,alpha=0.5,color='r')
plt.plot(xv,yv,label= 'Median',color='k',lw=3.0)

# plt.ylim(ymin=np.nanmedian(sgrmed)-4*np.nanmedian(sgrstd),ymax=np.nanmedian(sgrmed)+6*np.nanmedian(sgrstd))
ax=plt.gca()
for i0 in range(len(t_pert)):
    plt.vlines(t_pert[i0],ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],label=labels[i0],linestyle=linestyles[i0])
plt.legend(loc=[1.02,0.2])
plt.xlabel('Time (s)')
plt.ylabel(r'$\frac{1}{S}\frac{dS}{dt}$ ($s^{-1}$)')
# plt.title('Growth rate response to antibiotic perturbation')
fig.savefig('./outputs/'+expt_id+expt_id+'_sgr_SA.png',bbox_inches='tight',dpi=300)
plt.show()
np.save('./outputs/'+expt_id+expt_id+'_sgr_sa.npy',sgr)
plt.clf()

# Plotting a streamlined version of the data
fig=plt.figure(figsize=[8,5])
sgr_sa_smoothed=np.nanmedian(sgr_sa,axis=0)
sgr_sa_iqr_smoothed=scipy.stats.iqr(sgr_sa,axis=0,nan_policy='omit')
plt.plot(time,sgr_sa_smoothed,label='Smoothed Median',lw=3.0)
plt.fill_between(time,sgr_sa_smoothed-sgr_sa_iqr_smoothed,
                 sgr_sa_smoothed+sgr_sa_iqr_smoothed,alpha=0.2)
plt.ylim(ymin=-0.0005,ymax=0.0015)
ax=plt.gca()
for i0 in range(len(t_pert)):
    plt.vlines(t_pert[i0],ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],label=labels[i0],linestyle=linestyles[i0])
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel(r'$\frac{1}{S}\frac{dS}{dt}$ ($s^{-1}$)')
# plt.title('Growth rate response to antibiotic perturbation')
fig.savefig('./outputs/'+expt_id+expt_id+'_sgr_SA_iqr.png',bbox_inches='tight',dpi=300)
plt.clf()

# Plotting growth rate on better axis
trunc_time=time[np.nonzero(time<=max_time_truncation)[0]]
trunc_sgr_sa = sgr_sa[:,np.nonzero(time<=max_time_truncation)[0]]*60*60
fig=plt.figure(figsize=[8,5])
sgr_sa_smoothed=np.nanmedian(trunc_sgr_sa,axis=0)
err=np.nanstd(trunc_sgr_sa,axis=0)/np.sqrt(np.sum(~np.isnan(trunc_sgr_sa),axis=0))
plt.plot(trunc_time/60,sgr_sa_smoothed,label='Median',lw=3.0)
plt.fill_between(trunc_time/60,sgr_sa_smoothed-err,
                 sgr_sa_smoothed+err,alpha=0.2)
plt.ylim(ymin=0.0,ymax=3)
ax=plt.gca()
for i0 in range(len(t_pert)):
    plt.vlines(t_pert[i0]/60,ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],label=labels[i0],linestyle=linestyles[i0])
plt.legend()
plt.xlabel('Time (min)')
plt.ylabel(r'$\frac{1}{S}\frac{dS}{dt}$ ($h^{-1}$)')
# plt.title('Growth rate response to antibiotic perturbation')
fig.savefig('./outputs/'+expt_id+expt_id+'_sgr_SA_sem.png',bbox_inches='tight',dpi=300)
print('SA growth rate')
max_trunc=int(t_pert[0]/dt[0])+int(30*60.0/dt[0]) # gives the required time delay before calculating the final growth rate
temp=np.nanmean(trunc_sgr_sa[:,:int(t_pert[0]/dt[0])],axis=1)
print('Pre shock growth rate median cell average', np.around(np.nanmedian(temp),4))
print('Pre shock growth rate SD cell average', np.around(np.nanstd(temp),4))
print('Pre shock growth rate SEM', np.around(np.nanstd(temp)/np.sqrt(np.sum(~np.isnan(temp))),4))
temp=np.nanmean(trunc_sgr_sa[:,max_trunc:],axis=1)
print('Post shock growth rate median cell average', np.around(np.nanmedian(temp),4))
print('Post shock growth rate SD cell average', np.around(np.nanstd(temp),4))
print('Post shock growth rate SEM', np.around(np.nanstd(temp)/np.sqrt(np.sum(~np.isnan(temp))),4))
sys.stdout.close()
plt.clf()

