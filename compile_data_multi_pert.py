# import scipy
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

out_path='/Users/barber.527/Documents/GitHub/WTA_hydrolysis/outputs/compiled_data/'
in_path='/Users/barber.527/Documents/GitHub/WTA_hydrolysis/outputs/'

##################### Experiment specifics

##################### Values
tstep=20.0
# Note: If t_pert<0 then normalization is not going to be very helpful.

# expt_ids = ['/260506_bFB66_Tun_NLS_PBS_Mg_v2', '/260519_bFB66_Tun_NLS_PBS_Mg_v2', '/260519_bFB66_Tun_NLS_PBS_Mg']
# expt_label = 'bFB66_Tun_NLS_PBS_Mg'
# Celltype = r'WT Tun'
# t_pert = [(17-1)*20.0, (107-1)*20.0, (113-1)*20.0, (128-1)*20.0, (143-1)*20.0, (158-1)*20.0] # timepoints 32 and 40
# pert_comp_ind=1

# expt_ids = ['/260505_bFB66_NLS_PBS_Mg', '/260504_bFB66_NLS_PBS_Mg', '/260504_bFB66_NLS_PBS_Mg_v2', '/260520_bFB66_NLS_PBS_Mg']
# expt_label = 'bFB66_NLS_PBS_Mg'
# Celltype = r'WT'
# t_pert = [(32-1)*20.0, (38-1)*20.0, (53-1)*20.0, (68-1)*20.0, (83-1)*20.0]
# pert_comp_ind=0

var_ls=['sgr_l', 'sgr_sa', 'width']
yscale = [60*60.0, 60*60.0, 1.0, 1.0, 1.0]
# saving the experimental parameters
temp = {'expt_label':expt_label, 't_perts':t_pert, 'expt_ids':expt_ids, 'tstep':tstep,'vars':var_ls, 'Celltype':Celltype}
if 't_pert1' in locals():
    temp['t_pert1']=t_pert1
temp_path = out_path+expt_label+'_condition_parameters.pkl'
with open(temp_path, 'wb') as output:  # Overwrites any existing file.
    pickle.dump(temp, output, pickle.HIGHEST_PROTOCOL)

def compile_values_v1(temp_yvs, temp_xvs, show_out=False):
    # This function takes lists of arrays for each experiment and aligns them
    # this gives the maximal range for each data set around the perturbation timepoint.
    # Assumes the same timestep in each case
    temp1=[temp[0] for temp in temp_xvs]
    temp2=[temp[-1] for temp in temp_xvs]
    temp_xv=np.linspace(np.amin(temp1), np.amax(temp2), int((np.amax(temp2)-np.amin(temp1))/tstep+1))
    temp_out=np.empty([np.sum([int(temp.shape[0]) for temp in temp_yvs]), len(temp_xv)])
    temp_out[:]=np.nan
    num_points=0
    for i0 in range(len(temp_yvs)):
        start_ind = np.nonzero(temp_xv==temp_xvs[i0][0])[0][0]
        end_ind = np.nonzero(temp_xv==temp_xvs[i0][-1])[0][0]
        if show_out:
            print(start_ind, end_ind)
            print(temp_out.shape)
            print(temp_yvs[i0].shape)
            print(num_points)
        temp_out[num_points:num_points+temp_yvs[i0].shape[0], start_ind:end_ind+1]=temp_yvs[i0][:,:]
        num_points+=temp_yvs[i0].shape[0]
    return(temp_out, temp_xv)


def compile_values_v2(temp_yvs, temp_xvs, show_out=False):
    # This function takes lists of arrays for each experiment and aligns them conservatively, excluding timepoints with
    # incomplete coverage
    # Assumes the same timestep in each case
    temp1=[temp[0] for temp in temp_xvs]
    temp2=[temp[-1] for temp in temp_xvs]
    temp_xv = np.linspace(np.amax(temp1), np.amin(temp2), int((np.amin(temp2) - np.amax(temp1)) / tstep + 1))
    temp_out=np.empty([np.sum([int(temp.shape[0]) for temp in temp_yvs]), len(temp_xv)])
    temp_out[:]=np.nan
    num_points=0
    for i0 in range(len(temp_yvs)):
        start_ind = np.nonzero(temp_xvs[i0]>=temp_xv[0])[0][0]
        end_ind = np.nonzero(temp_xvs[i0]>=temp_xv[-1])[0][0]
        if show_out:
            print(start_ind, end_ind)
            print(temp_out.shape)
            print(temp_yvs[i0].shape)
            print(num_points)
        temp_out[num_points:num_points+temp_yvs[i0].shape[0], :] = temp_yvs[i0][:, start_ind:end_ind+1]
        num_points += temp_yvs[i0].shape[0]
    return(temp_out, temp_xv)

# Repeating the above but without the normalization step
for var in var_ls:
    xvs,yvs=[],[]
    for expt_id in expt_ids:
        temp=np.load(in_path+expt_id+expt_id+'_time.npy')
        temp -=t_pert[pert_comp_ind]
        xvs.append(temp)
        temp1=np.load(in_path+expt_id+expt_id+'_'+var+'.npy')*yscale[var_ls.index(var)]
        yvs.append(temp1)
    yv, xv=compile_values_v1(yvs,xvs)
    np.save(out_path+expt_label+'_'+var+'.npy', yv)
np.save(out_path+expt_label+'_time.npy', xv)

for var in var_ls:
    xvs,yvs=[],[]
    for expt_id in expt_ids:
        temp=np.load(in_path+expt_id+expt_id+'_time.npy')
        temp -=t_pert[pert_comp_ind]
        xvs.append(temp)
        temp1=np.load(in_path+expt_id+expt_id+'_'+var+'.npy')*yscale[var_ls.index(var)]
        yvs.append(temp1)
    yv, xv=compile_values_v2(yvs,xvs)
    # Filtering out rows that were only tracked in the timepoints we have now removed.
    # temp_yv=np.sum(np.isnan(yv),axis=1)
    # yv1=yv[np.nonzero(temp_yv!=yv.shape[1])[0],:]
    # print(yv1)
    # print(yv[np.nonzero(temp_yv==yv.shape[1])[0],:])
    # print(yv.shape, yv1.shape, xv.shape)
    np.save(out_path+expt_label+'_'+var+'_conservative.npy', yv)
np.save(out_path+expt_label+'_time_conservative.npy', xv)